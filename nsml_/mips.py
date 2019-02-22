import argparse
import json
import os
import random
from collections import namedtuple
from time import time

import h5py
import numpy as np
import faiss
from tqdm import tqdm


def int8_to_float(num, offset, factor):
    return num.astype(np.float32) / factor + offset


def adjust(each):
    last = each['context'].rfind(' [PAR] ', 0, each['start_pos'])
    last = 0 if last == -1 else last + len(' [PAR] ')
    next = each['context'].find(' [PAR] ', each['end_pos'])
    next = len(each['context']) if next == -1 else next
    each['context'] = each['context'][last:next]
    each['start_pos'] -= last
    each['end_pos'] -= last
    return each


def overlap(t1, t2, a1, a2, b1, b2):
    if t1[b1] > t2[a2] or t1[a1] > t2[b2]:
        return False
    return True


def sample_data(dump_paths, para=False, doc_sample_ratio=0.1, vec_sample_ratio=0.1, seed=29, max_norm=None):
    vecs = []
    random.seed(seed)
    np.random.seed(seed)
    dumps = [h5py.File(dump_path, 'r') for dump_path in dump_paths]
    for i, f in enumerate(tqdm(dumps)):
        doc_ids = f.keys()
        sampled_doc_ids = random.sample(doc_ids, int(doc_sample_ratio * len(doc_ids)))
        for doc_id in tqdm(sampled_doc_ids, desc='sampling from %d' % i):
            doc_group = f[doc_id]
            if para:
                groups = doc_group.values()
            else:
                groups = [doc_group]
            for group in groups:
                num_vecs, d = group['start'].shape
                sampled_vec_idxs = np.random.choice(num_vecs, int(vec_sample_ratio * num_vecs))
                cur_vecs = int8_to_float(group['start'][:],
                                         group.attrs['offset'], group.attrs['scale'])[sampled_vec_idxs]
                vecs.append(cur_vecs)
        break
    out = np.concatenate(vecs, 0)
    for dump in dumps:
        dump.close()

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if max_norm is None:
        max_norm = np.max(norms)
    consts = np.sqrt(np.maximum(0.0, max_norm ** 2 - norms ** 2))
    out = np.concatenate([consts, out], axis=1)
    return out, max_norm


def train_coarse_quantizer(data, quantizer_path, num_clusters, hnsw=False, niter=10):
    d = data.shape[1]

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    clus = faiss.Clustering(d, num_clusters)
    clus.verbose = True
    clus.niter = niter
    clus.train(data, gpu_index_flat)
    centroids = faiss.vector_float_to_array(clus.centroids)
    centroids = centroids.reshape(num_clusters, d)

    if hnsw:
        quantizer = faiss.IndexHNSWFlat(d, 32)
        quantizer.hnsw.efSearch = 128
        quantizer.train(centroids)
        quantizer.add(centroids)
    else:
        quantizer = faiss.IndexFlatL2(d)
        quantizer.add(centroids)

    faiss.write_index(quantizer, quantizer_path)


def train_index(data, quantizer_path, trained_index_path):
    quantizer = faiss.read_index(quantizer_path)
    trained_index = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
    trained_index.train(data)
    faiss.write_index(trained_index, trained_index_path)


def add_to_index(dump_path, trained_index_path, target_index_path, idx2id_path, max_norm, para=False):
    idx2doc_id = []
    idx2para_id = []
    idx2word_id = []
    phrase_dump = h5py.File(dump_path, 'r')
    print('reading %s' % trained_index_path)
    start_index = faiss.read_index(trained_index_path)

    print('adding %s' % dump_path)
    offset = 0
    if para:
        for i, (doc_idx, doc_group) in enumerate(tqdm(phrase_dump.items(), desc='faiss indexing')):
            for para_idx, group in doc_group.items():
                num_vecs = group['start'].shape[0]
                start = int8_to_float(group['start'][:], group.attrs['offset'], group.attrs['scale'])
                norms = np.linalg.norm(start, axis=1, keepdims=True)
                consts = np.sqrt(np.maximum(0.0, max_norm ** 2 - norms ** 2))
                start = np.concatenate([consts, start], axis=1)
                # self.start_index.add_with_ids(start, np.arange(offset, offset + start.shape[0]))
                start_index.add(start)
                idx2doc_id.extend([int(doc_idx)] * num_vecs)
                idx2para_id.extend([int(para_idx)] * num_vecs)
                idx2word_id.extend(list(range(num_vecs)))
                offset += start.shape[0]
            if i % 100 == 0:
                print('%d/%d' % (i + 1, len(phrase_dump.keys())))
    else:
        for i, (doc_idx, doc_group) in enumerate(tqdm(phrase_dump.items(), desc='faiss indexing')):
            num_vecs = doc_group['start'].shape[0]
            start = int8_to_float(doc_group['start'][:], doc_group.attrs['offset'],
                                  doc_group.attrs['scale'])
            norms = np.linalg.norm(start, axis=1, keepdims=True)
            consts = np.sqrt(max_norm ** 2 - norms ** 2)
            start = np.concatenate([consts, start], axis=1)
            start_index.add_with_ids(start, np.arange(offset, offset + start.shape[0]))
            idx2doc_id.extend([int(doc_idx)] * num_vecs)
            idx2word_id.extend(range(num_vecs))
            offset += start.shape[0]
            if i % 100 == 0:
                print('%d/%d' % (i + 1, len(phrase_dump.keys())))

    idx2doc_id = np.array(idx2doc_id, dtype=np.int32)
    idx2para_id = np.array(idx2para_id, dtype=np.int32)
    idx2word_id = np.array(idx2word_id, dtype=np.int32)

    with h5py.File(idx2id_path, 'w') as f:
        f.create_dataset('doc', data=idx2doc_id)
        f.create_dataset('para', data=idx2para_id)
        f.create_dataset('word', data=idx2word_id)
    faiss.write_index(start_index, target_index_path)


def merge_indexes(index_dir, idx2id_dir, target_index_path, target_idx2id_path):
    pass


class MIPS(object):
    def __init__(self, phrase_dump_dir, start_index_path, idx2id_path, max_answer_length, para=False):
        if os.path.isdir(phrase_dump_dir):
            self.phrase_dump_paths = [os.path.join(phrase_dump_dir, name) for name in os.listdir(phrase_dump_dir)]
            dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.phrase_dump_paths]
            self.dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
        else:
            self.phrase_dump_paths = [phrase_dump_dir]
        self.phrase_dumps = [h5py.File(path, 'r') for path in self.phrase_dump_paths]
        self.max_answer_length = max_answer_length
        self.para = para

        print('reading %s' % start_index_path)
        self.start_index = faiss.read_index(start_index_path)
        with h5py.File(idx2id_path, 'r') as f:
            self.idx2doc_id = f['doc'][:]
            self.idx2para_id = f['para'][:]
            self.idx2word_id = f['word'][:]

    def close(self):
        for phrase_dump in self.phrase_dumps:
            phrase_dump.close()

    def get_doc_group(self, doc_idx):
        if len(self.phrase_dumps) == 1:
            return self.phrase_dumps[0][str(doc_idx)]
        for dump_range, dump in zip(self.dump_ranges, self.phrase_dumps):
            if dump_range[0] <= int(doc_idx) < dump_range[1]:
                return dump[str(doc_idx)]
        raise ValueError('%d not found in dump list' % int(doc_idx))

    def search_start(self, query_start, doc_idxs=None, para_idxs=None, top_k=5, nprobe=16):
        # doc_idxs = [Q], para_idxs = [Q]
        assert self.start_index is not None
        query_start = query_start.astype(np.float32)

        if doc_idxs is None:
            query_start = np.concatenate([np.zeros([query_start.shape[0], 1]).astype(np.float32), query_start], axis=1)
            self.start_index.nprobe = nprobe
            start_scores, I = self.start_index.search(query_start, top_k)

            doc_idxs = self.idx2doc_id[I]
            start_idxs = self.idx2word_id[I]
            if self.para:
                para_idxs = self.idx2para_id[I]
        else:
            groups = [self.get_doc_group(doc_idx)[str(para_idx)] for doc_idx, para_idx in zip(doc_idxs, para_idxs)]
            starts = [group['start'][:, :] for group in groups]
            starts = [int8_to_float(start, groups[0].attrs['offset'], groups[0].attrs['scale']) for start in starts]
            all_scores = [np.squeeze(np.matmul(start, query_start[i:i + 1, :].transpose()), -1)
                          for i, start in enumerate(starts)]
            start_idxs = np.array([scores.argsort()[-top_k:][::-1]
                                   for scores in all_scores])
            start_scores = np.array([scores[idxs] for scores, idxs in zip(all_scores, start_idxs)])
            doc_idxs = np.tile(np.expand_dims(doc_idxs, -1), [1, top_k])
            para_idxs = np.tile(np.expand_dims(para_idxs, -1), [1, top_k])
        return start_scores, doc_idxs, para_idxs, start_idxs

    def search_phrase(self, query, doc_idxs, start_idxs, para_idxs=None, start_scores=None):
        # query = [Q, d]
        # doc_idxs = [Q]
        # start_idxs = [Q]
        # para_idxs = [Q]
        bs = int((query.shape[1] - 1) / 2)
        query_start, query_end, query_span_logit = query[:, :bs], query[:, bs:2 * bs], query[:, -1:]

        groups = [self.get_doc_group(doc_idx) for doc_idx in doc_idxs]
        if self.para:
            groups = [group[str(para_idx)] for group, para_idx in zip(groups, para_idxs)]

        def dequant(group, input_):
            if 'offset' in group.attrs:
                return int8_to_float(input_, group.attrs['offset'], group.attrs['scale'])
            return input_

        if start_scores is None:
            start = np.stack([group['start'][start_idx, :]
                              for group, start_idx in zip(groups, start_idxs)], 0)  # [Q, d]
            start = dequant(groups[0], start)
            start_scores = np.sum(query_start * start, 1)  # [Q]

        ends = [group['end'][:] for group in groups]
        spans = [group['span_logits'][:] for group in groups]

        end_idxs = [group['start2end'][start_idx, :] for group, start_idx in zip(groups, start_idxs)]  # [Q, L]
        end_mask = -1e9 * (np.array(end_idxs) < 0)  # [Q, L]
        end = np.stack([[each_end[each_end_idx, :] for each_end_idx in each_end_idxs]
                        for each_end, each_end_idxs in zip(ends, end_idxs)], 0)  # [Q, L, d]
        end = dequant(groups[0], end)
        span = np.stack([[each_span[start_idx, i] for i in range(len(each_end_idxs))]
                         for each_span, start_idx, each_end_idxs in zip(spans, start_idxs, end_idxs)], 0)  # [Q, L]

        end_scores = np.sum(np.expand_dims(query_end, 1) * end, 2)  # [Q, L]
        span_scores = query_span_logit * span  # [Q, L]
        scores = np.expand_dims(start_scores, 1) + end_scores + span_scores + end_mask  # [Q, L]
        pred_end_idxs = np.stack([each[idx] for each, idx in zip(end_idxs, np.argmax(scores, 1))], 0)  # [Q]
        max_scores = np.max(scores, 1)

        out = [{'context': group.attrs['context'],
                'title': group.attrs['title'],
                'doc_idx': doc_idx,
                'start_pos': group['word2char_start'][start_idx].item(),
                'end_pos': group['word2char_end'][end_idx].item(),
                'score': score}
               for doc_idx, group, start_idx, end_idx, score in zip(doc_idxs.tolist(), groups, start_idxs.tolist(),
                                                                    pred_end_idxs.tolist(), max_scores.tolist())]

        for each in out:
            each['answer'] = each['context'][each['start_pos']:each['end_pos']]

        out = [adjust(each) for each in out]
        # out = [each for each in out if len(each['context']) > 100 and each['score'] >= 30]

        return out

    def search(self, query, top_k=5, nprobe=64, doc_idxs=None, para_idxs=None):
        num_queries = query.shape[0]
        bs = int((query.shape[1] - 1) / 2)
        query_start = query[:, :bs]
        start_scores, doc_idxs, para_idxs, start_idxs = self.search_start(query_start, top_k=top_k, nprobe=nprobe,
                                                                          doc_idxs=doc_idxs, para_idxs=para_idxs)
        # reshape
        query = np.reshape(np.tile(np.expand_dims(query, 1), [1, top_k, 1]), [-1, query.shape[1]])
        idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k]), [-1])
        start_scores = np.reshape(start_scores, [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        para_idxs = np.reshape(para_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])

        out = self.search_phrase(query, doc_idxs, start_idxs, para_idxs=para_idxs)
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])

        return new_out


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_dir')
    parser.add_argument('stage')
    parser.add_argument('--index_name', default='default_index')
    parser.add_argument('--quantizer_path', default='quantizer.faiss')
    parser.add_argument('--max_norm_path', default='max_norm.json')
    parser.add_argument('--trained_index_path', default='trained.faiss')
    parser.add_argument('--dump_path', default='dump.hdf5')
    parser.add_argument('--merged_index_path', default='index.faiss')
    parser.add_argument('--num_clusters', type=int, default=4096)
    parser.add_argument('--hnsw', default=False, action='store_true')
    parser.add_argument('--max_norm', default=None, type=float)
    parser.add_argument('--para', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    from nsml import NSML_NFS_OUTPUT
    args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)

    if os.path.isdir(args.dump_dir):
        dump_names = os.listdir(args.dump_dir)
        dump_paths = [os.path.join(args.dump_dir, 'phrase', name) for name in dump_names]
        dump_path = os.path.join(args.dump_dir, 'phrase', args.dump_path)
    else:
        dump_paths = [args.dump_dir]
        dump_path = args.dump_dir

    args.out_dir = os.path.join(args.dump_dir, args.index_name)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    quantizer_path = os.path.join(args.out_dir, args.quantizer_path)
    max_norm_path = os.path.join(args.out_dir, args.max_norm_path)
    trained_index_path = os.path.join(args.out_dir, args.trained_index_path)

    index_dir = os.path.join(args.out_dir, 'index')
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    dump_name = os.path.splitext(os.path.basename(args.dump_path))[0]
    target_index_path = os.path.join(index_dir, '%s.faiss' % dump_name)
    idx2id_path = os.path.join(index_dir, '%s.hdf5' % dump_name)
    merged_index_path = os.path.join(args.out_dir, args.merged_index_path)

    if args.stage == 'coarse':
        data, max_norm = sample_data(dump_paths, max_norm=args.max_norm)
        with open(max_norm_path, 'w') as fp:
            json.dump(max_norm, fp)
        train_coarse_quantizer(data, quantizer_path, args.num_clusters)

    if args.stage == 'fine':
        with open(max_norm_path, 'r') as fp:
            max_norm = json.load(fp)
        data, _ = sample_data(dump_paths, max_norm=max_norm)
        train_index(data, quantizer_path, trained_index_path)

    if args.stage == 'add':
        with open(max_norm_path, 'r') as fp:
            max_norm = json.load(fp)
        add_to_index(dump_path, trained_index_path, target_index_path, idx2id_path,
                     max_norm=max_norm, para=args.para)

    if args.stage == 'merge':
        pass


if __name__ == '__main__':
    main()
