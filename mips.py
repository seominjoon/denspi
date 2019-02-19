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


def id2idxs(id_, para=False):
    if para:
        return (id_ / 1e7).astype(np.int64), ((id_ % int(1e7)) / 1e4).astype(np.int64), id_ % int(1e4)
    return (id_ / 1e6).astype(np.int64), id_ % int(1e6)


def idxs2id(doc_idx, first_idx, second_idx=None):
    if second_idx is None:
        assert np.all(first_idx < 1e6)
        return doc_idx * int(1e6) + first_idx
    assert np.all(second_idx < 1e4)
    assert np.all(first_idx < 1e3)
    return doc_idx * int(1e7) + first_idx * int(1e4) + second_idx


def find_max_norm(dumps, para=False, doc_sample_ratio=0.1, vec_sample_ratio=0.1, seed=29):
    random.seed(seed)
    np.random.seed(seed)
    max_norm = 0.0
    for i, f in enumerate(tqdm(dumps, desc='finding max norm')):
        doc_ids = f.keys()
        sampled_doc_ids = random.sample(doc_ids, int(doc_sample_ratio * len(doc_ids)))
        for doc_id in tqdm(sampled_doc_ids, desc=str(i)):
            if para:
                for para_id, group in f[doc_id].items():
                    start = group['start'][:]
                    num_vecs, d = start.shape
                    num_samples = int(vec_sample_ratio * num_vecs)
                    if num_samples == 0:
                        continue
                    sampled_vec_idxs = np.random.choice(num_vecs, num_samples)
                    start = start[sampled_vec_idxs]
                    start = int8_to_float(start, group.attrs['offset'],
                                          group.attrs['scale'])
                    max_norm = max(max_norm, np.linalg.norm(start, axis=1).max())

            else:
                group = f[doc_id]
                start = group['start'][:]
                num_vecs, d = start.shape
                num_samples = int(vec_sample_ratio * num_vecs)
                if num_samples == 0:
                    continue
                sampled_vec_idxs = np.random.choice(num_vecs, num_samples)
                start = start[sampled_vec_idxs]
                start = int8_to_float(start, group.attrs['offset'],
                                      group.attrs['scale'])
                max_norm = max(max_norm, np.linalg.norm(start, axis=1).max())
    return max_norm


def get_coarse_quantizer(dumps, num_clusters,
                         hnsw=False, niter=10, doc_sample_ratio=0.1, vec_sample_ratio=0.1, seed=29, max_norm=None,
                         para=False):
    vecs = []
    np.random.seed(seed)
    d = None
    for i, f in enumerate(tqdm(dumps)):
        doc_ids = f.keys()
        sampled_doc_ids = random.sample(doc_ids, int(doc_sample_ratio * len(doc_ids)))
        for doc_id in tqdm(sampled_doc_ids, desc=str(i)):
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
                if max_norm is not None:
                    norms = np.linalg.norm(cur_vecs, axis=1, keepdims=True)
                    consts = np.sqrt(np.maximum(0.0, max_norm ** 2 - norms ** 2))
                    cur_vecs = np.concatenate([consts, cur_vecs], axis=1)
                    d += 1
                vecs.append(cur_vecs)

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    clus = faiss.Clustering(d, num_clusters)
    clus.verbose = True
    clus.niter = niter
    vecs_cat = np.concatenate(vecs, 0)
    clus.train(vecs_cat, gpu_index_flat)
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
    print('is trained?: %s' % str(quantizer.is_trained))

    return quantizer


class MIPS(object):
    def __init__(self, phrase_dump_dir, start_index_path, idx2id_path, max_answer_length,
                 max_norm=None, quantizer_path=None, start_subindex_dir=None,
                 num_clusters=1048576, niter=10, para=False, doc_sample_ratio=0.1, vec_sample_ratio=0.1, seed=29):
        if os.path.isdir(phrase_dump_dir):
            self.phrase_dump_paths = [os.path.join(phrase_dump_dir, name) for name in os.listdir(phrase_dump_dir)]
            dump_names = [os.path.splitext(os.path.basename(path))[0] for path in self.phrase_dump_paths]
            self.dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
        else:
            self.phrase_dump_paths = [phrase_dump_dir]
        self.phrase_dumps = [h5py.File(path, 'r') for path in self.phrase_dump_paths]

        self.max_answer_length = max_answer_length
        self.para = para
        if os.path.exists(start_index_path):
            self.start_index = faiss.read_index(start_index_path)
            with h5py.File(idx2id_path, 'r') as f:
                self.idx2doc_id = f['doc'][:]
                self.idx2para_id = f['para'][:]
                self.idx2word_id = f['word'][:]
        else:
            assert quantizer_path is not None, 'Quantizer path needs to be specified.'
            if max_norm is None:
                max_norm = find_max_norm(self.phrase_dumps, para=para, doc_sample_ratio=doc_sample_ratio,
                                         vec_sample_ratio=vec_sample_ratio, seed=seed)
                max_norm *= 1.3
            print('max norm: %.2f' % max_norm)
            if os.path.exists(quantizer_path):
                quantizer = faiss.read_index(quantizer_path)
            else:
                quantizer = get_coarse_quantizer(self.phrase_dumps, num_clusters,
                                                 niter=niter, doc_sample_ratio=doc_sample_ratio,
                                                 vec_sample_ratio=vec_sample_ratio,
                                                 seed=seed, max_norm=max_norm, hnsw=True, para=para)

            self.start_index = faiss.IndexIVFFlat(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
            self.idx2doc_id = []
            self.idx2para_id = []
            self.idx2word_id = []

            for phrase_dump in self.phrase_dumps:
                if para:
                    for doc_idx, doc_group in tqdm(phrase_dump.items(), desc='faiss indexing'):
                        for para_idx, group in doc_group.items():
                            num_vecs = group['start'].shape[0]
                            start = int8_to_float(group['start'][:], group.attrs['offset'], group.attrs['scale'])
                            norms = np.linalg.norm(start, axis=1, keepdims=True)
                            consts = np.sqrt(np.maximum(0.0, max_norm ** 2 - norms ** 2))
                            start = np.concatenate([consts, start], axis=1)
                            self.start_index.add(start)
                            self.idx2doc_id.extend([int(doc_idx)] * num_vecs)
                            self.idx2para_id.extend([int(para_idx)] * num_vecs)
                            self.idx2word_id.extend(list(range(num_vecs)))
                else:
                    for doc_idx, doc_group in tqdm(phrase_dump.items(), desc='faiss indexing'):
                        num_vecs = doc_group['start'].shape[0]
                        start = int8_to_float(doc_group['start'][:], doc_group.attrs['offset'],
                                              doc_group.attrs['scale'])
                        norms = np.linalg.norm(start, axis=1, keepdims=True)
                        consts = np.sqrt(max_norm ** 2 - norms ** 2)
                        start = np.concatenate([consts, start], axis=1)
                        self.start_index.add(start)
                        self.idx2doc_id.extend([int(doc_idx)] * num_vecs)
                        self.idx2word_id.append(list(range(num_vecs)))

            self.idx2doc_id = np.array(self.idx2doc_id, dtype=np.int32)
            self.idx2para_id = np.array(self.idx2para_id, dtype=np.int32)
            self.idx2word_id = np.array(self.idx2word_id, dtype=np.int32)

            faiss.write_index(quantizer, quantizer_path)
            with h5py.File(idx2id_path, 'w') as f:
                f.create_dataset('doc', data=self.idx2doc_id)
                f.create_dataset('para', data=self.idx2para_id)
                f.create_dataset('word', data=self.idx2word_id)
            faiss.write_index(self.start_index, start_index_path)

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
    parser.add_argument('quantizer_path')
    parser.add_argument('num_clusters', type=int)
    parser.add_argument('--hnsw', default=False, action='store_true')
    parser.add_argument('--max_norm', default=None, type=float)
    return parser.parse_args()


def main():
    args = get_args()
    if os.path.isdir(args.dump_dir):
        dump_paths = [os.path.join(args.dump_dir, name) for name in os.listdir(args.dump_dir)]
    else:
        dump_paths = [args.dump_dir]
    start = time()
    if args.max_norm is None:
        max_norm = find_max_norm(dump_paths)
    else:
        max_norm = args.max_norm
    print('max norm: %.2f' % max_norm)
    quantizer = get_coarse_quantizer(dump_paths, args.num_clusters, hnsw=args.hnsw, max_norm=max_norm)
    faiss.write_index(quantizer, args.quantizer_path)
    end = time()
    print('total time: %d mins' % int((end - start) / 60))


if __name__ == '__main__':
    main()
