import argparse
import json
import os
import random

import faiss
import h5py
import numpy as np
from tqdm import tqdm

from mips import int8_to_float


def sample_data(dump_paths, para=False, doc_sample_ratio=0.1, vec_sample_ratio=0.1, seed=29, max_norm=None):
    vecs = []
    random.seed(seed)
    np.random.seed(seed)
    print(dump_paths)
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
    out = np.concatenate(vecs, 0)
    for dump in dumps:
        dump.close()

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if max_norm is None:
        max_norm = 1.3 * np.max(norms)
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


def add_to_index(dump_paths, trained_index_path, target_index_path, idx2id_path, max_norm, para=False):
    idx2doc_id = []
    idx2para_id = []
    idx2word_id = []
    dumps = [h5py.File(dump_path, 'r') for dump_path in dump_paths]
    print('reading %s' % trained_index_path)
    start_index = faiss.read_index(trained_index_path)

    print('adding %s' % dump_paths)
    offset = 0
    if para:
        for di, phrase_dump in enumerate(tqdm(dumps, desc='dumps')):
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
        for di, phrase_dump in enumerate(tqdm(dumps, desc='dumps')):
            for i, (doc_idx, doc_group) in enumerate(tqdm(phrase_dump.items(), desc='adding %d' % di)):
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

    print('index ntotal: %d' % start_index.ntotal)
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
    parser.add_argument('--doc_sample_ratio', default=0.2, type=float)
    parser.add_argument('--vec_sample_ratio', default=0.2, type=float)
    parser.add_argument('--fs', default='local')
    parser.add_argument('--add_all', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    if args.fs == 'nfs':
        from nsml import NSML_NFS_OUTPUT
        args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)

    # from nsml import NSML_NFS_OUTPUT
    # args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)

    phrase_path = os.path.join(args.dump_dir, 'phrase.hdf5')
    if os.path.exists(phrase_path):
        dump_paths = [phrase_path]
        dump_path = phrase_path
    else:
        dump_names = os.listdir(os.path.join(args.dump_dir, 'phrase'))
        dump_paths = [os.path.join(args.dump_dir, 'phrase', name) for name in dump_names if 'hdf5' in name]
        dump_path = os.path.join(args.dump_dir, 'phrase', args.dump_path)
    print(dump_paths)

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
        data, max_norm = sample_data(dump_paths, max_norm=args.max_norm, para=args.para,
                                     doc_sample_ratio=args.doc_sample_ratio, vec_sample_ratio=args.vec_sample_ratio)
        with open(max_norm_path, 'w') as fp:
            json.dump(max_norm, fp)
        train_coarse_quantizer(data, quantizer_path, args.num_clusters)

    if args.stage == 'fine':
        with open(max_norm_path, 'r') as fp:
            max_norm = json.load(fp)
        data, _ = sample_data(dump_paths, max_norm=max_norm, para=args.para,
                              doc_sample_ratio=args.doc_sample_ratio, vec_sample_ratio=args.vec_sample_ratio)
        train_index(data, quantizer_path, trained_index_path)

    if args.stage == 'add':
        with open(max_norm_path, 'r') as fp:
            max_norm = json.load(fp)
        if not args.add_all:
            dump_paths = [dump_path]
        add_to_index(dump_paths, trained_index_path, target_index_path, idx2id_path,
                     max_norm=max_norm, para=args.para)

    if args.stage == 'merge':
        pass


if __name__ == '__main__':
    main()
