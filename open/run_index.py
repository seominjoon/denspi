import argparse
import json
import os
import random

import faiss
import h5py
import numpy as np
from tqdm import tqdm

from mips import int8_to_float


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_dir')
    parser.add_argument('stage')

    parser.add_argument('--dump_paths', default=None,
                        help='Relative to `dump_dir/phrase`. '
                             'If specified, creates subindex dir and save there with same name')
    parser.add_argument('--subindex_name', default='index', help='used only if dump_path is specified.')
    parser.add_argument('--offset', default=0, type=int)

    # relative paths in dump_dir/index_name
    parser.add_argument('--quantizer_path', default='quantizer.faiss')
    parser.add_argument('--max_norm_path', default='max_norm.json')
    parser.add_argument('--trained_index_path', default='trained.faiss')
    parser.add_argument('--index_path', default='index.faiss')
    parser.add_argument('--idx2id_path', default='idx2id.hdf5')
    parser.add_argument('--inv_path', default='merged.invdata')

    # Adding options
    parser.add_argument('--add_all', default=False, action='store_true')

    # coarse, fine, add
    parser.add_argument('--num_clusters', type=int, default=4096)
    parser.add_argument('--hnsw', default=False, action='store_true')
    parser.add_argument('--fine_quant', default='SQ8',
                        help='SQ8|SQ4|PQ# where # is number of bytes per vector (for SQ it would be 480 Bytes)')
    # stable params
    parser.add_argument('--max_norm', default=None, type=float)
    parser.add_argument('--max_norm_cf', default=1.3, type=float)
    parser.add_argument('--norm_th', default=999, type=float)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--doc_sample_ratio', default=0.2, type=float)
    parser.add_argument('--vec_sample_ratio', default=0.2, type=float)

    parser.add_argument('--fs', default='local')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--num_dummy_zeros', default=0, type=int)
    parser.add_argument('--replace', default=False, action='store_true')
    parser.add_argument('--num_docs_per_add', default=1000, type=int)

    args = parser.parse_args()

    coarse = 'hnsw' if args.hnsw else 'flat'
    args.index_name = '%d_%s_%s' % (args.num_clusters, coarse, args.fine_quant)

    if args.fs == 'nfs':
        from nsml import NSML_NFS_OUTPUT
        args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)
    elif args.fs == 'nsml':
        pass

    args.index_dir = os.path.join(args.dump_dir, args.index_name)

    args.quantizer_path = os.path.join(args.index_dir, args.quantizer_path)
    args.max_norm_path = os.path.join(args.index_dir, args.max_norm_path)
    args.trained_index_path = os.path.join(args.index_dir, args.trained_index_path)
    args.inv_path = os.path.join(args.index_dir, args.inv_path)

    args.subindex_dir = os.path.join(args.index_dir, args.subindex_name)
    if args.dump_paths is None:
        args.index_path = os.path.join(args.index_dir, args.index_path)
        args.idx2id_path = os.path.join(args.index_dir, args.idx2id_path)
    else:
        args.dump_paths = [os.path.join(args.dump_dir, 'phrase', path) for path in args.dump_paths.split(',')]
        args.index_path = os.path.join(args.subindex_dir, '%d.faiss' % args.offset)
        args.idx2id_path = os.path.join(args.subindex_dir, '%d.hdf5' % args.offset)

    return args


def sample_data(dump_paths, para=False, doc_sample_ratio=0.2, vec_sample_ratio=0.2, seed=29,
                max_norm=None, max_norm_cf=1.3, num_dummy_zeros=0, norm_th=999):
    vecs = []
    random.seed(seed)
    np.random.seed(seed)
    print('sampling from:')
    for dump_path in dump_paths:
        print(dump_path)
    dumps = [h5py.File(dump_path, 'r') for dump_path in dump_paths]
    for i, f in enumerate(tqdm(dumps)):
        doc_ids = list(f.keys())
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
                cur_vecs = cur_vecs[np.linalg.norm(cur_vecs, axis=1) <= norm_th]
                vecs.append(cur_vecs)
    out = np.concatenate(vecs, 0)
    for dump in dumps:
        dump.close()

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if max_norm is None:
        max_norm = max_norm_cf * np.max(norms)
    consts = np.sqrt(np.maximum(0.0, max_norm ** 2 - norms ** 2))
    out = np.concatenate([consts, out], axis=1)
    if num_dummy_zeros > 0:
        out = np.concatenate([out, np.zeros([out.shape[0], num_dummy_zeros], dtype=out.dtype)], axis=1)
    return out, max_norm


def train_coarse_quantizer(data, quantizer_path, num_clusters, hnsw=False, niter=10, cuda=False):
    d = data.shape[1]

    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    if cuda:
        res = faiss.StandardGpuResources()
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    clus = faiss.Clustering(d, num_clusters)
    clus.verbose = True
    clus.niter = niter
    clus.train(data, index_flat)
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


def train_index(data, quantizer_path, trained_index_path, fine_quant='SQ8', cuda=False):
    quantizer = faiss.read_index(quantizer_path)
    if fine_quant == 'SQ8':
        trained_index = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
    elif fine_quant.startswith('PQ'):
        m = int(fine_quant[2:])
        trained_index = faiss.IndexIVFPQ(quantizer, quantizer.d, quantizer.ntotal, m, 8)
    else:
        raise ValueError(fine_quant)

    if cuda:
        if fine_quant.startswith('PQ'):
            print('PQ not supported on GPU; keeping CPU.')
        else:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, trained_index)
            gpu_index.train(data)
            trained_index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        trained_index.train(data)
    faiss.write_index(trained_index, trained_index_path)


def add_with_offset(index, data, offset, valids=None):
    ids = np.arange(data.shape[0]) + offset + index.ntotal
    if valids is not None:
        data = data[valids]
        ids = ids[valids]
    index.add_with_ids(data, ids)


def add_to_index(dump_paths, trained_index_path, target_index_path, idx2id_path, max_norm, para=False,
                 num_docs_per_add=1000, num_dummy_zeros=0, cuda=False, fine_quant='SQ8', offset=0, norm_th=999,
                 ignore_ids=None):
    idx2doc_id = []
    idx2para_id = []
    idx2word_id = []
    dumps = [h5py.File(dump_path, 'r') for dump_path in dump_paths]
    print('reading %s' % trained_index_path)
    start_index = faiss.read_index(trained_index_path)

    if cuda:
        if fine_quant.startswith('PQ'):
            print('PQ not supported on GPU; keeping CPU.')
        else:
            res = faiss.StandardGpuResources()
            start_index = faiss.index_cpu_to_gpu(res, 0, start_index)

    print('adding following dumps:')
    for dump_path in dump_paths:
        print(dump_path)
    if para:
        for di, phrase_dump in enumerate(tqdm(dumps, desc='dumps')):
            starts = []
            for i, (doc_idx, doc_group) in enumerate(tqdm(phrase_dump.items(), desc='faiss indexing')):
                for para_idx, group in doc_group.items():
                    num_vecs = group['start'].shape[0]
                    start = int8_to_float(group['start'][:], group.attrs['offset'], group.attrs['scale'])
                    norms = np.linalg.norm(start, axis=1, keepdims=True)
                    consts = np.sqrt(np.maximum(0.0, max_norm ** 2 - norms ** 2))
                    start = np.concatenate([consts, start], axis=1)
                    if num_dummy_zeros > 0:
                        start = np.concatenate(
                            [start, np.zeros([start.shape[0], num_dummy_zeros], dtype=start.dtype)], axis=1)
                    starts.append(start)
                    idx2doc_id.extend([int(doc_idx)] * num_vecs)
                    idx2para_id.extend([int(para_idx)] * num_vecs)
                    idx2word_id.extend(list(range(num_vecs)))
                if len(starts) > 0 and i % num_docs_per_add == 0:
                    print('concatenating')
                    concat = np.concatenate(starts, axis=0)
                    print('adding')
                    add_with_offset(start_index, concat, offset)
                    # start_index.add(concat)
                    print('done')
                    starts = []
                if i % 100 == 0:
                    print('%d/%d' % (i + 1, len(phrase_dump.keys())))
            print('adding leftover')
            add_with_offset(start_index, np.concatenate(starts, axis=0), offset)
            # start_index.add(np.concatenate(starts, axis=0))  # leftover
            print('done')
    else:
        for di, phrase_dump in enumerate(tqdm(dumps, desc='dumps')):
            starts = []
            valids = []
            for i, (doc_idx, doc_group) in enumerate(tqdm(phrase_dump.items(), desc='adding %d' % di)):
                if ignore_ids is not None and doc_idx in ignore_ids:
                    continue
                num_vecs = doc_group['start'].shape[0]
                start = int8_to_float(doc_group['start'][:], doc_group.attrs['offset'],
                                      doc_group.attrs['scale'])
                valid = np.linalg.norm(start, axis=1) <= norm_th
                norms = np.linalg.norm(start, axis=1, keepdims=True)
                consts = np.sqrt(np.maximum(0.0, max_norm ** 2 - norms ** 2))
                start = np.concatenate([consts, start], axis=1)
                if num_dummy_zeros > 0:
                    start = np.concatenate([start, np.zeros([start.shape[0], num_dummy_zeros], dtype=start.dtype)],
                                           axis=1)
                starts.append(start)
                valids.append(valid)
                idx2doc_id.extend([int(doc_idx)] * num_vecs)
                idx2word_id.extend(range(num_vecs))
                if len(starts) > 0 and i % num_docs_per_add == 0:
                    add_with_offset(start_index, np.concatenate(starts, axis=0), offset, np.concatenate(valids))
                    # start_index.add(np.concatenate(starts, axis=0))
                    starts = []
                    valids = []
                if i % 100 == 0:
                    print('%d/%d' % (i + 1, len(phrase_dump.keys())))
            add_with_offset(start_index, np.concatenate(starts, axis=0), offset, np.concatenate(valids))
            # start_index.add(np.concatenate(starts, axis=0))  # leftover

    for dump in dumps:
        dump.close()

    if cuda and not fine_quant.startswith('PQ'):
        print('moving back to cpu')
        start_index = faiss.index_gpu_to_cpu(start_index)

    print('index ntotal: %d' % start_index.ntotal)
    idx2doc_id = np.array(idx2doc_id, dtype=np.int32)
    idx2para_id = np.array(idx2para_id, dtype=np.int32)
    idx2word_id = np.array(idx2word_id, dtype=np.int32)

    print('writing index and metadata')
    with h5py.File(idx2id_path, 'w') as f:
        f.create_dataset('doc', data=idx2doc_id)
        f.create_dataset('para', data=idx2para_id)
        f.create_dataset('word', data=idx2word_id)
        f.attrs['offset'] = offset
    faiss.write_index(start_index, target_index_path)
    print('done')


def merge_indexes(subindex_dir, trained_index_path, target_index_path, target_idx2id_path, target_inv_path):
    # target_inv_path = merged_index.ivfdata
    names = os.listdir(subindex_dir)
    idx2id_paths = [os.path.join(subindex_dir, name) for name in names if name.endswith('.hdf5')]
    index_paths = [os.path.join(subindex_dir, name) for name in names if name.endswith('.faiss')]

    print('copying idx2id')
    with h5py.File(target_idx2id_path, 'w') as out:
        for idx2id_path in tqdm(idx2id_paths, desc='copying idx2id'):
            with h5py.File(idx2id_path, 'r') as in_:
                offset = str(in_.attrs['offset'])
                group = out.create_group(offset)
                group.create_dataset('doc', data=in_['doc'])
                group.create_dataset('para', data=in_['para'])
                group.create_dataset('word', data=in_['word'])

    print('loading invlists')
    ivfs = []
    for index_path in tqdm(index_paths, desc='loading invlists'):
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        index = faiss.read_index(index_path,
                                 faiss.IO_FLAG_MMAP)
        ivfs.append(index.invlists)

        # avoid that the invlists get deallocated with the index
        index.own_invlists = False

    # construct the output index
    index = faiss.read_index(trained_index_path)

    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    invlists = faiss.OnDiskInvertedLists(
        index.nlist, index.code_size,
        target_inv_path)

    # merge all the inverted lists
    print('merging')
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in tqdm(ivfs):
        ivf_vector.push_back(ivf)

    print("merge %d inverted lists " % ivf_vector.size())
    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
    print(ntotal)

    # now replace the inverted lists in the output index
    index.ntotal = ntotal
    index.replace_invlists(invlists)

    print('writing index')
    faiss.write_index(index, target_index_path)


def run_index(args):
    phrase_path = os.path.join(args.dump_dir, 'phrase.hdf5')
    if os.path.exists(phrase_path):
        dump_paths = [phrase_path]
    else:
        dump_names = os.listdir(os.path.join(args.dump_dir, 'phrase'))
        dump_paths = [os.path.join(args.dump_dir, 'phrase', name) for name in dump_names]

    data = None

    if args.stage in ['all', 'coarse']:
        if args.replace or not os.path.exists(args.quantizer_path):
            if not os.path.exists(args.index_dir):
                os.makedirs(args.index_dir)
            data, max_norm = sample_data(dump_paths, max_norm=args.max_norm, para=args.para,
                                         doc_sample_ratio=args.doc_sample_ratio, vec_sample_ratio=args.vec_sample_ratio,
                                         max_norm_cf=args.max_norm_cf, num_dummy_zeros=args.num_dummy_zeros,
                                         norm_th=args.norm_th)
            with open(args.max_norm_path, 'w') as fp:
                json.dump(max_norm, fp)
            train_coarse_quantizer(data, args.quantizer_path, args.num_clusters, cuda=args.cuda)

    if args.stage in ['all', 'fine']:
        if args.replace or not os.path.exists(args.trained_index_path):
            with open(args.max_norm_path, 'r') as fp:
                max_norm = json.load(fp)
            if data is None:
                data, _ = sample_data(dump_paths, max_norm=max_norm, para=args.para,
                                      doc_sample_ratio=args.doc_sample_ratio, vec_sample_ratio=args.vec_sample_ratio,
                                      num_dummy_zeros=args.num_dummy_zeros, norm_th=args.norm_th)
            train_index(data, args.quantizer_path, args.trained_index_path, fine_quant=args.fine_quant, cuda=args.cuda)

    if args.stage in ['all', 'add']:
        if args.replace or not os.path.exists(args.index_path):
            with open(args.max_norm_path, 'r') as fp:
                max_norm = json.load(fp)
            if args.dump_paths is not None:
                dump_paths = args.dump_paths
                if not os.path.exists(args.subindex_dir):
                    os.makedirs(args.subindex_dir)
            add_to_index(dump_paths, args.trained_index_path, args.index_path, args.idx2id_path,
                         max_norm=max_norm, para=args.para, num_dummy_zeros=args.num_dummy_zeros, cuda=args.cuda,
                         num_docs_per_add=args.num_docs_per_add, offset=args.offset, norm_th=args.norm_th)

    if args.stage == 'merge':
        if args.replace or not os.path.exists(args.index_path):
            merge_indexes(args.subindex_dir, args.trained_index_path, args.index_path, args.idx2id_path, args.inv_path)

    if args.stage == 'move':
        index = faiss.read_index(args.trained_index_path)
        invlists = faiss.OnDiskInvertedLists(
            index.nlist, index.code_size,
            args.inv_path)
        index.replace_invlists(invlists)
        faiss.write_index(index, args.index_path)


def main():
    args = get_args()
    run_index(args)


if __name__ == '__main__':
    main()
