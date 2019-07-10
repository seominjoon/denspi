"""Remove document ids that are considered as noise"""

import argparse
import json
import os

import faiss
import h5py
from tqdm import tqdm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('subindex_dir')
    parser.add_argument('ignore_path')
    parser.add_argument('--target_dir', default=None)
    parser.add_argument('--ratio', default=0.01, type=float)
    args = parser.parse_args()
    if args.target_dir is None:
        args.target_dir = args.subindex_dir
    return args


def remove_doc_ids(args):
    if os.path.isdir(args.subindex_dir):
        names = os.listdir(args.subindex_dir)
        index_names = [name for name in names if name.endswith('.faiss')]
        index_paths = [os.path.join(args.subindex_dir, name) for name in index_names]
        target_paths = [os.path.join(args.target_dir, name) for name in index_names]
        idx2id_paths = [path.replace('.faiss', '.hdf5') for path in index_paths]
        if not os.path.exists(args.target_dir):
            os.makedirs(args.target_dir)

        with open(args.ignore_path, 'r') as fp:
            ignore_counter = json.load(fp)
        count = sum(ignore_counter.values())
        th = count * args.ratio
        ignores = [int(key) for key, val in ignore_counter.items() if val > th]
        print('thresholding at %.1f, removing following document ids:' % th)
        for ignore in ignores:
            print(ignore)

        for idx2id_path, index_path, target_path in zip(idx2id_paths, tqdm(index_paths), target_paths):
            with h5py.File(idx2id_path, 'r') as f:
                doc_ids = f['doc'][:]
                offset = f.attrs['offset']
            idxs, = np.where(np.any(np.expand_dims(doc_ids, 1) == ignores, 1))
            if len(idxs) > 0:
                idxs = idxs + offset
                print('found %d ids to remove' % len(idxs))
                index = faiss.read_index(index_path)
                index.remove_ids(idxs)
                faiss.write_index(index, target_path)
            else:
                print('no ignore list found at %s' % index_path)
    else:
        index_path = args.subindex_dir
        target_path = args.target_dir
        idx2id_path = args.subindex_dir.replace('index.faiss', 'idx2id.hdf5')
        with open(args.ignore_path, 'r') as fp:
            ignores = np.array(list(map(int, json.load(fp))))
        with h5py.File(idx2id_path, 'r') as f:
            for offset, group in f.items():
                doc_ids = group['doc'][:]
                offset = int(offset)
                idxs, = np.where(np.any(np.expand_dims(doc_ids, 1) == ignores, 1))
                if len(idxs) > 0:
                    idxs = idxs + offset
                    print(idxs)
                    index = faiss.read_index(index_path)
                    index.remove_ids(idxs)
                    faiss.write_index(index, target_path)
                else:
                    print('no ignore list found at %d' % offset)


def main():
    args = get_args()
    remove_doc_ids(args)


if __name__ == '__main__':
    main()
