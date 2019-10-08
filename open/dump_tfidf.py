import json
import os
import argparse
import random
import h5py
import numpy as np
import scipy.sparse as sp
import logging

from tqdm import tqdm
from drqa import retriever
from drqa.retriever import utils

logger = logging.getLogger(__name__)


class MyTfidfDocRanker(retriever.get_class('tfidf')):
    def text2spvec(self, query, data_val=False):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        if data_val:
            return data, wids_unique

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec


def dump_tfidf(ranker, dumps, names, args):
    for phrase_dump, name in tqdm(zip(dumps, names)):
        with h5py.File(os.path.join(args.out_dir, name + '_tfidf.hdf5')) as f:
            for doc_id in tqdm(phrase_dump):
                if doc_id in f:
                    print('%s exists; replacing' % doc_id)
                    del f[doc_id]
                dg = f.create_group(doc_id)
                doc = phrase_dump[doc_id]
                paras = [k.strip() for k in doc.attrs['context'].split('[PAR]')]
                para_data = [ranker.text2spvec(para, data_val=True) for para in paras]
                for p_idx, data in enumerate(para_data):
                    if str(p_idx) in dg:
                        print('%s exists; replacing' % str(p_idx))
                        del dg[str(p_idx)]
                    pdg = dg.create_group(str(p_idx))
                    try:
                        pdg.create_dataset('vals', data=data[0])
                        pdg.create_dataset('idxs', data=data[1])
                    except Exception as e:
                        print('Exception occured {} {}'.format(str(e), data))
                        pdg.create_dataset('vals', data=[0])
                        pdg.create_dataset('idxs', data=[0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1, type=int)
    parser.add_argument('--ranker_path', default='docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz', type=str)
    parser.add_argument('--nfs', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    if args.nfs:
        from nsml import NSML_NFS_OUTPUT
        args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)
        args.out_dir = os.path.join(NSML_NFS_OUTPUT, args.out_dir)
        args.ranker_path = os.path.join(NSML_NFS_OUTPUT, args.ranker_path)
    os.makedirs(args.out_dir)
    assert os.path.isdir(args.dump_dir)
    dump_paths = sorted([os.path.join(args.dump_dir, name) for name in os.listdir(args.dump_dir) if 'hdf5' in name])[
                 args.start:args.end]
    print(dump_paths)
    dump_names = [os.path.splitext(os.path.basename(path))[0] for path in dump_paths]
    dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
    phrase_dumps = [h5py.File(path, 'r') for path in dump_paths]

    ranker = None
    ranker = MyTfidfDocRanker(
        tfidf_path=args.ranker_path,
        strict=False
    )

    print('Ranker shape {} from {}'.format(ranker.doc_mat.shape, args.ranker_path))
    # new_mat = ranker.doc_mat.T.tocsr()
    # sp.save_npz('doc_tfidf.npz', new_mat)
    dump_tfidf(ranker, phrase_dumps, dump_names, args)


if __name__ == '__main__':
    main()
