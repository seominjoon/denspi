import torch
import argparse
import scipy.sparse as sp
import numpy as np
import time

from tqdm import tqdm

QUERY_NUM = 10570
DATASET_WORD_SIZE = 178 # from 1783576
DVEC_SIZE = 961
PHRASE_HOT = 209
QUERY_HOT = 16
SVEC_SIZE = 16777216
SVEC_DENSITY = 1.2487489895663467e-05 # doc_mat nz = 1063277605 (query = 9.53674316e-7)
BATCH_SIZE = 128


def benchmark(args):
    # Dense
    phrase_d = torch.randn((DATASET_WORD_SIZE, DVEC_SIZE), dtype=torch.float32)
    query_d = torch.randn((QUERY_NUM, DVEC_SIZE), dtype=torch.float32)

    # Sparse
    if args.sparse:
        phrase_data = np.random.randn(DATASET_WORD_SIZE, PHRASE_HOT).flatten()
        phrase_idxs = np.random.randint(SVEC_SIZE, size=(DATASET_WORD_SIZE, PHRASE_HOT)).flatten()
        phrase_indptr = np.array(list(range(0, DATASET_WORD_SIZE * PHRASE_HOT, PHRASE_HOT)) + [DATASET_WORD_SIZE * PHRASE_HOT])
        phrase_s = sp.csr_matrix((phrase_data, phrase_idxs, phrase_indptr), shape=(DATASET_WORD_SIZE, SVEC_SIZE))
        query_data = np.random.randn(QUERY_NUM, QUERY_HOT).flatten()
        query_idxs = np.random.randint(SVEC_SIZE, size=(QUERY_NUM, QUERY_HOT)).flatten()
        query_indptr = np.array(list(range(0, QUERY_NUM * QUERY_HOT, QUERY_HOT)) + [QUERY_NUM * QUERY_HOT])
        query_s = sp.csr_matrix((query_data, query_idxs, query_indptr), shape=(QUERY_NUM, SVEC_SIZE))

    if args.start_top_k > DATASET_WORD_SIZE:
        args.start_top_k = DATASET_WORD_SIZE

    if not args.no_cuda:
        print('Running on GPU')
        phrase_d = phrase_d.cuda()
        query_d = query_d.cuda()
        assert phrase_d.is_cuda and query_d.is_cuda
    else:
        print('Running on CPU')
        assert not phrase_d.is_cuda and not query_d.is_cuda

    results = []
    start_time = time.time()

    for batch_idx in tqdm(range(0, QUERY_NUM, BATCH_SIZE)):
        batch_query_d = query_d[batch_idx:batch_idx+BATCH_SIZE, :]
        batch_result = torch.matmul(batch_query_d, torch.transpose(phrase_d, 0, 1))

        if not args.sparse:
            argmax = torch.topk(batch_result, args.top_k, dim=1)[1]
            results.append(argmax)
        else:
            batch_query_s = query_s[batch_idx:batch_idx+BATCH_SIZE, :]
            max1, argmax1 = torch.topk(batch_result, args.start_top_k, dim=1)
            sp_result = torch.FloatTensor((batch_query_s * phrase_s[:argmax1.size(1),:].T).toarray())
            if not args.no_cuda:
                sp_result = sp_result.cuda()
            argmax2 = torch.topk(sp_result + max1, args.top_k, dim=1)[1]
            results.append(argmax2)

    elapsed_time = time.time() - start_time
    print('Elapsed sec: {}'.format(elapsed_time))



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', default=10, type=int)
    parser.add_argument('--start-top-k', default=10, type=int)
    parser.add_argument('--sparse', default=False, action='store_true')
    parser.add_argument('--no-cuda', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    benchmark(args)


if __name__ == '__main__':
    main()
