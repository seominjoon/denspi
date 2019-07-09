import argparse
import json
import os

import h5py
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp

from mips import MIPS


def get_args():
    parser = argparse.ArgumentParser()
    # File paths
    parser.add_argument('dump_dir')
    parser.add_argument('wikipedia_dir')
    parser.add_argument('data_path')
    parser.add_argument('--question_dump_path', default='question.hdf5')
    parser.add_argument('--dump_path', default='phrase')
    parser.add_argument('--index_name', default='1048576_hnsw_SQ8')
    parser.add_argument('--index_path', default='index.faiss')
    parser.add_argument('--idx2id_path', default='idx2id.hdf5')
    parser.add_argument('--abs_path', default=False, action='store_true')
    parser.add_argument('--pred_dir', default='predictions')

    # MIPS params
    parser.add_argument('--max_answer_length', default=20, type=int)
    parser.add_argument('--start_top_k', default=1000, type=int)
    parser.add_argument('--mid_top_k', default=100, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--doc_top_k', default=5, type=int)
    parser.add_argument('--nprobe', default=64, type=int)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--num_dummy_zeros', default=0, type=int)
    parser.add_argument('--sparse_weight', default=0.05, type=float)
    parser.add_argument('--sparse_type', default='dp', type=str, help='dp|p|d|(empty_string)')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--filter', default=False, action='store_true')
    parser.add_argument('--search_strategy', default='dense_first')

    # Eval params
    parser.add_argument('--no_od', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--step_size', default=32, type=int)
    parser.add_argument('--fs', default='local')

    args = parser.parse_args()
    return args


def run_pred(args):
    # For NFS
    if args.fs == 'nfs':
        from nsml import NSML_NFS_OUTPUT
        args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)
        args.wikipedia_dir = os.path.join(NSML_NFS_OUTPUT, args.wikipedia_dir)
        args.data_path = os.path.join(NSML_NFS_OUTPUT, args.data_path)

    # MIPS files
    dump_dir = os.path.join(args.dump_dir, args.dump_path)
    if args.abs_path:
        index_dir = args.index_name
    else:
        index_dir = os.path.join(args.dump_dir, args.index_name)
    index_path = os.path.join(index_dir, args.index_path)
    idx2id_path = os.path.join(index_dir, args.idx2id_path)
    tfidf_dump_dir = os.path.join(args.dump_dir, 'tfidf')
    max_norm_path = os.path.join(index_dir, 'max_norm.json')
    ranker_path = os.path.join(args.wikipedia_dir, 'docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')

    # Evaluation files
    question_dump_path = os.path.join(args.dump_dir, args.question_dump_path)
    pred_dir = os.path.join(args.dump_dir, args.pred_dir)
    out_name = '%s_%s_%.2f_%d_%d_%d' % (args.index_name, args.sparse_type, args.sparse_weight, args.start_top_k,
                                        args.top_k, args.nprobe)
    od_out_path = os.path.join(pred_dir, 'od_%s.json' % out_name)
    cd_out_path = os.path.join(pred_dir, 'cd_%s.json' % out_name)
    counter_path = os.path.join(pred_dir, 'counter.json')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    with open(args.data_path, 'r') as fp:
        test_data = json.load(fp)
    pairs = []
    qid2text = {}
    for doc_idx, article in enumerate(test_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                id_ = qa['id']
                question = qa['question']
                qid2text[id_] = question
                pairs.append([doc_idx, para_idx, id_, question])

    with h5py.File(question_dump_path, 'r') as question_dump:
        vecs = []
        q_texts = []
        for doc_idx, para_idx, id_, question in tqdm(pairs):
            vec = question_dump[id_][0, :]
            vecs.append(vec)
            q_texts.append(qid2text[id_])

        query = np.stack(vecs, 0)
        if args.draft:
            query = query[:3]

    mips = MIPS(dump_dir, index_path, idx2id_path, ranker_path, args.max_answer_length,
                para=args.para, tfidf_dump_dir=tfidf_dump_dir, sparse_weight=args.sparse_weight,
                sparse_type=args.sparse_type, cuda=args.cuda, max_norm_path=max_norm_path,
                num_dummy_zeros=args.num_dummy_zeros)

    # recall at k
    cd_results = []
    od_results = []
    step_size = args.step_size
    is_ = range(0, query.shape[0], step_size)
    for i in tqdm(is_):
        each_query = query[i:i + step_size]
        each_q_text = q_texts[i:i + step_size]

        if args.no_od:
            doc_idxs, para_idxs, _, _ = zip(*pairs[i:i + step_size])
            each_results = mips.search(each_query, top_k=args.top_k, doc_idxs=doc_idxs, para_idxs=para_idxs,
                                       start_top_k=args.start_top_k, q_texts=each_q_text)
            cd_results.extend(each_results)

        else:
            each_results = mips.search(each_query, top_k=args.top_k, nprobe=args.nprobe, mid_top_k=args.mid_top_k,
                                       start_top_k=args.start_top_k, q_texts=each_q_text, filter_=args.filter,
                                       search_strategy=args.search_strategy,
                                       doc_top_k=args.doc_top_k)
            od_results.extend(each_results)

    top_k_answers = {query_id: [result['answer'] for result in each_results]
                     for (_, _, query_id, _), each_results in zip(pairs, od_results)}
    answers = {query_id: each_results[0]['answer']
               for (_, _, query_id, _), each_results in zip(pairs, cd_results)}

    if args.para:
        print('dumping %s' % cd_out_path)
        with open(cd_out_path, 'w') as fp:
            json.dump(answers, fp)

    print('dumping %s' % od_out_path)
    with open(od_out_path, 'w') as fp:
        json.dump(top_k_answers, fp)

    from collections import Counter
    counter = Counter(result['doc_idx'] for each in od_results for result in each)
    with open(counter_path, 'w') as fp:
        json.dump(counter, fp)


def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


def main():
    args = get_args()
    run_pred(args)


if __name__ == '__main__':
    main()
