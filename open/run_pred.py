import argparse
import json
import os

import h5py
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp

from mips import MIPS
from mips_sparse import MIPSSparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('dump_dir')
    parser.add_argument('--question_dump_path', default='question.hdf5')

    parser.add_argument('--index_name', default='default_index')
    parser.add_argument('--index_path', default='index.faiss')
    parser.add_argument('--idx2id_path', default='idx2id.hdf5')
    parser.add_argument('--pred_dir', default='predictions')
    parser.add_argument('--ranker_path', default='wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
    parser.add_argument('--doc_mat_path', default='wikipedia/doc_tfidf.npz')

    # MIPS params
    parser.add_argument('--sparse_weight', default=1e-1, type=float)
    parser.add_argument('--start_top_k', default=1000, type=int)
    parser.add_argument('--mid_top_k', default=100, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--sparse_type', default='dp', type=str)

    # stable MIPS params
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--sparse', default=False, action='store_true')
    parser.add_argument('--no_od', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--fs', default='local')
    parser.add_argument('--num_dummy_zeros', default=0, type=int)
    parser.add_argument('--cuda', default=False, action='store_true')

    parser.add_argument('--filter', default=False, action='store_true')
    args = parser.parse_args()

    if args.fs == 'nfs':
        from nsml import NSML_NFS_OUTPUT
        args.data_path = os.path.join(NSML_NFS_OUTPUT, args.data_path)
        args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)
    phrase_dump_path = os.path.join(args.dump_dir, 'phrase.hdf5')
    args.phrase_dump_dir = phrase_dump_path if os.path.exists(phrase_dump_path) else os.path.join(args.dump_dir,
                                                                                                  'phrase')
    args.tfidf_dump_dir = os.path.join(args.dump_dir, 'tfidf')
    args.ranker_path = os.path.join(os.path.dirname(args.data_path), args.ranker_path)
    args.doc_mat_path = os.path.join(os.path.dirname(args.data_path), args.doc_mat_path)

    args.index_dir = os.path.join(args.dump_dir, args.index_name)
    args.index_path = os.path.join(args.index_dir, args.index_path)
    args.question_dump_path = os.path.join(args.dump_dir, args.question_dump_path)
    args.idx2id_path = os.path.join(args.index_dir, args.idx2id_path)

    args.pred_dir = os.path.join(args.dump_dir, args.pred_dir)
    out_name = '%s_%s_%.1f_%d_%d_%d' % (args.index_name, args.sparse_type, args.sparse_weight, args.start_top_k,
                                        args.top_k, args.nprobe)
    args.od_out_path = os.path.join(args.pred_dir, 'od_%s.json' % out_name)
    args.cd_out_path = os.path.join(args.pred_dir, 'cd_%s.json' % out_name)
    args.counter_path = os.path.join(args.pred_dir, 'counter.json')

    return args


def run_pred(args):
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

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

    with h5py.File(args.question_dump_path, 'r') as question_dump:
        vecs = []
        q_texts = []
        for doc_idx, para_idx, id_, question in tqdm(pairs):
            vec = question_dump[id_][0, :]
            vecs.append(vec)

            if args.sparse:
                q_texts.append(qid2text[id_])

        query = np.stack(vecs, 0)
        if args.draft:
            query = query[:3]

    if not args.sparse:
        mips = MIPS(args.phrase_dump_dir, args.index_path, args.idx2id_path, args.max_answer_length, para=args.para,
                    num_dummy_zeros=args.num_dummy_zeros, cuda=args.cuda)
    else:
        from drqa import retriever
        if args.draft:
            text2spvec, doc_mat = None, None
        else:
            ranker = retriever.get_class('tfidf')(
                args.ranker_path,
                strict=False
            )
            text2spvec = ranker.text2spvec
            print('Ranker loaded from {}'.format(args.ranker_path))
            doc_mat = sp.load_npz(args.doc_mat_path)
            print('Doc TFIDF matrix loaded {}'.format(doc_mat.shape))

        mips = MIPSSparse(args.phrase_dump_dir, args.index_path, args.idx2id_path, args.max_answer_length,
                          para=args.para, tfidf_dump_dir=args.tfidf_dump_dir, sparse_weight=args.sparse_weight,
                          text2spvec=text2spvec, doc_mat=doc_mat, sparse_type=args.sparse_type, cuda=args.cuda)

    # recall at k
    cd_results = []
    od_results = []
    step_size = args.step_size
    is_ = range(0, query.shape[0], step_size)
    for i in tqdm(is_):
        each_query = query[i:i + step_size]
        if args.sparse:
            each_q_text = q_texts[i:i+step_size]

        if args.no_od:
            doc_idxs, para_idxs, _, _ = zip(*pairs[i:i + step_size])
            if not args.sparse:
                each_results = mips.search(each_query, top_k=args.top_k, doc_idxs=doc_idxs, para_idxs=para_idxs)
            else:
                each_results = mips.search(each_query, top_k=args.top_k, doc_idxs=doc_idxs, para_idxs=para_idxs,
                                           start_top_k=args.start_top_k, q_texts=each_q_text)
            cd_results.extend(each_results)

        else:
            if not args.sparse:
                each_results = mips.search(each_query, top_k=args.top_k, nprobe=args.nprobe)
            else:
                each_results = mips.search(each_query, top_k=args.top_k, nprobe=args.nprobe, mid_top_k=args.mid_top_k,
                                           start_top_k=args.start_top_k, q_texts=each_q_text, filter_=args.filter)
            od_results.extend(each_results)
        if i % 10 == 0:
            print('%d/%d' % (i+1, len(is_)))

    top_k_answers = {query_id: [result['answer'] for result in each_results]
                     for (_, _, query_id, _), each_results in zip(pairs, od_results)}
    answers = {query_id: each_results[0]['answer']
               for (_, _, query_id, _), each_results in zip(pairs, cd_results)}

    if args.para:
        print('dumping %s' % args.cd_out_path)
        with open(args.cd_out_path, 'w') as fp:
            json.dump(answers, fp)

    print('dumping %s' % args.od_out_path)
    with open(args.od_out_path, 'w') as fp:
        json.dump(top_k_answers, fp)

    from collections import Counter
    counter = Counter(result['doc_idx'] for each in od_results for result in each)
    with open(args.counter_path, 'w') as fp:
        json.dump(counter, fp)


def main():
    args = get_args()
    run_pred(args)


if __name__ == '__main__':
    main()
