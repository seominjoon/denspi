import argparse
import json
import os

import h5py
from tqdm import tqdm
import numpy as np

from mips import MIPS
from mips_sparse import MIPSSparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('dump_dir')
    parser.add_argument('index_dir')
    parser.add_argument('--index_path', default='default_index/index.faiss')
    parser.add_argument('--quantizer_path', default='quantizer.faiss')
    parser.add_argument('--question_dump_path', default='question.hdf5')
    parser.add_argument('--od_out_path', default='pred_od.json')
    parser.add_argument('--cd_out_path', default="pred_cd.json")
    parser.add_argument('--idx2id_path', default='default_index/idx2id.hdf5')
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--sparse_weight', default=3e+0, type=float)
    parser.add_argument('--start_top_k', default=100, type=int)
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--no_od', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--sparse', default=False, action='store_true')
    parser.add_argument('--nprobe', default=64, type=int)
    parser.add_argument('--fs', default='local')
    parser.add_argument('--step_size', default=10, type=int)
    args = parser.parse_args()
    return args


def run_pred(args):
    if args.fs == 'nfs':
        from nsml import NSML_NFS_OUTPUT
        args.data_path = os.path.join(NSML_NFS_OUTPUT, args.data_path)
        args.dump_dir = os.path.join(NSML_NFS_OUTPUT, args.dump_dir)
        args.index_dir = os.path.join(NSML_NFS_OUTPUT, args.index_dir)
    phrase_dump_path = os.path.join(args.dump_dir, 'phrase.hdf5')
    args.phrase_dump_dir = phrase_dump_path if os.path.exists(phrase_dump_path) else os.path.join(args.dump_dir,
                                                                                                  'phrase')
    args.index_path = os.path.join(args.index_dir, args.index_path)
    args.quantizer_path = os.path.join(args.index_dir, args.quantizer_path)
    args.question_dump_path = os.path.join(args.dump_dir, args.question_dump_path)
    # args.od_out_path = os.path.join(args.dir, args.od_out_path)
    # args.cd_out_path = os.path.join(args.dir, args.cd_out_path)
    args.idx2id_path = os.path.join(args.index_dir, args.idx2id_path)

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
        sparses = []
        input_idss = []
        for doc_idx, para_idx, id_, question in tqdm(pairs):
            vec = question_dump[id_][0, :]
            vecs.append(vec)

            if (id_ + '_sparse') in question_dump and args.sparse:
                sparse = question_dump[id_ + '_sparse'][:]
                input_ids = question_dump[id_ + '_input_ids'][:]
                sparses.append(sparse)
                input_idss.append(input_ids)
                # print(sparse)
                # print(qid2text[id_])

        query = np.stack(vecs, 0)
        if args.draft:
            query = query[:100]

    if not args.sparse:
        mips = MIPS(args.phrase_dump_dir, args.index_path, args.idx2id_path, args.max_answer_length, para=args.para)
    else:
        mips = MIPSSparse(args.phrase_dump_dir, args.index_path, args.idx2id_path, args.max_answer_length,
                          para=args.para)

    # recall at k
    cd_results = []
    od_results = []
    step_size = args.step_size
    for i in tqdm(range(0, query.shape[0], step_size)):
        each_query = query[i:i + step_size]

        if len(sparses) > 0:
            each_sparse = sparses[i:i + step_size]
            each_input_ids = input_idss[i:i + step_size]

        if args.no_od:
            doc_idxs, para_idxs, _, _ = zip(*pairs[i:i + step_size])
            if len(sparses) == 0:
                each_results = mips.search(each_query, top_k=args.top_k, doc_idxs=doc_idxs, para_idxs=para_idxs)
            else:
                each_results = mips.search(each_query, top_k=args.top_k, doc_idxs=doc_idxs, para_idxs=para_idxs,
                                           q_sparse=each_sparse, q_input_ids=each_input_ids,
                                           start_top_k=args.start_top_k, sparse_weight=args.sparse_weight)
            cd_results.extend(each_results)

        else:
            if len(sparses) == 0:
                each_results = mips.search(each_query, top_k=args.top_k, nprobe=args.nprobe)
            else:
                each_results = mips.search(each_query, top_k=args.top_k, nprobe=args.nprobe,
                                           q_sparse=each_sparse, q_input_ids=each_input_ids,
                                           start_top_k=args.start_top_k, sparse_weight=args.sparse_weight)
            od_results.extend(each_results)

    top_k_answers = {query_id: [result['answer'] for result in each_results]
                     for (_, _, query_id, _), each_results in zip(pairs, od_results)}
    answers = {query_id: each_results[0]['answer']
               for (_, _, query_id, _), each_results in zip(pairs, cd_results)}

    if args.para:
        print('dumping %s' % args.cd_out_path)
        with open(args.cd_out_path, 'w') as fp:
            json.dump(answers, fp)

    while os.path.exists(args.od_out_path):
        args.od_out_path = args.od_out_path + '_'
    print('dumping %s' % args.od_out_path)
    with open(args.od_out_path, 'w') as fp:
        json.dump(top_k_answers, fp)


def main():
    args = get_args()
    run_pred(args)


if __name__ == '__main__':
    main()
