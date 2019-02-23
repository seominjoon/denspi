import argparse
import json
import os

import h5py
from tqdm import tqdm
import numpy as np

from mips import MIPS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('dir')
    parser.add_argument('--phrase_dump_dir', default='phrase.hdf5')
    parser.add_argument('--index_path', default='default_index/index.faiss')
    parser.add_argument('--quantizer_path', default='quantizer.faiss')
    parser.add_argument('--question_dump_path', default='question.hdf5')
    parser.add_argument('--od_out_path', default='pred_od.json')
    parser.add_argument('--cd_out_path', default="pred_cd.json")
    parser.add_argument('--idx2id_path', default='default_index/idx2id.hdf5')
    parser.add_argument('--max_norm', default=None, type=float)
    parser.add_argument('--num_clusters', default=524288, type=int)
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--doc_sample_ratio', default=0.1, type=float)
    parser.add_argument('--vec_sample_ratio', default=0.1, type=float)
    parser.add_argument('--draft', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.phrase_dump_dir = os.path.join(args.dir, args.phrase_dump_dir)
    args.index_path = os.path.join(args.dir, args.index_path)
    args.quantizer_path = os.path.join(args.dir, args.quantizer_path)
    args.question_dump_path = os.path.join(args.dir, args.question_dump_path)
    args.od_out_path = os.path.join(args.dir, args.od_out_path)
    args.cd_out_path = os.path.join(args.dir, args.cd_out_path)
    args.idx2id_path = os.path.join(args.dir, args.idx2id_path)

    with open(args.data_path, 'r') as fp:
        test_data = json.load(fp)
    pairs = []
    for doc_idx, article in enumerate(test_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                id_ = qa['id']
                question = qa['question']
                pairs.append([doc_idx, para_idx, id_, question])

    question_dump = h5py.File(args.question_dump_path)

    mips = MIPS(args.phrase_dump_dir, args.index_path, args.idx2id_path, args.max_answer_length, para=args.para)

    vecs = []
    for doc_idx, para_idx, id_, question in tqdm(pairs):
        vec = question_dump[id_][0, :]
        vecs.append(vec)
    query = np.stack(vecs, 0)
    if args.draft:
        query = query[:100]

    # recall at k
    cd_results = []
    od_results = []
    step_size = 10
    for i in tqdm(range(0, query.shape[0], step_size)):
        each_query = query[i:i+step_size]

        if args.para:
            doc_idxs, para_idxs, _, _ = zip(*pairs[i:i+step_size])
            each_results = mips.search(each_query, top_k=args.top_k, doc_idxs=doc_idxs, para_idxs=para_idxs)
            cd_results.extend(each_results)

        each_results = mips.search(each_query, top_k=args.top_k)
        od_results.extend(each_results)
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


if __name__ == '__main__':
    main()
