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
    parser.add_argument('--phrase_index_path', default='phrase.hdf5')
    parser.add_argument('--faiss_path', default='index.faiss')
    parser.add_argument('--question_index_path', default='question.hdf5')
    parser.add_argument('--od_out_path', default='pred_od.json')
    parser.add_argument('--cd_out_path', default="pred_cd.json")
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--index_factory', default="IVF4096,SQ8")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.phrase_index_path = os.path.join(args.dir, args.phrase_index_path)
    args.faiss_path = os.path.join(args.dir, args.faiss_path)
    args.question_index_path = os.path.join(args.dir, args.question_index_path)
    args.od_out_path = os.path.join(args.dir, args.od_out_path)
    args.cd_out_path = os.path.join(args.dir, args.cd_out_path)

    with open(args.data_path, 'r') as fp:
        test_data = json.load(fp)
    pairs = []
    for doc_idx, article in enumerate(test_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                id_ = qa['id']
                question = qa['question']
                pairs.append([doc_idx, para_idx, id_, question])

    question_index = h5py.File(args.question_index_path)

    mips = MIPS(args.phrase_index_path, args.faiss_path, args.max_answer_length, load_to_memory=True, para=args.para,
                index_factory=args.index_factory)

    vecs = []
    for doc_idx, para_idx, id_, question in tqdm(pairs):
        vec = question_index[id_][0, :]
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
        with open(args.cd_out_path, 'w') as fp:
            json.dump(answers, fp)

    with open(args.od_out_path, 'w') as fp:
        json.dump(top_k_answers, fp)


if __name__ == '__main__':
    main()
