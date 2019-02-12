import argparse
import json
from collections import defaultdict

import h5py
from drqa.retriever import TfidfDocRanker
from requests_futures.sessions import FuturesSession
from flask import request, jsonify
from tqdm import tqdm
import numpy as np

from mips import DocumentPhraseMIPS, MIPS


def query2emb(query, api_port):
    emb_session = FuturesSession()
    r = emb_session.get('http://localhost:%d/api' % api_port, params={'query': query})

    # print(r.url)
    def map_():
        result = r.result()
        emb = result.json()
        return emb, result.elapsed.total_seconds() * 1000

    return map_


def get_answer_from_para(mips, query, doc_idx, para_idx, top_k_phrases):
    ret = mips.search_phrase(doc_idx, query, top_k=top_k_phrases, para_idx=para_idx)[0]
    answer = ret['context'][ret['start_pos']:ret['end_pos']]
    return answer


def get_answers(mips, query, top_k, ):
    rets = mips.search_phrase_global(query, top_k=top_k)
    answers = [ret['context'][ret['start_pos']:ret['end_pos']] for ret in rets]
    return answers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('phrase_index_path')
    parser.add_argument('query_index_path')
    parser.add_argument('test_path')
    parser.add_argument('pred_path')
    parser.add_argument('--tfidf_path', default=None, type=str)
    parser.add_argument('--faiss_path', default=None, type=str)
    parser.add_argument('--api_port', default=9009, type=int)
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--doc_score_cf', default=3e-2, type=float)
    parser.add_argument('--top_k_docs', default=5, type=int)
    parser.add_argument('--top_k_phrases', default=5, type=int)
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--ip2l2', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    phrase_index = h5py.File(args.phrase_index_path)
    query_index = h5py.File(args.query_index_path)

    with open(args.test_path, 'r') as fp:
        test_data = json.load(fp)
    preds = {}
    pairs = []
    for doc_idx, article in enumerate(test_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                id_ = qa['id']
                question = qa['question']
                pairs.append([doc_idx, para_idx, id_, question])
    # pairs = pairs[2112:]

    if args.tfidf_path is None:
        mips = DocumentPhraseMIPS(None, phrase_index, args.max_answer_length, args.doc_score_cf)
        for doc_idx, para_idx, id_, question in tqdm(pairs):
            vec = query_index[id_][:]
            boundary_size = int((vec.shape[1] - 1) / 2)
            query = (vec[:, :boundary_size], vec[:, boundary_size:2 * boundary_size], vec[:, -1:])
            answer = get_answer_from_para(mips, query, doc_idx, para_idx, args.top_k_phrases)
            preds[id_] = answer
    else:
        doc_ranker = TfidfDocRanker(args.tfidf_path)
        print('tfidf doc ranker loaded from %s' % args.tfidf_path)
        doc_mat = doc_ranker.doc_mat
        mips = DocumentPhraseMIPS(doc_mat, phrase_index, args.max_answer_length, args.doc_score_cf)

        q2d = {}
        for _, _, id_, question in tqdm(pairs):
            doc_results = mips.search_document(doc_ranker.text2spvec(question), args.top_k_docs)
            q2d[id_] = [result['doc_idx'] for result in doc_results]
        with open('q2d.json', 'w') as fp:
            json.dump(q2d, fp)

        for doc_idx, para_idx, id_, question in tqdm(pairs):
            break
            answer = get_answer(mips, question, args.top_k_docs, args.top_k_phrases, args.api_port)
            preds[id_] = answer

    with open(args.pred_path, 'w') as fp:
        json.dump(preds, fp)


def main2():
    args = get_args()
    phrase_index = h5py.File(args.phrase_index_path)
    query_index = h5py.File(args.query_index_path)

    with open(args.test_path, 'r') as fp:
        test_data = json.load(fp)
    preds = {}
    pairs = []
    for doc_idx, article in enumerate(test_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                id_ = qa['id']
                question = qa['question']
                pairs.append([doc_idx, para_idx, id_, question])
    # pairs = pairs[2112:]

    mips = DocumentPhraseMIPS(None, phrase_index, args.max_answer_length, args.doc_score_cf,
                              start_index=args.faiss_path, ip2l2=args.ip2l2)
    recall_at_k_dict = defaultdict(list)
    for doc_idx, para_idx, id_, question in tqdm(pairs):
        vec = query_index[id_][:]
        boundary_size = int((vec.shape[1] - 1) / 2)
        query = (vec[:, :boundary_size], vec[:, boundary_size:2 * boundary_size], vec[:, -1:])
        answer = get_answer(mips, query, args.top_k_phrases)
        preds[id_] = answer

        # recall at k
        doc_idxs, _, _ = mips.search_start(query[0], args.top_k_phrases)
        doc_idxs = list(map(int, doc_idxs))
        for k in range(1, args.top_k_phrases + 1):
            recall_at_k_dict[k].append(int(doc_idx in doc_idxs[:k]))

        if args.draft and doc_idx > 0:
            break

    recall_at_k = {key: np.mean(val) for key, val in recall_at_k_dict.items()}
    print(recall_at_k)

    with open(args.pred_path, 'w') as fp:
        json.dump(preds, fp)


def main3():
    args = get_args()
    phrase_index = h5py.File(args.phrase_index_path)
    query_index = h5py.File(args.query_index_path)

    with open(args.test_path, 'r') as fp:
        test_data = json.load(fp)
    preds = {}
    pairs = []
    for doc_idx, article in enumerate(test_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                id_ = qa['id']
                question = qa['question']
                pairs.append([doc_idx, para_idx, id_, question])
    # pairs = pairs[2112:]

    mips = DocumentPhraseMIPS(None, phrase_index, args.max_answer_length, args.doc_score_cf,
                              start_index=args.faiss_path, ip2l2=args.ip2l2)
    recall_at_k_dict = defaultdict(list)
    top_k_answers = defaultdict(list)

    vecs = []
    gt_doc_idxs = []
    queries = []
    query_ids = []
    for doc_idx, para_idx, id_, question in tqdm(pairs):
        vec = query_index[id_][:]
        boundary_size = int((vec.shape[1] - 1) / 2)
        query = (vec[:, :boundary_size], vec[:, boundary_size:2 * boundary_size], vec[:, -1:])
        vecs.append(query[0])
        gt_doc_idxs.append(doc_idx)
        queries.append(query)
        query_ids.append(id_)

    vecs = np.concatenate(vecs, axis=0)
    if args.draft:
        vecs = vecs[:100]

    # recall at k
    results = mips.search_start(vecs, args.top_k_phrases)
    for gt_doc_idx, (doc_idxs, start_idxs, _), query, query_id in tqdm(zip(gt_doc_idxs, results, queries, query_ids),
                                                                       total=len(gt_doc_idxs)):
        doc_idxs = list(map(int, doc_idxs))
        start_idxs = list(map(int, start_idxs))
        results = sum(
            [mips.search_phrase(doc_idx, query, top_k=args.top_k_phrases, start_idx=start_idx) for doc_idx, start_idx in
             zip(doc_idxs, start_idxs)], [])
        results = sorted(results, key=lambda result: -result['score'])[:args.top_k_phrases]
        top_k_answers[query_id] = [result['answer'] for result in results]
        for k in range(1, args.top_k_phrases + 1):
            recall_at_k_dict[k].append(int(gt_doc_idx in doc_idxs[:k]))

    recall_at_k = {key: np.mean(val) for key, val in recall_at_k_dict.items()}
    print(recall_at_k)

    with open(args.pred_path, 'w') as fp:
        json.dump(top_k_answers, fp)


def main4():
    args = get_args()

    with open(args.test_path, 'r') as fp:
        test_data = json.load(fp)
    pairs = []
    for doc_idx, article in enumerate(test_data['data']):
        for para_idx, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                id_ = qa['id']
                question = qa['question']
                pairs.append([doc_idx, para_idx, id_, question])

    query_index = h5py.File(args.query_index_path)

    # pairs = pairs[2112:]

    mips = MIPS(args.phrase_index_path, args.faiss_path, args.max_answer_length, load_to_memory=True)

    vecs = []
    for doc_idx, para_idx, id_, question in tqdm(pairs):
        vec = query_index[id_][0, :]
        vecs.append(vec)
    query = np.stack(vecs, 0)
    if args.draft:
        query = query[:100]

    # recall at k
    results = []
    step_size = 10
    for i in tqdm(range(0, query.shape[0], step_size)):
        each_query = query[i:i+step_size]
        each_results = mips.search(each_query, top_k=args.top_k_phrases)
        results.extend(each_results)
    top_k_answers = {query_id: [result['answer'] for result in each_results]
                     for (_, _, query_id, _), each_results in zip(pairs, results)}

    with open(args.pred_path, 'w') as fp:
        json.dump(top_k_answers, fp)


if __name__ == '__main__':
    main4()
