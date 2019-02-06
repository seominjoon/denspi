import argparse
import json

import h5py
from drqa.retriever import TfidfDocRanker
from requests_futures.sessions import FuturesSession
from flask import request, jsonify
from tqdm import tqdm

from mips import DocumentPhraseMIPS


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('phrase_index_path')
    parser.add_argument('query_index_path')
    parser.add_argument('test_path')
    parser.add_argument('pred_path')
    parser.add_argument('--tfidf_path', default=None, type=str)
    parser.add_argument('--api_port', default=9009, type=int)
    parser.add_argument('--max_answer_length', default=30, type=int)
    parser.add_argument('--doc_score_cf', default=3e-2, type=float)
    parser.add_argument('--top_k_docs', default=5, type=int)
    parser.add_argument('--top_k_phrases', default=-1, type=int)
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
            query = (vec[:, :boundary_size], vec[:, boundary_size:2*boundary_size], vec[:, -1:])
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


if __name__ == '__main__':
    main()
