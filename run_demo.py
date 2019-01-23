import argparse

import h5py
from drqa.retriever import TfidfDocRanker

from mips import DocumentPhraseMIPS

parser = argparse.ArgumentParser()
parser.add_argument('tfidf_path')
parser.add_argument('phrase_index_path')
parser.add_argument('--port', default=10001, type=int)
parser.add_argument('--api_port', default=9009, type=int)
parser.add_argument('--max_answer_length', default=15, type=int)
parser.add_argument('--doc_score_cf', default=3e-2, type=float)
parser.add_argument('--top_k_docs', default=5, type=int)
parser.add_argument('--top_k_phrases', default=10, type=int)
parser.add_argument('-m', '--memory', default=False, action='store_true')
parser.add_argument('--multithread', default=False, action='store_true')
args = parser.parse_args()

# doc_ranker = TfidfDocRanker(args.tfidf_path)
# doc_mat = doc_ranker.doc_mat
doc_mat = None
phrase_index = h5py.File(args.phrase_index_path)
mips = DocumentPhraseMIPS(doc_mat, phrase_index, args.max_answer_length, args.doc_score_cf)

from flask import Flask, request, jsonify
from flask_cors import CORS

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from requests_futures.sessions import FuturesSession

app = Flask(__name__, static_url_path='/static')

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
CORS(app)

# Doc Index routine

emb_session = FuturesSession()

GROUP_LENGTH = 0


def search(query, top_k_docs, top_k_phrases):
    try:
        doc_vec = doc_ranker.text2spvec(query)
        phrase_vec, _ = query2emb(query, args.api_port)()
        rets = mips.search(doc_vec, phrase_vec, top_k_docs=top_k_docs, top_k_phrases=top_k_phrases)
        return rets

    except RuntimeError:
        print('%s: error' % query)
        return []


def search_(query, top_k_docs, top_k_phrases):
    try:
        phrase_vec, _ = query2emb(query, args.api_port)()
        rets = mips.search_phrase(0, phrase_vec, top_k=top_k_phrases)
        return rets
    except RuntimeError:
        print('%s: error' % query)
        return []


def query2emb(query, api_port):
    r = emb_session.get('http://localhost:%d/api' % api_port, params={'query': query})

    # print(r.url)
    def map_():
        result = r.result()
        print('response status: {0}'.format(result))
        emb = result.json()
        return emb, result.elapsed.total_seconds() * 1000

    return map_


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/files/<path:path>')
def static_files(path):
    return app.send_static_file('files/' + path)


@app.route('/api', methods=['GET'])
def api():
    query = request.args['query']
    out = search_(query, args.top_k_docs, args.top_k_phrases)
    context, s, e = out[0]['context'], out[0]['start_pos'], out[0]['end_pos']
    answer = context[s:e]
    print(answer)
    return jsonify(out)


print('Starting server at %d' % args.port)
http_server = HTTPServer(WSGIContainer(app))
http_server.listen(args.port)
IOLoop.instance().start()
