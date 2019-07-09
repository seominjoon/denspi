import argparse
import os
import numpy as np
from time import time

from flask import Flask, request, jsonify
from flask_cors import CORS

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from requests_futures.sessions import FuturesSession

from mips import MIPS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump_dir')
    parser.add_argument('wikipedia_dir')
    parser.add_argument('--dump_path', default='phrase')
    parser.add_argument('--index_name', default='1048576_hnsw_SQ8')
    parser.add_argument('--index_path', default='index.faiss')
    parser.add_argument('--idx2id_path', default='idx2id.hdf5')
    parser.add_argument('--abs_path', default=False, action='store_true')
    parser.add_argument('--port', default=10001, type=int)
    parser.add_argument('--api_port', default=9009, type=int)
    parser.add_argument('--max_answer_length', default=20, type=int)
    parser.add_argument('--start_top_k', default=1000, type=int)
    parser.add_argument('--mid_top_k', default=100, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--doc_top_k', default=5, type=int)
    parser.add_argument('--nprobe', default=64, type=int)
    parser.add_argument('--para', default=False, action='store_true')
    parser.add_argument('--num_dummy_zeros', default=0, type=int)

    # MIPS params
    parser.add_argument('--sparse_weight', default=0.05, type=float)
    parser.add_argument('--sparse_type', default='dp', type=str, help='dp|p|d|(empty_string)')
    parser.add_argument('--cuda', default=False, action='store_true')

    parser.add_argument('--filter', default=False, action='store_true')

    parser.add_argument('--examples_path', default='static/examples.txt')
    args = parser.parse_args()
    return args


def run_demo(args):
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

    mips = MIPS(dump_dir, index_path, idx2id_path, ranker_path, args.max_answer_length,
                para=args.para, tfidf_dump_dir=tfidf_dump_dir, sparse_weight=args.sparse_weight,
                sparse_type=args.sparse_type, cuda=args.cuda, max_norm_path=max_norm_path,
                num_dummy_zeros=args.num_dummy_zeros)

    app = Flask(__name__, static_url_path='/static')

    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    CORS(app)

    emb_session = FuturesSession()

    def search(query, top_k, nprobe=64, search_strategy='dense_first', doc_top_k=5):
        t0 = time()
        (start, end, span), _ = query2emb([query]*1, args.api_port)()
        query_vec = np.concatenate([start, end, span], 1)
        rets = mips.search(query_vec, top_k=top_k, nprobe=nprobe, start_top_k=args.start_top_k,
                           mid_top_k=args.mid_top_k, q_texts=[query]*1, filter_=args.filter,
                           search_strategy=search_strategy, doc_top_k=doc_top_k)
        t1 = time()
        out = {'ret': rets[0], 'time': int(1000 * (t1 - t0))}
        return out

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
        strat = request.args['strat']
        out = search(query,
                     args.top_k,
                     args.nprobe,
                     search_strategy=strat,
                     doc_top_k=args.doc_top_k)
        return jsonify(out)

    @app.route('/get_examples', methods=['GET'])
    def get_examples():
        with open(args.examples_path, 'r') as fp:
            examples = [line.strip() for line in fp.readlines()]
        return jsonify(examples)

    print('Starting server at %d' % args.port)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(args.port)
    IOLoop.instance().start()


def main():
    args = get_args()
    run_demo(args)


if __name__ == "__main__":
    main()
