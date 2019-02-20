import argparse
import os

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from requests_futures.sessions import FuturesSession

from mips import MIPS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('phrase_dump_dir')
    parser.add_argument('index_dir')
    parser.add_argument('--start_index_path', default='target.faiss')
    parser.add_argument('--idx2id_path', default='idx2id.hdf5')
    parser.add_argument('--port', default=10001, type=int)
    parser.add_argument('--api_port', default=9009, type=int)
    parser.add_argument('--max_answer_length', default=20, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--nprobe', default=64, type=int)
    parser.add_argument('--para', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    start_index_path = os.path.join(args.index_dir, args.start_index_path)
    idx2id_path = os.path.join(args.index_dir, args.idx2id_path)

    mips = MIPS(args.phrase_dump_dir, start_index_path, idx2id_path, args.max_answer_length, para=args.para)

    app = Flask(__name__, static_url_path='/static')

    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    CORS(app)

    emb_session = FuturesSession()

    def search(query, top_k, nprobe=64):
        (start, end, span), _ = query2emb(query, args.api_port)()
        phrase_vec = np.concatenate([start, end, span], 1)
        rets = mips.search(phrase_vec, top_k=top_k, nprobe=nprobe)
        out = rets[0]
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
        out = search(query, args.top_k, args.nprobe)
        return jsonify(out)

    print('Starting server at %d' % args.port)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(args.port)
    IOLoop.instance().start()


if __name__ == "__main__":
    main()
