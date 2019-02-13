import argparse

import numpy as np

from mips import MIPS

parser = argparse.ArgumentParser()
parser.add_argument('phrase_index_path')
parser.add_argument('start_index_path')
parser.add_argument('--port', default=10001, type=int)
parser.add_argument('--api_port', default=9009, type=int)
parser.add_argument('--max_answer_length', default=20, type=int)
parser.add_argument('--top_k', default=10, type=int)
parser.add_argument('--nprobe', default=64, type=int)
parser.add_argument('-m', '--load_to_memory', default=False, action='store_true')
args = parser.parse_args()

mips = MIPS(args.phrase_index_path, args.start_index_path, args.max_answer_length, load_to_memory=args.load_to_memory)

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
