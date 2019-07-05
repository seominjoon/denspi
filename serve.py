from flask import Flask, request, jsonify
from flask_cors import CORS

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from time import time


def serve(get_vec, port):
    app = Flask(__name__)

    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    CORS(app)

    @app.route('/api', methods=['GET'])
    def api():
        # query = request.args['query']
        query = request.args.getlist('query')
        start = time()
        out = get_vec(query)
        end = time()
        print('latency: %dms' % int((end - start) * 1000))
        return jsonify(out)

    print('Starting server at %d' % port)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    IOLoop.instance().start()
