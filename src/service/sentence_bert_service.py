import os
import json
import time
import torch
import flask
import argparse
from gevent import pywsgi
from sentence_transformers import SentenceTransformer
import logging


app = flask.Flask(__name__)


@app.route('/sentence_bert', methods=["POST"])
def predict():
    data = {'success': False}

    logging.info(f"Request Content Type: {flask.request.headers.get('content_type')}")

    if flask.request.method == 'POST':
        request_json = flask.request.get_json(force=True)
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info(f"Server Received Json: ({time.strftime('%Y-%m-%d, %H:%M:%S')})")
        logging.info(json.dumps(request_json, indent=4, ensure_ascii=False))

        if 'sentence_list_1' not in request_json or 'sentence_list_2' not in request_json:
            data['scores'] = None
            data['message'] = f"request needs to contain `sentence_list_1` and `sentence_list_2`"
        elif not isinstance(request_json['sentence_list_1'], list) or not isinstance(request_json['sentence_list_2'], list):
            data['scores'] = None
            data['message'] = f"`sentence_list_1` and `sentence_list_2` should be two lists of strings, but got {type(request_json['sentence_list_1'])} and {type(request_json['sentence_list_2'])}"
        elif len(request_json['sentence_list_1']) == 0 or len(request_json['sentence_list_2']) == 0:
            data['scores'] = None
            data['message'] = f"`sentence_list_1` and `sentence_list_2` should be non-empty lists, but got length of {len(request_json['sentence_list_1'])} and {len(request_json['sentence_list_2'])}"
        elif not all([isinstance(sentence, str) and len(sentence) > 0 for sentence in request_json['sentence_list_1'] + request_json['sentence_list_2']]):
            data['scores'] = None
            data['message'] = f"each sentence in `sentence_list_1` and `sentence_list_2` should be non-empty strings"
        else:
            sentences = request_json['sentence_list_1'] + request_json['sentence_list_2']
            embeddings = model.encode(sentences)
            sim = model.similarity(embeddings[0:len(request_json['sentence_list_1'])], embeddings[len(request_json['sentence_list_1']):]).numpy().tolist()
            data['scores'] = sim
            data['success'] = True
        
        logging.info(f"Response Json:")
        logging.info(json.dumps(data, indent=4, ensure_ascii=False))
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        torch.cuda.empty_cache()
        
    return flask.jsonify(data)


def configure_log():
    log_path = f"sentence_bert-service-log/{time.strftime('%Y%m%d-%H%M%S')}.log"

    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_path,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence-bert-model-name-or-path', type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--port', type=int, default=65534)
    args = parser.parse_args()
    
    configure_log()
    
    model = SentenceTransformer(args.sentence_bert_model_name_or_path)
    
    server = pywsgi.WSGIServer(('0.0.0.0', args.port), app)
    server.serve_forever()
