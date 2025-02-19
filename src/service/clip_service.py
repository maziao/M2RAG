import os
import json
import time
import torch
import flask
import requests
import argparse
from io import BytesIO
from PIL import Image, ImageFile
from gevent import pywsgi
from transformers import CLIPModel, CLIPProcessor
import logging


app = flask.Flask(__name__)


def load_image_from_url(image_url: str) -> Image.Image:
    session = requests.Session()
    response = session.get(url=image_url, timeout=3.0)
    img = Image.open(BytesIO(response.content))
    img_format = img.format
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.format is None:
        tmp_img = BytesIO()
        img.save(tmp_img, format=img_format)
        img = Image.open(tmp_img)
    return img


def load_image_from_file(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    img_format = img.format
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.format is None:
        tmp_img = BytesIO()
        img.save(tmp_img, format=img_format)
        img = Image.open(tmp_img)
    return img


@app.route('/clip', methods=["POST"])
def predict():
    data = {'success': False}

    logging.info(f"Request Content Type: {flask.request.headers.get('content_type')}")

    if flask.request.method == 'POST':
        request_json = flask.request.get_json(force=True)
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info(f"Server Received Json: ({time.strftime('%Y-%m-%d, %H:%M:%S')})")
        logging.info(json.dumps(request_json, indent=4, ensure_ascii=False))

        # invalid request
        if 'image_urls' not in request_json and 'image_paths' not in request_json:
            return flask.jsonify(data)
        
        if 'image_urls' in request_json:
            pil_images = [load_image_from_url(image_url) for image_url in request_json['image_urls']]
        else:
            pil_images = [load_image_from_file(image_path) for image_path in request_json['image_paths']]
        reference_text = request_json['reference_text']
    
        scores = []
        for i in range(0, len(pil_images), args.batch_size):
            batch_pil_images = pil_images[i: i + args.batch_size]
            
            try:
                inputs = processor(text=[reference_text], images=batch_pil_images, return_tensors="pt", padding=True)
            except ValueError:
                batch_scores = [None] * len(batch_pil_images)
                scores.extend(batch_scores)
                continue

            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(args.device)
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            batch_scores = logits_per_image.detach().squeeze().cpu().numpy().tolist()
            if not isinstance(batch_scores, list):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
            
        data['scores'] = scores
        data['success'] = True
        
        logging.info(f"Response Json:")
        logging.info(json.dumps(data, indent=4, ensure_ascii=False))
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        torch.cuda.empty_cache()
        
    return flask.jsonify(data)


def configure_log():
    log_path = f"clip-service-log/{time.strftime('%Y%m%d-%H%M%S')}.log"

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
    parser.add_argument('--clip-model-name-or-path', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--port', type=int, default=65534)
    args = parser.parse_args()
    
    configure_log()
    
    assert args.batch_size >= 1
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model = CLIPModel.from_pretrained(args.clip_model_name_or_path).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_model_name_or_path)
    
    server = pywsgi.WSGIServer(('0.0.0.0', args.port), app)
    server.serve_forever()
