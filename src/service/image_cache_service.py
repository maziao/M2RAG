import os
import re
import json
import time
import flask
import base64
import socket
import urllib
import argparse
import cairosvg
from PIL import Image, UnidentifiedImageError, ImageFile
from io import BytesIO
from gevent import pywsgi
import logging


app = flask.Flask(__name__)


def base64_to_image(base64_str: str) -> Image.Image:
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def convert_image(image_url: str):
    socket.setdefaulttimeout(3)
    try:
        urllib.request.urlretrieve(image_url, "tmp_img")
    except urllib.error.HTTPError:
        return None
    except socket.timeout:
        return None
    except urllib.error.URLError:
        return None
    socket.setdefaulttimeout(None)
    
    try:
        img = Image.open("tmp_img")
        if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
            img.load()
            img = img.convert('RGB')
            img.save("tmp_img", "JPEG")
            img = Image.open("tmp_img")
    except UnidentifiedImageError:
        try:
            cairosvg.svg2png(url="tmp_img", write_to="tmp_img")
            img = Image.open("tmp_img")
            img.load()
            img = img.convert('RGB')
            img.save("tmp_img", "JPEG")
            img = Image.open("tmp_img")
        except Exception:
            img = None
        
    if img is not None:
        try:
            img.save("tmp_img", img.format)
        except Exception:
            img = None
    
    return img
    
    
@app.route('/image_cache', methods=["POST"])
def predict():
    data = {'success': False}

    logging.info(f"Request Content Type: {flask.request.headers.get('content_type')}")

    if flask.request.method == 'POST':
        request_json = flask.request.get_json(force=True)
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info(f"Server Received Json: ({time.strftime('%Y-%m-%d, %H:%M:%S')})")
        logging.info(json.dumps(request_json, indent=4, ensure_ascii=False))

        if request_json['service'] == 'save':
            if 'image_url' not in request_json and 'base64_image' not in request_json:
                return flask.jsonify(data)
            
            if 'image_url' in request_json:
                assert 'relative_basename' in request_json
                pil_image = convert_image(image_url=request_json['image_url'])
                if pil_image is None:
                    return flask.jsonify(data)
                relative_path = f"{request_json['relative_basename']}.{pil_image.format.lower()}"
            else:
                pil_image = base64_to_image(request_json['base64_image'])
                relative_path = request_json['relative_path']
            save_path = os.path.join(image_root, relative_path) 
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            pil_image.save(save_path, pil_image.format)
            data['cached_image_url'] = os.path.join(image_base_url, relative_path)
            data['success'] = True
        elif request_json['service'] == 'delete':
            delete_path = os.path.join(image_root, request_json['relative_path'])
            if os.path.exists(delete_path):
                os.remove(delete_path)
            data['success'] = True
        else:
            data['success'] = False
            data['error'] = f"service name `{request_json['service']}` not identified."
        
        logging.info(f"Response Json:")
        logging.info(json.dumps(data, indent=4, ensure_ascii=False))
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    else:
        logging.info(f"Error: invalid request {flask.request.get_json(force=True)}")
    return flask.jsonify(data)


def configure_log():
    log_path = f"image-cache-service-log/{time.strftime('%Y%m%d-%H%M%S')}.log"

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
    parser.add_argument('--image-root', type=str, default=None)
    parser.add_argument('--image-base-url', type=str, default=None)
    parser.add_argument('--port', type=int, default=65534)
    parser.add_argument('--num-threads', type=int, default=50)
    args = parser.parse_args()
    
    configure_log()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    if args.image_root is None:
        image_root = os.environ.get('IMAGE_ROOT')
    else:
        image_root = args.image_root
        
    if args.image_base_url is None:
        image_base_url = os.environ.get('IMAGE_URL')
    else:
        image_base_url = args.image_base_url
        
    server = pywsgi.WSGIServer(('0.0.0.0', args.port), app, threads=args.num_threads)
    server.serve_forever()
