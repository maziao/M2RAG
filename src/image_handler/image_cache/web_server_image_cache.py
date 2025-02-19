import os
import base64
import logging
import requests
from io import BytesIO
from PIL import Image
from typing import Optional
from src.image_handler.image_cache import IMAGE_CACHE
from src.image_handler.image_cache.image_cache import ImageCache


def image_to_base64(image: Image.Image, fmt='jpeg') -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{fmt};base64,' + base64_str


@IMAGE_CACHE.register_module
class WebServerImageCache(ImageCache):
    def __init__(
        self,
        image_cache_service_url: Optional[str] = None
    ) -> None:
        super().__init__()
        if image_cache_service_url is None:
            image_cache_service_url = os.environ.get('IMAGE_CACHE_SERVICE_URL')
        assert image_cache_service_url is not None
        self.image_cache_service_url = image_cache_service_url
        
    def save(
        self,
        pil_image: Image.Image,
        relative_path: str,
        **kwargs
    ):
        request_json = {
            "service": "save"
        }
        image_url = kwargs.pop('image_url', None)
        relative_basename = kwargs.pop('relative_basename', None)
        if image_url is None or relative_basename is None:
            request_json["base64_image"] = image_to_base64(image=pil_image, fmt=pil_image.format)
            request_json["relative_path"] = relative_path
        else:
            request_json["image_url"] = image_url
            request_json["relative_basename"] = relative_basename
        
        response = requests.post(
            self.image_cache_service_url,
            json=request_json,
            headers={'content-type': 'application/json'}
        )
        try:
            response = response.json()
        except Exception:
            logging.warning(f"cache image {relative_path} to web server failed, got response {response}. Ignore this image.")
            response = {'success': False}
        if response['success']:
            return response['cached_image_url']
        else:
            return None
    
    def delete(
        self,
        relative_path: str
    ):
        _ = requests.post(
            self.image_cache_service_url,
            json={
                "service": "delete",
                "relative_path": relative_path
            },
            headers={'content-type': 'application/json'}
        ).json()
        