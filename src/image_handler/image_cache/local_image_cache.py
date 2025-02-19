import os
from PIL import Image
from typing import Optional
from src.image_handler.image_cache import IMAGE_CACHE
from src.image_handler.image_cache.image_cache import ImageCache


@IMAGE_CACHE.register_module
class LocalImageCache(ImageCache):
    def __init__(
        self,
        image_root: Optional[str] = None,
        image_base_url: Optional[str] = None
    ) -> None:
        super().__init__()
        if image_root is None:
            image_root = os.environ.get('IMAGE_ROOT')
        assert image_root is not None
        self.image_root = image_root
        
        if image_base_url is None:
            image_base_url = os.environ.get('IMAGE_URL')
        assert image_base_url is not None
        self.image_base_url = image_base_url
        
    def save(
        self,
        pil_image: Image.Image,
        relative_path: str,
        **kwargs
    ):
        absolute_path = os.path.join(self.image_root, relative_path)
        save_dir = os.path.dirname(absolute_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        pil_image.save(absolute_path, pil_image.format)
        url = os.path.join(self.image_base_url, relative_path)
        del pil_image
        return url
    
    def delete(
        self,
        relative_path: str
    ):
        absolute_path = os.path.join(self.image_root, relative_path)
        if os.path.exists(absolute_path):
            os.remove(absolute_path)
