import os
import re
import urllib
import tempfile
import urllib.request
from abc import abstractmethod
from typing import List, Optional
from src.image_handler.image_dedup import IMAGE_DEDUP


@IMAGE_DEDUP.register_module
class ImageDeduplicateAgent:
    def __init__(self, image_root: Optional[str] = None, temp_dir: Optional[str] = None) -> None:
        if image_root is None:
            image_root = os.environ.get('IMAGE_ROOT')
        self.image_root = image_root
        
        if temp_dir is None:
            temp_dir = os.environ.get('TEMP_DIR')
        assert temp_dir is not None
        self.temp_dir = temp_dir
    
    @abstractmethod
    def get_duplicate_images(self, image_dir: str, **kwargs) -> List[str]:
        raise NotImplementedError
    
    def get_duplicate_images_remote(self, images: List[dict]) -> List[str]:
        temp_image_dir = tempfile.TemporaryDirectory(dir=self.temp_dir)
        for image in images:
            if image['valid'] and image['cached_image_url'] is not None:
                urllib.request.urlretrieve(
                    url=image['cached_image_url'],
                    filename=os.path.join(temp_image_dir.name, os.path.basename(image['cached_image_url']))
                )
        duplicate_images = self.get_duplicate_images(image_dir=temp_image_dir)
        return duplicate_images

    def deduplicate_images(self, images: List[dict]) -> List[dict]:
        if len(images) == 0:
            return images
        
        if self.image_root is None:
            duplicate_images = self.get_duplicate_images_remote(images=images)
        else:
            image_dir = os.path.join(self.image_root, os.path.dirname(images[0]['relative_path']))
            if os.path.exists(image_dir):
                duplicate_images = self.get_duplicate_images(image_dir=image_dir)
            else:
                duplicate_images = self.get_duplicate_images_remote(images=images)
        
        duplicate_image_ids = [int(re.findall(r"\d+", os.path.basename(image))[0]) for image in duplicate_images]
        for image in images:
            if image['id'] in duplicate_image_ids:
                image['valid'] = False
        return images
