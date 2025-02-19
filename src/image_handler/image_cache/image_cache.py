from PIL import Image
from abc import abstractmethod
from src.image_handler.image_cache import IMAGE_CACHE


@IMAGE_CACHE.register_module
class ImageCache:
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def save(
        self,
        pil_image: Image.Image,
        relative_path: str,
        **kwargs
    ) -> str:
        """Save an image to cache

        Args:
            pil_image (Image.Image): image to be saved
            relative_path (str): relative path based on the image root

        Return:
            cache_image_url: str
            
        Raises:
            NotImplementedError
        """
        raise NotImplementedError
    
    @abstractmethod
    def delete(
        self,
        relative_path: str
    ) -> None:
        """Remove an image from cache

        Args:
            relative_path (str): relative path of the cached image

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    