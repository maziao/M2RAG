from typing import List, Optional
from src.image_handler.image_scorer import IMAGE_SCORER
from src.image_handler.image_scorer.image_scorer import ImageScorer


@IMAGE_SCORER.register_module
class PixelImageScorer(ImageScorer):
    def __init__(
        self,
        filter_config: dict = {},
        enable_cache: bool = False,
        name: str = 'pixel',
        image_url: Optional[str] = None,
        image_root: Optional[str] = None
    ) -> None:
        super().__init__(
            filter_config=filter_config,
            enable_cache=enable_cache,
            name=name,
            image_url=image_url,
            image_root=image_root
        )
        
    def get_image_score_single(self, image_url: str, reference_text: Optional[str] = None) -> float:
        image = self.load_image_from_url(image_url)
        score = image.size[0] * image.size[1]
        return {'score': score}
    
    def _get_image_scores(self, image_urls: List[str], reference_text: Optional[str] = None) -> List[float]:
        images = [self.load_image_from_url(image_url) for image_url in image_urls]
        scores = [image.size[0] * image.size[1] for image in images]
        return [{'score': score} for score in scores]
