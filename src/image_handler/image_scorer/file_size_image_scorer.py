import os
import tempfile
from typing import List, Optional
from src.image_handler.image_scorer import IMAGE_SCORER
from src.image_handler.image_scorer.image_scorer import ImageScorer


@IMAGE_SCORER.register_module
class FileSizeImageScorer(ImageScorer):
    def __init__(
        self,
        filter_config: dict = {},
        enable_cache: bool = False,
        name: str = 'file_size',
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        temp_dir: Optional[str] = None
    ) -> None:
        super().__init__(
            filter_config=filter_config,
            enable_cache=enable_cache,
            name=name,
            image_url=image_url,
            image_root=image_root
        )
        if temp_dir is None:
            temp_dir = os.environ.get('TEMP_DIR')
        assert temp_dir is not None
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        
    def get_image_score_single(self, image_url: str, reference_text: Optional[str] = None) -> float:
        image = self.load_image_from_url(image_url)    
        file = tempfile.NamedTemporaryFile(mode='w+b', dir=self.temp_dir, suffix=image.format)
        image.save(file, format=image.format)
        num_bytes = os.path.getsize(file.name)
        return {'score': num_bytes}
    
    def _get_image_scores(self, image_urls: List[str], reference_text: Optional[str] = None) -> List[float]:
        scores = [self.get_image_score_single(image_url=image_url, reference_text=reference_text) for image_url in image_urls]
        return scores
