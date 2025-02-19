import os
import copy
import logging
import requests
from typing import List, Optional
try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = None
    CLIPProcessor = None
    logging.warning(f"module `transformers` is not currently imported, `LocalCLIPImageScorer` cannot be used.")
from src.image_handler.image_scorer import IMAGE_SCORER
from src.image_handler.image_scorer.image_scorer import ImageScorer


@IMAGE_SCORER.register_module
class LocalCLIPImageScorer(ImageScorer):
    def __init__(
        self,
        filter_config: dict = {},
        enable_cache: bool = False,
        name: str = 'CLIP',
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        clip_model_name_or_path: str = 'openai/clip-vit-large-patch14'
    ) -> None:
        super().__init__(
            filter_config=filter_config,
            enable_cache=enable_cache,
            name=name,
            image_url=image_url,
            image_root=image_root
        )
        self.model = CLIPModel.from_pretrained(clip_model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name_or_path)
        
    def get_image_score_single(self, image_url: str, reference_text: Optional[str] = None) -> float:
        image = self.load_image_from_url(image_url)

        try:
            inputs = self.processor(text=[reference_text], images=[image], return_tensors="pt", padding=True)
        except ValueError:
            return {'score': 0}

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        score = logits_per_image.detach().squeeze().cpu().numpy().tolist()
        if isinstance(score, list):
            score = score[0]
        return {'score': score}
    
    def _get_image_scores(self, image_urls: List[str], reference_text: Optional[str] = None) -> List[float]:
        if len(image_urls) == 0:
            return []
        
        images = [self.load_image_from_url(image_url) for image_url in image_urls]

        try:
            inputs = self.processor(text=[reference_text], images=images, return_tensors="pt", padding=True)
        except ValueError:
            return [copy.deepcopy({'score': 0})] * len(images)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        scores = logits_per_image.detach().squeeze().cpu().numpy().tolist()
        if not isinstance(scores, list):
            scores = [scores]
        return [{'score': score} for score in scores]
    

@IMAGE_SCORER.register_module
class WebServerCLIPImageScorer(ImageScorer):
    def __init__(
        self,
        filter_config: dict = {},
        enable_cache: bool = False,
        name: str = 'CLIP',
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        clip_service_url: Optional[str] = None
    ) -> None:
        super().__init__(
            filter_config=filter_config,
            enable_cache=enable_cache,
            name=name,
            image_url=image_url,
            image_root=image_root
        )
        if clip_service_url is None:
            clip_service_url = os.environ.get('CLIP_SERVICE_URL')
        assert clip_service_url is not None
        self.clip_service_url = clip_service_url
        
    def get_image_score_single(self, image_url: str, reference_text: Optional[str] = None) -> float:
        if self.image_root is not None and self.image_url is not None:
            request_json = {
                'image_paths': [self.get_image_path_from_url(image_url=image_url)],
                'reference_text': reference_text
            }
        else:
            request_json = {
                'image_urls': image_url,
                'reference_text': reference_text
            }
            
        response = requests.post(
            self.clip_service_url,
            json=request_json,
            headers={'content-type': 'application/json'}
        ).json()
        if response['success']:
            return {'score': response['scores'][0]}
        else:
            return {'score': None}
    
    def _get_image_scores(self, image_urls: List[str], reference_text: str = None) -> List[float]:
        if len(image_urls) == 0:
            return []
        
        if self.image_root is not None and self.image_url is not None:
            request_json = {
                'image_paths': [self.get_image_path_from_url(image_url=image_url) for image_url in image_urls],
                'reference_text': reference_text
            }
        else:
            request_json = {
                'image_urls': image_urls,
                'reference_text': reference_text
            }
        
        response = requests.post(
            self.clip_service_url,
            json=request_json,
            headers={'content-type': 'application/json'}
        ).json()
        
        if response['success']:
            return [{'score': score} for score in response['scores']]
        else:
            return [copy.deepcopy({'score': None})] * len(image_urls)
