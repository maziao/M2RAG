import os
import re
import math
import copy
import requests
from io import BytesIO
import numpy as np
from PIL import ImageFile, Image
from abc import abstractmethod
from typing import List, Optional, Union
from src.image_handler.image_scorer import IMAGE_SCORER, build_image_scorer


@IMAGE_SCORER.register_module
class ImageScorer:
    def __init__(
        self,
        filter_config: dict = {},
        enable_cache: bool = False,
        name: Optional[str] = None,
        image_url: Optional[str] = None,
        image_root: Optional[str] = None
    ) -> None:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.filter = ScoreFilter(**filter_config)
        self.pil_image_cache = {}
        self.enable_cache = enable_cache
        self.name = name
        
        if image_url is None:
            image_url = os.environ.get('IMAGE_URL')
        self.image_url = image_url
        if image_root is None:
            image_root = os.environ.get('IMAGE_ROOT')
        self.image_root = image_root
    
    @abstractmethod
    def get_image_score_single(self, image_url: str, reference_text: Optional[str] = None) -> dict:
        raise NotImplementedError
        
    @abstractmethod
    def _get_image_scores(self, image_urls: List[str], reference_text: Optional[str] = None) -> List[dict]:
        """Get scores of a list of images

        Args:
            image_urls (List[str]): image urls that can be accessed through the public network
            reference_text (Optional[str], optional): reference text for scoring. Defaults to None.
            
        Return:
            scores: a list of scores
        """
        raise NotImplementedError
    
    def get_image_scores(self, image_urls: List[str], reference_text: Optional[str] = None) -> List[dict]:
        scores = self._get_image_scores(image_urls=image_urls, reference_text=reference_text)
        scores = self.filter.filter(score_list=scores)
        return scores
    
    def get_image_path_from_url(self, image_url: str) -> str:
        if self.image_root is None or self.image_url is None:
            raise ValueError(f"Mirroring remote images to local archive should have non-empty `image_root` and `image_url_base`, but got `{self.image_root}` and `{self.image_url}` respectively.")
        relative_path = re.sub(f'^{re.escape(self.image_url)}', '', image_url)
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        image_path = os.path.join(self.image_root, relative_path)
        return image_path
    
    def load_image_from_url(self, image_url: str) -> Image.Image:
        if image_url in self.pil_image_cache:
            img = self.pil_image_cache[image_url]
        elif self.image_root is not None and self.image_url is not None:
            image_path = self.get_image_path_from_url(image_url=image_url)
            img = Image.open(image_path)
        else:
            session = requests.Session()
            response = session.get(url=image_url, timeout=3.0)
            img = Image.open(BytesIO(response.content))
            
        # transform
        img_format = img.format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.format is None:
            tmp_img = BytesIO()
            img.save(tmp_img, format=img_format)
            img = Image.open(tmp_img)
            
        # cache image
        if self.enable_cache and image_url not in self.pil_image_cache:
            self.pil_image_cache[image_url] = img
        
        return img


class ScoreFilter:
    def __init__(
        self,
        top_p: float = 1.0,
        top_k: int = -1,
        upper_limit: Optional[float] = None,
        lower_limit: Optional[float] = None,
        filtered_value: float = -100
    ) -> None:
        self.top_p = top_p
        self.top_k = top_k
        if upper_limit is not None and lower_limit is not None:
            assert upper_limit > lower_limit
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.filtered_value = filtered_value
    
    def filter(self, score_list: List[Union[float, dict]]):
        scores = [score['score'] for score in score_list]
        for i in range(len(scores)):
            if scores[i] is None:
                scores[i] = self.filtered_value
        
        rank = np.argsort(-np.array(scores))
        
        cutoff = math.ceil(self.top_p * len(scores))
        if self.top_k > 0:
            cutoff = min(cutoff, self.top_k)
        
        for index in rank[cutoff:]:
            score_list[index]['score'] = None
        
        for index in rank[:cutoff]:
            lower_flag = self.lower_limit is not None and scores[index] < self.lower_limit
            upper_flag = self.upper_limit is not None and scores[index] > self.upper_limit
            filterd_flag = self.filtered_value == scores[index]
            if lower_flag or upper_flag or filterd_flag:
                score_list[index]['score'] = None
        return score_list


@IMAGE_SCORER.register_module
class ImageScorerPipeline(ImageScorer):
    def __init__(
        self,
        filter_config: dict = {},
        enable_cache: bool = True,
        name: str = 'pipeline',
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        image_scorer_config_list: List[dict] = []
    ) -> None:
        super().__init__(
            filter_config=filter_config,
            enable_cache=enable_cache,
            name=name,
            image_url=image_url,
            image_root=image_root
        )
        self.image_scorer_list = [build_image_scorer(config) for config in image_scorer_config_list]
        
    def get_image_scores(self, image_urls: List[str], reference_text: Optional[str] = None) -> List[Union[float, dict]]:
        score_mapper = {image_url: {} for image_url in image_urls}
        temp_image_urls = copy.deepcopy(image_urls)
        
        # cache images
        for image_url in image_urls:
            _ = self.load_image_from_url(image_url=image_url)
        
        for image_scorer in self.image_scorer_list:
            # copy pil image cache to each scorer
            for image_url in temp_image_urls:
                image_scorer.pil_image_cache[image_url] = self.pil_image_cache[image_url]
            
            scores = image_scorer.get_image_scores(image_urls=temp_image_urls, reference_text=reference_text)
            
            # delete pil image cache in each scorer
            for image_url in temp_image_urls:
                del image_scorer.pil_image_cache[image_url]
            
            new_temp_image_urls = []
            for score, image_url in zip(scores, temp_image_urls):
                score_mapper[image_url][f'score({image_scorer.name})'] = score['score']
                if score['score'] is None:
                    score_mapper[image_url]['filtered_reason'] = image_scorer.name
                else:
                    new_temp_image_urls.append(image_url)
            temp_image_urls = new_temp_image_urls
        
        # assign the score in the last step as the final score
        for image_url in score_mapper:
            if image_url in temp_image_urls:
                score_mapper[image_url]['score'] = score_mapper[image_url][f'score({self.image_scorer_list[-1].name})']
            else:
                score_mapper[image_url]['score'] = None
        
        # delete pil image cache in pipeline scorer
        for image_url in image_urls:
            del self.pil_image_cache[image_url]
        
        final_scores = []
        for image_url in image_urls:
            final_scores.append(score_mapper[image_url])
        return final_scores
