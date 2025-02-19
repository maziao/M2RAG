import os
import re
from typing import Optional, Dict, List, Any
from src.utils import MultithreadManager, convert_image_url_to_pil_image
from src.image_handler import IMAGE_HANDLER
from src.image_handler.image_cache import build_image_cache
from src.image_handler.image_dedup import build_image_dedup
from src.image_handler.image_scorer import build_image_scorer
from src.image_handler.image_caption import build_image_caption


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"


@IMAGE_HANDLER.register_module
class ImageHandler:
    def __init__(
        self,
        image_cache_config: Dict,
        image_dedup_config: Optional[Dict] = None,
        image_scorer_config: Optional[Dict] = None,
        image_caption_config: Optional[Dict] = None,
        multi_thread_config: Dict = {},
        max_images: int = None
    ) -> None:
        self.image_cache = build_image_cache(image_cache_config)
        
        if image_dedup_config is not None:
            self.image_dedup = build_image_dedup(image_dedup_config)
        else:
            self.image_dedup = None
            
        if image_scorer_config is not None:
            self.image_scorer = build_image_scorer(image_scorer_config)
        else:
            self.image_scorer = None
            
        if image_caption_config is not None:
            self.image_caption = build_image_caption(image_caption_config)
        else:
            self.image_caption = None
            
        self.manager = MultithreadManager(**multi_thread_config)
        self.max_images = max_images
    
    def _cache_image_single(self, image_url: str, image_relative_dir: str, image_id: Any):
        pil_image = convert_image_url_to_pil_image(image_url=image_url)
        if pil_image is not None:
            relative_path = os.path.join(image_relative_dir, f"{image_id}.{pil_image.format.lower()}")
            cached_image_url = self.image_cache.save(pil_image=pil_image, relative_path=relative_path)
            return {
                'cached_image_url': cached_image_url,
                'relative_path': relative_path
            }
        else:
            return None
        
    def cache_images(self, images: List[str], image_relative_dir: str):
        for image in images:
            self.manager.add_task(
                self._cache_image_single,
                None,
                image['image_id'],
                image_url=image['image_url'],
                image_relative_dir=image_relative_dir,
                image_id=image['image_id']
            )
            
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for result in results:
            image_id = result['id']
            if result['success'] and result['result'] is not None:
                update_dict = result['result']
                update_dict['valid'] = True
            else:
                update_dict = {
                    'cached_image_url': None,
                    'relative_path': None,
                    'valid': False
                }
            
            for image in images:
                if image['image_id'] == image_id:
                    image.update(**update_dict)
                    break
        
        return images
    
    @staticmethod
    def _remove_image_placeholders(text: str):
        image_indices = re.findall(r"<IMAGE_PLACEHOLDER>\[(\d+)\]", text)
        for index in image_indices:
            text = text.replace(f"{IMAGE_PLACEHOLDER}[{index}]", '')
        text = text.replace(IMAGE_PLACEHOLDER, '')
        return text
    
    def process(
        self,
        cleaned_webpage_content: str,
        images: List[Dict],
        image_relative_dir: str,
        ref_text: Optional[str] = None,
        cache_file: Optional[str] = None
    ):
        # cache images
        images = self.cache_images(images=images, image_relative_dir=image_relative_dir)
        
        # deduplicate images
        if self.image_dedup is not None:
            images = self.image_dedup.deduplicate_images(images=images)
        
        # score images
        if self.image_scorer is not None:
            scores = self.image_scorer.get_image_scores(image_urls=[image['cached_image_url'] for image in images if image['valid']], reference_text=ref_text)
            index = 0
            for image in images:
                if image['valid']:
                    image['scores'] = scores[index]
                    image['final_score'] = scores[index]['score']
                    if image['final_score'] is None:
                        image['valid'] = False
                    index += 1
                else:
                    image['scores'] = None
                    image['final_score'] = None
                    
        if self.image_caption is not None:
            for image in images:
                if image['valid']:
                    target_placeholder = f"{IMAGE_PLACEHOLDER}[{image['image_id']}]"
                    full_webpage_splits = cleaned_webpage_content.split(target_placeholder)
                    assert len(full_webpage_splits) == 2, f"{full_webpage_splits} {image}"
                    
                    context_above = self._remove_image_placeholders(text=full_webpage_splits[0])
                    context_below = self._remove_image_placeholders(text=full_webpage_splits[1])
                    
                    self.manager.add_task(
                        self.image_caption.get_image_caption_single,
                        cache_file,
                        image['image_id'],
                        image_url=image['cached_image_url'],
                        ref_text=None,
                        context=context_above + IMAGE_PLACEHOLDER + context_below
                    )
                    
            results = self.manager.execute_tasks()
            self.manager.clear_tasks()
            
            for result in results:
                if result['success'] and result['result'] is not None:
                    image_id = result['id']
                    for image in images:
                        if image['image_id'] == image_id:
                            image['detailed_image_caption'] = result['result']['response']
                            break
            
        return images
