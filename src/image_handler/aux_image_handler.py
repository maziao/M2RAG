from typing import Optional, Dict, List
from src.image_handler import IMAGE_HANDLER
from src.image_handler.image_handler import ImageHandler

IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"


@IMAGE_HANDLER.register_module
class AuxiliaryImageHandler(ImageHandler):
    def __init__(
        self,
        image_cache_config: Dict,
        image_scorer_config: Optional[Dict] = None,
        image_caption_config: Optional[Dict] = None,
        multi_thread_config: Dict = {},
        max_images: int = None
    ) -> None:
        super().__init__(
            image_cache_config=image_cache_config,
            image_dedup_config=None,
            image_scorer_config=image_scorer_config,
            image_caption_config=image_caption_config,
            multi_thread_config=multi_thread_config,
            max_images=max_images
        )
    
    def process(
        self,
        images: List[Dict],
        image_relative_dir: str,
        ref_text: Optional[str] = None,
        cache_file: Optional[str] = None
    ):
        # cache images
        images = self.cache_images(images=images, image_relative_dir=image_relative_dir)
        
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
                    self.manager.add_task(
                        self.image_caption.get_image_caption_single,
                        cache_file,
                        image['image_id'],
                        image_url=image['cached_image_url'],
                        ref_text=ref_text,
                        context=None
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
