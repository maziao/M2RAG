import os
import re
import logging
from abc import abstractmethod
from src.model import MODEL
from typing import List, Dict, Tuple, Optional
from src.utils import convert_image_url_to_base64, convert_image_path_to_base64


@MODEL.register_module
class BaseModel:
    def __init__(self, name: str) -> None:
        self.name = name
    
    @abstractmethod
    def chat(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        raise NotImplementedError
    
    @abstractmethod
    def chat_core(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        """Core method for LLM / VLM inference.

        Args:
            messages (List[Dict]): a list of messages in OpenAI format.
            [
                {
                    "type": "text",
                    "text": str,
                },
                {
                    "type": "image_url", # for multi-modal models only
                    "image_url": {
                        "url": str
                    }
                },
                {
                    "type": "image_caption", # for text-only models only
                    "image_caption": {
                        "caption": str
                    }
                }
                ...
            ]

        Raises:
            NotImplementedError: any subclass of `BaseModel` should implement `chat_core` method.

        Returns:
            Dict: {
                "response": str,
                "usage": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int
                }
            }
        """
        raise NotImplementedError
    
    
class BaseModelMixin:        
    @abstractmethod
    def num_tokens_from_text(self, text: str) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def trim_text(self, text: str, max_tokens: int, side: str = 'right') -> str:
        raise NotImplementedError
    
    
@MODEL.register_module
class LLM(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name=f"llm-{name}")
        
    def simplify_text_messages(self, messages: List[Dict]) -> str:
        prompt = ""
        for message in messages:
            if message['type'] == 'text':
                prompt += message['text']
            elif message['type'] == 'image_caption':
                prompt += message['image_caption']['caption']
            else:
                raise ValueError(f"Unrecognized message type `{message['type']}`, expected values include `text` and `image_caption`.")
        return [{
            'type': 'text',
            'text': prompt
        }]
        
    def chat(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        simplified_messages = self.simplify_text_messages(messages=messages)
        return self.chat_core(messages=simplified_messages, generate_config=generate_config, **kwargs)
    
    
@MODEL.register_module
class VLM(BaseModel):
    def __init__(
        self,
        name: str,
        use_local_image_file: bool = False,
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        image_to_base64: bool = False,
        resize_image: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(name=f"vlm-{name}")
        self.use_local_image_file = use_local_image_file
        
        if image_url is None:
            image_url = os.environ.get('IMAGE_URL')
        self.image_url = image_url
        if image_root is None:
            image_root = os.environ.get('IMAGE_ROOT')
        self.image_root = image_root
        
        if use_local_image_file:
            assert image_url is not None and image_root is not None
            
        self.image_to_base64 = image_to_base64
        self.resize_image = resize_image
        assert resize_image is None or isinstance(resize_image, tuple) or isinstance(resize_image, list), f"keyword argument `resize_image` should be None or a tuple, but got {resize_image}"
        if isinstance(resize_image, tuple) or isinstance(resize_image, list):
            assert len(resize_image) == 2 and all([isinstance(item, int) and item > 0 for item in resize_image]), f"keyword argument `resize_image` should be None or a tuple consisting of 2 positive integers, but got {resize_image}"
        
        if resize_image is not None and image_to_base64 is False:
            logging.warning(f"keyword argument `resize_image` only takes effect when `image_to_base64` is set to True, but it is set to False in the current run")
    
    def get_image_path_from_url(self, image_url: str) -> str:
        if self.image_root is None or self.image_url is None:
            raise ValueError(f"Mirroring remote images to local archive should have non-empty `image_root` and `image_url_base`, but got `{self.image_root}` and `{self.image_url}` respectively.")
        relative_path = re.sub(f'^{re.escape(self.image_url)}', '', image_url)
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        image_path = os.path.join(self.image_root, relative_path)
        return image_path
    
    def convert_image_to_base64(self, image_url_or_path: str) -> str:
        if image_url_or_path.startswith('http'):
            return convert_image_url_to_base64(image_url=image_url_or_path, size=self.resize_image)
        elif image_url_or_path.startswith('file://'):
            return convert_image_path_to_base64(image_path=image_url_or_path, size=self.resize_image)
        elif image_url_or_path.startswith('data:image'):
            return image_url_or_path
        else:
            raise ValueError(f"Invalid image URL or path in the messages: {image_url_or_path}")
        
    def chat(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        for i in range(len(messages)):
            if messages[i]['type'] == 'image_caption':
                messages[i] = {
                    'type': 'text',
                    'text': messages[i]['image_caption']['caption']
                }
                logging.warning(f"`image_caption` is included in the messages for VLM, please check if this is a harmful behaviour.")
        
        if self.use_local_image_file:
            for i in range(len(messages)):
                if messages[i]['type'] == 'image_url':
                    messages[i]['image_url']['url'] = f"file://{self.get_image_path_from_url(image_url=messages[i]['image_url']['url'])}"
        
        if self.image_to_base64:
            for i in range(len(messages)):
                if messages[i]['type'] == 'image_url':
                    messages[i]['image_url']['url'] = self.convert_image_to_base64(image_url_or_path=messages[i]['image_url']['url'])
        return self.chat_core(messages=messages, generate_config=generate_config, **kwargs)
