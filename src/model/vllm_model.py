import os
import logging
from openai import OpenAI
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None
    logging.warning(f"`AutoTokenizer` is not currently imported, `num_tokens_from_text` will always return -1 and `trim_text` will always return the input string.")
from src.model import MODEL
from typing import Dict, List, Optional, Tuple
from src.model.base import BaseModelMixin, LLM, VLM


class VLLMMixin(BaseModelMixin):
    def __init__(
        self,
        model: Optional[str],
        tokenizer_path: Optional[str],
        vllm_api_key: Optional[str],
        vllm_service_url: Optional[str]
    ) -> None:
        if vllm_api_key is None:
            vllm_api_key = os.environ.get('VLLM_API_KEY', 'pseudo-api-key')
        
        if vllm_service_url is None:
            vllm_service_url = os.environ.get('VLLM_SERVICE_URL', None)
        assert vllm_service_url is not None, f"`VLLM_SERVICE_URL` is not set"
        
        self.api_key = vllm_api_key
        self.base_url = vllm_service_url
        self.client = OpenAI(api_key=vllm_api_key, base_url=vllm_service_url)
        
        self.model = self.client.models.list().data[0].id
        if model is not None:
            self.model_name = model
        else:
            self.model_name = self.model
            
        if tokenizer_path is not None and AutoTokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        else:
            if tokenizer_path is None:
                logging.warning(f"`tokenizer_path` is not provided, `num_tokens_from_text` will always return -1 and `trim_text` will always return the input string.")
            self.tokenizer = None

    def num_tokens_from_text(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text=text).input_ids)
        else:
            return -1
    
    def trim_text(self, text: str, max_tokens: int, side: str = 'right') -> str:
        if self.tokenizer is not None:
            tokens = self.tokenizer(text).input_ids
            if len(tokens) > max_tokens:
                if side == 'left':
                    tokens = tokens[-max_tokens:]
                elif side == 'right':
                    tokens = tokens[:max_tokens]
                else:
                    raise ValueError(f"Invalid value for argument `side`. Expected `left` or `right`, but got `{side}` instead.")
            trimmed_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return trimmed_text
        else:
            return text
        
        
@MODEL.register_module
class VLLMLLM(VLLMMixin, LLM):
    def __init__(
        self,
        model: Optional[str],
        tokenizer_path: Optional[str],
        vllm_api_key: Optional[str],
        vllm_service_url: Optional[str]
    ) -> None:
        super().__init__(model=model, tokenizer_path=tokenizer_path, vllm_api_key=vllm_api_key, vllm_service_url=vllm_service_url)
        super(VLLMMixin, self).__init__(name=f"vllm-{self.model_name}")
    
    def chat_core(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": messages
                }
            ],
            **generate_config
        )

        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }


@MODEL.register_module
class VLLMVLM(VLLMMixin, VLM):
    def __init__(
        self,
        model: Optional[str],
        tokenizer_path: Optional[str],
        vllm_api_key: Optional[str],
        vllm_service_url: Optional[str],
        use_local_image_file: bool = False,
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        image_to_base64: bool = False,
        resize_image: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(model=model, tokenizer_path=tokenizer_path, vllm_api_key=vllm_api_key, vllm_service_url=vllm_service_url)
        super(VLLMMixin, self).__init__(
            name=f"vllm-{self.model_name}",
            use_local_image_file=use_local_image_file,
            image_url=image_url,
            image_root=image_root,
            image_to_base64=image_to_base64,
            resize_image=resize_image
        )
        
    def chat_core(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": messages
                }
            ],
            **generate_config
        )

        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
