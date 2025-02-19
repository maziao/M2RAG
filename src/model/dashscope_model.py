import os
import logging
import dashscope
from http import HTTPStatus
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None
    logging.warning(f"`AutoTokenizer` is not currently imported, `num_tokens_from_text` will always return -1 and `trim_text` will always return the input string.")
from src.model import MODEL
from typing import Dict, List, Optional, Tuple
from src.model.base import BaseModelMixin, LLM, VLM


class DashScopeMixin(BaseModelMixin):
    def __init__(
        self,
        model: str,
        tokenizer_path: Optional[str],
        dashscope_api_key: Optional[str]
    ) -> None:
        self.model = model
        
        if dashscope_api_key is None:
            dashscope_api_key = os.environ.get('DASHSCOPE_API_KEY', None)
        assert dashscope_api_key is not None
        self.api_key = dashscope_api_key
        self.base_url = None
        
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
class DashScopeLLM(DashScopeMixin, LLM):
    def __init__(
        self,
        model: str = 'qwen2.5-72b-instruct',
        tokenizer_path: Optional[str] = None,
        dashscope_api_key: Optional[str] = None
    ) -> None:
        super().__init__(model=model, tokenizer_path=tokenizer_path, dashscope_api_key=dashscope_api_key)
        super(DashScopeMixin, self).__init__(name=f"dashscope-{model}")
    
    def chat_core(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        response = dashscope.Generation.call(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': messages[0]['text']
                }
            ],
            result_format='message',
            api_key=self.api_key,
            **generate_config
        )

        if response.status_code == HTTPStatus.OK:
            return {
                "response": response.output.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        else:
            return None


@MODEL.register_module
class DashScopeVLM(DashScopeMixin, VLM):
    def __init__(
        self,
        model: str = 'qwen-vl-max-0809',
        tokenizer_path: Optional[str] = None,
        dashscope_api_key: Optional[str] = None,
        use_local_image_file: bool = False,
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        image_to_base64: bool = False,
        resize_image: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(model=model, tokenizer_path=tokenizer_path, dashscope_api_key=dashscope_api_key)
        super(DashScopeMixin, self).__init__(
            name=f"dashscope-{model}",
            use_local_image_file=use_local_image_file,
            image_url=image_url,
            image_root=image_root,
            image_to_base64=image_to_base64,
            resize_image=resize_image
        )
        
    def chat_core(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        message_content = []
        for message in messages:
            if message['type'] == 'text':
                message_content.append({'text': message['text']})
            elif message['type'] == 'image_url':
                message_content.append({'image': message['image_url']['url']})
            else:
                raise ValueError(f"Unrecognized message type `{message['type']}`, expected values include `text` and `image_url`.")
        
        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': message_content
                }
            ],
            result_format='message',
            api_key=self.api_key,
            **generate_config
        )
        
        if response.status_code == HTTPStatus.OK:
            return {
                "response": response.output.choices[0].message.content[0]['text'],
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        else:
            return None
