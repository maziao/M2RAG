import os
import logging
from openai import OpenAI
import tiktoken
from src.model import MODEL
from typing import Dict, List, Optional, Tuple
from src.model.base import BaseModelMixin, LLM, VLM


class OpenAIMixin(BaseModelMixin):
    def __init__(
        self,
        model: str,
        openai_api_key: Optional[str],
        openai_base_url: Optional[str]
    ) -> None:
        self.model = model
        
        if openai_api_key is None:
            openai_api_key = os.environ.get('OPENAI_API_KEY')
        assert openai_api_key is not None
        
        if openai_base_url is None:
            openai_base_url = os.environ.get('OPENAI_BASE_URL')
        
        self.api_key = openai_api_key
        self.base_url = openai_base_url
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logging.warning(f"tokenizer for model {model} is not found. Using cl100k_base tokenizer.")

    def num_tokens_from_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text=text))
    
    def trim_text(self, text: str, max_tokens: int, side: str = 'right') -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            if side == 'left':
                tokens = tokens[-max_tokens:]
            elif side == 'right':
                tokens = tokens[:max_tokens]
            else:
                raise ValueError(f"Invalid value for argument `side`. Expected `left` or `right`, but got `{side}` instead.")
        trimmed_text = self.tokenizer.decode(tokens)
        return trimmed_text
        
        
@MODEL.register_module
class OpenAILLM(OpenAIMixin, LLM):
    def __init__(
        self,
        model: str = 'gpt-4o-2024-08-06',
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None
    ) -> None:
        super().__init__(model=model, openai_api_key=openai_api_key, openai_base_url=openai_base_url)
        super(OpenAIMixin, self).__init__(name=f"openai-{model}")
    
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
class OpenAIVLM(OpenAIMixin, VLM):
    def __init__(
        self,
        model: str = 'gpt-4o-2024-08-06',
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        use_local_image_file: bool = False,
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        image_to_base64: bool = False,
        resize_image: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(model=model, openai_api_key=openai_api_key, openai_base_url=openai_base_url)
        super(OpenAIMixin, self).__init__(
            name=f"openai-{model}",
            use_local_image_file=use_local_image_file,
            image_url=image_url,
            image_root=image_root,
            image_to_base64=image_to_base64,
            resize_image=resize_image
        )
        
    def chat_core(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        new_messages = []
        for message in messages:
            if message['type'] == 'image_url':
                message['image_url']['detail'] = 'low'
            new_messages.append(message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": new_messages
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
