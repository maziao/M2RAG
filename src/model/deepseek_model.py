import os
import logging
from openai import OpenAI
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None
    logging.warning(f"`AutoTokenizer` is not currently imported, `num_tokens_from_text` will always return -1 and `trim_text` will always return the input string.")
from src.model import MODEL
from typing import Dict, List, Optional
from src.model.base import BaseModelMixin, LLM


class DeepSeekMixin(BaseModelMixin):
    def __init__(
        self,
        model: str,
        tokenizer_path: Optional[str],
        deepseek_api_key: Optional[str]
    ) -> None:
        self.model = model
        
        if deepseek_api_key is None:
            deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
        assert deepseek_api_key is not None
        
        self.api_key = deepseek_api_key
        self.base_url = "https://api.deepseek.com"
        self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
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
class DeepSeekLLM(DeepSeekMixin, LLM):
    def __init__(
        self,
        model: str = 'deepseek-chat',
        tokenizer_path: Optional[str] = None,
        deepseek_api_key: Optional[str] = None
    ) -> None:
        super().__init__(model=model, tokenizer_path=tokenizer_path, deepseek_api_key=deepseek_api_key)
        super(DeepSeekMixin, self).__init__(name=f"deepseek-{model}")
    
    def chat_core(self, messages: List[Dict], generate_config: Dict = {}, **kwargs) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": messages
                },
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
