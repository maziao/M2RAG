import os
import json
import requests
from typing import Optional
from src.crawler import CRAWLER
from src.crawler.crawler import Crawler


@CRAWLER.register_module
class JinaAICralwer(Crawler):
    def __init__(self, multi_thread_config: dict = {}, jina_ai_api_key: Optional[str] = None) -> None:
        super().__init__(multi_thread_config)
        if jina_ai_api_key is None:
            jina_ai_api_key = os.environ.get('JINA_AI_API_KEY')
        self.jina_ai_api_key = jina_ai_api_key
        
    def download_webpage(self, url: str, **kwargs) -> dict:
        jina_ai_url = f"https://r.jina.ai/{url}"
        headers = {
            'Accept': 'application/json',
            'X-Token-Budget': '32768'
        }
        if self.jina_ai_api_key is not None:
            headers.update({'Authorization': f'Bearer {self.jina_ai_api_key}'})
            
        response = requests.get(jina_ai_url, headers=headers)

        response_json = json.loads(response.text)
        
        if response_json['code'] == 200:
            title = response_json['data'].pop('title')
            content = response_json['data'].pop('content')
            result = {
                "title": title,
                "content": content,
                **response_json['data']
            }
        else:
            result = None
        
        return result
