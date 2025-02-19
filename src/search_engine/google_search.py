import os
import json
import requests
from typing import Dict, List, Optional
from src.search_engine import SEARCH_ENGINE
from src.search_engine.search_engine import SearchEngine


@SEARCH_ENGINE.register_module
class GoogleSearch(SearchEngine):
    def __init__(
        self,
        top_k: int = 10,
        google_cse_id: Optional[str] = None,
        google_api_key: Optional[str] = None,
        multi_thread_config: Dict = {}
    ) -> None:
        super().__init__(top_k, multi_thread_config)
        if google_cse_id is None:
            google_cse_id = os.environ.get('GOOGLE_CSE_ID', None)
        assert google_cse_id is not None, f"`GOOGLE_CSE_ID` must be provided for GoogleSearch, but got None"
        self.google_cse_id = google_cse_id
        
        if google_api_key is None:
            google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        assert google_api_key is not None, f"`GOOGLE_API_KEY` must be provided for GoogleSearch, but got None"
        self.google_api_key = google_api_key
        
        self.top_k = top_k
    
    def search_single(self, content: str, **kwargs) -> List:
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": content,
            "num": self.top_k
        }
        response = requests.get(
            url='https://www.googleapis.com/customsearch/v1',
            params=params
        )
        results = json.loads(response.content)
        
        # handle empty image search result
        if 'items' not in results:
            return None
        
        raw_search_result = results['items']
        search_result = []
        for i, webpage in enumerate(raw_search_result):
            url = webpage.pop('link')
            search_result.append({
                "webpage_id": i,
                "url": url,
                **webpage
            })
        return search_result

    def search_image_single(self, content: str, **kwargs) -> List:
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": content,
            "searchType": "image",
            "num": self.top_k
        }
        response = requests.get(
            url='https://www.googleapis.com/customsearch/v1',
            params=params
        )
        results = json.loads(response.content)
        
        # handle empty image search result
        if 'items' not in results:
            return None
        
        results = results['items']
        for i, result in enumerate(results):
            result.update({
                "image_id": f"aux-{i}",
                "image_url": result['link'],
                "webpage_title": result['title'],
                "webpage_url": result['image']['contextLink'],
            })
        return results
