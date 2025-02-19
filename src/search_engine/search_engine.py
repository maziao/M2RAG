import os
import copy
from tqdm import tqdm
from abc import abstractmethod
from typing import Dict, List
from src.utils import MultithreadManager
from src.search_engine import SEARCH_ENGINE


@SEARCH_ENGINE.register_module
class SearchEngine:
    def __init__(self, top_k: int = 10, multi_thread_config: Dict = {}) -> None:
        self.top_k = top_k
        
        assert isinstance(multi_thread_config, dict), f"Expected `dict` object for parameter `multi_thread_config`, but got {multi_thread_config}."
        self.manager = MultithreadManager(**multi_thread_config)
    
    @abstractmethod
    def search_single(self, content: str, **kwargs) -> List:
        """Get search results (url list) for a single user request.

        Args:
            content (str): user query string

        Returns:
            List:
            [
                {
                    "webpage_id": Any,
                    "url": str,
                    ...
                },
                ...
            ]
            
        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def search_image_single(self, content: str, **kwargs) -> List:
        """Get images for a single user request.

        Args:
            content (str): user query string

        Raises:
            NotImplementedError

        Returns:
            List:
            [
                {
                    "image_id": Any,
                    "image_url": str,
                    "webpage_title": str,
                    "webpage_url": str,
                    ...
                },
                ...
            ]
        """
        raise NotImplementedError
    