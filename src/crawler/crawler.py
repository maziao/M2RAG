from abc import abstractmethod
from src.crawler import CRAWLER
from src.utils import MultithreadManager


@CRAWLER.register_module
class Crawler:
    def __init__(self, multi_thread_config: dict = {}) -> None:
        assert isinstance(multi_thread_config, dict), f"Expected `dict` object for parameter `multi_thread_config`, but got {multi_thread_config}."
        self.manager = MultithreadManager(**multi_thread_config)
    
    @abstractmethod
    def download_webpage(self, url: str, **kwargs) -> dict:
        """Get dict result from webpage

        Args:
            url (str): url of the target webpage

        Returns:
            dict: {
                "title": str,
                "content": str,
                ...
            }
        """
        raise NotImplementedError
