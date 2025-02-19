from typing import Any
from utils.registry import build_from_config, RegistryList
from src.crawler import CRAWLER
from src.evaluator import EVALUATOR
from src.image_handler import IMAGE_HANDLER
from src.image_handler.image_cache import IMAGE_CACHE
from src.image_handler.image_caption import IMAGE_CAPTION
from src.image_handler.image_dedup import IMAGE_DEDUP
from src.image_handler.image_scorer import IMAGE_SCORER
from src.question_filter import QUESTION_FILTER
from src.search_engine import SEARCH_ENGINE
from src.summarizer import SUMMARIZER
from src.task_watcher import TASK_WATCHER
from src.webpage_cleaner import WEBPAGE_CLEANER


MODULE = RegistryList(name='module', registries=[
    CRAWLER,
    EVALUATOR,
    IMAGE_HANDLER,
    IMAGE_CACHE,
    IMAGE_CAPTION,
    IMAGE_DEDUP,
    IMAGE_SCORER,
    QUESTION_FILTER,
    SEARCH_ENGINE,
    SUMMARIZER,
    TASK_WATCHER,
    WEBPAGE_CLEANER
])


def build_module(config: Any):
    return build_from_config(config, MODULE)
