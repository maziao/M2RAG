import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

CRAWLER = Registry('crawler')


def build_crawler(config: Any):
    return build_from_config(config, CRAWLER)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.crawler." + file_name)
