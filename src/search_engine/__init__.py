import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

SEARCH_ENGINE = Registry('search_engine')


def build_search_engine(config: Any):
    return build_from_config(config, SEARCH_ENGINE)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.search_engine." + file_name)
