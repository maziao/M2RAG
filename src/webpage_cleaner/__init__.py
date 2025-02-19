import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

WEBPAGE_CLEANER = Registry('webpage_cleaner')


def build_webpage_cleaner(config: Any):
    return build_from_config(config, WEBPAGE_CLEANER)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.webpage_cleaner." + file_name)
