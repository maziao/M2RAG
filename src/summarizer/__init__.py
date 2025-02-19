import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

SUMMARIZER = Registry('summarizer')


def build_summarizer(config: Any):
    return build_from_config(config, SUMMARIZER)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.summarizer." + file_name)
    elif os.path.isdir(os.path.join(os.path.dirname(__file__), file)) and not file.startswith("_"):
        importlib.import_module("src.summarizer." + file)
