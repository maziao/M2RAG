import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

QUESTION_FILTER = Registry('question_filter')


def build_question_filter(config: Any):
    return build_from_config(config, QUESTION_FILTER)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.question_filter." + file_name)
