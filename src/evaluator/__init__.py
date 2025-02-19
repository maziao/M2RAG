import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

EVALUATOR = Registry('evaluator')


def build_evaluator(config: Any):
    return build_from_config(config, EVALUATOR)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.evaluator." + file_name)
