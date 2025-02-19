import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

MODEL = Registry('model')


def build_model(config: Any):
    return build_from_config(config, MODEL)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.model." + file_name)
