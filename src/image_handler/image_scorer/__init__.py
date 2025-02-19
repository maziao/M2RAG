import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

IMAGE_SCORER = Registry('image_scorer')


def build_image_scorer(config: Any):
    return build_from_config(config, IMAGE_SCORER)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.image_handler.image_scorer." + file_name)
