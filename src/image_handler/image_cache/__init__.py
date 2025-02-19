import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

IMAGE_CACHE = Registry('image_cache')


def build_image_cache(config: Any):
    return build_from_config(config, IMAGE_CACHE)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_") and "service" not in file:
        file_name = file[: file.find(".py")]
        importlib.import_module("src.image_handler.image_cache." + file_name)
