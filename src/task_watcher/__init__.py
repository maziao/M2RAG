import os
import importlib
from typing import Any
from src.utils.registry import Registry, build_from_config

TASK_WATCHER = Registry('task_watcher')


def build_task_watcher(config: Any):
    return build_from_config(config, TASK_WATCHER)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("src.task_watcher." + file_name)
