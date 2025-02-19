import inspect
from typing import List
import logging

logger = logging.getLogger(__name__)


class Registry:
    """
    Register modules automatically by adding a decoration to the class.

    Args:
        name (str): name of a module series.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module_class (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            logging.warning('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_config(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg: Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(default_args, dict) or default_args is None
    obj_type = cfg.get('type')
    if isinstance(obj_type, str):
        obj_type = registry.get(obj_type)
        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(obj_type, registry.name))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
    return obj_type(**cfg.get('params'))


class RegistryList:

    def __init__(self, name, registries: List[Registry] = None):
        self._name = name
        self._registries = registries

    @property
    def name(self) -> str:
        return self._name

    def add_registry(self, registry: Registry):
        self._registries.append(registry)

    def get_registry(self, registry_name: str):
        for registry in self._registries:
            if registry_name == registry.name:
                return registry
        return None

    def get_module(self, module_name: str, registry_name: str = None):
        if registry_name is not None:
            registry = self.get_registry(registry_name)
            if registry is None:
                logger.warning(
                    f'Registry {registry_name} is not found. Try to find {module_name} in the whole registry list.'
                )
            else:
                module = registry.get(module_name)
                return module
        for registry in self._registries:
            module = registry.get(module_name)
            if module is not None:
                return module
        logger.warning(
            f'Module {module_name} is not found in the whole registry list.')
