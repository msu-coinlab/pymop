import os

__all__ = []

import inspect
import pkgutil
import os


problem_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'problems')

for loader, name, is_pkg in pkgutil.walk_packages([problem_dir]):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue
        globals()[name] = value
        __all__.append(name)