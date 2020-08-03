import glob
from os.path import basename, dirname, isfile, join

__all__ = [
    basename(f)[:-3] for f in glob.glob(join(dirname(__file__), "*.py"))
    if isfile(f) and not f.endswith('__init__.py')
]
[exec('import {}.{}'.format(__name__, module)) for module in __all__]
