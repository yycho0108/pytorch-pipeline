#!/usr/bin/env python3

import os
import sys
import json
import yaml

from functools import partial
from collections import OrderedDict
from pathlib import Path


def file_cache(name_fn, load_fn, dump_fn, binary: bool = True):
    """ Decorator for caching a result from a function to a file. """
    def call_or_load(compute):
        def wrapper(*args, **kwargs):
            filename = name_fn(compute, *args, **kwargs)
            cache_file = Path(filename)
            # Compute if non-existent.
            if not cache_file.exists():
                # Ensure directory exists.
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                result = compute(*args, **kwargs)
                mode = 'wb' if binary else 'w'
                with open(filename, mode) as f:
                    dump_fn(result, f)
                return result

            # O.W. Return from cache.
            mode = 'rb' if binary else 'r'
            with open(filename, mode) as f:
                return load_fn(f)

        return wrapper
    return call_or_load


try:
    import json
    json_load = partial(json.load, object_pairs_hook=OrderedDict)
    json_file_cache = partial(file_cache,
                              load_fn=json.load,
                              dump_fn=json.dump,
                              binary=False)
except ImportError:
    pass

try:
    import yaml
    yaml_file_cache = partial(file_cache,
                              load_fn=yaml.safe_load,
                              dump_fn=yaml.safe_dump,
                              binary=False)
except ImpotError:
    pass

try:
    import pickle
    pickle_file_cache = partial(file_cache,
                                load_fn=pickle.load,
                                dump_fn=pickle.dump,
                                binary=True)
except ImportError:
    pass
