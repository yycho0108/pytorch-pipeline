#!/usr/bin/env python3

import sys
from dataclasses import dataclass
import argcomplete
from simple_parsing import ArgumentParser, Serializable
from typing import Callable, List, Type, TypeVar

D = TypeVar("D")


def with_args(cls: Type[D] = None, argv: List[str] = None, filename=None):
    """
    Decorator for automatically adding parsed args from cli.
    """
    def decorator(main: Callable[[Type[D]], None]):

        # NOTE(ycho):
        # if `cls` is None, try to infer them from `main` signature.
        nonlocal cls
        if cls is None:
            # TODO(ycho): cleanup this mess
            import inspect
            sig = inspect.signature(main)
            if len(sig.parameters) == 1:
                key = next(iter(sig.parameters))
                cls = sig.parameters[key].annotation
            else:
                raise ValueError(
                    'More than one arg in main({}): Cannot infer param.'
                    .format(sig))

        # Load from file -> cli override.
        instance = None
        if isinstance(cls, Serializable) and filename is not None:
            instance = cls.load(filename)

        def wrapper():
            parser = ArgumentParser()
            parser.add_arguments(cls, dest='opts', default=instance)
            argcomplete.autocomplete(parser)
            args = parser.parse_args(sys.argv[1:] if argv is None else argv)
            return main(args.opts)
        return wrapper
    return decorator
