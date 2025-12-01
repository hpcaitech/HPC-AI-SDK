# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from __future__ import annotations

from typing import Any
from typing_extensions import override

from ._proxy import LazyProxy


class ResourcesProxy(LazyProxy[Any]):
    """A proxy for the `hpcai.resources` module.

    This is used so that we can lazily import `hpcai.resources` only when
    needed *and* so that users can just import `hpcai` and reference `hpcai.resources`
    """

    @override
    def __load__(self) -> Any:
        import importlib

        mod = importlib.import_module("hpcai.resources")
        return mod


resources = ResourcesProxy().__as_proxied__()
