"""OmegaConf custom resolvers used by config files.

Registering a resolver here makes it usable as ``${name:args}`` in any YAML
config composed through Hydra. Resolvers are registered as an import
side effect, so anything that needs them just needs to import this module
(every entry point does, via ``hydra_cli``).
"""

import yaml
from omegaconf import OmegaConf


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


OmegaConf.register_new_resolver("load_yaml", _load_yaml, replace=True)
