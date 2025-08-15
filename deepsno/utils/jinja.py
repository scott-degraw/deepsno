from datetime import datetime
from io import StringIO
from pathlib import Path

import jinja2
from jsonargparse._loaders_dumpers import get_loader_exceptions, yaml_load


def get_exceptions():
    exceptions = get_loader_exceptions("yaml")
    for obj in dir(jinja2.exceptions):
        if isinstance(obj, type) and issubclass(obj, Exception):
            exceptions.append(obj)

    return exceptions


def model_save_directory(checkpoint_dir: str | Path) -> str:
    checkpoint_dir = Path(checkpoint_dir)

    date_string = datetime.now().strftime(r"%Y-%m-%d")
    time_string = datetime.now().strftime(r"%H-%M-%S")

    return str(checkpoint_dir / date_string / time_string)


def jinja_yaml_loader(stream):
    env = jinja2.Environment()
    env.filters["model_save_directory"] = model_save_directory
    rendered_yaml = env.from_string(stream).render()
    return yaml_load(rendered_yaml)