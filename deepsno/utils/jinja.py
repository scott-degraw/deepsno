from datetime import datetime
from pathlib import Path

import jinja2
from jsonargparse._loaders_dumpers import get_loader_exceptions, yaml_load

now = datetime.now()
env = jinja2.Environment()


def get_exceptions():
    exceptions = get_loader_exceptions("yaml")
    for obj in dir(jinja2.exceptions):
        if isinstance(obj, type) and issubclass(obj, Exception):
            exceptions.append(obj)

    return exceptions


def add_filter(func, name: str | None = None) -> callable:
    env.filters[name or func.__name__] = func
    return func


@add_filter
def model_save_directory(checkpoint_dir: str | Path) -> str:
    checkpoint_dir = Path(checkpoint_dir)

    date_string = now.strftime(r"%Y-%m-%d")
    time_string = now.strftime(r"%H-%M-%S")

    return str(checkpoint_dir / date_string / time_string)


@add_filter
def path_join(paths) -> str:
    return str(Path(*paths))


def jinja_yaml_loader(stream):
    if not isinstance(stream, str):
        stream = stream.read()
    rendered_yaml = env.from_string(stream).render()
    return yaml_load(rendered_yaml)
