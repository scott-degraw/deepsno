"""Shared helpers for the train/predict/bench entry points."""

import subprocess
from pathlib import Path


class Tee:
    """Write to both a file and another stream simultaneously."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def fileno(self):
        return self._stream.fileno()


class UncommittedChangesError(RuntimeError):
    pass


class MismatchedGitHash(RuntimeError):
    pass


def get_git_hash(raise_exception: bool = False) -> str:
    """Return the short git hash of HEAD.

    If *raise_exception* is ``True`` and the working tree has uncommitted
    changes, an :class:`UncommittedChangesError` is raised.
    """
    repo_directory = Path(__file__).parent.parent.resolve()

    has_uncommitted = subprocess.run(["git", "diff", "--quiet"], cwd=repo_directory).returncode != 0
    if raise_exception and has_uncommitted:
        raise UncommittedChangesError("Working tree is not clean. Please commit all changes.")

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo_directory, capture_output=True, text=True, check=True
    ).stdout.strip()

    return git_hash


def resolve_git_hash(cfg: dict) -> str:
    """Validate and return the git hash."""
    git_hash = get_git_hash(raise_exception=not cfg.get("force", False))
    if cfg.get("git_hash") is not None and git_hash != cfg["git_hash"]:
        raise MismatchedGitHash(
            f"Git hash '{cfg['git_hash']}' does not match the git hash of the current working tree: '{git_hash}'"
        )
    return git_hash
