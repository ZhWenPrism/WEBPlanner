"""Staged file replacement for saved artifacts."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def staged_path(path):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=target.suffix)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path, text):
    with staged_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
