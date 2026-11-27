"""Staged file replacement for saved artifacts."""
from contextlib import contextmanager
import os
import stat
from pathlib import Path
import tempfile


@contextmanager
def staged_path(path):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".nii.gz" if target.name.lower().endswith(".nii.gz") else target.suffix)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if target.exists():
            temporary.chmod(stat.S_IMODE(target.stat().st_mode))
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()


def write_text_atomic(path, text):
    with staged_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
