import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def chroma_lock(persist_dir: Path) -> Iterator[None]:
    lock_path = persist_dir / ".chroma.lock"
    persist_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
