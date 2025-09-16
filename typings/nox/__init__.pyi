from collections.abc import Callable
from typing import Any

class Session:
    python: str

    def install(self, *args: str, **kwargs: Any) -> None: ...
    def run(self, *args: str, **kwargs: Any) -> None: ...
    def skip(self, msg: str = "") -> None: ...
    def notify(self, session: str, posargs: list[str] | None = None) -> None: ...

def session(
    func: Callable[..., Any] | None = None,
    *,
    python: str | list[str] | bool | None = None,
    reuse_venv: bool | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    venv_backend: str | None = None,
) -> Callable[..., Any]: ...

class Options:
    sessions: list[str]
    pythons: list[str]
    default_venv_backend: str
    reuse_existing_virtualenvs: bool

options: Options
