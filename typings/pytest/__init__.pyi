from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, overload

@overload
def fixture[F: Callable[..., Any]](func: F) -> F: ...
@overload
def fixture[F: Callable[..., Any]](
    *,
    scope: str = "function",
    params: list[Any] | None = None,
    autouse: bool = False,
    ids: list[str] | None = None,
    name: str | None = None,
) -> Callable[[F], F]: ...
def raises(
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    *,
    match: str | None = None,
) -> AbstractContextManager[Any]: ...

class Mark:
    def skip(self, *, reason: str = "") -> Any: ...
    def parametrize(
        self,
        argnames: str | list[str],
        argvalues: list[Any],
        *,
        ids: list[str] | None = None,
    ) -> Any: ...
    def xfail(self, *, reason: str = "") -> Any: ...

mark: Mark

class MonkeyPatch:
    def setenv(self, name: str, value: str) -> None: ...
    def delenv(self, name: str, raising: bool = True) -> None: ...
    def setattr(self, target: Any, name: str, value: Any) -> None: ...
    def delattr(self, target: Any, name: str) -> None: ...

class LogRecord:
    levelname: str
    message: str
    msg: str

class LogCaptureFixture:
    def set_level(self, level: int | str) -> None: ...
    records: list[LogRecord]
