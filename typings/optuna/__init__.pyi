
from typing import Any

class TrialState:
    FAIL: TrialState

class Trial:
    number: int

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = ...,
        step: float | None = ...,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = ...,
        log: bool = ...,
    ) -> int: ...

    def suggest_categorical(self, name: str, choices: tuple[Any, ...]) -> Any: ...

    def set_user_attr(self, key: str, value: Any) -> None: ...

class Study:
    study_name: str | None

    def ask(self) -> Trial: ...

    def tell(
        self,
        trial: Trial,
        value: float | None = ...,
        *,
        state: TrialState | None = ...,
    ) -> None: ...

class TPESampler:
    def __init__(self, *, seed: int | None = ...) -> None: ...

class Samplers:
    TPESampler: type[TPESampler]


def create_study(
    *,
    direction: str,
    study_name: str | None = ...,
    **kwargs: Any,
) -> Study: ...
