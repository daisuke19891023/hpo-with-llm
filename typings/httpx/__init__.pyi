from typing import Any

class Response:
    status_code: int
    headers: dict[str, str]
    text: str

    def json(self) -> Any: ...
