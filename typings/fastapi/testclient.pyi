from typing import Any

import httpx

class TestClient:
    def __init__(self, app: Any, **kwargs: Any) -> None: ...
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        auth: Any = None,
        follow_redirects: bool = True,
        timeout: Any = None,
        extensions: dict[str, Any] | None = None,
    ) -> httpx.Response: ...
    def post(
        self,
        url: str,
        *,
        content: Any = None,
        data: dict[str, Any] | None = None,
        files: Any = None,
        json: Any = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        auth: Any = None,
        follow_redirects: bool = True,
        timeout: Any = None,
        extensions: dict[str, Any] | None = None,
    ) -> httpx.Response: ...
