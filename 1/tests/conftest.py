"""自动测试不读取用户的真实密钥、不发起真实API调用。"""
from pathlib import Path

import pytest
import requests
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest


@pytest.fixture
def workspace_app():
    """通过真实封面按钮进入工作区，供已有功能测试复用。"""
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button(key="start_conversation").click().run()
    assert not app.exception
    return app


@pytest.fixture(autouse=True)
def isolate_credentials_and_network(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("APP_ACCESS_CODE", raising=False)
    original_read = Path.read_text

    def read_without_local_key(path, *args, **kwargs):
        if path.name == "local_model.json" and path.parent.name == ".streamlit":
            return "{}"
        return original_read(path, *args, **kwargs)

    def no_live_request(*args, **kwargs):
        raise AssertionError("自动测试禁止发起真实API调用；请显式注入模拟transport。")

    monkeypatch.setattr(Path, "read_text", read_without_local_key)
    monkeypatch.setattr(Secrets, "get", lambda self, key, default=None: default)
    monkeypatch.setattr(requests, "post", no_live_request)
