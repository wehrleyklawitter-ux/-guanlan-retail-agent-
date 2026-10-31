import json
from pathlib import Path
import sys

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import AnalysisError, load_data, Query
from agent import Usage, ask_deepseek, call_api, dispatch


@pytest.fixture(scope="module")
def df():
    return load_data()


class FakeResponse:
    status_code = 200
    def __init__(self, message): self.message = message
    def json(self):
        return {"choices": [{"message": self.message}], "usage": {"prompt_tokens": 100, "completion_tokens": 40}}


def tool_message(name="compare_stores", args=None):
    args = args or {"stores": [1, 20], "start": "2011-01-01", "end": "2011-12-31", "holiday": "all"}
    return {"role": "assistant", "content": None, "tool_calls": [{"id": "call_test", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_complete_tool_roundtrip_with_mock(df):
    payloads = []
    def transport(url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(tool_message() if len(payloads) == 1 else {"content": "阅读时请同时关注累计金额与样本覆盖情况。"})
    usage = Usage()
    reply = ask_deepseek("比较1号店和20号店2011年销售", df, key="mock-key", usage=usage, transport=transport)
    assert reply.result["records"] == 104 and reply.context.stores == (1, 20)
    assert reply.explanation and usage.requests == 2
    assert usage.prompt_tokens == 200
    assert payloads[0]["thinking"] == {"type": "disabled"}
    assert payloads[0]["tool_choice"] == "required"
    assert "parallel_tool_calls" not in payloads[0]
    assert payloads[1]["tool_choice"] == "auto"
    assert payloads[1]["messages"][-1]["role"] == "tool"
    evidence = json.loads(payloads[1]["messages"][-1]["content"])
    assert "trend" not in evidence


def test_model_clarification_preserves_context(df):
    old = Query((1,), "2011-01-01", "2011-12-31")
    message = tool_message("clarify", {"question": "你想比较哪两家门店？"})
    reply = ask_deepseek("比较一下", df, old, key="mock-key", transport=lambda *a, **k: FakeResponse(message))
    assert reply.result is None and reply.context is old


def test_no_key_does_not_contact_api():
    def prohibited(*a, **k): raise AssertionError("No request should be made")
    with pytest.raises(AnalysisError, match="尚未配置"):
        call_api("", "test", [], Usage(), transport=prohibited)


def test_cap_does_not_contact_api():
    with pytest.raises(AnalysisError, match="请求上限"):
        call_api("mock", "test", [], Usage(requests=30), transport=lambda *a, **k: pytest.fail("called"))


@pytest.mark.parametrize("status,match", [(401,"密钥无效"),(402,"余额不足"),(429,"限流"),(500,"暂时不可用")])
def test_http_errors_sanitized(status, match):
    response = FakeResponse({})
    response.status_code = status
    with pytest.raises(AnalysisError, match=match) as exc:
        call_api("secret-never-print", "test", [], Usage(), transport=lambda *a, **k: response)
    assert "secret-never-print" not in str(exc.value)


def test_timeout_sanitized():
    def timeout(*a, **k): raise requests.Timeout("secret")
    with pytest.raises(AnalysisError, match="请求超时"):
        call_api("mock", "test", [], Usage(), transport=timeout)


@pytest.mark.parametrize("name,args", [("run_python", {}), ("sales_summary", {"stores": [1]}),
    ("sales_summary", {"stores": [99], "start": "2011-01-01", "end": "2011-12-31", "holiday": "all"}),
    ("compare_holidays", {"stores": [1], "start": "2011-01-01", "end": "2011-12-31", "holiday": "holiday"})])
def test_bad_model_tools_rejected(df, name, args):
    with pytest.raises(AnalysisError): dispatch(name, args, df)


def test_invented_model_numbers_not_displayed(df):
    count = 0
    def transport(*a, **k):
        nonlocal count
        count += 1
        return FakeResponse(tool_message() if count == 1 else {"content": "销售额是999999999，促销导致增长。"})
    reply = ask_deepseek("比较门店", df, key="mock", transport=transport)
    assert reply.explanation == "" and reply.result is not None
    assert "999999999" not in reply.text


def test_explanation_failure_keeps_computed_result(df):
    count = 0
    def transport(*a, **k):
        nonlocal count
        count += 1
        if count == 2: raise requests.Timeout()
        return FakeResponse(tool_message())
    reply = ask_deepseek("比较门店", df, key="mock", transport=transport)
    assert reply.result is not None and reply.notices


def test_no_tool_is_not_passed_as_analysis(df):
    with pytest.raises(AnalysisError, match="可核验"):
        ask_deepseek("查询", df, key="mock", transport=lambda *a, **k: FakeResponse({"content": "我算出了100"}))


@pytest.mark.parametrize("message", [
    None,
    {"tool_calls": [None]},
    {"tool_calls": [{}]},
    {"tool_calls": {"invalid": True}},
    {"tool_calls": [{"id": "", "function": {"name": "sales_summary", "arguments": "{}"}}]},
])
def test_malformed_tool_envelope(df, message):
    with pytest.raises(AnalysisError):
        ask_deepseek("查询", df, key="mock", transport=lambda *a, **k: FakeResponse(message))
