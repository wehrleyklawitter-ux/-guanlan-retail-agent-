import csv
import io
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import AnalysisError, Query, analyze, load_data, parse_local, report
from agent import Usage, dispatch
from data_import import (MAX_BYTES, TEMPLATE_BYTES, activate_dataset, apply_pending_dataset,
                         dataset_identity, parse_sales_csv)
from diagnosis import diagnose_sales_decline, local_diagnosis_request
from example import run_example


def csv_bytes(rows, headers=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers or ["Store", "Date", "Weekly_Sales", "Holiday_Flag"])
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


@pytest.fixture
def uploaded():
    return parse_sales_csv(TEMPLATE_BYTES, "my_sales.csv", "元")[0]


def test_template_roundtrip_and_known_amount(uploaded):
    assert len(uploaded) == 18 and sorted(uploaded.Store.unique()) == [101, 202]
    assert uploaded.Date.min() == pd.Timestamp("2025-01-03")
    assert uploaded.Date.max() == pd.Timestamp("2025-02-28")
    assert uploaded.attrs["sales_unit"] == "元"
    result = analyze(uploaded, Query((101,), "2025-01-03", "2025-02-28"))
    assert result["total"] == sum(10000 + i * 200 for i in range(8)) + 9000
    assert "元" in report(result) and "美元" not in report(result)
    workflow = run_example(uploaded)
    assert workflow.results[0]["selected_stores"] == [101]
    case = diagnose_sales_decline(uploaded, [101])["cases"][0]
    assert case["previous_sales"] == 11400 and case["current_sales"] == 9000
    assert case["history"]["mean"] == 10700
    assert not case["factors"]


def test_unknown_currency_not_relabelled_usd():
    df, _ = parse_sales_csv(TEMPLATE_BYTES)
    assert "原始金额单位" in report(analyze(df, Query((101,), "2025-01-03", "2025-02-28")))


def test_chinese_headers_gb18030_and_legacy_dates():
    raw = "门店编号,销售周日期,周销售额,节假日标记,温度\n101,03-01-2025,100,0,\n101,10-01-2025,90,0,5\n".encode("gb18030")
    frame, _ = parse_sales_csv(raw, unit="元")
    assert frame.Date.iloc[0] == pd.Timestamp("2025-01-03")
    assert pd.isna(frame.Temperature.iloc[0])


@pytest.mark.parametrize("raw,match", [
    (b"", "文件为空"), (b"x" * (MAX_BYTES + 1), "超过5MB"),
    (b"Store,Date,Weekly_Sales,Holiday_Flag\n", "只有表头"),
    (b"Store,Date,Weekly_Sales\n1,2025-01-03,100\n", "缺少必填列"),
    (b"Store,Date,Weekly_Sales,Holiday_Flag,Store\n1,2025-01-03,100,0,1\n", "重复列名"),
    (b"Store,Date,Weekly_Sales,Holiday_Flag,Customer\n1,2025-01-03,100,0,A\n", "暂不支持"),
    (b"Store,Date,Weekly_Sales,Holiday_Flag\n1,2025-01-03,100\n", "列数与表头不同"),
    (csv_bytes([[1, "2025-02-30", 100, 0]]), "无效日期"),
    (csv_bytes([[1, "2025-01-03", 100, 0], [1, "10-01-2025", 90, 0]]), "统一使用"),
    (csv_bytes([[1, "2025-01-03", 100, 0], [1, "2025-01-04", 90, 0]]), "同一星期几"),
    (csv_bytes([[1.5, "2025-01-03", 100, 0]]), "整数编号"),
    (csv_bytes([[1, "2025-01-03", "NaN", 0]]), "有限数字"),
    (csv_bytes([[1, "2025-01-03", "inf", 0]]), "有限数字"),
    (csv_bytes([[1, "2025-01-03", -1, 0]]), "非负销售额"),
    (csv_bytes([[1, "2025-01-03", "", 0]]), "不能为空"),
    (csv_bytes([[1, "2025-01-03", 100, 2]]), "只支持0或1"),
    (csv_bytes([[1, "2025-01-03", 100, 0], [1, "2025-01-03", 90, 0]]), "重复记录"),
], ids=["empty", "oversize", "headers_only", "missing_column", "duplicate_header", "extra_column", "ragged_row", "bad_date", "mixed_date_formats", "daily_rows", "fractional_store", "nan", "infinity", "negative_sales", "blank_sales", "bad_holiday", "duplicate_week"])
def test_invalid_file_rejected(raw, match):
    with pytest.raises(AnalysisError, match=match):
        parse_sales_csv(raw)


def test_missing_weeks_not_zero_filled():
    df, notices = parse_sales_csv(csv_bytes([[101, "2025-01-03", 100, 0], [101, "2025-01-17", 90, 0]]))
    assert len(df) == 2 and any("缺失周" in n for n in notices)
    with pytest.raises(AnalysisError, match="缺少目标周"):
        diagnose_sales_decline(df, [101])


def test_large_store_id_is_not_treated_as_year_and_dates_are_dynamic(uploaded):
    uploaded.loc[uploaded.Store.eq(101), "Store"] = 202501
    q = parse_local("202501号店最近2周的销售", uploaded)
    assert q.stores == (202501,) and q.start == "2025-02-21"
    q = parse_local("202501号店2025-01-03至2025-01-10的销售", uploaded)
    assert q.stores == (202501,) and q.end == "2025-01-10"
    result = local_diagnosis_request("排查202501号店最后一周下降原因", uploaded)
    assert result["cases"][0]["store"] == 202501


def test_switch_clears_old_context_but_keeps_credentials_and_usage(uploaded):
    usage = Usage(requests=9)
    state = {"chat_history": [{"text": "old"}], "context": "old", "business_notes": {"16|2012-10-26": {}},
             "quick_result": {}, "quick_filters": {}, "diagnosis_result": {}, "diagnosis_controls": {},
             "note_16|2012-10-26_stockout": "yes", "diagnostic_store": 16, "chat_prompt": "old",
             "usage": usage, "api_key": "dummy", "mode": "本地练习", "access_ok": True}
    identity = dataset_identity(TEMPLATE_BYTES, "元")
    assert activate_dataset(state, uploaded, identity, "sales.csv", "upload")
    assert state["chat_history"] == [] and state["context"] is None and state["business_notes"] == {}
    assert not any(k.startswith(("note_", "diagnostic_")) for k in state)
    assert "quick_result" not in state and "diagnosis_result" not in state and "chat_prompt" not in state
    assert state["usage"] is usage and usage.requests == 9 and state["api_key"] == "dummy" and state["access_ok"]
    state["chat_history"] = ["new"]
    assert not activate_dataset(state, uploaded, identity, "same.csv", "upload")
    assert state["chat_history"] == ["new"]
    state["_pending_dataset"] = {"kind": "example"}
    assert apply_pending_dataset(state, load_data())
    assert state["dataset_kind"] == "example" and len(state["dataset_frame"]) == 6435
    assert usage.requests == 9 and not state["chat_history"]


def test_uploaded_dataset_runs_all_pages_without_sample_ids(uploaded, workspace_app):
    app = workspace_app
    app.session_state._pending_dataset = {"kind": "upload", "frame": uploaded,
        "identity": dataset_identity(TEMPLATE_BYTES, "元"), "label": "sales.csv"}
    app.run()
    assert not app.exception and app.session_state.dataset_kind == "upload"
    app.button(key="example_compare").click().run()
    assert not app.exception and "101号店" in app.session_state.chat_history[0]["text"]
    assert app.session_state.context.stores == (101, 202)
    app.radio(key="nav_page").set_value("筛选分析").run()
    assert not app.exception and app.multiselect[0].value == [101, 202]
    assert app.session_state.quick_result["unit_name"] == "元"
    app.radio(key="nav_page").set_value("下降诊断").run()
    assert not app.exception and app.selectbox(key="diagnostic_store").value == 101
    app.button(key="FormSubmitter:business_note_form-保存补充并排查").click().run()
    assert not app.exception and app.session_state.diagnosis_result["cases"][0]["store"] == 101
    app.radio(key="nav_page").set_value("数据与方法").run()
    assert not app.exception


def test_single_store_single_week_can_query(workspace_app):
    one, notices = parse_sales_csv(csv_bytes([[999, "2025-01-03", 100, 0]]))
    assert len(notices) >= 2
    app = workspace_app
    app.session_state._pending_dataset = {"kind": "upload", "frame": one, "identity": "one", "label": "one.csv"}
    app.run()
    app.radio(key="nav_page").set_value("筛选分析").run()
    assert not app.exception and app.session_state.quick_result["total"] == 100
