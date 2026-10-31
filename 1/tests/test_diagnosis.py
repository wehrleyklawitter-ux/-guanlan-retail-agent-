import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import DATA_PATH, AnalysisError, Query, load_data, report
from diagnosis import (BUSINESS_STATES, DIAGNOSIS_QUESTION, diagnose_sales_decline,
                       local_diagnosis_request, note_key)
from agent import ask_deepseek, dispatch, safe_explanation
from test_agent import FakeResponse, tool_message


@pytest.fixture(scope="module")
def df():
    return load_data()


def independent_records():
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as f:
        return [{**r, "date": datetime.strptime(r["Date"], "%d-%m-%Y").date().isoformat(),
                 "store": int(r["Store"]), "sales": float(r["Weekly_Sales"])} for r in csv.DictReader(f)]


def test_diagnosis_matches_source_and_past_only_baseline(df):
    case = diagnose_sales_decline(df, [16])["cases"][0]
    rows = independent_records()
    before = next(r for r in rows if r["store"] == 16 and r["date"] == "2012-10-19")
    after = next(r for r in rows if r["store"] == 16 and r["date"] == "2012-10-26")
    historical = [r["sales"] for r in rows if r["store"] == 16 and "2012-08-31" <= r["date"] < "2012-10-26"]
    assert case["previous_sales"] == before["sales"] and case["current_sales"] == after["sales"]
    assert case["change_pct"] == pytest.approx(100 * (after["sales"] / before["sales"] - 1))
    assert case["history"]["mean"] == pytest.approx(statistics.mean(historical))
    assert case["history"]["records"] == 8 and case["history"]["change_samples"] == 7
    assert len(case["trend"]) == 9
    values = {r["field"]: r for r in case["factors"]}
    for field in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        assert values[field]["current"] == float(after[field])
        assert values[field]["delta"] == pytest.approx(float(after[field]) - float(before[field]))
    assert case["holiday"] == {"previous": int(before["Holiday_Flag"]), "current": int(after["Holiday_Flag"])}


def test_peers_exclude_target_and_use_per_store_median(df):
    case = diagnose_sales_decline(df, [16])["cases"][0]
    rows = independent_records()
    before = {r["store"]: r["sales"] for r in rows if r["date"] == "2012-10-19" and r["store"] != 16}
    after = {r["store"]: r["sales"] for r in rows if r["date"] == "2012-10-26" and r["store"] != 16}
    changes = [100 * (after[s] / before[s] - 1) for s in before]
    assert case["peers"]["count"] == 44
    assert case["peers"]["declining_count"] == sum(c < 0 for c in changes)
    assert case["peers"]["median_change_pct"] == pytest.approx(statistics.median(changes))
    assert 16 not in [r["store"] for r in case["peer_changes"]]


def test_future_records_do_not_affect_past_diagnosis(df):
    original = diagnose_sales_decline(df, [16], "2011-10-21")
    changed = df.copy()
    changed.loc[changed.Date.gt("2011-10-21"), "Weekly_Sales"] = 999999999.0
    changed.loc[changed.Date.gt("2011-10-21"), "Temperature"] = 99999.0
    assert diagnose_sales_decline(changed, [16], "2011-10-21") == original


def test_missing_history_and_peer_pair_are_disclosed(df):
    broken = df[~((df.Store.eq(16) & df.Date.eq(pd.Timestamp("2012-09-14"))) |
                  (df.Store.eq(1) & df.Date.eq(pd.Timestamp("2012-10-19"))))]
    case = diagnose_sales_decline(broken, [16])["cases"][0]
    assert case["history"]["records"] == 7 and case["history"]["change_samples"] == 5
    assert case["peers"]["count"] == 43 and case["peers"]["excluded_stores"] == [1]
    assert any("实际只有7条" in w for w in case["warnings"])


def test_unknown_business_facts_are_not_invented(df):
    case = diagnose_sales_decline(df, [16])["cases"][0]
    assert set(case["business_note"].values()) == {"unknown"}
    business_hypotheses = [h for h in case["hypotheses"] if h["id"] in case["business_note"]]
    assert len(business_hypotheses) == 3
    assert all(h["status"] == "待补充记录" for h in business_hypotheses)
    assert all("原始CSV没有" in h["evidence"] for h in business_hypotheses)


def test_user_note_changes_priority_and_is_bound_to_store_and_week(df):
    notes = {note_key(16, "2012-10-26"): {"promotion_ended": "yes", "stockout": "no"}}
    result = diagnose_sales_decline(df, [16, 43], business_notes=notes)
    first, other = result["cases"]
    assert first["hypotheses"][0]["id"] == "promotion_ended"
    assert "用户填报支持" in first["hypotheses"][0]["status"]
    assert "未独立核验" in first["hypotheses"][0]["evidence"]
    assert first["hypotheses"][-1]["id"] == "stockout"
    assert set(other["business_note"].values()) == {"unknown"}
    another_week = diagnose_sales_decline(df, [16], "2012-10-19", business_notes=notes)["cases"][0]
    assert set(another_week["business_note"].values()) == {"unknown"}


def test_no_decline_does_not_generate_downturn_hypotheses(df):
    changed = df.copy()
    before = float(changed.loc[changed.Store.eq(16) & changed.Date.eq(pd.Timestamp("2012-10-19")), "Weekly_Sales"].iloc[0])
    changed.loc[changed.Store.eq(16) & changed.Date.eq(pd.Timestamp("2012-10-26")), "Weekly_Sales"] = before * 1.1
    result = diagnose_sales_decline(changed, [16])
    assert result["cases"][0]["change_pct"] > 0
    assert not result["cases"][0]["hypotheses"]
    assert "没有环比下降" in report(result)


@pytest.mark.parametrize("kwargs", [
    {"stores": []}, {"stores": [True]}, {"stores": [99]}, {"stores": [16, 16]},
    {"history_weeks": True}, {"history_weeks": 3}, {"history_weeks": 27},
    {"end_date": "2012-10-25"}, {"end_date": "2010-02-05"},
    {"business_notes": {"16|2012-10-26": {"stockout": "confirmed"}}},
    {"business_notes": {"16|2012-10-26": {"invented": "yes"}}},
])
def test_bad_diagnosis_parameters(df, kwargs):
    with pytest.raises(AnalysisError):
        diagnose_sales_decline(df, **{"stores": [16], **kwargs})


def test_model_cannot_inject_business_facts_as_tool_arguments(df):
    args = {"stores": [16], "end_date": "", "history_weeks": 8, "business_notes": {"stockout": "yes"}}
    with pytest.raises(AnalysisError):
        dispatch("diagnose_sales_decline", args, df)


def test_mock_model_ranks_then_diagnoses_actual_stores(df):
    calls = 0
    def transport(*a, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse(tool_message("rank_sales_declines", {"stores": [], "end_date": "", "top_n": 5, "metric": "percent"}))
        if calls == 2:
            evidence = json.loads(kwargs["json"]["messages"][-1]["content"])
            return FakeResponse(tool_message("diagnose_sales_decline", {"stores": evidence["selected_stores"], "end_date": evidence["actual_end"], "history_weeks": 8}))
        return FakeResponse({"content": '{"focus":["history","business_records"]}'})
    reply = ask_deepseek("找出最后一周下降最多的5家店并排查原因", df, key="fake", transport=transport)
    assert [s["tool"] for s in reply.steps] == ["rank_sales_declines", "diagnose_sales_decline"]
    assert [c["store"] for c in reply.result["cases"]] == reply.results[0]["selected_stores"]
    assert reply.explanation and not reply.notices


def test_diagnosis_reading_advice_cannot_add_free_claims():
    evidence = {"type": "sales_diagnosis"}
    assert safe_explanation('{"focus":["history","business_records"]}', evidence)
    assert not safe_explanation('{"focus":["促销导致销售下降"]}', evidence)
    assert not safe_explanation('{"focus":["history"],"claim":"缺货导致下降"}', evidence)
    assert not safe_explanation("促销导致下降99%", evidence)


def test_local_diagnosis_and_followup_date(df):
    result = local_diagnosis_request(DIAGNOSIS_QUESTION, df)
    assert result["cases"][0]["store"] == 16
    old = Query((16,), "2011-10-01", "2011-10-21")
    result = local_diagnosis_request("为什么下降", df, old)
    assert result["actual_end"] == "2011-10-21"
    with pytest.raises(AnalysisError):
        local_diagnosis_request("16号店2011年下降原因", df)


def test_local_diagnosis_button_and_form_update(workspace_app):
    app = workspace_app
    app.button(key="diagnosis_example").click().run()
    assert not app.exception
    assert app.session_state.chat_history[-1]["result"]["type"] == "sales_diagnosis"
    app.radio(key="nav_page").set_value("下降诊断").run()
    app.selectbox(key="note_16|2012-10-26_promotion_ended").select("yes")
    app.button(key="FormSubmitter:business_note_form-保存补充并排查").click().run()
    assert not app.exception
    case = app.session_state.diagnosis_result["cases"][0]
    assert case["business_note"]["promotion_ended"] == "yes"
    assert case["hypotheses"][0]["id"] == "promotion_ended"
    app.selectbox(key="diagnostic_store").select(43).run()
    assert not app.exception
    assert app.selectbox(key="note_43|2012-10-26_promotion_ended").value == "unknown"
