import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import DATA_PATH, AnalysisError, load_data, rank_sales_declines, get_sales_trend
from agent import Usage, ask_deepseek
from example import DEMO_QUESTION, run_example
from test_agent import FakeResponse, tool_message


@pytest.fixture(scope="module")
def df():
    return load_data()


def independent_ranking(metric):
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    weeks = sorted({datetime.strptime(r["Date"], "%d-%m-%Y") for r in records})
    recent = {int(r["Store"]): float(r["Weekly_Sales"]) for r in records if datetime.strptime(r["Date"], "%d-%m-%Y") == weeks[-1]}
    previous = {int(r["Store"]): float(r["Weekly_Sales"]) for r in records if datetime.strptime(r["Date"], "%d-%m-%Y") == weeks[-2]}
    dropped = [(s, previous[s] - recent[s], 100 * (previous[s] - recent[s]) / previous[s]) for s in recent if recent[s] < previous[s]]
    return sorted(dropped, key=lambda r: (-r[2 if metric == "percent" else 1], r[0]))[:5]


@pytest.mark.parametrize("metric", ["percent", "amount"])
def test_ranking_matches_independent_csv_computation(df, metric):
    result = rank_sales_declines(df, [], metric=metric)
    expected = independent_ranking(metric)
    assert result["actual_start"] == "2012-10-19"
    assert result["actual_end"] == "2012-10-26"
    assert result["compared_stores"] == 45
    assert result["selected_stores"] == [r[0] for r in expected]
    for actual, independent in zip(result["ranking"], expected):
        assert actual["drop_amount"] == pytest.approx(independent[1], abs=.01)
        assert actual["drop_pct"] == pytest.approx(independent[2])


def test_trend_matches_exact_dates_and_source(df):
    ranked = rank_sales_declines(df, [])
    trend = get_sales_trend(df, ranked["selected_stores"], weeks=8)
    assert trend["actual_start"] == "2012-09-07"
    assert trend["actual_end"] == "2012-10-26"
    assert trend["records"] == 40
    assert all(row["weeks"] == 8 for row in trend["summary"])
    for row in trend["trend"]:
        source = df[df.Store.eq(row["store"]) & df.Date.eq(pd.Timestamp(row["date"]))]
        assert row["sales"] == float(source.iloc[0].Weekly_Sales)


def test_missing_week_is_excluded_not_compared_across_gap(df):
    broken = df[~(df.Store.eq(1) & df.Date.eq(pd.Timestamp("2012-10-19")))]
    result = rank_sales_declines(broken, [1, 2])
    assert result["compared_stores"] == 1
    assert result["excluded"] == [{"store": 1, "reason": "缺少本周或相隔7天的上周记录"}]


def test_zero_baseline_and_no_declines(df):
    small = df[df.Store.eq(1) & df.Date.ge("2012-10-19")].copy()
    small.loc[small.Date.eq(pd.Timestamp("2012-10-19")), "Weekly_Sales"] = 0
    result = rank_sales_declines(small, [1])
    assert not result["ranking"] and result["excluded"]
    small["Weekly_Sales"] = 100.0
    result = rank_sales_declines(small, [1])
    assert result["declining_stores"] == 0 and not result["ranking"]


@pytest.mark.parametrize("kwargs", [dict(top_n=True), dict(top_n=11), dict(metric=[]), dict(stores=[99]), dict(stores=[True]), dict(end_date="2026-10-17"), dict(end_date="2012-10-25"), dict(end_date="2010-02-05")])
def test_ranking_rejects_bad_inputs(df, kwargs):
    with pytest.raises(AnalysisError):
        rank_sales_declines(df, **{"stores": [], **kwargs})


def test_partial_trend_discloses_missing_weeks(df):
    result = get_sales_trend(df, [1], "2010-02-12", 8)
    assert result["records"] == 2 and result["warnings"]


def test_fixed_example_is_two_real_tools(df):
    result = run_example(df)
    assert len(result.steps) == 2 and len(result.results) == 2
    assert result.steps[1]["arguments"]["stores"] == result.results[0]["selected_stores"]
    assert result.result["records"] == 40
    assert "无模型" in result.tool


def test_model_selects_two_tools_using_returned_evidence(df):
    count = 0
    def transport(*a, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return FakeResponse(tool_message("rank_sales_declines", {"stores": [], "end_date": "", "top_n": 5, "metric": "percent"}))
        if count == 2:
            evidence = json.loads(kwargs["json"]["messages"][-1]["content"])
            return FakeResponse(tool_message("get_sales_trend", {"stores": evidence["selected_stores"], "end_date": evidence["actual_end"], "weeks": 8}))
        return FakeResponse({"content": "请结合排名与历史走势，核实经营记录。"})
    usage = Usage()
    reply = ask_deepseek(DEMO_QUESTION, df, key="fake", usage=usage, transport=transport)
    assert usage.requests == 3
    assert len(reply.steps) == 2 and len(reply.results) == 2
    assert reply.explanation and reply.context.stores == tuple(reply.results[0]["selected_stores"])


def test_fabricated_ranked_stores_stop_second_tool(df):
    count = 0
    def transport(*a, **kwargs):
        nonlocal count
        count += 1
        return FakeResponse(tool_message("rank_sales_declines", {"stores": [], "end_date": "", "top_n": 5, "metric": "percent"}) if count == 1 else
                            tool_message("get_sales_trend", {"stores": [99], "end_date": "", "weeks": 8}))
    reply = ask_deepseek(DEMO_QUESTION, df, key="fake", transport=transport)
    assert len(reply.results) == 1 and "不一致" in reply.notices[-1]


def test_second_request_failure_preserves_partial_evidence(df):
    count = 0
    def transport(*a, **kwargs):
        nonlocal count
        count += 1
        if count > 1:
            raise requests.Timeout()
        return FakeResponse(tool_message("rank_sales_declines", {"stores": [], "end_date": "", "top_n": 5, "metric": "percent"}))
    reply = ask_deepseek(DEMO_QUESTION, df, key="fake", transport=transport)
    assert len(reply.results) == 1 and "尚未全部完成" in reply.notices[-1]


def test_loop_is_bounded(df):
    usage = Usage()
    reply = ask_deepseek("查询", df, key="fake", usage=usage, transport=lambda *a, **k: FakeResponse(tool_message()))
    assert usage.requests == 5 and len(reply.steps) == 4 and reply.notices


def test_early_model_stop_reports_missing_requested_trend(df):
    count = 0
    def transport(*a, **kwargs):
        nonlocal count
        count += 1
        return FakeResponse(tool_message("rank_sales_declines", {"stores": [], "end_date": "", "top_n": 5, "metric": "percent"}) if count == 1 else {"content": "请关注销售变化。"})
    reply = ask_deepseek(DEMO_QUESTION, df, key="fake", transport=transport)
    assert len(reply.results) == 1 and any("尚未调用" in notice for notice in reply.notices)
