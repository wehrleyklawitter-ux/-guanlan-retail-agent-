import csv
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import DATA_PATH, AnalysisError, Query, analyze, data_info, load_data, parse_local


@pytest.fixture(scope="module")
def df():
    return load_data()


def independent_rows():
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def test_dataset_shape(df):
    assert data_info(df) == {"rows": 6435, "stores": list(range(1, 46)), "weeks": 143,
                             "start": "2010-02-05", "end": "2012-10-26"}


def test_01_single_store_matches_independent_csv_sum(df):
    q = parse_local("1号店2011年的销售表现", df)
    result = analyze(df, q)
    rows = [r for r in independent_rows() if r["Store"] == "1" and r["Date"].endswith("2011")]
    assert result["total"] == pytest.approx(sum(float(r["Weekly_Sales"]) for r in rows), abs=.01)
    assert result["records"] == len(rows) == 52


def test_02_store_comparison(df):
    result = analyze(df, parse_local("比较1号店和20号店2011年的销售表现", df))
    assert result["query"]["kind"] == "comparison"
    assert len(result["summary"]) == 2
    for summary in result["summary"]:
        rows = [r for r in independent_rows() if int(r["Store"]) == summary["store"] and r["Date"].endswith("2011")]
        assert summary["total"] == pytest.approx(sum(float(r["Weekly_Sales"]) for r in rows), abs=.01)
        assert summary["average"] == pytest.approx(summary["total"] / len(rows), abs=.01)


def test_03_holiday_comparison(df):
    result = analyze(df, parse_local("比较1号店2011年节假日与普通周的平均销售额", df))
    assert result["query"]["kind"] == "holidays"
    counts = {r["flag"]: r["records"] for r in result["holiday_summary"]}
    assert counts == {0: 48, 1: 4}


def test_04_followup_group_one_initial(df):
    q = parse_local("比较1号店和20号店2011年的销售表现", df)
    assert q.stores == (1, 20) and q.start == "2011-01-01"


def test_05_followup_group_one_holidays(df):
    original = parse_local("比较1号店和20号店2011年的销售表现", df)
    followup = parse_local("只看节假日呢", df, original)
    assert followup.stores == original.stores and followup.start == original.start
    assert followup.holiday == "holiday"
    assert analyze(df, followup)["records"] == 8


def test_06_followup_group_two_initial(df):
    q = parse_local("1号店最近8周的销售趋势", df)
    result = analyze(df, q)
    assert q.start == "2012-09-07" and q.end == "2012-10-26"
    assert result["records"] == 8


def test_07_followup_group_two_override_year(df):
    old = parse_local("1号店最近8周的销售趋势", df)
    new = parse_local("改看2011年", df, old)
    assert new.stores == (1,) and new.start == "2011-01-01"
    assert analyze(df, new)["records"] == 52


@pytest.mark.parametrize("question, expected", [
    ("比较一下", "哪家门店"),
    ("99号店2011年销售", "不存在门店"),
    ("1号店2026年销售", "所选时间没有数据"),
])
def test_08_to_10_ambiguous_and_out_of_range(df, question, expected):
    with pytest.raises(AnalysisError, match=expected):
        parse_local(question, df)


def test_full_store_ranking(df):
    result = analyze(df, parse_local("全部门店销售排名", df))
    assert result["summary"][0]["store"] == 20
    assert result["summary"][0]["total"] == 301397792.46
    assert result["total"] == 6737218987.11


@pytest.mark.parametrize("text", ["1号店2011年13月", "1号店2011-02-30", "1号店2011-12-31至2011-01-01", "1号店最近0周", "1号店本月销售"])
def test_bad_dates(df, text):
    with pytest.raises(AnalysisError):
        parse_local(text, df)


def test_no_data_and_partial_year_are_disclosed(df):
    with pytest.raises(AnalysisError, match="没有记录"):
        analyze(df, Query((1,), "2011-01-01", "2011-01-02"))
    result = analyze(df, parse_local("1号店2010年销售", df))
    assert result["actual_start"] == "2010-02-05"
    assert result["warnings"]


def test_holiday_toggle(df):
    q = parse_local("1号店2011年节假日销售", df)
    q = parse_local("只看非节假日", df, q)
    assert q.holiday == "ordinary"
    q = parse_local("取消周类型筛选", df, q)
    assert q.holiday == "all"


@pytest.mark.parametrize("text", ["1号店利润", "预测1号店销量", "1号店商品补货", "油价导致1号店下降吗"])
def test_unsupported_analysis(df, text):
    with pytest.raises(AnalysisError, match="第一版支持"):
        parse_local(text, df)


def test_single_holiday_type_does_not_invent_difference(df):
    result = analyze(df, Query((1,), "2011-01-07", "2011-01-07", kind="holidays"))
    assert result["holiday_difference_pct"] is None
    assert result["warnings"]


def test_duplicate_and_invalid_parameters(df):
    for q in [Query((1, 1), "2011-01-01", "2011-12-31"), Query((True,), "2011-01-01", "2011-12-31"),
              Query((1,), "2011-01-01", "2011-12-31", holiday=[]), Query((1,), "2011-01-01", "2011-12-31", kind="comparison")]:
        with pytest.raises(AnalysisError):
            analyze(df, q)
