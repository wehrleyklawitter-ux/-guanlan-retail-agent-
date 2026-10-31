"""知识关联不能变造事实、改变计算，报告引用须可复现。"""
from copy import deepcopy

import pytest

from analysis import load_data, report
from diagnosis import diagnose_sales_decline, note_key
from knowledge import associate_knowledge, load_knowledge


def test_catalog_distinguishes_external_sources_and_project_rules():
    catalog = load_knowledge()
    cards = catalog["cards"]
    assert len({c["id"] for c in cards}) == len(cards) == 8
    for card in cards:
        assert card["principle"] and card["application"] and card["boundary"] and card["records"]
        if card["source_url"]:
            assert card["source_url"].startswith("https://")
            assert "公开资料" in card["kind"]
        else:
            assert card["source_org"] == "项目自行设计"
    assert {c["id"] for c in cards if c["source_url"]} == {"K05", "K06", "K07"}


@pytest.mark.parametrize("state", ["unknown", "yes", "no"])
def test_business_evidence_status_and_knowledge_stay_separate(state):
    df = load_data()
    baseline = diagnose_sales_decline(df, [16])
    result = diagnose_sales_decline(df, [16], business_notes={note_key(16, "2012-10-26"): {"promotion_ended": state}})
    case = result["cases"][0]
    promotion = next(h for h in case["hypotheses"] if h["id"] == "promotion_ended")
    assert promotion["knowledge_ids"] == ["K06"]
    for key in ("previous_sales", "current_sales", "change_pct", "change_amount", "history", "peers"):
        assert case[key] == baseline["cases"][0][key]
    if state == "unknown":
        assert promotion["status"] == "待补充记录" and "原始CSV没有" in promotion["evidence"]
    else:
        assert "未独立核验" in promotion["evidence"]
        assert (case["hypotheses"][0]["id"] == "promotion_ended") == (state == "yes")
    text = report(result)
    assert "知识依据：K06" in text
    assert "一般促销知识不证明本店" in text


def test_no_decline_does_not_attach_business_cause_cards():
    df = load_data()
    df.loc[df.Store.eq(16) & df.Date.eq(df.Date.max()), "Weekly_Sales"] = 9999999.0
    result = diagnose_sales_decline(df, [16])
    assert not result["cases"][0]["hypotheses"]
    assert not {"K05", "K06", "K07", "K08"}.intersection(c["id"] for c in result["knowledge_cards"])


def test_report_contains_only_resolved_citations_and_snapshot(monkeypatch):
    result = diagnose_sales_decline(load_data(), [16])
    cards = {c["id"]: c for c in result["knowledge_cards"]}
    for hypothesis in result["cases"][0]["hypotheses"]:
        assert set(hypothesis["knowledge_ids"]) <= cards.keys()
    assert "K04" not in cards
    assert "https://www.business.qld.gov.au/" in report(result)
    original = deepcopy(result)
    changed_catalog = load_knowledge()
    monkeypatch.setattr("knowledge.load_knowledge", lambda: changed_catalog)
    snapshot = associate_knowledge(deepcopy(result["cases"]))
    changed_catalog["cards"][0]["title"] = "changed after export"
    assert snapshot["knowledge_cards"][0]["title"] != "changed after export"
    assert result == original


def test_preview_ui_shows_source_and_keeps_request_count(workspace_app):
    app = workspace_app
    app.radio(key="nav_page").set_value("下降诊断").run()
    app.button(key="FormSubmitter:business_note_form-保存补充并排查").click().run()
    assert not app.exception
    assert any("促销结束" in e.label and "K06" in e.label for e in app.expander)
    assert any("Business Queensland" in e.value for e in app.markdown)
    assert app.session_state.usage.requests == 0
    app.selectbox(key="note_16|2012-10-26_promotion_ended").select("yes")
    app.button(key="FormSubmitter:business_note_form-保存补充并排查").click().run()
    assert not app.exception
    assert app.session_state.diagnosis_result["cases"][0]["hypotheses"][0]["knowledge_ids"] == ["K06"]
    app.radio(key="nav_page").set_value("数据与方法").run()
    assert not app.exception
    assert len([e for e in app.expander if e.label.startswith("K")]) == 8
    assert app.session_state.usage.requests == 0
