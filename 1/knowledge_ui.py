"""知识来源预览；只读渲染已有诊断结果，不请求模型。"""
from html import escape
import streamlit as st

from knowledge import load_knowledge


def render_card_details(card, reviewed_on):
    source = escape(card["source_title"])
    if card["source_url"]:
        source = f'<a href="{escape(card["source_url"], quote=True)}" target="_blank" rel="noopener noreferrer">{escape(card["source_org"])} · {source} ↗</a>'
    records = "".join("<li>" + escape(record) + "</li>" for record in card["records"])
    st.markdown('<div class="knowledge-detail">'
        f'<div class="knowledge-reference"><span>{escape(card["id"])}</span><strong>{escape(card["title"])}</strong></div>'
        f'<p class="knowledge-kind">{escape(card["kind"])}</p>'
        f'<div class="knowledge-row"><strong>为什么这样查</strong><p>{escape(card["principle"])}</p></div>'
        f'<div class="knowledge-row"><strong>项目核查方法</strong><p>{escape(card["application"])}</p></div>'
        f'<div class="knowledge-row"><strong>请准备的记录</strong><ul>{records}</ul></div>'
        f'<div class="knowledge-boundary"><strong>适用边界</strong><br>{escape(card["boundary"])}</div>'
        f'<p class="knowledge-source">来源：{source}</p>'
        f'<p class="knowledge-kind">定位：{escape(card["source_section"])} · 查阅/整理：{escape(reviewed_on)}</p></div>', unsafe_allow_html=True)


def render_diagnosis_knowledge(result, case, unique_key):
    if not case.get("hypotheses") or not result.get("knowledge_cards"):
        return
    by_id = {card["id"]: card for card in result["knowledge_cards"]}
    with st.container(key=f"{unique_key}_knowledge"):
        st.subheader("下一步核查与知识依据")
        st.caption("按当前证据和经营补充安排顺序。展开建议，可查看需要的记录与资料来源。")
        for hypothesis in case["hypotheses"]:
            ids = [key for key in hypothesis.get("knowledge_ids", []) if key in by_id]
            label = f"{hypothesis['name']} · {hypothesis['status']}"
            if ids:
                label += " ｜ " + "、".join(ids)
            with st.expander(label, expanded="优先核查" in hypothesis["status"]):
                st.markdown('<div class="knowledge-observation">'
                    f'<div><strong>当前证据</strong><p>{escape(hypothesis["evidence"])}</p></div>'
                    f'<div><strong>下一步核查</strong><p>{escape(hypothesis["next_check"])}</p></div></div>', unsafe_allow_html=True)
                for card_id in ids:
                    render_card_details(by_id[card_id], result["knowledge_reviewed_on"])


def render_knowledge_catalog():
    catalog = load_knowledge()
    st.subheader("经营知识与来源")
    st.write("知识库预览包含8条：3条参考公开资料编写，其余为项目自行设计的分析或核查规则。通过固定类别关联到诊断建议。")
    st.caption("公开资料提供一般背景；核查清单由项目整理。知识条目不会补造门店事件，也不代表根因已经确认。")
    for card in catalog["cards"]:
        with st.expander(f"{card['id']} · {card['title']} · {card['kind']}"):
            render_card_details(card, catalog["reviewed_on"])
