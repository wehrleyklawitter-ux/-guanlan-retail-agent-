"""界面布局与主题；不改变分析计算和模型调用。"""
from html import escape
import base64
import json
from pathlib import Path

import streamlit as st

from agent import DEFAULT_MODEL
from data_import_ui import open_import
from cover import return_to_cover

NAVIGATION = ["对话分析", "筛选分析", "下降诊断", "数据与方法"]


def apply_theme():
    assets = Path(__file__).parent / "assets"
    css = (assets / "interface.css").read_text(encoding="utf-8")
    css += (assets / "workspace.css").read_text(encoding="utf-8")
    scene = base64.b64encode((assets / "workspace-scene.png").read_bytes()).decode("ascii")
    css += '.stApp:not(:has(.landing-cover)){background-image:linear-gradient(rgba(242,247,255,.18),rgba(242,247,255,.7)),url("data:image/png;base64,' + scene + '");}'
    for key, name in (("example_compare", "chart-bar"), ("diagnosis_example", "chart-line-up"),
                      ("example_ranking", "trophy"), ("complete_example", "report")):
        icon_svg = (assets / f"{name}.svg").read_text(encoding="utf-8").replace('fill="currentColor"', 'fill="#3475ff"')
        icon = base64.b64encode(icon_svg.encode("utf-8")).decode("ascii")
        css += f'.st-key-{key} button::before{{background-image:url("data:image/svg+xml;base64,{icon}");}}'
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def brand_image():
    mark = base64.b64encode((Path(__file__).parent / "assets" / "cover-logo.svg").read_bytes()).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{mark}" alt="" aria-hidden="true">'


def reset_chat():
    st.session_state.chat_history = []
    st.session_state.context = None
    st.session_state.nav_page = "对话分析"


def render_sidebar(info, setting):
    configured_key = setting("DEEPSEEK_API_KEY")
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark" aria-hidden="true">' + brand_image() + '</div><div><div class="brand-name">观澜</div><div class="brand-sub">零售经营分析智能体</div></div></div>', unsafe_allow_html=True)
        st.button("新建对话", icon=":material/edit_square:", key="new_chat", width="stretch", on_click=reset_chat)
        st.markdown('<div class="sidebar-label">工作空间</div>', unsafe_allow_html=True)
        page = st.radio("功能导航", NAVIGATION, key="nav_page", label_visibility="collapsed")
        if st.session_state.chat_history:
            title = next((item["text"] for item in st.session_state.chat_history if item["role"] == "user"), "当前对话")
            st.markdown('<div class="sidebar-label">当前对话</div><div class="current-chat">' + escape(title[:60]) + '</div>', unsafe_allow_html=True)
        label = st.session_state.get("dataset_label", "Walmart 历史销售样本")
        st.markdown('<div class="sidebar-label">当前数据</div><div class="data-badge"><strong>' + escape(label) + '</strong><br>' +
                    f"{len(info['stores'])} 家门店 · {info['weeks']} 周<br>{info['start']} — {info['end']}" + '</div>', unsafe_allow_html=True)
        with st.expander("设置与模型", icon=":material/tune:"):
            if "mode" not in st.session_state:
                st.session_state.mode = "DeepSeek智能体" if configured_key else "本地练习"
            mode = st.radio("对话模式", ["本地练习", "DeepSeek智能体"], key="mode")
            st.caption("本地练习不调用模型；智能体模式按量计费。")
            entered = st.text_input("DeepSeek API密钥", type="password", key="api_key", help="已保存密钥不回显；留空使用本机配置。")
            api_key = entered.strip() or configured_key
            if "model_name" not in st.session_state:
                st.session_state.model_name = setting("DEEPSEEK_MODEL", DEFAULT_MODEL)
            model = st.text_input("模型名称", key="model_name")
            if st.get_option("server.address") == "127.0.0.1" and entered:
                if st.button("保存密钥到本机", key="save_key"):
                    path = Path(__file__).parent / ".streamlit" / "local_model.json"
                    path.parent.mkdir(exist_ok=True)
                    try:
                        path.write_text(json.dumps({"DEEPSEEK_API_KEY": entered.strip(), "DEEPSEEK_MODEL": model}, ensure_ascii=False), encoding="utf-8")
                        st.success("已保存到本机。")
                    except OSError:
                        st.error("保存失败，本次仍可使用输入的密钥。")
            st.caption("密钥已配置，提问后才会请求模型。" if api_key else "尚未配置密钥，可以先用本地练习。")
            usage = st.session_state.usage
            st.caption(f"本会话 {usage.requests}/{usage.max_requests} 次请求；输入 {usage.prompt_tokens}、输出 {usage.completion_tokens} tokens。每个问题最多5次请求。")
            st.caption("新建对话保留计费计数和已填写的经营信息。")
        st.button("返回封面", icon=":material/home:", key="back_to_cover", on_click=return_to_cover, width="stretch")
        st.markdown('<div class="sidebar-foot">历史数据分析与经营排查<br>每个结论，都有可查看的计算依据。</div>', unsafe_allow_html=True)
    return page, mode, api_key, model


def render_topbar(page, mode, configured):
    text = "智能体 · 已配置" if mode == "DeepSeek智能体" and configured else "智能体 · 待配置" if mode == "DeepSeek智能体" else "本地练习 · 免费"
    dot = "status-dot" if mode == "DeepSeek智能体" and configured else "status-dot local"
    with st.container(key="workspace_topbar"):
        heading, action = st.columns([6, 2.2], vertical_alignment="center")
        with heading:
            st.markdown('<div class="topbar"><div class="topbar-title">观澜<span class="topbar-path"> / ' +
                        escape(page) + '</span></div><span class="status-pill"><span class="' + dot + '"></span>' + text + '</span></div>', unsafe_allow_html=True)
        with action:
            st.button("导入销售数据", icon=":material/upload_file:", key="open_import", width="stretch", on_click=open_import)


def welcome():
    st.markdown('<div class="welcome"><div class="welcome-icon" aria-hidden="true">' + brand_image() + '</div><h1>今天想了解哪家门店？</h1><p>每周销售复盘：找到重点门店，查看证据，明确下一步核查。</p></div>', unsafe_allow_html=True)


def page_intro(eyebrow, title, description):
    st.markdown('<div class="page-eyebrow">' + escape(eyebrow) + '</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown('<div class="page-description">' + escape(description) + '</div>', unsafe_allow_html=True)


def diagnosis_summary(result):
    lines = []
    for case in result["cases"]:
        lines.append(f"### {case['store']}号店 · 销售下降排查")
        lines.append(" ".join(case["interpretation"]))
        supported = [h["name"] for h in case["hypotheses"] if "优先核查" in h["status"]]
        if supported:
            lines.append("**优先核查**：" + "、".join(supported) + "。这些是用户补充的情况，仍需记录核验。")
        elif case["hypotheses"]:
            lines.append("**接下来**：核对促销、缺货和营业记录。在“下降诊断”补充已掌握的信息，可重新排查。")
    return "\n\n".join(lines)


def style_figure(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(color="#18304A", family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif", size=13),
                      colorway=["#315FA8", "#277F97", "#A45F22", "#8666A7", "#6A789B", "#A15167"],
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#BDCEE3", font=dict(color="#18304A", size=13)),
                      legend=dict(bgcolor="#FFFFFF", font=dict(color="#526A82")),
                      margin=dict(l=18,r=18,t=32,b=24), height=340)
    fig.update_xaxes(gridcolor="#E2EAF5", zerolinecolor="#CCDCEC", automargin=True,
                     tickfont_color="#526A82", title_font_color="#526A82")
    fig.update_yaxes(gridcolor="#E2EAF5", zerolinecolor="#CCDCEC", automargin=True,
                     tickfont_color="#526A82", title_font_color="#526A82")
    fig.update_annotations(font_color="#526A82")
    return fig
