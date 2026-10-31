"""观澜封面与工作区入口；仅切换页面，不调用模型或清空分析状态。"""
import base64
from pathlib import Path

import streamlit as st


def background_image_url(name):
    """大背景图走独立静态请求，不阻塞页面文字与按钮的交互消息。"""
    if name not in {"cover-scene.png", "workspace-scene.png"}:
        raise ValueError("未知背景图片。")
    return f"app/static/{name}" if st.get_option("server.enableStaticServing") else ""


def enter_workspace():
    # 控件重新挂载时主动同步保存值，避免浏览器先显示控件的默认模式。
    for key in ("mode", "api_key", "model_name"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    st.session_state.workspace_entered = True
    st.session_state.nav_page = "对话分析"


def return_to_cover():
    # 保留暂时不再渲染的模型控件值，避免返回时改变模式或丢失本次输入。
    for key in ("mode", "api_key", "model_name"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    st.session_state.workspace_entered = False


def render_cover():
    assets = Path(__file__).parent / "assets"
    css = (assets / "cover.css").read_text(encoding="utf-8")
    brand_mark = base64.b64encode((assets / "cover-logo.svg").read_bytes()).decode("ascii")
    # 背景仅在封面使用，文字与入口始终由网页渲染，支持选择和键盘操作。
    scene = background_image_url("cover-scene.png")
    icons = {name: (assets / f"{name}.svg").read_text(encoding="utf-8")
             for name in ("chart-bar", "chart-line-up", "magnifying-glass")}
    # 关闭静态服务时保留可用的浅色页面，不退回发送大幅Base64图片。
    scene_css = '.stApp:has(.landing-cover) { background-image: url("' + scene + '"); }' if scene else ""
    st.markdown(f'<style>{css}{scene_css}</style>', unsafe_allow_html=True)
    # 封面多次重跑时仍保留工作区设置；数据、对话及计数由原会话机制保存。
    for key in ("mode", "api_key", "model_name"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    with st.container(key="cover_shell"):
        st.html(f'''<div class="landing-cover">
            <header class="cover-header">
                <div class="cover-brand"><span class="cover-monogram" aria-hidden="true"><img src="data:image/svg+xml;base64,{brand_mark}" alt=""></span>
                    <span>观澜</span><span class="cover-brand-description">零售经营分析智能体</span></div>
                <span class="cover-header-note">以数据为据 · 以经营为题</span>
            </header></div>''')
        with st.container(key="cover_hero"):
            st.markdown('''<div class="cover-copy">
                    <h1 class="cover-title">观澜</h1>
                    <p class="cover-statement">观数据之变，<br>察经营之势。</p>
                    <p class="cover-description">查询门店销售，比较经营表现，<br>循着证据，梳理销售下降的线索。</p>
                </div>''', unsafe_allow_html=True)
            st.button("开始对话", type="primary", key="start_conversation", on_click=enter_workspace)
            st.markdown('<p class="cover-entry-note">支持示例体验与 CSV 数据导入</p>', unsafe_allow_html=True)
        st.markdown(f'''<div class="cover-capabilities">
            <div class="cover-feature"><span class="cover-feature-icon" aria-hidden="true">{icons['chart-bar']}</span><div class="cover-feature-copy"><h2>看清销售</h2><p>查询门店表现，读懂历史走势。</p></div></div>
            <div class="cover-feature"><span class="cover-feature-icon" aria-hidden="true">{icons['chart-line-up']}</span><div class="cover-feature-copy"><h2>读懂差异</h2><p>对照门店与时段，核对关键数字。</p></div></div>
            <div class="cover-feature"><span class="cover-feature-icon" aria-hidden="true">{icons['magnifying-glass']}</span><div class="cover-feature-copy"><h2>循证排查</h2><p>梳理下降线索，明确下一步核查。</p></div></div>
            </div>''', unsafe_allow_html=True)
