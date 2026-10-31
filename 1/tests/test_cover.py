"""封面入口与工作区往返必须保留当前会话。"""
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_import import TEMPLATE_BYTES, dataset_identity, parse_sales_csv


def test_cover_is_default_and_start_opens_conversation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.button(key="start_conversation").label == "开始对话"
    assert not app.chat_input and not app.radio

    app.button(key="start_conversation").click().run()
    assert not app.exception
    assert app.session_state.workspace_entered
    assert app.session_state.nav_page == "对话分析" and len(app.chat_input) == 1
    assert app.session_state.usage.requests == 0


def test_return_to_cover_preserves_uploaded_data_conversation_and_usage(workspace_app):
    app = workspace_app
    uploaded, _ = parse_sales_csv(TEMPLATE_BYTES, "sales.csv", "元")
    app.session_state._pending_dataset = {
        "kind": "upload", "frame": uploaded,
        "identity": dataset_identity(TEMPLATE_BYTES, "元"), "label": "sales.csv",
    }
    app.run()
    app.button(key="example_compare").click().run()
    assert not app.exception and app.session_state.context.stores == (101, 202)
    history = list(app.session_state.chat_history)
    context = app.session_state.context
    app.session_state.usage.requests = 7
    app.session_state.business_notes = {"101|2025-02-28": {"stockout": "unknown"}}
    app.radio(key="nav_page").set_value("数据与方法").run()
    app.radio(key="mode").set_value("DeepSeek智能体")
    app.text_input(key="api_key").set_value("test-only-key")
    app.text_input(key="model_name").set_value("test-only-model").run()
    assert not app.exception and app.session_state.usage.requests == 7

    app.button(key="back_to_cover").click().run()
    assert not app.exception and not app.session_state.workspace_entered
    assert not app.chat_input and not app.radio
    assert app.session_state.chat_history == history
    assert app.session_state.context == context
    assert app.session_state.dataset_kind == "upload"
    assert app.session_state.dataset_label == "sales.csv"
    assert app.session_state.usage.requests == 7
    app.run()
    assert not app.exception and not app.chat_input and not app.radio
    assert app.session_state.mode == "DeepSeek智能体"
    assert app.session_state.api_key == "test-only-key"
    assert app.session_state.model_name == "test-only-model"
    assert app.session_state.usage.requests == 7

    app.button(key="start_conversation").click().run()
    assert not app.exception and app.session_state.nav_page == "对话分析"
    assert len(app.chat_input) == 1
    assert app.session_state.chat_history == history and app.session_state.context == context
    assert app.session_state.dataset_frame.attrs["sales_unit"] == "元"
    assert app.session_state.business_notes == {"101|2025-02-28": {"stockout": "unknown"}}
    assert app.session_state.usage.requests == 7
    assert app.radio(key="mode").value == "DeepSeek智能体"
    assert app.text_input(key="api_key").value == "test-only-key"
    assert app.text_input(key="model_name").value == "test-only-model"
