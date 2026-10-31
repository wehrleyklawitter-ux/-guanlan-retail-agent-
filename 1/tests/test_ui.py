def test_initial_page_and_filtering(workspace_app):
    app = workspace_app
    assert not app.exception
    assert app.session_state.nav_page == "对话分析" and len(app.chat_input) == 1
    app.radio(key="nav_page").set_value("筛选分析").run()
    assert app.session_state.quick_result["records"] == 104
    app.multiselect[0].set_value([1])
    app.selectbox[0].select("销售查询")
    app.button(key="FormSubmitter:filter_form-开始分析").click().run()
    assert not app.exception
    assert app.session_state.quick_result["records"] == 52


def test_chat_followup_and_error_preserve_memory(workspace_app):
    app = workspace_app
    app.chat_input[0].set_value("比较1号店和20号店2011年的销售表现").run()
    assert not app.exception
    app.chat_input[0].set_value("只看节假日呢").run()
    assert not app.exception
    assert app.session_state.context.holiday == "holiday"
    assert app.session_state.chat_history[-1]["result"]["records"] == 8
    old = app.session_state.context
    app.chat_input[0].set_value("99号店2011年销售").run()
    assert not app.exception
    assert "不存在门店" in app.session_state.chat_history[-1]["text"]
    assert app.session_state.context == old


def test_model_mode_without_key_is_honest(workspace_app):
    app = workspace_app
    app.radio(key="mode").set_value("DeepSeek智能体").run()
    app.chat_input[0].set_value("1号店2011年销售").run()
    assert not app.exception
    assert "尚未配置" in app.session_state.chat_history[-1]["text"]
    assert app.session_state.usage.requests == 0


def test_complete_example_in_local_mode(workspace_app):
    app = workspace_app
    app.button(key="complete_example").click().run()
    assert not app.exception
    item = app.session_state.chat_history[-1]
    assert "未调用大模型" in item["text"]
    assert len(item["steps"]) == 2 and len(item["results"]) == 2
    assert item["results"][-1]["records"] == 40


def test_navigation_preserves_conversation_and_new_chat_retains_usage(workspace_app):
    app = workspace_app
    app.chat_input[0].set_value("1号店2011年销售").run()
    assert not app.exception
    context = app.session_state.context
    history = list(app.session_state.chat_history)
    app.radio(key="nav_page").set_value("数据与方法").run()
    assert not app.exception and not app.chat_input
    app.radio(key="nav_page").set_value("对话分析").run()
    assert not app.exception and app.session_state.context == context
    assert app.session_state.chat_history == history
    app.session_state.usage.requests = 7
    app.button(key="new_chat").click().run()
    assert not app.exception
    assert not app.session_state.chat_history and app.session_state.context is None
    assert app.session_state.usage.requests == 7 and len(app.chat_input) == 1


def test_filter_controls_survive_navigation(workspace_app):
    app = workspace_app
    app.radio(key="nav_page").set_value("筛选分析").run()
    app.multiselect[0].set_value([1])
    app.selectbox[0].select("销售查询")
    app.button(key="FormSubmitter:filter_form-开始分析").click().run()
    app.radio(key="nav_page").set_value("对话分析").run()
    app.radio(key="nav_page").set_value("筛选分析").run()
    assert not app.exception
    assert app.multiselect[0].value == [1] and app.selectbox[0].value == "销售查询"
    assert app.session_state.quick_result["records"] == 52
