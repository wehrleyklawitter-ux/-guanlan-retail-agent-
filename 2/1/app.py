from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from analysis import AnalysisError, HOLIDAYS, Query, analyze, data_info, load_data, parse_local, report, unit_label, sales_unit
from agent import DEFAULT_MODEL, Usage, ask_deepseek
from example import DEMO_QUESTION, run_example
from diagnosis import (DIAGNOSIS_QUESTION, BUSINESS_FIELDS, BUSINESS_STATES, note_key,
                       diagnose_sales_decline, local_diagnosis_request)

from interface import (apply_theme, render_sidebar, render_topbar, welcome, page_intro, style_figure, diagnosis_summary)
from data_import import apply_pending_dataset
from data_import_ui import render_import_dialog
from cover import render_cover
from knowledge_ui import render_diagnosis_knowledge, render_knowledge_catalog
from deployment import cloud_mode, DailyRequestBudget
from sample_data import load_default_data

st.set_page_config(page_title="观澜 · 零售经营分析", page_icon="⌁", layout="wide", initial_sidebar_state="auto")
apply_theme()

def setting(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    local_settings = Path(__file__).parent / ".streamlit" / "local_model.json"
    if not cloud_mode() and name in {"DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"} and local_settings.exists():
        try:
            value = json.loads(local_settings.read_text(encoding="utf-8")).get(name)
            if isinstance(value, str) and value:
                return value
        except (OSError, ValueError):
            pass
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


access_code = setting("APP_ACCESS_CODE")
if access_code:
    if not st.session_state.get("access_ok"):
        st.title("零售门店经营分析")
        entered = st.text_input("试用口令", type="password")
        if st.button("进入应用", type="primary"):
            if hmac.compare_digest(entered.encode("utf-8"), str(access_code).encode("utf-8")):
                st.session_state.access_ok = True
                st.rerun()
            else:
                st.error("口令不正确，请重新输入。")
        st.stop()


if not st.session_state.get("workspace_entered", False):
    render_cover()
    st.stop()


@st.cache_data
def cached_data(public_demo=False):
    return load_default_data(public_demo=public_demo)


@st.cache_resource
def shared_request_budget(limit):
    return DailyRequestBudget(limit)


def hosted_transport(*args, **kwargs):
    try:
        limit = int(setting("APP_DAILY_REQUEST_LIMIT", "100"))
        budget = shared_request_budget(limit)
    except (ValueError, TypeError):
        raise AnalysisError("智能体试用配置暂不可用，请使用筛选分析或下降诊断。") from None
    budget.reserve()
    return requests.post(*args, **kwargs)


try:
    sample_df = cached_data(cloud_mode())
    apply_pending_dataset(st.session_state, sample_df)
    df = st.session_state.get("dataset_frame", sample_df)
except (AnalysisError, OSError, ValueError) as exc:
    st.error(f"数据暂时无法读取：{exc}")
    st.stop()
info = data_info(df)
st.session_state.setdefault("dataset_kind", sample_df.attrs.get("source_kind", "example"))
st.session_state.setdefault("dataset_label", sample_df.attrs.get("label", "Walmart 历史销售样本"))
for key, value in {"chat_history": [], "context": None, "usage": Usage(), "business_notes": {}}.items():
    if key not in st.session_state:
        st.session_state[key] = value


page, mode, api_key, model = render_sidebar(info, setting)
render_topbar(page, mode, bool(api_key))
if df.attrs.get("source_kind") == "synthetic":
    st.info(df.attrs["source_notice"])
if st.session_state.get("dataset_notice"):
    st.success(st.session_state.pop("dataset_notice"))


def draw_result(result, unique_key, *, show_report=False):
    unit = unit_label(result)
    if show_report:
        if result.get("type") == "sales_diagnosis":
            st.markdown(diagnosis_summary(result))
            with st.expander("查看完整文字报告"):
                st.markdown(report(result))
        else:
            st.markdown(report(result))
    if result.get("type") == "sales_diagnosis":
        for index, case in enumerate(result["cases"]):
            st.markdown(f"**{case['store']}号店：诊断证据图**")
            cols = st.columns(3)
            cols[0].metric("周环比", f"{case['change_pct']:+.2f}%")
            cols[1].metric(f"本周销售额 / {unit}", f"{case['current_sales']:,.2f}")
            cols[2].metric("其他门店下降数", f"{case['peers']['declining_count']}/{case['peers']['count']}")
            trend = pd.DataFrame(case["trend"])
            trend["日期"] = pd.to_datetime(trend.date)
            trend[f"销售额 / {unit}"] = trend.sales
            fig = px.line(trend, x="日期", y=f"销售额 / {unit}", markers=True, color_discrete_sequence=["#315FA8"])
            if case["history"]["mean"] is not None:
                fig.add_hline(y=case["history"]["mean"], line_dash="dot", line_color="#526A82", annotation_text="此前窗口均值（不含目标周）")
            st.plotly_chart(style_figure(fig), theme=None, width="stretch", key=f"{unique_key}_diagnosis_{index}")
            render_diagnosis_knowledge(result, case, f"{unique_key}_{index}")
            with st.expander(f"查看{case['store']}号店排查证据与待核实事项"):
                st.write(" ".join(case["interpretation"]))
                if case["hypotheses"]:
                    st.dataframe(pd.DataFrame(case["hypotheses"])[["name", "status", "evidence", "next_check"]].rename(columns={"name": "可能原因", "status": "当前状态", "evidence": "依据与来源", "next_check": "核查动作"}), hide_index=True)
                if case["factors"]:
                    st.dataframe(pd.DataFrame(case["factors"])[["label", "previous", "current", "delta", "unit"]].rename(columns={"label": "环境字段", "previous": "上周", "current": "本周", "delta": "变化", "unit": "单位口径"}), hide_index=True)
                st.caption(case["business_note_source"] + "。 " + " ".join(case["warnings"]))
        st.download_button("下载原因排查报告", report(result), file_name="sales_diagnosis.md", key=f"{unique_key}_download")
        return
    if result.get("type") == "decline_ranking":
        rows = pd.DataFrame(result["ranking"])
        st.caption(f"比较 {result['actual_start']} → {result['actual_end']}；按{'降幅百分比' if result['metric'] == 'percent' else '下降金额'}排名。")
        if not rows.empty:
            rows["门店"] = rows.store.map(lambda s: f"{s}号店")
            rows["降幅 / %"] = rows.drop_pct
            rows[f"下降金额 / {unit}"] = rows.drop_amount
            y = "降幅 / %" if result["metric"] == "percent" else f"下降金额 / {unit}"
            chart = px.bar(rows, x="门店", y=y, color_discrete_sequence=["#AD4F36"])
            st.plotly_chart(style_figure(chart), theme=None, width="stretch", key=f"{unique_key}_rank")
            st.dataframe(rows[["门店", "previous_sales", "current_sales", f"下降金额 / {unit}", "降幅 / %"]].rename(columns={"previous_sales": f"上周销售额 / {unit}", "current_sales": f"本周销售额 / {unit}"}), hide_index=True)
        for warning in result["warnings"]:
            st.caption(warning)
        if result["excluded"]:
            st.dataframe(pd.DataFrame(result["excluded"]).rename(columns={"store": "排除门店", "reason": "原因"}), hide_index=True)
        st.download_button("下载排名报告", report(result), file_name="sales_declines.md", key=f"{unique_key}_download")
        return
    columns = st.columns(3)
    divisor = 1e6 if abs(result["total"]) >= 1e6 else 1
    prefix = "百万" if divisor == 1e6 else ""
    columns[0].metric(f"销售额合计 / {prefix}{unit}", f"{result['total'] / divisor:,.2f}")
    columns[1].metric("门店周记录", f"{result['records']:,}")
    columns[2].metric("覆盖周数", result["observed_weeks"])
    q = result["query"]
    st.caption(f"查询：{q['start']} 至 {q['end']} · {HOLIDAYS[q['holiday']]}。实际记录：{result['actual_start']} 至 {result['actual_end']}。")
    for warning in result["warnings"]:
        st.warning(warning)
    summary = pd.DataFrame(result["summary"])
    if q["kind"] == "holidays":
        holiday_table = pd.DataFrame(result["holiday_by_store"])
        holiday_table["门店"] = holiday_table.store.map(lambda s: f"{s}号店")
        holiday_table["周类型"] = holiday_table.flag.map({0: "普通周", 1: "节假日周"})
        holiday_table[f"平均周销售额 / {prefix}{unit}"] = holiday_table.average / divisor
        chart = px.bar(holiday_table, x="门店", y=f"平均周销售额 / {prefix}{unit}", color="周类型", barmode="group",
                       color_discrete_sequence=["#277F97", "#315FA8"], hover_data={"weeks": True})
        st.plotly_chart(style_figure(chart), theme=None, width="stretch", key=f"{unique_key}_holiday")
        st.caption("样本数量可能不同，使用平均值比较；差异不代表因果关系。")
    else:
        trend = pd.DataFrame(result["trend"])
        if len(q["stores"]) <= 6:
            trend["门店"] = trend.store.map(lambda s: f"{s}号店")
            trend["日期"] = pd.to_datetime(trend.date)
            trend[f"周销售额 / {prefix}{unit}"] = trend.sales / divisor
            chart = px.line(trend, x="日期", y=f"周销售额 / {prefix}{unit}", color="门店",
                            color_discrete_sequence=["#315FA8", "#277F97", "#A45F22", "#8666A7", "#6A789B", "#A15167"])
        else:
            top = summary.head(10).copy()
            top["门店"] = top.store.map(lambda s: f"{s}号店")
            top[f"累计销售额 / {prefix}{unit}"] = top.total / divisor
            chart = px.bar(top, x="门店", y=f"累计销售额 / {prefix}{unit}", color_discrete_sequence=["#315FA8"])
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", margin=dict(l=20,r=20,t=15,b=20), height=330)
        st.plotly_chart(style_figure(chart), theme=None, width="stretch", key=f"{unique_key}_chart")
    with st.expander("查看计算表与统计口径"):
        display = summary.rename(columns={"store": "门店", "weeks": "记录周数", "total": "销售额合计", "average": "平均周销售额", "minimum": "最低周销售额", "maximum": "最高周销售额"})
        st.dataframe(display, hide_index=True, width="stretch")
        if q["kind"] == "holidays":
            st.dataframe(pd.DataFrame(result["holiday_summary"]).rename(columns={"label":"周类型","records":"门店周记录数","average":"平均门店周销售额"}), hide_index=True)
        st.caption(f"合计=sum(Weekly_Sales)；平均周销售额=所选门店的销售额合计÷该门店记录数。多门店汇总的平均值按门店周记录计算。金额单位：{unit}。")
    st.download_button("下载这次分析报告", report(result), file_name="retail_analysis.md", mime="text/markdown", key=f"{unique_key}_download")


def render_quick():
    page_intro("门店销售查询", "筛选分析", "选好门店和时间，即可查看销售表现。适合先浏览数据，或核对对话中的计算结果。")
    saved = st.session_state.get("quick_filters", {})
    is_sample = df.attrs.get("source_kind", "example") == "example"
    default_stores = [1, 20] if is_sample else info["stores"][:2]
    default_dates = (pd.Timestamp("2011-01-01").date(), pd.Timestamp("2011-12-31").date()) if is_sample else (df.Date.min().date(), df.Date.max().date())
    widget_suffix = st.session_state.get("dataset_id", "example:v1")
    with st.form("filter_form"):
        c1, c2 = st.columns([2, 1])
        selected_stores = c1.multiselect("选择门店", info["stores"], default=saved.get("stores", default_stores), format_func=lambda s: f"{s}号店", key=f"filter_stores_{widget_suffix}")
        task = c2.selectbox("分析任务", ["销售查询", "门店比较", "节假日比较"], index=saved.get("task_index", 1 if len(info["stores"]) > 1 else 0), key=f"filter_task_{widget_suffix}")
        c3, c4 = st.columns([2, 1])
        dates = c3.date_input("时间范围", value=saved.get("dates", default_dates), min_value=df.Date.min().date(), max_value=df.Date.max().date(), format="YYYY-MM-DD", key=f"filter_dates_{widget_suffix}")
        holiday = c4.selectbox("周类型", list(HOLIDAYS), index=saved.get("holiday_index", 0), format_func=lambda s: HOLIDAYS[s], key=f"filter_holiday_{widget_suffix}")
        submitted = st.form_submit_button("开始分析", type="primary")
    if "quick_result" not in st.session_state or submitted:
        try:
            if len(dates) != 2:
                raise AnalysisError("请在日历中选择开始和结束日期。")
            query = Query(tuple(selected_stores), dates[0].isoformat(), dates[1].isoformat(),
                          "all" if task == "节假日比较" else holiday,
                          {"销售查询": "sales", "门店比较": "comparison", "节假日比较": "holidays"}[task])
            st.session_state.quick_result = analyze(df, query)
            st.session_state.quick_filters = {"stores": selected_stores, "dates": dates,
                "task_index": ["销售查询", "门店比较", "节假日比较"].index(task), "holiday_index": list(HOLIDAYS).index(holiday)}
            st.session_state.quick_error = ""
        except AnalysisError as exc:
            st.session_state.quick_result = None
            st.session_state.quick_error = str(exc)
    if st.session_state.get("quick_error"):
        st.error(st.session_state.quick_error)
    if st.session_state.get("quick_result"):
        draw_result(st.session_state.quick_result, "quick")

def starter_questions():
    selected = None
    is_sample = df.attrs.get("source_kind", "example") == "example"
    first, last = (1, 20) if is_sample else (info["stores"][0], info["stores"][-1])
    target = 16 if is_sample else first
    period = "2011年" if is_sample else f"{info['start']}至{info['end']}"
    compare_title = "比较两家门店" if len(info["stores"]) > 1 else "查看门店表现"
    with st.container(key="starter_cards"):
        left, right = st.columns(2)
        if left.button(f"**{compare_title}**\n\n查看{first}号店" + (f"与{last}号店" if first != last else "") + "的销售表现", key="example_compare", width="stretch"):
            selected = f"比较{first}号店和{last}号店{period}的销售表现" if first != last else f"{first}号店{period}的销售表现"
        if right.button(f"**排查销售下降**\n\n看看{target}号店最后一周发生了什么", key="diagnosis_example", width="stretch"):
            selected = f"请排查{target}号店最后一周销售下降的可能原因，给出证据和下一步核查建议。"
        left, right = st.columns(2)
        if left.button("**查看销售排名**\n\n了解全部门店的历史销售表现", key="example_ranking", width="stretch"):
            selected = "全部门店销售排名，使用全部时间和全部周"
        if right.button("**运行完整例子**\n\n找出下降最多的5家店，再看走势", key="complete_example", width="stretch"):
            selected = DEMO_QUESTION
    with st.expander("更多示例", expanded=False):
        if st.button("比较节假日与普通周", key="example_holiday", width="stretch"):
            selected = f"比较{first}号店{period}节假日与普通周的平均销售额"
    return selected


def render_chat():
    example = None
    if not st.session_state.chat_history:
        welcome()
        example = starter_questions()
    else:
        with st.expander("试试其他问题", expanded=False):
            example = starter_questions()
    context = st.session_state.context
    if context:
        st.caption(f"记住的条件：门店{list(context.stores)}；{context.start}至{context.end}；{HOLIDAYS[context.holiday]}。")
    for i, item in enumerate(st.session_state.chat_history):
        with st.chat_message(item["role"]):
            diagnostic = next((r for r in item.get("results", []) if r.get("type") == "sales_diagnosis"), None)
            if diagnostic is None and (item.get("result") or {}).get("type") == "sales_diagnosis":
                diagnostic = item["result"]
            if diagnostic:
                st.markdown(diagnosis_summary(diagnostic))
                with st.expander("查看完整文字报告"):
                    st.markdown(item["text"])
            else:
                st.markdown(item["text"])
            for j, result in enumerate(item.get("results") or ([item["result"]] if item.get("result") else [])):
                draw_result(result, f"history_{i}_{j}")
            if item.get("explanation"):
                st.markdown(f"**模型补充说明**\n\n{item['explanation']}")
            for notice in item.get("notices", []):
                st.caption(notice)
            if item.get("tool"):
                with st.expander("查看本次工具与参数"):
                    st.write(item["tool"])
                    if item.get("steps"):
                        for number, step in enumerate(item["steps"], 1):
                            st.markdown(f"**第{number}步：{step['tool']}**")
                            st.json(step["arguments"])
                            st.json(step["evidence"])
                    elif item.get("result"):
                        st.json(item["result"]["query"])
    if mode == "本地练习" and st.session_state.chat_history:
        st.caption("当前为本地规则练习。要体验模型自主选择工具，可在左侧“设置与模型”切换模式。")
    target = 16 if df.attrs.get("source_kind", "example") == "example" else info["stores"][0]
    prompt = st.chat_input(f"在这里输入问题，例如：为什么{target}号店最后一周销售下降？", key="chat_prompt", max_chars=2000, height=132, submit_mode="disable") or example
    if prompt:
        st.session_state.chat_history.append({"role": "user", "text": prompt})
        with st.spinner("正在核对条件并计算…"):
            try:
                if mode == "本地练习" and prompt == DEMO_QUESTION:
                    reply = run_example(df)
                    item = {"role": "assistant", "text": "**固定工具演练：本次未调用大模型。**\n\n" + reply.text,
                            "result": reply.result, "results": reply.results, "steps": reply.steps, "tool": reply.tool}
                    st.session_state.context = reply.context
                elif mode == "本地练习":
                    result = local_diagnosis_request(prompt, df, st.session_state.context, st.session_state.business_notes)
                    if result is None:
                        query = parse_local(prompt, df, st.session_state.context)
                        result = analyze(df, query)
                    item = {"role": "assistant", "text": report(result), "result": result, "tool": "本地规则解析 → Python分析工具"}
                    st.session_state.context = Query(**{**result["query"], "stores": tuple(result["query"]["stores"])})
                else:
                    reply = ask_deepseek(prompt, df, st.session_state.context, key=api_key, model=model,
                                         usage=st.session_state.usage, business_notes=st.session_state.business_notes,
                                         transport=hosted_transport if cloud_mode() else None)
                    item = {"role": "assistant", "text": reply.text, "result": reply.result,
                            "results": reply.results, "steps": reply.steps,
                            "tool": reply.tool, "explanation": reply.explanation, "notices": reply.notices}
                    st.session_state.context = reply.context
            except AnalysisError as exc:
                item = {"role": "assistant", "text": str(exc)}
            st.session_state.chat_history.append(item)
        st.rerun()

def render_diagnosis():
    page_intro("销售变化排查", "下降诊断", "选择门店和销售周，查看排查证据。补充已掌握的经营信息后，可以重新安排核查顺序。")
    st.write("此页直接调用Python诊断工具，不产生模型费用。对话分析中的DeepSeek也可选择同一工具。")
    st.caption("历史参照窗口不含目标周；当前CSV未含促销、库存和营业记录。下方补充信息只保留在当前会话，并绑定到所选门店与销售周。")
    target = 16 if df.attrs.get("source_kind", "example") == "example" else info["stores"][0]
    saved_controls = st.session_state.get("diagnosis_controls", {"store": target, "date": df.Date.max().date(), "weeks": 8})
    dc1, dc2, dc3 = st.columns(3)
    diagnostic_store = dc1.selectbox("排查门店", info["stores"], index=info["stores"].index(saved_controls["store"]), key="diagnostic_store", format_func=lambda s: f"{s}号店")
    diagnostic_date = dc2.date_input("目标销售周", value=saved_controls["date"], min_value=df.Date.min().date(), max_value=df.Date.max().date(), key="diagnostic_date")
    history_weeks = dc3.selectbox("此前参照周数（不含目标周）", [4, 8, 12, 26], index=[4, 8, 12, 26].index(saved_controls["weeks"]), key="diagnostic_weeks")
    bound_key = note_key(diagnostic_store, diagnostic_date.isoformat())
    saved_note = st.session_state.business_notes.get(bound_key, {})
    with st.form("business_note_form"):
        st.markdown("**补充经营信息（没有资料时保持‘尚未提供’）**")
        note = {}
        for field, (label, question) in BUSINESS_FIELDS.items():
            options = list(BUSINESS_STATES)
            note[field] = st.selectbox(label, options, index=options.index(saved_note.get(field, "unknown")),
                                      format_func=lambda value: BUSINESS_STATES[value], key=f"note_{bound_key}_{field}", help=question)
        diagnosis_submitted = st.form_submit_button("保存补充并排查", type="primary")
    if diagnosis_submitted:
        try:
            result = diagnose_sales_decline(df, [diagnostic_store], diagnostic_date.isoformat(), history_weeks,
                                           {bound_key: note})
            st.session_state.diagnosis_controls = {"store": diagnostic_store, "date": diagnostic_date, "weeks": history_weeks}
            st.session_state.business_notes[bound_key] = note
            st.session_state.diagnosis_result = result
            st.session_state.diagnosis_error = ""
        except AnalysisError as exc:
            st.session_state.diagnosis_error = str(exc)
            st.session_state.diagnosis_result = None
    if st.session_state.get("diagnosis_error"):
        st.error(st.session_state.diagnosis_error)
    diagnostic_result = st.session_state.get("diagnosis_result")
    if diagnostic_result:
        if diagnostic_result["query"]["stores"] != (diagnostic_store,) or diagnostic_result["actual_end"] != diagnostic_date.isoformat() or diagnostic_result["cases"][0]["history"]["requested_weeks"] != history_weeks:
            st.info("已切换门店或销售周，请点击‘保存补充并排查’重新计算。")
        else:
            draw_result(diagnostic_result, "diagnosis_form", show_report=True)

def render_about():
    page_intro("数据范围与统计口径", "数据与方法", "了解样本范围、计算口径和目前能回答的问题。")
    label = st.session_state.get("dataset_label", "Walmart 历史销售样本")
    st.write(f"当前数据：{label}；{info['rows']:,}条记录，{len(info['stores'])}家门店，共{info['weeks']}个销售周，金额单位：{sales_unit(df)}。")
    st.write("支持门店销售查询、比较及下降原因排查。排查结果区分计算事实、待核实假设和用户补充，无法仅凭销售样本确认根因。数据缺少商品、库存、成本和顾客字段，无法计算利润或具体商品补货。")
    st.write("节假日比较采用平均门店周销售额，同时展示样本数量。观察到差异不能证明因果关系。")
    st.write(f"最近几周以数据最新日期{info['end']}为参照；查询范围超出覆盖时，明确披露实际使用的记录。")
    st.subheader("数据字段")
    fields = pd.DataFrame({"字段": ["Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"],
        "含义": ["门店编号", "销售周日期", "周销售额", "是否节假日周", "当地温度", "当地燃油价格", "消费者价格指数", "失业率"],
        "用途": ["筛选与比较", "筛选与趋势", "金额统计", "节假日背景核查", "诊断时展示同期变化", "诊断时展示同期变化", "诊断时展示同期变化", "诊断时展示同期变化"]})
    fields["当前数据包含"] = fields["字段"].isin(df.columns).map({True: "是", False: "否"})
    st.dataframe(fields, hide_index=True, width="stretch")
    st.subheader("计算与回答")
    st.write("自然语言请求转为门店、日期和周类型参数，程序验证参数，再调用固定Python函数。金额报告和图表共用计算结果；模型补充说明不承担数值计算。")
    if df.attrs.get("source_kind", "example") == "example":
        st.caption("示例来源线索：Kaggle / mikhail1681 / Walmart Sales，依据课程题目。原站许可标注Other，公开再发布前需进一步核对授权。示例金额按课程说明使用美元。")
    elif df.attrs.get("source_kind") == "synthetic":
        st.caption(df.attrs["source_notice"] + " 来源：本项目生成，生成方法与使用说明随代码保存；没有促销、库存、营业或宏观环境实测记录。")
    else:
        st.caption("当前数据由用户上传，未独立核验真实性；分析仅使用本会话确认的数据。上传不调用模型，DeepSeek对话会使用分析所需的统计结果及部分逐周记录。")
    st.caption("官方接口文档：https://api-docs.deepseek.com/zh-cn/guides/tool_calls/")
    render_knowledge_catalog()

if page == "对话分析":
    render_chat()
elif page == "筛选分析":
    render_quick()
elif page == "下降诊断":
    render_diagnosis()
else:
    render_about()

if st.session_state.get("show_import_dialog"):
    render_import_dialog(sample_df.attrs.get("source_kind", "example"))
