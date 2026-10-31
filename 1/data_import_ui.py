"""数据导入对话框：校验成功且确认后，下一次完整运行才切换数据。"""
import streamlit as st

from analysis import AnalysisError, data_info
from data_import import TEMPLATE_BYTES, UNITS, parse_sales_csv, dataset_identity


def open_import():
    st.session_state.show_import_dialog = True


def close_import():
    st.session_state.show_import_dialog = False


@st.dialog("导入销售数据", width="medium", on_dismiss=close_import)
def render_import_dialog():
    st.write("下载模板、替换销售记录，再上传CSV。预览确认后才会切换当前数据。")
    st.download_button("下载CSV模板", TEMPLATE_BYTES, file_name="sales_template.csv", mime="text/csv", key="download_template")
    with st.expander("查看字段与格式要求"):
        st.markdown("必填：**Store**（整数门店编号）、**Date**（YYYY-MM-DD销售周日期）、**Weekly_Sales**（非负周销售额）、**Holiday_Flag**（0或1）。")
        st.caption("支持对应中文列名：门店编号、销售周日期、周销售额、节假日标记。温度、燃油价格、CPI和失业率为选填。最多5MB、50000条记录、100家门店；所有销售周须落在同一星期几。")
        st.caption("模板的101、202号店数据为虚构教学记录，金额单位请自行确认。仅支持CSV，不接收Excel文件。")
    uploaded = st.file_uploader("上传CSV销售数据", type=["csv"], max_upload_size=5, key="sales_csv_upload")
    unit = st.selectbox("这份数据的金额单位", UNITS, key="import_unit", help="单位只作口径标注，不进行汇率换算。")
    candidate = None
    raw = None
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            candidate, notices = parse_sales_csv(raw, uploaded.name, unit)
            info = data_info(candidate)
            st.success("格式检查通过，请核对以下数据后确认。")
            c1, c2, c3 = st.columns(3)
            c1.metric("销售记录", info["rows"])
            c2.metric("门店", len(info["stores"]))
            c3.metric("销售周", info["weeks"])
            st.caption(f"{info['start']} 至 {info['end']} · 金额单位：{unit}")
            preview = candidate.head(5).copy()
            preview["Date"] = preview.Date.dt.strftime("%Y-%m-%d")
            st.dataframe(preview, hide_index=True, width="stretch")
            for notice in notices:
                st.warning(notice)
        except AnalysisError as exc:
            st.error(str(exc))
    st.caption("上传与预览不调用模型。文件仅用于当前会话；使用DeepSeek分析时，模型会收到所需的统计结果和部分逐周数据。")
    st.warning("切换数据会清空旧对话、筛选结果和经营补充，保留模型请求计数。")
    left, right = st.columns(2)
    if left.button("使用这份数据", type="primary", disabled=candidate is None, key="confirm_upload", width="stretch"):
        st.session_state._pending_dataset = {"kind": "upload", "frame": candidate,
            "identity": dataset_identity(raw, unit), "label": candidate.attrs["label"]}
        close_import()
        st.rerun()
    if right.button("恢复Walmart示例", key="restore_example", width="stretch", disabled=st.session_state.get("dataset_kind", "example") == "example"):
        st.session_state._pending_dataset = {"kind": "example"}
        close_import()
        st.rerun()
