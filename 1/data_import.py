"""销售CSV导入与会话切换。上传内容只存当前会话，不写入原始数据目录。"""
from __future__ import annotations

import csv
import hashlib
import io
import re

import numpy as np
import pandas as pd

from analysis import AnalysisError, data_info

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 50000
MAX_STORES = 100
UNITS = ["未注明", "元", "美元"]
REQUIRED = ["Store", "Date", "Weekly_Sales", "Holiday_Flag"]
OPTIONAL = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
ALIASES = {"门店编号": "Store", "销售周日期": "Date", "周销售额": "Weekly_Sales", "节假日标记": "Holiday_Flag",
           "温度": "Temperature", "燃油价格": "Fuel_Price", "消费者价格指数": "CPI", "失业率": "Unemployment"}


def bad_rows(mask):
    return "、".join(str(int(i) + 2) for i in mask[mask].index[:5])


def parse_sales_csv(raw, filename="sales.csv", unit="未注明"):
    if not isinstance(raw, bytes) or not raw:
        raise AnalysisError("文件为空，请上传包含表头和销售记录的CSV。")
    if len(raw) > MAX_BYTES:
        raise AnalysisError("文件超过5MB，请按门店或时间范围缩小数据后重试。")
    if not isinstance(unit, str) or unit not in UNITS:
        raise AnalysisError("请选择金额单位：元、美元或未注明。")
    text = None
    for encoding in ["utf-8-sig", "gb18030"]:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    if text is None or "\x00" in text:
        raise AnalysisError("无法读取文件编码，请另存为UTF-8编码的CSV。")
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        records = [row for row in reader if any(cell.strip() for cell in row)]
    except csv.Error:
        raise AnalysisError("CSV引号或分隔格式有误，请按模板重新导出。") from None
    if len(records) < 2:
        raise AnalysisError("文件只有表头或没有销售记录。")
    if len(records) - 1 > MAX_ROWS:
        raise AnalysisError("记录超过50000行，请缩小数据范围。")
    headers = [ALIASES.get(c.strip(), c.strip()) for c in records[0]]
    if len(headers) != len(set(headers)):
        raise AnalysisError("存在重复列名，或中英文列名对应同一字段，请保留一列。")
    missing = [c for c in REQUIRED if c not in headers]
    if missing:
        raise AnalysisError("缺少必填列：" + "、".join(missing) + "。请下载模板核对。")
    unknown = [c for c in headers if c not in REQUIRED + OPTIONAL]
    if unknown:
        raise AnalysisError("暂不支持这些列：" + "、".join(unknown[:5]) + "。请仅保留模板字段。")
    if any(len(row) != len(headers) for row in records[1:]):
        raise AnalysisError("部分记录列数与表头不同，请检查逗号和引号。")
    df = pd.DataFrame(records[1:], columns=headers).map(lambda value: value.strip())
    for column in REQUIRED:
        empty = df[column].eq("")
        if empty.any():
            raise AnalysisError(f"{column}不能为空，文件记录行：{bad_rows(empty)}。")
    dates = df.Date
    if dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        date_format = "%Y-%m-%d"
    elif dates.str.fullmatch(r"\d{2}-\d{2}-\d{4}").all():
        date_format = "%d-%m-%Y"
    else:
        raise AnalysisError("Date请统一使用YYYY-MM-DD格式（也兼容整列DD-MM-YYYY），不要混用格式或加入时分秒。")
    parsed = pd.to_datetime(dates, format=date_format, errors="coerce")
    if parsed.isna().any():
        raise AnalysisError(f"Date存在无效日期，文件记录行：{bad_rows(parsed.isna())}。")
    if not parsed.dt.year.between(1900, 2100).all():
        raise AnalysisError("Date请使用1900年至2100年之间的销售周日期。")
    df["Date"] = parsed
    for column in ["Store", "Weekly_Sales", "Holiday_Flag"] + [c for c in OPTIONAL if c in df]:
        original = df[column]
        values = pd.to_numeric(original.replace("", np.nan), errors="coerce")
        invalid = ~np.isfinite(values) & (original.ne("") if column in OPTIONAL else True)
        if invalid.any():
            raise AnalysisError(f"{column}应为有限数字，不能含文字、NaN或无穷值，文件记录行：{bad_rows(invalid)}。")
        if values.abs().gt(1e15).any():
            raise AnalysisError(f"{column}存在超大数值，请检查金额或指标单位（单值绝对值上限10的15次方）。")
        df[column] = values
    invalid_store = (df.Store <= 0) | (df.Store > 999999999) | (df.Store % 1 != 0)
    if invalid_store.any():
        raise AnalysisError(f"Store应为1至999999999之间的整数编号，文件记录行：{bad_rows(invalid_store)}。")
    df["Store"] = df.Store.astype("int64")
    if df.Store.nunique() > MAX_STORES:
        raise AnalysisError("第一版每份数据最多支持100家门店，请缩小范围。")
    if (df.Weekly_Sales < 0).any():
        raise AnalysisError("Weekly_Sales应为非负销售额，负值需要先确认业务口径。")
    invalid_flag = ~df.Holiday_Flag.isin([0, 1])
    if invalid_flag.any():
        raise AnalysisError(f"Holiday_Flag只支持0或1，文件记录行：{bad_rows(invalid_flag)}。")
    df["Holiday_Flag"] = df.Holiday_Flag.astype("int64")
    duplicates = df.duplicated(["Store", "Date"], keep=False)
    if duplicates.any():
        raise AnalysisError(f"同一门店、同一销售周有重复记录，文件记录行：{bad_rows(duplicates)}。")
    if df.Date.dt.dayofweek.nunique() != 1:
        raise AnalysisError("每行应是一家门店的一周销售；销售周日期须统一在同一星期几，不能混入日销售记录。")
    df = df[REQUIRED + [c for c in OPTIONAL if c in df]].sort_values(["Date", "Store"]).reset_index(drop=True)
    label = re.split(r"[\\/]", str(filename))[-1][:100]
    df.attrs.update(sales_unit=unit, source_kind="upload", label=label)
    notices = []
    if df.Date.nunique() < 2:
        notices.append("只有一个销售周，可查询销售额；周环比和下降诊断还需要前一周记录。")
    if df.Store.nunique() == 1:
        notices.append("只有一家门店，可查询和诊断；门店间比较缺少参照。")
    expected_dates = pd.date_range(df.Date.min(), df.Date.max(), freq="7D")
    gaps = sum(len(expected_dates.difference(rows.Date)) for _, rows in df.groupby("Store"))
    if gaps:
        notices.append(f"门店记录在整体时间范围内有{gaps}个缺失周位置；缺失值不会当作零销售，也不会跨周计算环比。")
    return df, notices


def dataset_identity(raw, unit):
    return "upload:" + hashlib.sha256(raw + b"\x00" + unit.encode()).hexdigest()


def activate_dataset(state, frame, identity, label, kind):
    """应在页面小组件创建前调用，原子切换并清理旧数据上下文。"""
    if state.get("dataset_id", "example:v1") == identity:
        return False
    exact = {"chat_history", "context", "business_notes", "quick_result", "quick_error", "quick_filters",
             "diagnosis_result", "diagnosis_error", "diagnosis_controls", "chat_prompt"}
    for key in list(state):
        if key in exact or str(key).startswith(("note_", "diagnostic_", "filter_")):
            del state[key]
    state["chat_history"], state["context"], state["business_notes"] = [], None, {}
    state["dataset_frame"] = frame.copy(deep=True)
    state["dataset_id"], state["dataset_label"], state["dataset_kind"] = identity, label, kind
    state["nav_page"] = "对话分析"
    state["dataset_notice"] = "数据已切换，旧对话、筛选结果和经营补充已清空；模型请求计数保留。"
    return True


def apply_pending_dataset(state, example_frame):
    pending = state.pop("_pending_dataset", None)
    if not pending:
        return False
    if pending["kind"] == "example":
        return activate_dataset(state, example_frame, "example:v1", "Walmart 历史销售样本", "example")
    return activate_dataset(state, pending["frame"], pending["identity"], pending["label"], "upload")


def make_template():
    """教学用虚构销售记录，方便直接上传验证；金额单位由导入时指定。"""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(REQUIRED)
    for week, date in enumerate(pd.date_range("2025-01-03", periods=9, freq="7D")):
        writer.writerow([101, date.date().isoformat(), 10000 + week * 200 if week < 8 else 9000, int(week == 3)])
        writer.writerow([202, date.date().isoformat(), 20000 + week * 100, int(week == 3)])
    return buffer.getvalue().encode("utf-8-sig")


TEMPLATE_BYTES = make_template()
