"""可独立运行的销售分析工具。所有金额均由这里计算。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import calendar
import math
import re

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "Walmart_Sales.csv"
HOLIDAYS = {"all": "全部周", "holiday": "节假日周", "ordinary": "普通周"}
MAX_STORES = 100


def sales_unit(df):
    return df.attrs.get("sales_unit", "美元")


def unit_label(result):
    unit = result.get("unit_name", "美元")
    return "原始金额单位" if unit == "未注明" else unit


class AnalysisError(ValueError):
    """可直接向用户解释的输入问题。"""


@dataclass(frozen=True)
class Query:
    stores: tuple[int, ...]
    start: str
    end: str
    holiday: str = "all"
    kind: str = "sales"

    def to_dict(self):
        return asdict(self)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Store", "Date", "Weekly_Sales", "Holiday_Flag"}
    if not required.issubset(df.columns):
        raise AnalysisError("数据缺少门店、日期、销售额或节假日字段。")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="raise")
    if df[list(required)].isna().any().any():
        raise AnalysisError("关键字段存在缺失值，请先检查原始数据。")
    if df.duplicated(["Store", "Date"]).any():
        raise AnalysisError("同一门店同一周存在重复记录，已停止汇总。")
    if not df["Weekly_Sales"].map(lambda x: isinstance(x, (int, float)) and math.isfinite(x)).all():
        raise AnalysisError("销售额存在无效数字。")
    df = df.sort_values(["Date", "Store"]).reset_index(drop=True)
    df.attrs.update(sales_unit="美元", source_kind="example", label="Walmart 历史销售样本")
    return df


def data_info(df):
    return {"rows": len(df), "stores": sorted(int(s) for s in df.Store.unique()),
            "weeks": int(df.Date.nunique()), "start": df.Date.min().date().isoformat(),
            "end": df.Date.max().date().isoformat()}


def validate_query(df, query: Query) -> Query:
    if not query.stores or any(type(s) is not int for s in query.stores):
        raise AnalysisError("请提供有效的门店编号，例如1号店。")
    if len(query.stores) != len(set(query.stores)):
        raise AnalysisError("门店编号重复，请选择不同门店。")
    missing = sorted(set(query.stores) - set(df.Store.unique()))
    if missing:
        available = sorted(int(s) for s in df.Store.unique())
        raise AnalysisError(f"不存在门店：{', '.join(map(str, missing))}。当前可用编号：{', '.join(map(str, available[:20]))}" + ("等。" if len(available) > 20 else "。"))
    if not isinstance(query.holiday, str) or not isinstance(query.kind, str) or query.holiday not in HOLIDAYS or query.kind not in {"sales", "comparison", "holidays"}:
        raise AnalysisError("请选择销售查询、门店比较或节假日比较。")
    if query.kind == "comparison" and len(query.stores) < 2:
        raise AnalysisError("比较门店需要至少两家门店，请再指定一家。")
    try:
        if not all(isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in (query.start, query.end)):
            raise ValueError()
        start, end = pd.Timestamp(query.start), pd.Timestamp(query.end)
    except (ValueError, TypeError):
        raise AnalysisError("日期无效，请使用有效的年月日。") from None
    if start > end:
        raise AnalysisError("开始日期晚于结束日期，请调整时间范围。")
    info = data_info(df)
    if end < df.Date.min() or start > df.Date.max():
        raise AnalysisError(f"所选时间没有数据。数据范围为{info['start']}至{info['end']}，不能查询当前或未来销售额。")
    return query


def analyze(df: pd.DataFrame, query: Query) -> dict:
    validate_query(df, query)
    selected = df[df.Store.isin(query.stores) & df.Date.between(query.start, query.end)].copy()
    if query.kind != "holidays" and query.holiday != "all":
        selected = selected[selected.Holiday_Flag.eq(1 if query.holiday == "holiday" else 0)]
    if selected.empty:
        raise AnalysisError("这个时间和筛选条件下没有记录，请扩大时间范围。")
    missing_stores = sorted(set(query.stores) - set(selected.Store.unique()))
    if missing_stores:
        raise AnalysisError(f"筛选后门店{missing_stores}没有记录，无法进行同口径比较。")
    if query.kind == "holidays":
        query = Query(query.stores, query.start, query.end, "all", "holidays")
    summary = []
    for store, rows in selected.groupby("Store"):
        summary.append({"store": int(store), "weeks": len(rows),
                        "total": round(float(rows.Weekly_Sales.sum()), 2),
                        "average": round(float(rows.Weekly_Sales.mean()), 2),
                        "minimum": round(float(rows.Weekly_Sales.min()), 2),
                        "maximum": round(float(rows.Weekly_Sales.max()), 2)})
    summary.sort(key=lambda row: row["total"], reverse=True)
    holiday_rows = []
    holiday_by_store = []
    for (store, flag), rows in selected.groupby(["Store", "Holiday_Flag"]):
        holiday_by_store.append({"store": int(store), "flag": int(flag), "weeks": len(rows),
                                 "average": round(float(rows.Weekly_Sales.mean()), 2)})
    for flag, rows in selected.groupby("Holiday_Flag"):
        holiday_rows.append({"flag": int(flag), "label": "节假日周" if flag else "普通周",
                             "records": len(rows), "average": round(float(rows.Weekly_Sales.mean()), 2)})
    info = data_info(df)
    actual_start, actual_end = selected.Date.min().date().isoformat(), selected.Date.max().date().isoformat()
    warnings = []
    if query.start < info["start"] or query.end > info["end"]:
        warnings.append("所选范围包含未覆盖日期，结果仅使用实际存在的记录。")
    if len({row["weeks"] for row in summary}) > 1:
        warnings.append("各门店记录周数不同，比较总额时请同时参考平均周销售额。")
    if query.kind == "holidays" and len(holiday_rows) < 2:
        warnings.append("所选范围只包含一种周类型，不能计算节假日与普通周差异。")
    difference = None
    if len(holiday_rows) == 2:
        means = {int(k): float(v) for k, v in selected.groupby("Holiday_Flag").Weekly_Sales.mean().items()}
        if means[0] != 0:
            difference = 100 * (means[1] / means[0] - 1)
    trend = [{"date": row.Date.date().isoformat(), "store": int(row.Store),
              "sales": float(row.Weekly_Sales)} for row in selected.itertuples()]
    return {"query": query.to_dict(), "actual_start": actual_start, "actual_end": actual_end,
            "records": len(selected), "observed_weeks": int(selected.Date.nunique()),
            "total": round(float(selected.Weekly_Sales.sum()), 2), "summary": summary,
            "trend": trend, "holiday_summary": holiday_rows, "holiday_by_store": holiday_by_store,
            "holiday_difference_pct": difference, "warnings": warnings,
            "unit": sales_unit(df), "unit_name": sales_unit(df), "available_store_count": int(df.Store.nunique())}


def report(result: dict) -> str:
    if result.get("type") == "sales_diagnosis":
        from diagnosis import diagnosis_report
        return diagnosis_report(result)
    if result.get("type") == "decline_ranking":
        return decline_report(result)
    q = result["query"]
    unit = unit_label(result)
    count = result.get("available_store_count", 45)
    stores = f"全部{count}家门店" if len(q["stores"]) == count else "、".join(f"{s}号店" for s in q["stores"])
    lines = [f"**{stores} · {HOLIDAYS[q['holiday']]}**",
             f"查询范围：{q['start']} 至 {q['end']}。",
             f"实际记录日期：{result['actual_start']} 至 {result['actual_end']}，共{result['records']}条门店周记录。",
             f"销售额合计 **{result['total']:,.2f}** {unit}。"]
    for row in result["summary"][:10]:
        lines.append(f"- {row['store']}号店：合计 **{row['total']:,.2f}** {unit}；平均每周 **{row['average']:,.2f}** {unit}；{row['weeks']}周记录。")
    if len(result["summary"]) > 10:
        lines.append("以上列出累计销售额前10家门店，完整结果见计算表。")
    if q["kind"] == "comparison":
        first, second = result["summary"][:2]
        lines.append(f"在所选范围内，{first['store']}号店累计销售额比{second['store']}号店高 **{first['total'] - second['total']:,.2f}** {unit}。")
    if q["kind"] == "holidays":
        for row in result["holiday_summary"]:
            lines.append(f"- {row['label']}：平均每条门店周记录 **{row['average']:,.2f}** {unit}，样本数{row['records']}。")
        pct = result["holiday_difference_pct"]
        if pct is not None:
            lines.append(f"节假日周平均值比普通周{'高' if pct >= 0 else '低'} **{abs(pct):.2f}%**。")
        lines.append("该差异是历史样本的描述性比较，不能据此认定节假日导致销售变化。")
    lines.extend(f"提示：{w}" for w in result["warnings"])
    lines.append(f"口径：一条记录代表一家门店某一周；金额单位：{unit}；只分析当前数据实际覆盖的记录。")
    return "\n\n".join(lines)


def resolve_week(df, end_date=""):
    """周环比仅比较相隔7天的两条周记录，不跨缺失周。"""
    if end_date == "":
        return df.Date.max()
    if not isinstance(end_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date):
        raise AnalysisError("请使用YYYY-MM-DD格式的销售周日期。")
    try:
        end = pd.Timestamp(end_date)
    except ValueError:
        raise AnalysisError("销售周日期无效。") from None
    if end not in set(df.Date):
        raise AnalysisError("所选日期没有销售周记录，请使用数据中的周日期；留空表示数据最后一周。")
    return end


def rank_sales_declines(df, stores, end_date="", top_n=5, metric="percent"):
    if not isinstance(stores, list) or len(stores) > MAX_STORES:
        raise AnalysisError("门店参数应为门店编号列表。")
    if type(top_n) is not int or not 1 <= top_n <= 10:
        raise AnalysisError("排名数量应为1至10家。")
    if not isinstance(metric, str) or metric not in {"percent", "amount"}:
        raise AnalysisError("下降排名支持降幅百分比或下降金额。")
    end = resolve_week(df, end_date)
    start = end - pd.Timedelta(days=7)
    requested = stores or data_info(df)["stores"]
    query = Query(tuple(requested), start.date().isoformat(), end.date().isoformat())
    validate_query(df, query)
    current = df[df.Date.eq(end) & df.Store.isin(requested)].set_index("Store")
    previous = df[df.Date.eq(start) & df.Store.isin(requested)].set_index("Store")
    common = sorted(set(current.index) & set(previous.index))
    ranking, excluded = [], []
    for store in requested:
        if store not in common:
            excluded.append({"store": store, "reason": "缺少本周或相隔7天的上周记录"})
            continue
        before, after = float(previous.loc[store, "Weekly_Sales"]), float(current.loc[store, "Weekly_Sales"])
        if before <= 0 and metric == "percent":
            excluded.append({"store": store, "reason": "上周销售额非正，无法计算降幅百分比"})
            continue
        drop = before - after
        if drop > 0:
            ranking.append({"store": int(store), "previous_sales": before, "current_sales": after,
                            "drop_amount": round(drop, 2), "drop_pct": 100 * drop / before if before > 0 else None,
                            "previous_holiday": int(previous.loc[store, "Holiday_Flag"]),
                            "current_holiday": int(current.loc[store, "Holiday_Flag"])})
    ranking.sort(key=lambda r: (-(r["drop_pct"] if metric == "percent" else r["drop_amount"]), r["store"]))
    if not common:
        raise AnalysisError("没有门店同时具备本周和相隔7天的上周记录，无法计算周环比。")
    warnings = ["环比下降用于筛选需要复盘的门店，不能单独证明经营异常或确定下降原因。"]
    if excluded:
        warnings.append(f"有{len(excluded)}家门店未参与排名，详见排除记录。")
    rows = ranking[:top_n]
    return {"type": "decline_ranking", "query": query.to_dict(), "actual_start": query.start,
            "actual_end": query.end, "ranking": rows, "metric": metric, "top_n": top_n,
            "compared_stores": len(common) - sum(r["reason"].startswith("上周") for r in excluded),
            "declining_stores": len(ranking), "excluded": excluded, "warnings": warnings,
            "selected_stores": [r["store"] for r in rows], "unit": sales_unit(df), "unit_name": sales_unit(df)}


def get_sales_trend(df, stores, end_date="", weeks=8):
    if not isinstance(stores, list) or not stores or len(stores) > 10:
        raise AnalysisError("趋势分析请选择1至10家门店。")
    if type(weeks) is not int or not 2 <= weeks <= 52:
        raise AnalysisError("趋势窗口应为2至52周。")
    end = resolve_week(df, end_date)
    start = end - pd.Timedelta(weeks=weeks - 1)
    result = analyze(df, Query(tuple(stores), start.date().isoformat(), end.date().isoformat()))
    result["type"] = "sales_trend"
    result["requested_weeks"] = weeks
    if any(r["weeks"] != weeks for r in result["summary"]):
        result["warnings"].append("部分门店未覆盖完整趋势窗口，请查看各门店实际记录周数。")
    return result


def decline_report(result):
    unit = unit_label(result)
    metric = "周环比降幅百分比" if result["metric"] == "percent" else "周环比下降金额"
    lines = [f"### 销售下降排名：按{metric}",
             f"比较销售周：{result['actual_start']} → {result['actual_end']}（相隔7天）。",
             f"有效比较{result['compared_stores']}家门店，其中{result['declining_stores']}家销售下降；展示前{len(result['ranking'])}家。"]
    for i, row in enumerate(result["ranking"], 1):
        pct = f"，降幅 **{row['drop_pct']:.2f}%**" if row["drop_pct"] is not None else "（百分比不可计算）"
        lines.append(f"{i}. **{row['store']}号店**：上周{row['previous_sales']:,.2f} → 本周{row['current_sales']:,.2f}{unit}；下降 **{row['drop_amount']:,.2f}** {unit}{pct}。")
    if not result["ranking"]:
        lines.append("有效比较门店中没有销售下降的门店。")
    lines.extend(f"提示：{w}" for w in result["warnings"])
    lines.append("默认以降幅百分比排名；也可要求按下降金额排名。‘最后一周’指数据最后一周，历史数据不能代表当前经营情况。")
    return "\n\n".join(lines)


def parse_local(text: str, df: pd.DataFrame, previous: Query | None = None) -> Query:
    """离线练习用规则解析器，不冒充大模型。"""
    text = text.strip()
    if not text or len(text) > 2000:
        raise AnalysisError("请输入不超过2000字的销售分析问题。")
    if any(w in text for w in ["利润", "库存", "补货", "预测", "商品", "顾客", "客户", "油价", "温度", "失业率", "CPI", "促销"]):
        raise AnalysisError("第一版支持门店销售、趋势和节假日比较。当前工具无法回答商品、库存、利润、预测或外部因素归因问题。")
    info = data_info(df)
    if re.search(r"今天|今年|明年|昨天|明天|上周|本周|本月|上个月|去年|现在|当前", text):
        raise AnalysisError(f"这里是历史数据，范围为{info['start']}至{info['end']}。请指定历史年份，或说“最近8周”（以数据末日为准）。")
    ids = re.findall(r"(?<![\d-])(\d{1,9})\s*号(?:门)?店", text)
    ids += re.findall(r"(?:门店|店铺|(?<!号)店)\s*(\d{1,9})(?!\d|年|[-/])", text)
    time_text = re.sub(r"(?<![\d-])\d{1,9}\s*号(?:门)?店|(?:门店|店铺|(?<!号)店)\s*\d{1,9}(?!\d|年|[-/])", "", text)
    stores = tuple(dict.fromkeys(int(s) for s in ids))
    all_stores = bool(re.search(r"全部|所有|各门店|各家|排名|哪.*店.*(?:最高|最好)", text))
    if all_stores:
        stores = tuple(info["stores"])
    elif not stores:
        if previous is not None:
            stores = previous.stores
        else:
            raise AnalysisError("你想分析哪家门店？例如“1号店2011年的销售表现”，或“全部门店销售排名”。")
    start, end = (previous.start, previous.end) if previous else (info["start"], info["end"])
    if "全部时间" in text or "所有年份" in text:
        start, end = info["start"], info["end"]
    else:
        recent = re.search(r"最近\s*(\d+)\s*(?:周|星期)", text)
        dates = re.findall(r"((?:19|20|21)\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?", time_text)
        months = re.findall(r"((?:19|20|21)\d{2})年(\d{1,2})月", time_text)
        years = list(dict.fromkeys(re.findall(r"(?:19|20|21)\d{2}", time_text)))
        try:
            if recent:
                count = int(recent.group(1))
                if not 1 <= count <= info["weeks"]:
                    raise AnalysisError(f"最近周数应在1至{info['weeks']}之间。")
                end = info["end"]
                start = (pd.Timestamp(end) - pd.Timedelta(weeks=count - 1)).date().isoformat()
            elif "最近" in text:
                raise AnalysisError("请说明最近几周，例如“1号店最近8周”，参照数据最新日期。")
            elif dates:
                if len(dates) > 2:
                    raise AnalysisError("一次请选择一个起止日期范围。")
                parsed = [pd.Timestamp(year=int(y), month=int(m), day=int(d)).date().isoformat() for y, m, d in dates]
                start, end = parsed[0], parsed[-1]
            elif months:
                if len(months) > 1:
                    raise AnalysisError("一次请选择一个月份，或使用完整的起止日期。")
                year, month = map(int, months[0])
                start, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
            elif len(years) == 1:
                start, end = f"{years[0]}-01-01", f"{years[0]}-12-31"
            elif len(years) > 1:
                raise AnalysisError("一次请选择一个年份，或使用完整的起止日期。")
            elif re.search(r"\d+月|季度|半年|前\s*\d+周|\d+[-/]\d+", text):
                raise AnalysisError("请提供完整年份和时间，例如“2011年3月”或“2011-03-01至2011-04-30”。")
        except AnalysisError:
            raise
        except (ValueError, OverflowError):
            raise AnalysisError("日期无效，请检查月份、日期和范围。") from None
    holiday = previous.holiday if previous else "all"
    kind = "comparison" if len(stores) >= 2 and not all_stores else "sales"
    if re.search(r"节假日.*(?:普通|非节假日)|(?:普通|非节假日).*节假日|节假日对比|节假日比较", text):
        holiday, kind = "all", "holidays"
    elif re.search(r"普通周|非节假日|平常周|非假日", text):
        holiday = "ordinary"
    elif "节假日" in text or "假日周" in text:
        holiday = "holiday"
    elif re.search(r"全部周|所有周|不限节假日|取消.*筛选", text):
        holiday = "all"
    if re.search(r"比较|对比", text) and kind != "holidays" and len(stores) < 2:
        raise AnalysisError("你想比较哪两家门店？例如“比较1号店和20号店2011年的销售”。")
    if not ids and not all_stores and not re.search(r"(?:19|20|21)\d{2}|最近|普通周|节假日|所有周|全部周|全部时间|平均|趋势|合计|总额|取消", text):
        raise AnalysisError("请说明要查询的门店和时间，或输入“只看节假日”等后续筛选条件。")
    return validate_query(df, Query(stores, start, end, holiday, kind))
