"""历史销售下降排查：计算事实、提出待核实假设，保留信息来源。"""
from __future__ import annotations

import math
import re

import pandas as pd

from analysis import AnalysisError, Query, data_info, resolve_week, validate_query, sales_unit, unit_label
from knowledge import associate_knowledge, knowledge_report

DIAGNOSIS_QUESTION = "请排查16号店最后一周销售下降的可能原因，给出证据和下一步核查建议。"
BUSINESS_FIELDS = {
    "promotion_ended": ("促销结束", "上周有活动、本周已经结束吗？请核对活动日期、参与商品及折扣。"),
    "stockout": ("缺货", "本周是否缺货？请提供缺货商品、发生时段及缺货天数。"),
    "reduced_hours": ("营业受限", "本周是否缩短营业、停业或施工？请提供实际营业时长。"),
}
BUSINESS_STATES = {"unknown": "尚未提供", "yes": "用户填报有此情况", "no": "用户填报无此情况"}


def note_key(store, end_date):
    return f"{store}|{end_date}"


def validate_business_note(note):
    if note is None:
        return {field: "unknown" for field in BUSINESS_FIELDS}
    if not isinstance(note, dict) or set(note) - set(BUSINESS_FIELDS):
        raise AnalysisError("经营补充信息格式无效。")
    for state in note.values():
        if not isinstance(state, str) or state not in BUSINESS_STATES:
            raise AnalysisError("经营补充信息应为尚未提供、有或无。")
    return {field: note.get(field, "unknown") for field in BUSINESS_FIELDS}


def percent_change(before, after):
    return 100 * (after / before - 1) if before > 0 else None


def diagnose_sales_decline(df, stores, end_date="", history_weeks=8, business_notes=None):
    """基线只使用目标周以前的记录；其他门店只是参照群体，不是因果对照组。"""
    if not isinstance(stores, list) or not 1 <= len(stores) <= 5:
        raise AnalysisError("每次请选择1至5家门店进行下降排查。")
    if type(history_weeks) is not int or not 4 <= history_weeks <= 26:
        raise AnalysisError("历史参照窗口应为4至26周。")
    if business_notes is not None and not isinstance(business_notes, dict):
        raise AnalysisError("经营补充记录格式无效。")
    end = resolve_week(df, end_date)
    previous_date = end - pd.Timedelta(days=7)
    start = end - pd.Timedelta(weeks=history_weeks)
    query = Query(tuple(stores), start.date().isoformat(), end.date().isoformat())
    validate_query(df, query)
    cases = []
    for store in stores:
        rows = df[df.Store.eq(store)].sort_values("Date")
        pair = rows[rows.Date.isin([previous_date, end])].set_index("Date")
        if len(pair) != 2:
            raise AnalysisError(f"{store}号店缺少目标周或相隔7天的上周记录，无法排查周环比下降。")
        previous, current = pair.loc[previous_date], pair.loc[end]
        before, after = float(previous.Weekly_Sales), float(current.Weekly_Sales)
        change = percent_change(before, after)
        if change is None:
            raise AnalysisError(f"{store}号店上周销售额非正，无法计算有效环比。")
        historical = rows[rows.Date.ge(start) & rows.Date.lt(end)].copy()
        series = historical.set_index("Date").Weekly_Sales
        changes = [percent_change(float(series.loc[date - pd.Timedelta(days=7)]), float(value))
                   for date, value in series.items() if date - pd.Timedelta(days=7) in series.index]
        changes = [value for value in changes if value is not None]
        mean = float(series.mean()) if len(series) else None
        earlier = series[series.index < previous_date]
        earlier_median = float(earlier.median()) if len(earlier) else None
        recent_low = len(changes) >= 4 and change < min(changes)
        high_base_pattern = (change < 0 and earlier_median is not None and before > earlier_median and after >= earlier_median)
        # 其他门店同期环比，排除目标店、缺周和非正基数，避免把大店体量当作普遍变化。
        others = df[~df.Store.eq(store) & df.Date.isin([previous_date, end])]
        pivot = others.pivot(index="Store", columns="Date", values="Weekly_Sales")
        peer_changes = []
        excluded_peers = []
        for peer in sorted(s for s in df.Store.unique() if s != store):
            if peer not in pivot.index or any(d not in pivot.columns or pd.isna(pivot.loc[peer, d]) for d in [previous_date, end]):
                excluded_peers.append(int(peer))
                continue
            value = percent_change(float(pivot.loc[peer, previous_date]), float(pivot.loc[peer, end]))
            if value is None:
                excluded_peers.append(int(peer))
            else:
                peer_changes.append({"store": int(peer), "change_pct": value})
        peer_values = [r["change_pct"] for r in peer_changes]
        peers = {"count": len(peer_values), "declining_count": sum(v < 0 for v in peer_values),
                 "median_change_pct": float(pd.Series(peer_values).median()) if peer_values else None,
                 "excluded_stores": excluded_peers}
        factors = []
        for field, label in [("Temperature", "温度"), ("Fuel_Price", "燃油价格"), ("CPI", "CPI"), ("Unemployment", "失业率")]:
            if field not in rows.columns:
                continue
            a, b = previous[field], current[field]
            if pd.notna(a) and pd.notna(b) and math.isfinite(float(a)) and math.isfinite(float(b)):
                factors.append({"field": field, "label": label, "previous": float(a), "current": float(b),
                                "delta": float(b) - float(a), "unit": "原始数据单位"})
        holiday = {"previous": int(previous.Holiday_Flag), "current": int(current.Holiday_Flag)}
        note = validate_business_note((business_notes or {}).get(note_key(store, query.end)))
        hypotheses = []
        if change >= 0:
            interpretation = ["目标周销售没有环比下降，本次按销售变化复盘，不生成已发生下降的结论。"]
        else:
            interpretation = []
            if recent_low:
                interpretation.append("本次环比低于所选历史窗口内所有有效周环比，值得优先复盘；这不是已确认的经营异常。")
            elif len(changes) >= 4:
                interpretation.append("此前窗口也出现过不高于本次环比的变化，仅凭本次降幅不足以确认异常。")
            else:
                interpretation.append("可用历史环比样本不足，暂不判断本次下降是否超出以往波动。")
            if peer_values:
                widespread = peers["declining_count"] > peers["count"] / 2
                interpretation.append("同期过半其他门店也下降，可先核查共同日历和经营背景，再看本店差异。" if widespread else
                                      "同期下降门店未超过其他门店的一半，可优先补查本店经营记录；这不能排除共同因素。")
            hypotheses.append({"id": "high_base", "name": "上周高基数后回落",
                               "status": "有形态线索，待核实" if high_base_pattern else "上周偏高且本周偏低，需并查" if earlier_median is not None and before > earlier_median else "当前形态依据不足",
                               "evidence": ("上周高于更早历史中位数，本周虽下降但仍不低于该中位数。" if high_base_pattern else
                                            "上周高于更早历史中位数，本周已低于该中位数；高基数回落不足以覆盖全部走势，需并查本周经营。" if earlier_median is not None and before > earlier_median else
                                            "上周并未高于更早历史中位数，或更早样本不足，当前缺少高基数回落的形态依据。"),
                               "next_check": "核对上周是否存在活动、集中采购或其他一次性高销售记录。"})
            hypotheses.append({"id": "holiday", "name": "节假日背景变化",
                               "status": "标记发生变化，待核实" if holiday["previous"] != holiday["current"] else "标记相同，当前切换依据不足",
                               "evidence": f"上周Holiday_Flag={holiday['previous']}，本周={holiday['current']}；此字段不覆盖所有当地日历事件。",
                               "next_check": "核对实际节庆、活动日历和当地事件；标记变化不能单独解释下降。"})
        if change < 0:
            for field, (name, question) in BUSINESS_FIELDS.items():
                state = note[field]
                hypotheses.append({"id": field, "name": name,
                    "status": {"unknown": "待补充记录", "yes": "用户填报支持，优先核查", "no": "用户填报无此情况，暂降核查顺序"}[state],
                    "evidence": "原始CSV没有这项记录。" if state == "unknown" else f"{BUSINESS_STATES[state]}；来源为用户手动补充，未独立核验。",
                    "next_check": question})
            hypotheses.sort(key=lambda item: 0 if "优先核查" in item["status"] else 2 if "暂降" in item["status"] else 1)
        warnings = ["参照群体为其他有完整同期记录的门店，未进行相似门店匹配或因果对照。",
                    "环境字段仅展示同期变化，不能据此归因；温度单位未核实，保留原始值。"]
        if len(historical) != history_weeks:
            warnings.append(f"历史窗口请求{history_weeks}周，实际只有{len(historical)}条记录；不跨缺失周计算环比。")
        chart_rows = rows[rows.Date.between(start, end)]
        cases.append({"store": store, "previous_date": previous_date.date().isoformat(), "end_date": query.end,
            "previous_sales": before, "current_sales": after, "change_pct": change, "change_amount": after - before,
            "history": {"requested_weeks": history_weeks, "records": len(historical), "mean": mean,
                        "current_vs_mean_pct": percent_change(mean, after) if mean is not None else None,
                        "earlier_median": earlier_median, "change_samples": len(changes),
                        "minimum_past_change_pct": min(changes) if changes else None,
                        "below_all_past_changes": recent_low, "high_base_pattern": high_base_pattern},
            "peers": peers, "holiday": holiday, "factors": factors, "business_note": note,
            "business_note_source": "用户手动填报，未独立核验" if any(v != "unknown" for v in note.values()) else "尚无用户补充",
            "interpretation": interpretation, "hypotheses": hypotheses, "warnings": warnings,
            "trend": [{"date": r.Date.date().isoformat(), "store": store, "sales": float(r.Weekly_Sales)} for r in chart_rows.itertuples()],
            "peer_changes": peer_changes})
    knowledge = associate_knowledge(cases)
    return {"type": "sales_diagnosis", "query": query.to_dict(), "actual_start": query.start, "actual_end": query.end,
            "cases": cases, "method": "描述性排查：目标周与上周、过去窗口、同期其他门店及业务补充对照。历史基线不使用目标周或未来数据。",
            "unit": sales_unit(df), "unit_name": sales_unit(df), **knowledge}


def diagnosis_report(result):
    unit = unit_label(result)
    lines = ["### 销售下降原因排查", "以下区分计算事实、待核实假设和用户补充，所有原因假设均需经营记录验证。"]
    for case in result["cases"]:
        history, peers = case["history"], case["peers"]
        lines += [f"#### {case['store']}号店 · {case['end_date']}",
                  f"**已计算的事实**：上周（{case['previous_date']}）{case['previous_sales']:,.2f} → 本周{case['current_sales']:,.2f}{unit}；周环比 **{case['change_pct']:+.2f}%**。"]
        if history["mean"] is not None:
            comparison = f"；本周比该均值 **{history['current_vs_mean_pct']:+.2f}%**" if history["current_vs_mean_pct"] is not None else "；历史均值非正，比例不适用"
            lines.append(f"- 此前{history['records']}周平均销售额{history['mean']:,.2f}{unit}{comparison}。基线不含本周。")
        if peers["count"]:
            lines.append(f"- 其他{peers['count']}家有效参照门店中，{peers['declining_count']}家下降；门店环比中位数 **{peers['median_change_pct']:+.2f}%**。")
        if history["change_samples"]:
            lines.append(f"- 此前窗口有{history['change_samples']}个有效环比，最低为 **{history['minimum_past_change_pct']:+.2f}%**。")
        lines += ["**线索解读**：" + " ".join(case["interpretation"])]
        if case["hypotheses"]:
            lines.append("**可能原因与下一步核查**：")
            for hypothesis in case["hypotheses"]:
                ids = "、".join(hypothesis.get("knowledge_ids", []))
                citation = f" 知识依据：{ids}。" if ids else ""
                lines.append(f"- **{hypothesis['name']}｜{hypothesis['status']}**：{hypothesis['evidence']} 下一步：{hypothesis['next_check']}{citation}")
        if case["factors"]:
            lines.append("**环境背景（同期变化，未证明因果）**：" + "；".join(
                f"{f['label']} {f['previous']:.3f}→{f['current']:.3f}（原始单位）" for f in case["factors"]) + "。")
        lines += ["补充记录来源：" + case["business_note_source"] + "。",
                  "口径提示：" + " ".join(case["warnings"])]
    lines.append("本结果用于历史复盘和安排核查，不是根因确认，也没有估算任何原因的销售损失贡献。")
    if result.get("knowledge_cards"):
        lines.append(knowledge_report(result["knowledge_cards"], result["knowledge_reviewed_on"]))
    return "\n\n".join(lines)


def local_diagnosis_request(text, df, previous=None, business_notes=None):
    """有限规则入口，显式标为本地练习；含糊年份/日期先澄清。"""
    if not re.search(r"原因|为何|为什么|诊断|排查", text):
        return None
    if re.search(r"今天|今年|本周|上周|昨天|现在|当前", text):
        raise AnalysisError("这里仅有历史样本，请指定销售周日期，或说‘最后一周’。")
    ids = re.findall(r"(\d+)号(?:门)?店|门店\s*(\d+)", text)
    stores = list(dict.fromkeys(int(a or b) for a, b in ids)) or (list(previous.stores) if previous else [])
    if not stores:
        raise AnalysisError("请说明要排查哪家门店，例如‘排查16号店最后一周下降的原因’。")
    time_text = re.sub(r"\d+号(?:门)?店|门店\s*\d+", "", text)
    dates = re.findall(r"(?:19|20|21)\d{2}-\d{2}-\d{2}", time_text)
    if len(dates) > 1 or (re.search(r"(?:19|20|21)\d{2}", time_text) and not dates):
        raise AnalysisError("原因排查需要一个具体销售周，请使用YYYY-MM-DD格式。")
    end_date = dates[0] if dates else "" if re.search(r"最后|末周", text) or not previous else previous.end
    weeks = re.search(r"(?:前|过去|历史)\s*(\d+)周", text)
    return diagnose_sales_decline(df, stores, end_date, int(weeks.group(1)) if weeks else 8, business_notes)
