"""按诊断类别关联本地知识快照，不进行自由引用或根因推断。"""
from copy import deepcopy
import json
from pathlib import Path

KNOWLEDGE_PATH = Path(__file__).parent / "data" / "retail_knowledge.json"
HYPOTHESIS_KNOWLEDGE = {
    "high_base": "K02", "holiday": "K05", "promotion_ended": "K06",
    "stockout": "K07", "reduced_hours": "K08",
}


def load_knowledge():
    return json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))


def associate_knowledge(cases):
    """标识跟随诊断结果，完整条目在结果根部去重保存，便于复现和下载。"""
    catalog = load_knowledge()
    by_id = {card["id"]: card for card in catalog["cards"]}
    used = set()
    for case in cases:
        ids = ["K01", "K02"]
        if case["peers"]["count"]:
            ids.append("K03")
        if case["peers"]["excluded_stores"] or case["history"]["records"] != case["history"]["requested_weeks"]:
            ids.append("K04")
        for hypothesis in case["hypotheses"]:
            card_id = HYPOTHESIS_KNOWLEDGE.get(hypothesis["id"])
            hypothesis["knowledge_ids"] = [card_id] if card_id in by_id else []
            if card_id in by_id:
                ids.append(card_id)
        case["knowledge_ids"] = list(dict.fromkeys(ids))
        used.update(ids)
    return {
        "knowledge_version": catalog["version"],
        "knowledge_reviewed_on": catalog["reviewed_on"],
        "knowledge_cards": [deepcopy(card) for card in catalog["cards"] if card["id"] in used],
    }


def knowledge_report(cards, reviewed_on):
    lines = ["### 经营知识与来源", f"资料查阅日期：{reviewed_on}。一般知识用于安排核查，不能证明本店发生了相应事件。"]
    for card in cards:
        source = f"[{card['source_title']}]({card['source_url']})" if card["source_url"] else card["source_title"]
        lines += [f"**{card['id']} · {card['title']}｜{card['kind']}**",
                  card["principle"], "项目应用：" + card["application"],
                  "需核对：" + "、".join(card["records"]) + "。",
                  "适用边界：" + card["boundary"],
                  f"来源：{card['source_org']} · {source}；定位：{card['source_section']}。"]
    return "\n\n".join(lines)
