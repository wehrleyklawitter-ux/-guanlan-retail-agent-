"""固定工具演练：不调用模型，用于核对完整例子的计算。"""
from analysis import Query, report, rank_sales_declines, get_sales_trend
from agent import Reply, compact_evidence

DEMO_QUESTION = "帮我找出最后一周销售下降最多的5家门店，再看看它们之前8周的表现。"


def run_example(df):
    ranking_args = {"stores": [], "end_date": "", "top_n": 5, "metric": "percent"}
    ranking = rank_sales_declines(df, **ranking_args)
    results = [ranking]
    steps = [{"tool": "rank_sales_declines", "arguments": ranking_args, "evidence": compact_evidence(ranking)}]
    if ranking["selected_stores"]:
        trend_args = {"stores": ranking["selected_stores"], "end_date": ranking["actual_end"], "weeks": 8}
        trend = get_sales_trend(df, **trend_args)
        results.append(trend)
        steps.append({"tool": "get_sales_trend", "arguments": trend_args, "evidence": compact_evidence(trend)})
    last = results[-1]
    context = Query(**{**last["query"], "stores": tuple(ranking["selected_stores"] or last["query"]["stores"])})
    return Reply("\n\n---\n\n".join(report(item) for item in results), last, context,
                 "固定工具演练（无模型）：" + " → ".join(step["tool"] for step in steps),
                 results=results, steps=steps)
