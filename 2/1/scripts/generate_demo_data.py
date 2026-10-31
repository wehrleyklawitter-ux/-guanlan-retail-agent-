"""生成独立虚构教学样本，不读取课程数据或真实商家记录。"""
import csv
from datetime import date, timedelta
from math import sin
from pathlib import Path


def generate(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    final_ratios = {1: .72, 6: .78, 10: .84, 16: .90, 20: .95}
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["Store", "Date", "Weekly_Sales", "Holiday_Flag"])
        for week in range(26):
            sales_date = date(2025, 1, 3) + timedelta(weeks=week)
            holiday = int(week in {4, 12, 18})  # 演示标记，不是真实节庆日历。
            for store in range(1, 21):
                base = 60000 + store * 4200
                sales = round(base * (1 + .002 * week + .025 * sin(week + store)) * (1.12 if holiday else 1), 2)
                if week == 25:
                    ratio = final_ratios.get(store, 1.01 if store % 2 else .995)
                    sales = round(previous[store] * ratio, 2)
                writer.writerow([store, sales_date.isoformat(), sales, holiday])
                previous[store] = sales


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "data" / "Retail_Demo.csv")
