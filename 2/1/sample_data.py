"""本机优先使用已有课程数据；云端使用随项目生成的虚构样本。"""
from pathlib import Path

from analysis import DATA_PATH, load_data
from data_import import parse_sales_csv

DEMO_PATH = Path(__file__).parent / "data" / "Retail_Demo.csv"
DEMO_NOTICE = "虚构演示数据：销售记录与节假日标记均为项目生成，仅用于体验分析流程，不代表真实商家或经营成效。"


def load_default_data(*, public_demo=False):
    if not public_demo and DATA_PATH.exists():
        return load_data(DATA_PATH)
    frame, _ = parse_sales_csv(DEMO_PATH.read_bytes(), "Retail_Demo.csv", "元")
    frame.attrs.update(source_kind="synthetic", label="虚构零售演示数据", source_notice=DEMO_NOTICE)
    return frame
