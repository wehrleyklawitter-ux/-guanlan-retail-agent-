"""云端入口识别与跨会话请求计数；不保存用户数据或密钥。"""
from datetime import datetime, timezone
import os
from threading import Lock

from analysis import AnalysisError


def cloud_mode():
    return os.environ.get("GUANLAN_CLOUD") == "1"


class DailyRequestBudget:
    """单个运行实例内共享的UTC日请求上限；实例重建后重置。"""

    def __init__(self, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("请求上限须为1至1000的整数。")
        self.limit = limit
        self._day = ""
        self._requests = 0
        self._lock = Lock()

    def reserve(self, day=None):
        day = day or datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            if day != self._day:
                self._day, self._requests = day, 0
            if self._requests >= self.limit:
                raise AnalysisError("今日智能体试用请求已达上限，请稍后再试；筛选分析和下降诊断仍可使用。")
            self._requests += 1  # 失败请求也占用额度，不因网络重试无限增加支出。
            return self._requests
