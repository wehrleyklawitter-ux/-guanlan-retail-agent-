"""云端部署入口，兼容仓库内已有的1文件夹。"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "1" / "cloud_app.py"), run_name="__main__")
