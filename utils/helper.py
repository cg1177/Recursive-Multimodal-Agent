import os
from PIL import Image
import cv2
import numpy as np
import base64
import io

def timesec_hms(value, out=None):
    """
    双向/强制转换：
    - float 秒 <-> "HH:MM:SS" / "HH:MM:SS.mmm"

    Args:
        value: float/int 秒数 或 "HH:MM:SS(.mmm)" 字符串
        out:
            None        -> 自动互转（按输入类型）
            "float"     -> 强制输出秒（float）
            "hms"       -> 强制输出时间字符串

    Examples:
        timesec_hms(3661.5)              -> "01:01:01.500"
        timesec_hms("01:01:01.500")      -> 3661.5
        timesec_hms("01:01:01", out="hms") -> "01:01:01"
        timesec_hms(3661.5, out="float") -> 3661.5
    """

    def sec_to_hms(sec: float) -> str:
        sec = float(sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        if s.is_integer():
            return f"{h:02d}:{m:02d}:{int(s):02d}"
        else:
            return f"{h:02d}:{m:02d}:{s:06.3f}"

    def hms_to_sec(hms: str) -> float:
        parts = hms.split(":")
        if len(parts) != 3:
            raise ValueError("时间字符串格式必须为 HH:MM:SS 或 HH:MM:SS.mmm")
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)

    # ---------- 自动模式 ----------
    if out is None:
        if isinstance(value, (int, float)):
            return sec_to_hms(value)
        elif isinstance(value, str):
            return hms_to_sec(value)
        else:
            raise TypeError("输入必须为 float 秒数或 HH:MM:SS 字符串")

    # ---------- 强制输出模式 ----------
    if out == "float":
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return hms_to_sec(value)
        else:
            raise TypeError("输入必须为 float 秒数或 HH:MM:SS 字符串")

    if out == "hms":
        if isinstance(value, (int, float)):
            return sec_to_hms(value)
        elif isinstance(value, str):
            # 先转秒再规范化格式（防止奇怪输入）
            return sec_to_hms(hms_to_sec(value))
        else:
            raise TypeError("输入必须为 float 秒数或 HH:MM:SS 字符串")

    raise ValueError("out 只能为 None, 'float', 或 'hms'")

def image_to_base64(img, size=(256, 256), quality=95):
    """将图像转换为Base64编码"""
    if isinstance(img, Image.Image):
        pil_img = img.convert("RGB")
    elif isinstance(img, np.ndarray):
        # 假设输入是 BGR (OpenCV 默认)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("ndarray must be HxWx3")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(img)
        pil_img = Image.open(img).convert("RGB")
    else:
        raise TypeError(f"Unsupported type: {type(img)}")

    pil_img = pil_img.resize(size, Image.BICUBIC)
    
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"