import os
import time
import json
import base64
import mimetypes
from copy import deepcopy

import jwt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SENSENOVA_AK = os.getenv("SENSENOVA_AK")
SENSENOVA_SK = os.getenv("SENSENOVA_SK")
SENSENOVA_API_KEY = os.getenv("SENSENOVA_API_KEY")
MODEL_ID = os.getenv("SENSENOVA_OPENAI_MODEL_ID") or os.getenv("SENSENOVA_MODEL_ID") or "SenseNova-V6-5-Pro"
IMAGE_FILE = os.getenv("SENSENOVA_IMAGE_FILE", "测试图片.png")
PROMPT = os.getenv("SENSENOVA_IMAGE_PROMPT", "请描述这张图片。")
BASE_URL = "https://api.sensenova.cn/compatible-mode/v2"


def log_step(message):
    print(f"[*] {message}")


def dump_json(title, payload):
    print(f"\n【{title}】")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def validate_config():
    missing = []
    if not SENSENOVA_API_KEY:
        if not SENSENOVA_AK:
            missing.append("SENSENOVA_AK")
        if not SENSENOVA_SK:
            missing.append("SENSENOVA_SK")
    if not os.path.exists(IMAGE_FILE):
        missing.append(f"图片文件不存在: {IMAGE_FILE}")
    if missing:
        raise RuntimeError(f"启动失败: {', '.join(missing)}")


def get_auth_token():
    if SENSENOVA_API_KEY:
        return SENSENOVA_API_KEY
    payload = {
        "iss": SENSENOVA_AK,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(
        payload,
        SENSENOVA_SK,
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"},
    )


def encode_image_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def build_image_data_url(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return f"data:{mime_type or 'image/png'};base64,{encode_image_base64(file_path)}"


def build_payload():
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": build_image_data_url(IMAGE_FILE),
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ],
        "temperature": 0.8,
        "stream": False,
        "max_completion_tokens": 4049,
        "reasoning_effort": "low",
        "frequency_penalty": 1.05,
        "top_p": 0.7,
    }


def sanitize_payload_for_log(payload):
    sanitized = deepcopy(payload)
    for message in sanitized.get("messages", []):
        for content in message.get("content", []):
            if content.get("type") == "image_url":
                image_url = content.get("image_url", {}).get("url", "")
                content["image_url"] = {"url": f"<data-url length={len(image_url)}>"}
    return sanitized


def chat_with_image():
    client = OpenAI(api_key=get_auth_token(), base_url=BASE_URL)
    payload = build_payload()

    log_step(f"当前模型: {MODEL_ID}")
    log_step(f"图片文件: {IMAGE_FILE}")
    log_step(f"OpenAI SDK base_url: {BASE_URL}")
    dump_json("对话请求体", sanitize_payload_for_log(payload))
    log_step("正在请求模型分析图片...")

    response = client.chat.completions.create(**payload)
    response_json = response.model_dump()
    dump_json("对话响应", response_json)

    choices = response_json.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        print(f"\n【AI 分析】:\n{message.get('content', '无')}")


if __name__ == "__main__":
    validate_config()
    chat_with_image()
