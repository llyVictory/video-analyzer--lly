import os
import time
import json
import base64
from copy import deepcopy

import jwt
import requests
from dotenv import load_dotenv

load_dotenv()

SENSENOVA_AK = os.getenv("SENSENOVA_AK")
SENSENOVA_SK = os.getenv("SENSENOVA_SK")
SENSENOVA_API_KEY = os.getenv("SENSENOVA_API_KEY")
MODEL_ID = os.getenv("SENSENOVA_MODEL_ID")
IMAGE_FILE = os.getenv("SENSENOVA_IMAGE_FILE", "测试图片.png")
PROMPT = os.getenv("SENSENOVA_IMAGE_PROMPT", "请描述这张图片。")
CHAT_URL = "https://api.sensenova.cn/v1/llm/chat-completions"


def log_step(message):
    print(f"[*] {message}")


def log_error(message):
    print(f"[Error] {message}")


def dump_json(title, payload):
    print(f"\n【{title}】")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def sanitize_payload_for_log(payload):
    sanitized = deepcopy(payload)
    for message in sanitized.get("messages", []):
        for content in message.get("content", []):
            if content.get("type") == "image_base64" and "image_base64" in content:
                content["image_base64"] = f"<base64 length={len(content['image_base64'])}>"
    return sanitized


def validate_config():
    missing = []
    if not SENSENOVA_API_KEY:
        if not SENSENOVA_AK:
            missing.append("SENSENOVA_AK")
        if not SENSENOVA_SK:
            missing.append("SENSENOVA_SK")
    if not MODEL_ID:
        missing.append("SENSENOVA_MODEL_ID")
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


def build_payload():
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_base64",
                        "image_base64": encode_image_base64(IMAGE_FILE),
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ],
        "thinking": {"enabled": False},
        "max_new_tokens": 1024,
    }


def chat_with_image():
    token = get_auth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = build_payload()
    log_step(f"当前模型: {MODEL_ID}")
    log_step(f"图片文件: {IMAGE_FILE}")
    dump_json("对话请求体", sanitize_payload_for_log(payload))
    log_step("正在请求模型分析图片...")
    response = requests.post(CHAT_URL, headers=headers, json=payload)
    log_step(f"对话接口状态码: {response.status_code}")
    request_id = response.headers.get("x-request-id")
    if request_id:
        log_step(f"对话接口 x-request-id: {request_id}")

    if response.status_code != 200:
        log_error(f"接口报错: {response.text}")
        return

    response_json = response.json()
    dump_json("对话响应", response_json)
    data = response_json.get("data", {})
    choice = data.get("choices", [{}])[0]
    print(f"\n【AI 分析】:\n{choice.get('message', '无')}")


if __name__ == "__main__":
    validate_config()
    chat_with_image()
