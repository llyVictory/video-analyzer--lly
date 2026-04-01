import os
import time
import json

import jwt
import requests
from dotenv import load_dotenv

load_dotenv()

SENSENOVA_AK = os.getenv("SENSENOVA_AK")
SENSENOVA_SK = os.getenv("SENSENOVA_SK")
SENSENOVA_API_KEY = os.getenv("SENSENOVA_API_KEY")
MODELS_URL = "https://api.sensenova.cn/v1/llm/models"


def log_step(message):
    print(f"[*] {message}")


def log_error(message):
    print(f"[Error] {message}")


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
    if missing:
        raise RuntimeError(f"缺少必要环境变量: {', '.join(missing)}")


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


def list_models():
    token = get_auth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    log_step(f"请求模型列表: {MODELS_URL}")
    response = requests.get(MODELS_URL, headers=headers)
    log_step(f"接口状态码: {response.status_code}")

    request_id = response.headers.get("x-request-id")
    if request_id:
        log_step(f"x-request-id: {request_id}")

    if response.status_code != 200:
        log_error(f"接口报错: {response.text}")
        return

    response_json = response.json()
    dump_json("模型列表响应", response_json)

    models = response_json.get("data", [])
    print("\n【模型摘要】")
    if not models:
        print("未返回任何模型。")
        return

    for index, model in enumerate(models, start=1):
        print(f"{index}. id: {model.get('id', '')}")
        print(f"   type: {model.get('type', '')}")
        print(f"   owned_by: {model.get('owned_by', '')}")
        print(f"   created_at: {model.get('created_at', '')}")
        permissions = model.get("permission", [])
        if permissions:
            print(f"   permission: {json.dumps(permissions, ensure_ascii=False)}")
        else:
            print("   permission: []")


if __name__ == "__main__":
    validate_config()
    list_models()
