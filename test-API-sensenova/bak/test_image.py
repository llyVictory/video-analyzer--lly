import os
import time
import json
import mimetypes
import base64
from copy import deepcopy
import requests
import jwt
from dotenv import load_dotenv

# 1. 加载配置
load_dotenv()
SENSENOVA_AK = os.getenv("SENSENOVA_AK")
SENSENOVA_SK = os.getenv("SENSENOVA_SK")
SENSENOVA_API_KEY = os.getenv("SENSENOVA_API_KEY") 
MODEL_ID = os.getenv("SENSENOVA_MODEL_ID")
IMAGE_FILE = "../测试图片.png"
FILES_URL = "https://file.sensenova.cn/v1/files"
CHAT_URL = "https://api.sensenova.cn/v1/llm/chat-completions"
VALIDATION_WAIT_SECONDS = 5
IMAGE_INPUT_MODE = os.getenv("SENSENOVA_IMAGE_INPUT_MODE", "base64")

def log_step(message):
    print(f"[*] {message}")

def log_ok(message):
    print(f"[+] {message}")

def log_warn(message):
    print(f"[!] {message}")

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
                base64_text = content["image_base64"]
                content["image_base64"] = f"<base64 length={len(base64_text)}>"
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
    if missing:
        raise RuntimeError(f"缺少必要环境变量: {', '.join(missing)}")

def get_auth_token():
    if SENSENOVA_API_KEY: return SENSENOVA_API_KEY
    payload = {"iss": SENSENOVA_AK, "exp": int(time.time()) + 1800, "nbf": int(time.time()) - 5}
    return jwt.encode(payload, SENSENOVA_SK, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"})

def guess_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def encode_image_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")

def upload_image_file(token, file_path):
    if not os.path.exists(file_path):
        log_error(f"图片文件不存在: {file_path}")
        return None
    headers = {"Authorization": f"Bearer {token}"}
    mime_type = guess_mime_type(file_path)
    file_size = os.path.getsize(file_path)
    data = {'scheme': 'MULTIMODAL_1'} # 文档要求 scheme 参数
    log_step(f"正在上传图片: {file_path}")
    log_step(f"文件大小: {file_size} bytes")
    log_step(f"使用 MIME 类型: {mime_type}")
    dump_json("上传请求参数", {"url": FILES_URL, "data": data, "file_name": os.path.basename(file_path), "mime_type": mime_type})
    with open(file_path, "rb") as image_file:
        files = {'file': (os.path.basename(file_path), image_file, mime_type)}
        response = requests.post(FILES_URL, headers=headers, files=files, data=data)
    log_step(f"上传接口状态码: {response.status_code}")
    request_id = response.headers.get("x-request-id")
    if request_id:
        log_step(f"上传接口 x-request-id: {request_id}")
    if response.status_code == 200:
        body = response.json()
        payload = body.get("data", {}) or body
        file_id = payload.get("id")
        file_status = payload.get("status")
        dump_json("上传响应", body)
        log_ok(f"图片上传成功! ID: {file_id}")
        if file_status:
            log_step(f"上传返回状态: {file_status}")
        return file_id
    log_error(f"上传失败: {response.text}")
    return None

def wait_for_validation_window():
    log_warn("api.txt 说明上传后会经历格式校验，但未提供查询文件状态的接口文档。")
    log_step(f"固定等待 {VALIDATION_WAIT_SECONDS} 秒，尽量避开 UPLOADED 校验窗口")
    time.sleep(VALIDATION_WAIT_SECONDS)

def build_image_content(file_path, file_id, image_input_mode):
    if image_input_mode == "base64":
        log_step("图片输入模式: image_base64")
        return {
            "type": "image_base64",
            "image_base64": encode_image_base64(file_path)
        }
    log_step("图片输入模式: image_file_id")
    return {
        "type": "image_file_id",
        "image_file_id": file_id
    }

def extract_message_text(response_json):
    data = response_json.get("data", {})
    choice = data.get("choices", [{}])[0]
    return choice.get("message", "")

def is_missing_image_response(message_text):
    markers = (
        "无法直接查看或接收图片",
        "没有收到任何图片附件",
        "无法为您描述图片内容",
        "无法直接查看图片",
    )
    return any(marker in message_text for marker in markers)

def chat_with_image(token, file_path, file_id, image_input_mode):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    image_content = build_image_content(file_path, file_id, image_input_mode)
    payload = {
        "model": MODEL_ID,
        "messages": [{
            "role": "user", 
            "content": [
                image_content,
                {"type": "text", "text": "请描述这张图片。"}
            ]
        }],
        "thinking": {"enabled": False},
        "max_new_tokens": 1024
    }
    log_step(f"当前模型: {MODEL_ID}")
    dump_json("对话请求体", sanitize_payload_for_log(payload))
    log_step("正在请求模型分析图片...")
    response = requests.post(CHAT_URL, headers=headers, json=payload)
    log_step(f"对话接口状态码: {response.status_code}")
    request_id = response.headers.get("x-request-id")
    if request_id:
        log_step(f"对话接口 x-request-id: {request_id}")
    if response.status_code == 200:
        response_json = response.json()
        dump_json("对话响应", response_json)
        data = response_json.get("data", {})
        choice = data.get("choices", [{}])[0]
        print(f"\n【AI 思考】:\n{choice.get('reasoning_content', '无')}")
        print(f"\n【AI 分析】:\n{choice.get('message', '无')}")
        return response_json
    else:
        log_error(f"接口报错: {response.text}")
    return None

if __name__ == "__main__":
    validate_config()
    token = get_auth_token()
    fid = upload_image_file(token, IMAGE_FILE)
    if fid:
        wait_for_validation_window()
        response_json = chat_with_image(token, IMAGE_FILE, fid, IMAGE_INPUT_MODE)
        if (
            IMAGE_INPUT_MODE == "file_id"
            and response_json
            and is_missing_image_response(extract_message_text(response_json))
        ):
            log_warn("模型未识别到 image_file_id，对话将自动回退到 image_base64 重试。")
            chat_with_image(token, IMAGE_FILE, fid, "base64")
