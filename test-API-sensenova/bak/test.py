import os
import time
import warnings
import json
import requests
import jwt # pip install PyJWT
from jwt.exceptions import InvalidKeyError
from dotenv import load_dotenv # pip install python-dotenv

# 忽略 JWT 长度警告 (HMAC key below 32 bytes)
warnings.filterwarnings("ignore", category=UserWarning, module="jwt")

# 1. 加载环境变量 (Config from .env)
load_dotenv()

# 从 .env 读取配置
SENSENOVA_AK = os.getenv("SENSENOVA_AK")
SENSENOVA_SK = os.getenv("SENSENOVA_SK")
# 如果用户提供了直接的 API_KEY，则不生成 JWT
SENSENOVA_API_KEY = os.getenv("SENSENOVA_API_KEY") 

MODEL_ID = os.getenv("SENSENOVA_MODEL_ID", "SenseChat-V")
VIDEO_FILE = os.getenv("SENSENOVA_VIDEO_FILE", "640.mp4")
FILES_URL = os.getenv("SENSENOVA_FILES_URL", "https://api.sensenova.cn/v1/files")
CHAT_URL = os.getenv("SENSENOVA_API_URL", "https://api.sensenova.cn/v1/llm/chat-completions")

def get_auth_token():
    """
    获取鉴权 Token。
    方案 A: 直接使用 API_KEY。
    方案 B: 使用 AK/SK 通过 HS256 算法生成 JWT。
    """
    if SENSENOVA_API_KEY:
        return SENSENOVA_API_KEY

    if not SENSENOVA_AK or not SENSENOVA_SK:
        print("[Error] 缺少配置。请在 .env 中设置 SENSENOVA_AK/SK 或 SENSENOVA_API_KEY。")
        return None

    # 生成 JWT Token (有效期 30 分钟)
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": SENSENOVA_AK,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5
    }
    try:
        token = jwt.encode(payload, SENSENOVA_SK, algorithm="HS256", headers=headers)
        return token
    except Exception as e:
        print(f"[Error] JWT 生成失败: {e}")
        return None

def upload_video_file(token, file_path):
    """
    调用 /v1/files 接口上传本地视频。
    获取 video_file_id 供对话接口使用。
    """
    if not os.path.exists(file_path):
        print(f"[Error] 本地视频文件不存在: {file_path}")
        return None

    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # 根据 SenseNova 最新文档，文件上传需要指定 scheme:
    # MULTIMODAL_1: 图像
    # MULTIMODAL_2: 视频
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), 'video/mp4')
    }
    data = {
        'purpose': 'chat',
        'scheme': 'MULTIMODAL_2' # 核心参数：视频类
    }

    print(f"[*] 正在尝试上传视频至: {FILES_URL}")
    print(f"[*] 文件: '{file_path}' (大小: {os.path.getsize(file_path)/1024:.2f} KB)...")
    try:
        response = requests.post(FILES_URL, headers=headers, files=files, data=data)
        if response.status_code == 200:
            res_json = response.json()
            # 解析不同可能的返回结构 (通常在 data.id 或直接 id)
            file_id = res_json.get("data", {}).get("id") or res_json.get("id")
            if file_id:
                print(f"[+] 上传成功! File ID: {file_id}")
                return file_id
            else:
                print(f"[-] 上传返回异常: {res_json}")
                return None
        else:
            print(f"[Error] 上传失败 Code: {response.status_code}, Msg: {response.text}")
            return None
    except Exception as e:
        print(f"[Exception] 上传过程发生异常: {e}")
        return None

def chat_with_sensenova(token, file_id):
    """
    核心接口调用: 发送视频 file_id 和文本 Prompt 给模型。
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 根据 api.txt 配置请求体
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_file_id",
                        "video_file_id": file_id
                    },
                    {
                        "type": "text",
                        "text": "这是一段视频文件。请根据上传的视频内容，详细描述一下这段视频的场景、人物和动作。"
                    }
                ]
            }
        ],
        "thinking": {"enabled": False}, # 开启深度思考
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "stream": False # 非流式测试
    }

    print(f"[*] 正在调用模型 '{MODEL_ID}' 分析视频内容 (File ID: {file_id})...")
    # print(f"[*] Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}") # 如需排查格式可取消注释
    try:
        start_time = time.time()
        response = requests.post(CHAT_URL, headers=headers, json=payload)
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json().get("data", {})
            choices = data.get("choices", [])
            
            if choices:
                msg = choices[0].get("message", "")
                thought = choices[0].get("reasoning_content", "")
                
                print("-" * 50)
                if thought:
                    print(f"【深度思考内容】:\n{thought}\n")
                print(f"【AI 分析结果 (耗时 {latency:.2f}s)】:\n{msg}")
                print("-" * 50)
                
                # 打印使用量统计
                usage = data.get("usage", {})
                print(f"[Usage Stats]: Total Tokens: {usage.get('total_tokens')}, Prompt: {usage.get('prompt_tokens')}")
            else:
                print(f"[-] 模型未返回内容: {response.text}")
        else:
            print(f"[Error] 模型接口报错 ({response.status_code}): {response.text}")

    except Exception as e:
        print(f"[Exception] 模型请求发生异常: {e}")

if __name__ == "__main__":
    print("=== 商汤 Sensnova API 视频理解自动化测试 ===")
    
    # 步骤 1: 鉴权
    auth_token = get_auth_token()
    
    if auth_token:
        # 步骤 2: 上传视频文件 (通过本地 640.mp4)
        video_id = upload_video_file(auth_token, VIDEO_FILE)
        
        if video_id:
            # 步骤 3: 调用视觉模型
            chat_with_sensenova(auth_token, video_id)
        else:
            print("[!] 未能获取到视频 ID，停止分析。")
    else:
        print("[!] 鉴权失败，请检查 .env 配置。")
