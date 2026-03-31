import os
import sys
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def image_to_data_url(image_path: str) -> str:
    """
    把本地图片转成 data URL，供视觉模型传入 image_url.url
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as f:
        image_data = f.read()

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/png"

    base64_encoded = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded}"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value


def build_messages(prompt: str, image_path: str):
    image_data_url = image_to_data_url(image_path)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                    },
                },
            ],
        }
    ]


def print_basic_info(model_id: str, image_path: str, prompt: str):
    print("=== ModelScope 视觉模型测试 ===")
    print(f"模型: {model_id}")
    print(f"图片: {Path(image_path).resolve()}")
    print(f"提示词: {prompt}")
    print()


def test_non_stream(client: OpenAI, model_id: str, messages, temperature: float, max_tokens: int):
    print("=== 非流式测试开始 ===")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        print("\n--- 模型返回 ---")
        content = response.choices[0].message.content
        print(content if content else "[空返回]")

        usage = getattr(response, "usage", None)
        if usage:
            print("\n--- Token 用量 ---")
            print(usage)

        print("\n=== 非流式测试结束 ===\n")

    except Exception as e:
        print(f"\n非流式调用失败: {e}\n")


def test_stream(client: OpenAI, model_id: str, messages, temperature: float, max_tokens: int):
    print("=== 流式测试开始 ===")
    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        print("\n--- 流式输出 ---\n")
        has_output = False

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)

            if reasoning:
                print(reasoning, end="", flush=True)
                has_output = True
            elif content:
                print(content, end="", flush=True)
                has_output = True

        if not has_output:
            print("[无流式文本输出]", end="")

        print("\n\n=== 流式测试结束 ===\n")

    except Exception as e:
        print(f"\n流式调用失败: {e}\n")


def main():
    load_dotenv()

    api_key = require_env("MODELSCOPE_API_KEY")
    model_id = require_env("MODELSCOPE_MODEL_ID")
    image_path = require_env("IMAGE_PATH")

    prompt = os.getenv("PROMPT", "请描述这张图片")
    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("MAX_TOKENS", "1024"))

    if not os.path.isfile(image_path):
        print(f"图片不存在: {image_path}")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api-inference.modelscope.cn/v1",
    )

    messages = build_messages(prompt, image_path)

    print_basic_info(model_id, image_path, prompt)
    test_non_stream(client, model_id, messages, temperature, max_tokens)
    test_stream(client, model_id, messages, temperature, max_tokens)


if __name__ == "__main__":
    main()