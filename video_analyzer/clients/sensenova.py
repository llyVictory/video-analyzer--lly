import json
import time
from typing import Optional, Dict, Any

import requests

from .llm_client import LLMClient

import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
RATE_LIMIT_WAIT_TIME = 25
DEFAULT_WAIT_TIME = 25


class SenseNovaClient(LLMClient):
    def __init__(self, api_key: str, api_url: str, max_retries: int = DEFAULT_MAX_RETRIES):
        self.api_key = api_key
        self.generate_url = api_url
        self.compat_generate_url = "https://api.sensenova.cn/compatible-mode/v2/chat/completions"
        self.max_retries = max_retries

    def _extract_native_message_text(self, choice: Dict[str, Any]) -> str:
        message = choice.get("message", "")
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(text)
                return "".join(parts)
        return ""

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "SenseNova-V6-5-Pro-20251215",
        temperature: float = 0.2,
        num_predict: int = 256,
    ) -> Dict[Any, Any]:
        if image_path is None:
            return self._generate_text_only(
                prompt=prompt,
                stream=stream,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
            )

        content = []
        if image_path:
            content.append(
                {
                    "type": "image_base64",
                    "image_base64": self.encode_image(image_path),
                }
            )
        content.append({"type": "text", "text": prompt})

        data = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"enabled": False},
            "max_new_tokens": num_predict,
            "temperature": temperature,
            "stream": stream,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.generate_url, headers=headers, json=data)

                if response.status_code != 200:
                    logger.error(f"SenseNova request failed with status {response.status_code}: {response.text}")
                    response.raise_for_status()

                try:
                    json_response = response.json()
                    if stream:
                        return self._handle_streaming_response(response)

                    data_payload = json_response.get("data", {})
                    choices = data_payload.get("choices", [])
                    if not choices:
                        raise Exception("No choices in response")

                    text = self._extract_native_message_text(choices[0]).strip()
                    if text:
                        return {"response": text}

                    finish_reason = choices[0].get("finish_reason")
                    raise Exception(f"Empty SenseNova response (finish_reason={finish_reason})")
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {response.text}")

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"An error occurred: {str(e)}")

                if image_path is not None and data.get("max_new_tokens", 0) < 1536:
                    data["max_new_tokens"] = min(1536, max(1024, int(data["max_new_tokens"] * 1.5)))
                    logger.warning(f"Increasing SenseNova image max_new_tokens to {data['max_new_tokens']} for retry")

                wait_time = RATE_LIMIT_WAIT_TIME
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                    if "Retry-After" in e.response.headers:
                        try:
                            wait_time = int(e.response.headers["Retry-After"])
                            logger.info(f"Using Retry-After header value: {wait_time} seconds")
                        except (ValueError, TypeError):
                            logger.warning("Invalid Retry-After header value, using default wait time")
                else:
                    wait_time = DEFAULT_WAIT_TIME

                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                logger.warning(f"Waiting {wait_time} seconds before retry")
                time.sleep(wait_time)

    def _generate_text_only(
        self,
        prompt: str,
        stream: bool,
        model: str,
        temperature: float,
        num_predict: int,
    ) -> Dict[Any, Any]:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.compat_generate_url, headers=headers, json=data)
                if response.status_code != 200:
                    logger.error(f"SenseNova compatible text request failed with status {response.status_code}: {response.text}")
                    response.raise_for_status()

                json_response = response.json()
                choices = json_response.get("choices", [])
                if not choices:
                    raise Exception("No choices in compatible response")

                message = choices[0].get("message", {})
                content = message.get("content")
                if content is None:
                    raise Exception("No content in compatible response")

                text = content.strip() if isinstance(content, str) else ""
                if not text:
                    raise Exception("Empty content in compatible response")
                return {"response": text}
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"An error occurred: {str(e)}")
                wait_time = DEFAULT_WAIT_TIME
                logger.warning(f"Compatible text request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                logger.warning(f"Waiting {wait_time} seconds before retry")
                time.sleep(wait_time)

    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        accumulated_response = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    decoded = decoded[5:].strip()
                if decoded == "[DONE]":
                    continue
                json_response = json.loads(decoded)
                data_payload = json_response.get("data", {})
                choices = data_payload.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        accumulated_response += content
            except json.JSONDecodeError:
                continue

        return {"response": accumulated_response}
