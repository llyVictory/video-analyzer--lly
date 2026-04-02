import argparse
from pathlib import Path
import json
from typing import Any
import logging
import pkg_resources
import os
import time
import jwt
from dotenv import load_dotenv

# 依照官方设计文档，我们采用灵活的环境变量加载方式
# 尝试加载当前根目录下的 .env 以及测试目录下的 .env（如果存在）
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parents[1] / "test-API-Inference" / ".env")
load_dotenv(dotenv_path=Path(__file__).parents[1] / "test-API-sensenova" / ".env")

logger = logging.getLogger(__name__)


def build_sensenova_auth_token() -> str | None:
    api_key = os.getenv("SENSENOVA_API_KEY")
    if api_key:
        return api_key.strip()

    ak = os.getenv("SENSENOVA_AK")
    sk = os.getenv("SENSENOVA_SK")
    if not ak or not sk:
        return None

    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(
        payload,
        sk,
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"},
    )

class Config:
    def __init__(self, config_dir: str = "config"):
        # Handle user-provided config directory
        self.config_dir = Path(config_dir)
        self.user_config = self.config_dir / "config.json"
        
        # First try to find default_config.json in the user-provided directory
        self.default_config = self.config_dir / "default_config.json"
        
        # If not found, fallback to package's default config
        if not self.default_config.exists():
            try:
                default_config_path = pkg_resources.resource_filename('video_analyzer', 'config/default_config.json')
                self.default_config = Path(default_config_path)
                logger.debug(f"Using packaged default config from {self.default_config}")
            except Exception as e:
                logger.error(f"Error finding default config: {e}")
                raise
            
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file with cascade:
        1. Try user config (config.json)
        2. Fall back to default config (default_config.json)
        """
        try:
            if self.user_config.exists():
                logger.debug(f"Loading user config from {self.user_config}")
                with open(self.user_config) as f:
                    self.config = json.load(f)
            else:
                logger.debug(f"No user config found, loading default config from {self.default_config}")
                with open(self.default_config) as f:
                    self.config = json.load(f)
            
            # Ensure clients section exists
            if "clients" not in self.config:
                self.config["clients"] = {"default": "ollama"}
            
            # --- START: MODELSCOPE INJECTION ---
            # 依照用户要求，按照官方 openai_api 风格进行环境变量注入，而不改动原有客户端逻辑
            ms_api_key = os.getenv("MODELSCOPE_API_KEY")
            if ms_api_key:
                logger.info("检测到 MODELSCOPE_API_KEY，正在自动配置为 openai_api 模式...")
                self.config["clients"]["default"] = "openai_api"
                if "openai_api" not in self.config["clients"]:
                    self.config["clients"]["openai_api"] = {}
                
                self.config["clients"]["openai_api"]["api_key"] = ms_api_key
                self.config["clients"]["openai_api"]["api_url"] = "https://api-inference.modelscope.cn/v1"
                
                ms_model = os.getenv("MODELSCOPE_MODEL_ID")
                if ms_model:
                    self.config["clients"]["openai_api"]["model"] = ms_model
                elif not self.config["clients"]["openai_api"].get("model"):
                    self.config["clients"]["openai_api"]["model"] = "Qwen/Qwen2-VL-7B-Instruct"
            # --- END: MODELSCOPE INJECTION ---

            sensenova_api_key = build_sensenova_auth_token()
            if sensenova_api_key:
                logger.info("检测到 SenseNova 环境变量，已启用 sensenova 预设。")
                if "sensenova" not in self.config["clients"]:
                    self.config["clients"]["sensenova"] = {}
                self.config["clients"]["sensenova"]["api_key"] = sensenova_api_key
                self.config["clients"]["sensenova"]["api_url"] = "https://api.sensenova.cn/v1/llm/chat-completions"
                sensenova_model = (
                    os.getenv("SENSENOVA_OPENAI_MODEL_ID")
                    or os.getenv("SENSENOVA_MODEL_ID")
                )
                if sensenova_model:
                    self.config["clients"]["sensenova"]["model"] = sensenova_model.strip()
                elif not self.config["clients"]["sensenova"].get("model"):
                    self.config["clients"]["sensenova"]["model"] = "SenseNova-V6-5-Pro-20251215"

            # Ensure prompts is a list
            if not isinstance(self.config.get("prompts", []), list):
                logger.warning("Prompts in config is not a list, setting to empty list")
                self.config["prompts"] = []
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration with command line arguments."""
        for key, value in vars(args).items():
            if value is not None:  # Only update if argument was provided
                # 对字符串类型的值进行首尾空格清理，防止 API 报错
                if isinstance(value, str):
                    value = value.strip()
                    
                if key == "client":
                    # --- 官方风格注入：将 modelscope 虚拟客户端映射到 openai_api ---
                    if value == "modelscope":
                        self.config["clients"]["default"] = "openai_api"
                        if "openai_api" not in self.config["clients"]:
                            self.config["clients"]["openai_api"] = {}
                        self.config["clients"]["openai_api"]["api_url"] = "https://api-inference.modelscope.cn/v1"
                        
                        # 强力注入：优先使用 .env 中的成功 Model ID
                        ms_model = os.getenv("MODELSCOPE_MODEL_ID")
                        if ms_model:
                            logger.info(f"Using Model ID from ENV: {ms_model}")
                            self.config["clients"]["openai_api"]["model"] = ms_model
                    elif value == "sensenova":
                        self.config["clients"]["default"] = "sensenova"
                        if "sensenova" not in self.config["clients"]:
                            self.config["clients"]["sensenova"] = {}
                        self.config["clients"]["sensenova"]["api_url"] = "https://api.sensenova.cn/v1/llm/chat-completions"
                        sensenova_model = (
                            os.getenv("SENSENOVA_OPENAI_MODEL_ID")
                            or os.getenv("SENSENOVA_MODEL_ID")
                        )
                        if sensenova_model:
                            logger.info(f"Using SenseNova Model ID from ENV: {sensenova_model}")
                            self.config["clients"]["sensenova"]["model"] = sensenova_model.strip()
                    else:
                        self.config["clients"]["default"] = value
                elif key == "ollama_url":
                    self.config["clients"]["ollama"]["url"] = value
                elif key == "api_key":
                    self.config["clients"]["openai_api"]["api_key"] = value
                    # If key is provided but no client specified, use OpenAI API
                    if not args.client:
                        self.config["clients"]["default"] = "openai_api"
                elif key == "api_url":
                    self.config["clients"]["openai_api"]["api_url"] = value
                elif key == "model":
                    client = self.config["clients"]["default"]
                    self.config["clients"][client]["model"] = value
                elif key == "prompt":
                    self.config["prompt"] = value
                #overide audio config
                elif key == "whisper_model":
                    self.config["audio"]["whisper_model"] = value  # default is 'medium'
                elif key == "language":
                    if value is not None:
                        self.config["audio"]["language"] = value
                elif key == "device":
                    self.config["audio"]["device"] = value
                elif key == "temperature":
                    self.config["clients"]["temperature"] = value
                elif key == "output":
                    self.config["output_dir"] = value
                elif key not in ["start_stage", "max_frames"]:  # Ignore these as they're command-line only
                    self.config[key] = value

        # --- 最终否决权 (Final Override) ---
        # 确保无论 CLI 怎么传参，只要有有效的 MODELSCOPE 环境配置，就强制同步模型 ID
        ms_api_key = os.getenv("MODELSCOPE_API_KEY")
        if ms_api_key and self.config["clients"]["default"] == "openai_api":
            ms_model = os.getenv("MODELSCOPE_MODEL_ID")
            if ms_model:
                self.config["clients"]["openai_api"]["model"] = ms_model.strip()
        if self.config["clients"]["default"] == "sensenova":
            sensenova_api_key = build_sensenova_auth_token()
            if sensenova_api_key:
                self.config["clients"]["sensenova"]["api_key"] = sensenova_api_key
            sensenova_model = os.getenv("SENSENOVA_OPENAI_MODEL_ID") or os.getenv("SENSENOVA_MODEL_ID")
            if sensenova_model:
                self.config["clients"]["sensenova"]["model"] = sensenova_model.strip()

    def save_user_config(self):
        """Save current configuration to user config file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.user_config, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug(f"Saved user config to {self.user_config}")
        except Exception as e:
            logger.error(f"Error saving user config: {e}")
            raise

def get_client(config: Config) -> dict:
    """获取 client 的配置字典。"""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    
    if client_type == "ollama":
        return {"url": client_config.get("url", "http://localhost:11434")}
    elif client_type == "openai_api" or client_type == "modelscope":
        # 依照官方设计，魔搭通过 openai_api client 实现调用
        api_key = os.getenv("MODELSCOPE_API_KEY") or client_config.get("api_key")
        api_url = client_config.get("api_url") or "https://api-inference.modelscope.cn/v1"
        
        if not api_key:
            # 如果是 UI 传来的 modelscope 或者是已经重定义为 openai_api
            raise ValueError(
                "API KEY 缺失！请确保:\n"
                "1. 在项目根目录创建了 .env 文件\n"
                "2. .env 中设置了 MODELSCOPE_API_KEY=你的Token\n"
                "3. 或者在运行命令中提供 --api-key"
            )
        if not api_url:
            raise ValueError("API URL 缺失，请检查配置或提供 --api-url")
        return {
            "api_key": api_key,
            "api_url": api_url
        }
    elif client_type == "sensenova":
        api_key = build_sensenova_auth_token() or client_config.get("api_key")
        api_url = client_config.get("api_url") or "https://api.sensenova.cn/v1/llm/chat-completions"
        if not api_key:
            raise ValueError(
                "SenseNova API KEY 缺失！请确保至少满足以下任一条件:\n"
                "1. 在 .env 中设置 SENSENOVA_API_KEY\n"
                "2. 或者同时设置 SENSENOVA_AK 与 SENSENOVA_SK"
            )
        return {
            "api_key": api_key,
            "api_url": api_url
        }
    else:
        raise ValueError(f"未知的客户端类型: {client_type}")

def get_model(config: Config) -> str:
    """获取模型 ID。"""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    return client_config.get("model", "llama3.2-vision")
