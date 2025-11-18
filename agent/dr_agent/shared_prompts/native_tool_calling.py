"""
Prompts for native tool calling mode (OpenAI-style function calling)
"""

import yaml
from pathlib import Path


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


SYSTEM_PROMPT = load_yaml((Path(__file__).parent / "native_tool_calling.yaml"))

STRUCTURED_PROMPTS = {
    "v20250907": SYSTEM_PROMPT,
}
