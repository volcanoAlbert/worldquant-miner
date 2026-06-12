"""
OpenAI-compatible remote LLM helpers for generation_two.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


DEFAULT_REMOTE_MODEL = "gpt-4o-mini"


@dataclass
class RemoteLLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: int = 120


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def resolve_credentials_path(credentials_path: Optional[str] = None) -> Optional[Path]:
    names = []
    if credentials_path:
        requested = Path(credentials_path).expanduser()
        if requested.exists() or requested.is_absolute():
            return requested
        names.append(requested.name)

    names.extend(["credential.txt", "credentials.txt"])
    search_roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    seen = set()

    for root in search_roots:
        for parent in [root, *root.parents]:
            parent_key = str(parent)
            if parent_key in seen:
                continue
            seen.add(parent_key)
            for name in names:
                candidate = parent / name
                if candidate.exists():
                    return candidate

    return None


def parse_credentials_settings(credentials_path: Optional[str] = None) -> Tuple[Optional[list], Dict[str, str], Optional[Path]]:
    resolved_path = resolve_credentials_path(credentials_path)
    if not resolved_path:
        return None, {}, None

    text = resolved_path.read_text(encoding="utf-8")
    settings: Dict[str, str] = {}
    credentials = None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) >= 2:
            credentials = [parsed[0], parsed[1]]
        elif isinstance(parsed, dict):
            username = parsed.get("username") or parsed.get("email")
            password = parsed.get("password")
            if username and password:
                credentials = [username, password]
            settings = {
                str(key): str(value)
                for key, value in parsed.items()
                if value is not None
            }
    except json.JSONDecodeError:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if credentials is None and line.startswith("["):
                parsed_line = json.loads(line)
                if isinstance(parsed_line, list) and len(parsed_line) >= 2:
                    credentials = [parsed_line[0], parsed_line[1]]
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                settings[key.strip()] = strip_quotes(value)

    return credentials, settings, resolved_path


def setting(settings: Dict[str, str], *names: str) -> Optional[str]:
    normalized = {key.upper(): value for key, value in settings.items()}
    for name in names:
        value = settings.get(name)
        if value:
            return value

        value = normalized.get(name.upper())
        if value:
            return value

        value = os.getenv(name)
        if value:
            return value
    return None


def default_remote_model(base_url: str) -> str:
    base_url_lower = base_url.lower()
    if "deepseek" in base_url_lower:
        return "deepseek-chat"
    if "moonshot" in base_url_lower or "kimi" in base_url_lower:
        return "moonshot-v1-8k"
    if "openrouter" in base_url_lower:
        return "openai/gpt-4o-mini"
    return DEFAULT_REMOTE_MODEL


def chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def load_remote_llm_config(
    credentials_path: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 120,
) -> Optional[RemoteLLMConfig]:
    _, settings, _ = parse_credentials_settings(credentials_path)

    base_url = base_url or setting(settings, "base_url", "OPENAI_BASE_URL", "LLM_BASE_URL")
    api_key = api_key or setting(settings, "OPENAI_API_KEY", "LLM_API_KEY", "API_KEY")

    if not base_url or not api_key:
        return None

    model = model or setting(settings, "OPENAI_MODEL", "LLM_MODEL", "MODEL") or default_remote_model(base_url)
    timeout_raw = setting(settings, "OPENAI_TIMEOUT", "LLM_TIMEOUT")
    if timeout_raw:
        try:
            timeout = int(timeout_raw)
        except ValueError:
            pass

    return RemoteLLMConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
