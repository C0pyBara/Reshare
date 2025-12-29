import subprocess
import json
import re
import os
from pathlib import Path

# Базовая директория проекта (рядом с этим файлом)
BASE_DIR = Path(__file__).resolve().parent

# Пути к бинарнику и модели
if os.name == "nt":
    # Windows: llama-cli.exe
    LLAMA_BIN = BASE_DIR / "llama.cpp" / "llama-cli.exe"
else:
    # Unix-подобные: ./llama-cli
    LLAMA_BIN = BASE_DIR / "llama.cpp" / "llama-cli"

# Сначала пробуем модель из models/, потом из llama.cpp/models/
# Qwen3-8B-GGUF
MODEL_PATH = BASE_DIR / "models" / "Qwen3-8B-Q4_K_M.gguf"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "models" / "qwen3-8b-q4_k_m.gguf"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "llama.cpp" / "models" / "Qwen3-8B-Q4_K_M.gguf"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "llama.cpp" / "models" / "qwen3-8b-q4_k_m.gguf"

def qwen_classify(text: str) -> dict:
    # Формируем промпт для Qwen3 с правильным форматом
    system_msg = (
        "Ты классификатор Telegram-постов. "
        "Определи, является ли текст рекламой или спамом. "
        "score: 0.0 = точно не спам, 1.0 = точно спам. "
        "Ответь ТОЛЬКО валидным JSON без дополнительного текста:\n"
        '{"score": 0.5, "is_spam": false, "reason": "описание"}'
    )
    
    user_msg = f"Текст для анализа:\n{text[:1000]}"
    
    # Формат для Qwen3 (использует Jinja2 template, но можно использовать простой формат)
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    # Быстрая проверка наличия бинарника и модели
    if not LLAMA_BIN.exists():
        return {
            "is_spam": False,
            "score": 0.0,
            "reason": f"llm_error: llama binary not found at {LLAMA_BIN}"
        }

    if not MODEL_PATH.exists():
        return {
            "is_spam": False,
            "score": 0.0,
            "reason": f"llm_error: model not found at {MODEL_PATH}"
        }

    # Параметры для Qwen3 согласно документации
    # Для non-thinking mode рекомендуется temperature=0.7, но для классификации используем 0.6
    cmd = [
        str(LLAMA_BIN),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", "200",  # max tokens
        "--temp", "0.6",  # Рекомендуемая температура
        "--top-k", "20",
        "--top-p", "0.95",
        "--min-p", "0",
        "--presence-penalty", "1.5",  # Важно для Qwen3
        "--no-display-prompt",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return {
                "is_spam": False,
                "score": 0.0,
                "reason": f"llm_error: llama-cli exit {result.returncode}, stderr: {result.stderr.strip()}"
            }

        output = result.stdout
        match = re.search(r"\{[\s\S]*\}", output)
        if match:
            return json.loads(match.group(0))

    except Exception as e:
        return {
            "is_spam": False,
            "score": 0.0,
            "reason": f"llm_error: {e}"
        }

    return {
        "is_spam": False,
        "score": 0.0,
        "reason": "no_llm_output"
    }
