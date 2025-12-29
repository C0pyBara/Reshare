"""Модуль для логирования сообщений для последующей разметки ML датасета."""
import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_MESSAGES_FILE = DATA_DIR / "raw_messages.jsonl"

# Создаем директорию если не существует
DATA_DIR.mkdir(exist_ok=True)


def log_message_for_ml(text: str, heuristic_score: float, channel: str = None, message_id: int = None):
    """Логирует сообщение для последующей разметки ML датасета.
    
    Args:
        text: Текст сообщения
        heuristic_score: Оценка эвристики (0-10+)
        channel: Название канала (опционально)
        message_id: ID сообщения (опционально)
    """
    if not text or not text.strip():
        return  # Пропускаем пустые сообщения
    
    record = {
        "text": text.strip(),
        "heuristic_score": float(heuristic_score),
        "timestamp": time.time(),
        "channel": channel,
        "message_id": message_id
    }
    
    try:
        with open(RAW_MESSAGES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Не падаем если не удалось залогировать
        print(f"Ошибка логирования сообщения: {e}")

