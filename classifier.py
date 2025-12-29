# classifier.py — объединяет эвристику + ruBERT + Qwen (по очереди)
import logging
from typing import Dict, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Попытаемся загрузить transformers (если установлены)
if TYPE_CHECKING:
    # Подсказка для линтера, чтобы он знал о модуле transformers
    from transformers import pipeline as _pipeline  # type: ignore[import-not-found]

try:
    from transformers import pipeline  # type: ignore[import-not-found]
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# конфигурация модели берётся из config.py чтобы не дублировать
try:
    from config import RUBERT_MODEL, USE_RUBERT, USE_QWEN, QWEN_MODEL
except Exception:
    RUBERT_MODEL = 'cointegrated/rubert-tiny2'
    USE_RUBERT = True
    USE_QWEN = False  # Отключаем старую модель Qwen-3B-Chat, используем Qwen3-8B через spam_model
    QWEN_MODEL = 'Qwen3-8B-Q4_K_M.gguf'  # Fallback модель

bert_classifier = None
llm_generator = None

if HAS_TRANSFORMERS and USE_RUBERT:
    try:
        bert_classifier = pipeline('text-classification', model=RUBERT_MODEL)
        logger.info('Loaded bert classifier: %s', RUBERT_MODEL)
    except Exception:
        logger.exception('Не удалось загрузить BERT-классификатор')

# LLM: будем использовать генерацию с инструкцией, если модель доступна
if HAS_TRANSFORMERS and USE_QWEN:
    try:
        # используем text-generation если доступно
        llm_generator = pipeline('text-generation', model=QWEN_MODEL)
        logger.info('Loaded LLM generator: %s', QWEN_MODEL)
    except Exception:
        logger.exception('Не удалось загрузить LLM модель')

def llm_classify(text: str) -> Dict:
    # Генерируем JSON-ответ от LLM с простой инструкцией
    prompt = (
        "You are a classifier. Decide whether the following Telegram post is advertisement or spam. "
        "Return a JSON object with keys: is_spam (true/false), score (0-1), reason (string).\\n\\nPOST:\\n"
        + text + "\\n\\nJSON:\\n"
    )
    try:
        if llm_generator:
            out = llm_generator(prompt, max_new_tokens=150)
            txt = out[0].get('generated_text') or out[0].get('text') or ''
            # попытка извлечь JSON из конца
            import re, json
            m = re.search(r'\{[\s\S]*\}$', txt)
            if m:
                j = json.loads(m.group(0))
                return {
                    'is_spam': bool(j.get('is_spam', False)),
                    'score': float(j.get('score', 0.0)),
                    'reason': j.get('reason', '')
                }
    except Exception:
        logger.exception('Ошибка при вызове LLM')

    return {'is_spam': False, 'score': 0.0, 'reason': 'llm_failed'}

from spam_rules import heuristic_spam_score

def classify_text(text: str) -> dict:
    text = text or ""

    # 1️⃣ Быстрая эвристика
    score = heuristic_spam_score(text)

    if score >= 6:
        return {"is_spam": True, "score": score, "reason": "heuristic_high"}

    if score <= 1:
        return {"is_spam": False, "score": score, "reason": "heuristic_low"}

    # 2️⃣ Если есть BERT — используем его
    if bert_classifier:
        try:
            res = bert_classifier(text[:512])
            if isinstance(res, list) and len(res) > 0:
                lab = res[0]
                label = lab.get('label')
                sc = lab.get('score', 0.0)
                # Здесь я предполагаю, что модель помечает спам как LABEL_1 или содержит 'spam' в label.
                is_spam = sc > 0.75 and ('spam' in str(label).lower() or 'label_1' in str(label).lower())
                return {'is_spam': bool(is_spam), 'score': float(sc), 'reason': 'bert'}
        except Exception:
            logger.exception('Ошибка при вызове BERT')

    # 3️⃣ LLM (Qwen GGUF через llama-cli) - только если доступен
    try:
        from llm_qwen import qwen_classify
        return qwen_classify(text)
    except ImportError:
        logger.debug('llm_qwen недоступен, пропускаем llama-cli метод')
    except Exception:
        logger.exception('Ошибка при вызове Qwen через llama-cli')
    
    # 4️⃣ Fallback: LLM через transformers (медленнее) - только если включен
    if USE_QWEN and llm_generator:
        return llm_classify(text)
    
    # 5️⃣ Финальный fallback: возвращаем результат эвристики
    return {"is_spam": score >= 6, "score": min(1.0, score / 10.0) if score > 1.0 else score, "reason": "heuristic_final"}
