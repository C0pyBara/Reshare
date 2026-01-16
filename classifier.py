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
bert_is_trained = False  # Флаг, что модель обучена
llm_generator = None

if HAS_TRANSFORMERS and USE_RUBERT:
    try:
        bert_classifier = pipeline('text-classification', model=RUBERT_MODEL)
        logger.info('Loaded bert classifier: %s', RUBERT_MODEL)
        
        # Проверяем, обучена ли модель, делая тестовый запрос
        # Если модель не обучена, score будет около 0.5 для всех текстов
        try:
            test_result = bert_classifier("тест")
            if isinstance(test_result, list) and len(test_result) > 0:
                test_score = test_result[0].get('score', 0.0)
                # Если score сильно отличается от 0.5, модель может быть обучена
                # Но для надежности лучше проверить на нескольких примерах
                test_result2 = bert_classifier("реклама скидка купить")
                if isinstance(test_result2, list) and len(test_result2) > 0:
                    test_score2 = test_result2[0].get('score', 0.0)
                    # Если оба результата около 0.5, модель скорее всего не обучена
                    if abs(test_score - 0.5) < 0.1 and abs(test_score2 - 0.5) < 0.1:
                        logger.warning(
                            'BERT модель %s не обучена на классификации спама. '
                            'Все предсказания будут около 0.5. BERT будет пропущен.',
                            RUBERT_MODEL
                        )
                        bert_is_trained = False
                    else:
                        bert_is_trained = True
                        logger.info('BERT модель обучена, будет использоваться для классификации')
        except Exception:
            logger.warning('Не удалось проверить обученность BERT модели, предполагаем что не обучена')
            bert_is_trained = False
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

    # 2️⃣ Если есть BERT и он обучен — используем его
    if bert_classifier and bert_is_trained:
        try:
            res = bert_classifier(text[:512])
            if isinstance(res, list) and len(res) > 0:
                lab = res[0]
                label = str(lab.get('label', '')).lower()
                sc = lab.get('score', 0.0)
                
                # Логика для бинарной классификации:
                # Если score близок к 0.5 (0.4-0.6), модель не уверена - пропускаем BERT
                if 0.4 <= sc <= 0.6:
                    logger.debug(f'BERT неуверен (score={sc:.3f}, label={label}), пропускаем')
                    # Продолжаем к следующему методу
                else:
                    # Модель уверена - интерпретируем результат
                    # LABEL_1 обычно означает спам, LABEL_0 - не спам
                    is_spam_bert = False
                    normalized_score = sc
                    
                    if 'label_1' in label or 'spam' in label:
                        # LABEL_1 = спам
                        is_spam_bert = (sc > 0.5)  # Если score > 0.5, считаем спамом
                        normalized_score = sc
                    elif 'label_0' in label or 'not_spam' in label or 'ham' in label:
                        # LABEL_0 = не спам
                        is_spam_bert = (sc < 0.5)  # Если score < 0.5, считаем спамом (инвертированная логика)
                        normalized_score = 1.0 - sc  # Инвертируем score для консистентности
                    else:
                        # Неизвестный label - используем score напрямую
                        is_spam_bert = (sc > 0.5)
                        normalized_score = sc if is_spam_bert else (1.0 - sc)
                    
                    logger.debug(f'BERT результат: is_spam={is_spam_bert}, score={normalized_score:.3f}, label={label}, raw_score={sc:.3f}')
                    return {
                        'is_spam': bool(is_spam_bert),
                        'score': float(normalized_score),
                        'reason': 'bert'
                    }
        except Exception:
            logger.exception('Ошибка при вызове BERT')
    elif bert_classifier and not bert_is_trained:
        # BERT загружен, но не обучен - пропускаем без логирования (чтобы не засорять логи)
        pass

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
