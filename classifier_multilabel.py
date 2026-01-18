"""
Мультиметочный классификатор: эвристика + BERT
Поддерживает категории: ads, crypto, scam, casino
"""
import logging
from typing import Dict, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Попытка загрузить transformers
if TYPE_CHECKING:
    from transformers import pipeline as _pipeline  # type: ignore[import-not-found]

try:
    from transformers import pipeline  # type: ignore[import-not-found]
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    logger.warning("transformers не установлен, BERT недоступен")

# Конфигурация из config.py
try:
    from config import RUBERT_MODEL, USE_RUBERT
except Exception:
    RUBERT_MODEL = 'cointegrated/rubert-tiny2'
    USE_RUBERT = True

# Импорт эвристики
from spam_rules_multilabel import (
    heuristic_multilabel_score,
    heuristic_multilabel_predict
)

# BERT классификатор (будет загружен при первом использовании)
bert_classifier = None
bert_pipeline_type = None  # 'zero-shot-classification' или 'text-classification'

# Категории для классификации
LABELS = ["ads", "crypto", "scam", "casino"]

# Пороги по умолчанию
DEFAULT_THRESHOLDS = {
    "ads": 0.4,
    "crypto": 0.5,
    "scam": 0.5,
    "casino": 0.5
}


def _load_bert_classifier():
    """Ленивая загрузка BERT классификатора."""
    global bert_classifier, bert_pipeline_type
    
    if bert_classifier is not None:
        return bert_classifier, bert_pipeline_type
    
    if not HAS_TRANSFORMERS or not USE_RUBERT:
        logger.debug("BERT недоступен (HAS_TRANSFORMERS=%s, USE_RUBERT=%s)", 
                    HAS_TRANSFORMERS, USE_RUBERT)
        return None, None
    
    try:
        logger.info("Загрузка BERT классификатора: %s", RUBERT_MODEL)
        # Пробуем zero-shot classification (лучше для мультиметочной классификации)
        try:
            bert_classifier = pipeline(
                'zero-shot-classification',
                model=RUBERT_MODEL,
                device=-1  # CPU
            )
            bert_pipeline_type = 'zero-shot-classification'
            logger.info("✓ BERT классификатор загружен (zero-shot-classification)")
        except Exception as e1:
            # Fallback: используем text-classification если zero-shot не поддерживается
            logger.warning("zero-shot-classification не поддерживается: %s", str(e1))
            logger.info("Пробуем text-classification...")
            try:
                bert_classifier = pipeline(
                    'text-classification',
                    model=RUBERT_MODEL,
                    device=-1  # CPU
                )
                bert_pipeline_type = 'text-classification'
                logger.info("✓ BERT классификатор загружен (text-classification)")
            except Exception as e2:
                logger.exception("Не удалось загрузить BERT классификатор: %s", str(e2))
                return None, None
        
        return bert_classifier, bert_pipeline_type
    except Exception as e:
        logger.exception("Не удалось загрузить BERT классификатор: %s", str(e))
        return None, None


def classify_with_bert(text: str, candidate_labels: list = None) -> Dict[str, float]:
    """
    Классификация через BERT (zero-shot или text-classification).
    
    Args:
        text: Текст для классификации
        candidate_labels: Список категорий (по умолчанию: LABELS)
    
    Returns:
        Dict с оценками для каждой категории (0.0 - 1.0)
    """
    if candidate_labels is None:
        candidate_labels = LABELS
    
    classifier, pipeline_type = _load_bert_classifier()
    if classifier is None:
        return {label: 0.0 for label in candidate_labels}
    
    try:
        if pipeline_type == 'zero-shot-classification':
            # Zero-shot классификация возвращает оценки для всех категорий
            result = classifier(text[:512], candidate_labels, multi_label=True)
            
            # Преобразуем результат в словарь
            scores = {}
            if isinstance(result, dict) and "labels" in result and "scores" in result:
                for label, score in zip(result["labels"], result["scores"]):
                    scores[label] = float(score)
            
            # Убеждаемся, что все категории присутствуют
            for label in candidate_labels:
                if label not in scores:
                    scores[label] = 0.0
            
            return scores
        else:
            # Для text-classification модели (rubert-tiny2) используем эвристический подход
            # Классифицируем текст и пытаемся определить категории по меткам
            result = classifier(text[:512])
            
            # text-classification возвращает список с одним результатом
            if isinstance(result, list) and len(result) > 0:
                label = result[0].get('label', '').lower()
                score = result[0].get('score', 0.0)
                
                # Пытаемся сопоставить метку с категориями
                scores = {cat: 0.0 for cat in candidate_labels}
                
                # Простая эвристика: если метка содержит "spam" или "positive", распределяем score
                if 'spam' in label or 'positive' in label or '1' in label:
                    # Если модель определила как спам, распределяем score равномерно
                    # (это не идеально, но лучше чем ничего)
                    for cat in candidate_labels:
                        scores[cat] = score * 0.5  # Умеренная оценка для всех категорий
                else:
                    # Если не спам, все категории получают низкую оценку
                    for cat in candidate_labels:
                        scores[cat] = (1.0 - score) * 0.1
                
                return scores
            else:
                return {label: 0.0 for label in candidate_labels}
    except Exception as e:
        logger.exception("Ошибка при вызове BERT: %s", str(e))
        return {label: 0.0 for label in candidate_labels}


def classify_multilabel(
    text: str,
    use_heuristics: bool = True,
    use_bert: bool = True,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Мультиметочная классификация текста.
    
    Args:
        text: Текст для классификации
        use_heuristics: Использовать эвристику
        use_bert: Использовать BERT
        thresholds: Пороги для каждой категории
    
    Returns:
        Dict с ключами:
            - "scores": Dict[str, float] - оценки для каждой категории
            - "predictions": Dict[str, int] - бинарные предсказания (0/1)
            - "methods": Dict[str, str] - какие методы использовались
    """
    if not text:
        return {
            "scores": {label: 0.0 for label in LABELS},
            "predictions": {label: 0 for label in LABELS},
            "methods": {"heuristics": "skipped", "bert": "skipped"}
        }
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()
    
    # Получаем оценки от эвристики
    heuristic_scores = {}
    if use_heuristics:
        try:
            heuristic_scores = heuristic_multilabel_score(text)
        except Exception as e:
            logger.exception("Ошибка эвристики: %s", str(e))
            heuristic_scores = {label: 0.0 for label in LABELS}
    else:
        heuristic_scores = {label: 0.0 for label in LABELS}
    
    # Получаем оценки от BERT
    bert_scores = {}
    if use_bert:
        try:
            bert_scores = classify_with_bert(text)
        except Exception as e:
            logger.exception("Ошибка BERT: %s", str(e))
            bert_scores = {label: 0.0 for label in LABELS}
    else:
        bert_scores = {label: 0.0 for label in LABELS}
    
    # Объединяем оценки (среднее арифметическое)
    final_scores = {}
    methods_used = {}
    
    for label in LABELS:
        h_score = heuristic_scores.get(label, 0.0)
        b_score = bert_scores.get(label, 0.0)
        
        # Если оба метода доступны, берем среднее
        if use_heuristics and use_bert and h_score > 0 and b_score > 0:
            final_scores[label] = (h_score + b_score) / 2.0
            methods_used[label] = "heuristics+bert"
        elif use_heuristics and h_score > 0:
            final_scores[label] = h_score
            methods_used[label] = "heuristics"
        elif use_bert and b_score > 0:
            final_scores[label] = b_score
            methods_used[label] = "bert"
        else:
            final_scores[label] = 0.0
            methods_used[label] = "none"
    
    # Применяем пороги для бинарных предсказаний
    predictions = {}
    for label in LABELS:
        predictions[label] = 1 if final_scores[label] >= thresholds.get(label, 0.4) else 0
    
    return {
        "scores": final_scores,
        "predictions": predictions,
        "methods": methods_used,
        "heuristic_scores": heuristic_scores,
        "bert_scores": bert_scores
    }


# Функция для обратной совместимости (единая оценка спама)
def classify_text(text: str) -> dict:
    """
    Классификация текста (обратная совместимость).
    Возвращает единую оценку спама.
    """
    result = classify_multilabel(text, use_heuristics=True, use_bert=True)
    
    # Вычисляем общую оценку спама (максимум из всех категорий)
    max_score = max(result["scores"].values())
    
    # Проверяем, есть ли хотя бы одна категория с предсказанием 1
    is_spam = any(result["predictions"].values())
    
    return {
        "is_spam": is_spam,
        "score": max_score,
        "reason": f"multilabel_max={max_score:.3f}",
        "categories": result["predictions"]
    }


if __name__ == "__main__":
    # Тесты
    test_texts = [
        "🔥 УСПЕЙ ЗАБРАТЬ! Мест осталось мало. Моя секретная стратегия p2p арбитража в закрепе: t.me/link",
        "Новое казино с бонусом за регистрацию! Фриспины и джекпот ждут вас!",
        "Гарантированный доход без риска! Схема заработка на крипте. Быстрые деньги!",
        "Обычная новость о событиях в городе. Ничего особенного."
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Текст: {text[:80]}...")
        result = classify_multilabel(text)
        print(f"Оценки: {result['scores']}")
        print(f"Предсказания: {result['predictions']}")
        print(f"Методы: {result['methods']}")
