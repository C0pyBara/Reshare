"""Финальный multilabel pipeline классификации спама: Heuristics -> ML -> LLM (Active Learning)."""
import logging
import asyncio
import json
import time
from typing import Dict, Optional
from pathlib import Path

from spam_rules import heuristic_spam_score
from ml_model_multilabel import SpamMLMultilabelClassifier, LABELS
from spam_model import classify_parallel

logger = logging.getLogger(__name__)

# Пороги неопределенности для active learning
UNCERTAIN_LOW = 0.4
UNCERTAIN_HIGH = 0.6

# Директория для сохранения данных active learning
DATA_DIR = Path(__file__).parent / "data"
ACTIVE_LEARNING_FILE = DATA_DIR / "active_learning_data.jsonl"
DATA_DIR.mkdir(exist_ok=True)


class SpamMultilabelPipeline:
    """Финальный multilabel pipeline классификации спама с active learning."""
    
    def __init__(self):
        """Инициализирует pipeline."""
        self.heuristics_available = True
        self.ml_classifier = SpamMLMultilabelClassifier()
        self.ml_available = self.ml_classifier.is_available()
        self.llm_available = True  # LLM всегда доступен через spam_model
        
        logger.info("SpamMultilabelPipeline инициализирован:")
        logger.info(f"  Heuristics: {'✓' if self.heuristics_available else '✗'}")
        logger.info(f"  ML Classifier: {'✓' if self.ml_available else '✗'}")
        logger.info(f"  LLM: {'✓' if self.llm_available else '✗'}")
    
    async def classify_async(self, text: str, use_active_learning: bool = True) -> Dict[str, any]:
        """Классифицирует текст через pipeline: Heuristics -> ML -> LLM (Active Learning).
        
        Args:
            text: Текст для классификации
            use_active_learning: Использовать active learning (сохранять данные для LLM)
            
        Returns:
            Словарь с результатами:
            {
                "labels": {"ads": 0/1, "crypto": 0/1, "scam": 0/1, "casino": 0/1},
                "method": "heuristics" | "ml" | "llm",
                "heuristic_score": float,
                "ml_probabilities": Dict[str, float],
                "llm_labels": Optional[Dict[str, int]],
                "confidence": "high" | "medium" | "low"
            }
        """
        if not text or not text.strip():
            return {
                "labels": {label: 0 for label in LABELS},
                "method": "heuristics",
                "heuristic_score": 0.0,
                "ml_probabilities": None,
                "llm_labels": None,
                "confidence": "high"
            }
        
        # ЭТАП 1: Heuristics (быстрая проверка)
        h_score = heuristic_spam_score(text)
        
        # Явно не реклама - уверенное решение
        if h_score <= 2:
            return {
                "labels": {label: 0 for label in LABELS},
                "method": "heuristics",
                "heuristic_score": h_score,
                "ml_probabilities": None,
                "llm_labels": None,
                "confidence": "high"
            }
        
        # ЭТАП 2: ML классификатор (если доступен)
        ml_probs = None
        if self.ml_available:
            ml_probs = self.ml_classifier.predict_proba(text)
            
            # Проверяем, уверена ли ML модель
            if not self.ml_classifier.needs_llm(ml_probs, UNCERTAIN_LOW, UNCERTAIN_HIGH):
                # ML уверена - используем ее предсказания
                ml_labels = self.ml_classifier.predict(text)
                return {
                    "labels": ml_labels,
                    "method": "ml",
                    "heuristic_score": h_score,
                    "ml_probabilities": ml_probs,
                    "llm_labels": None,
                    "confidence": "high"
                }
            else:
                logger.debug(f"ML не уверена, используем LLM (probs: {ml_probs})")
        
        # ЭТАП 3: LLM (Active Learning - только если ML не уверена)
        logger.info("Использование LLM для классификации (ML не уверена)")
        
        llm_labels = await self._classify_with_llm(text)
        
        # Сохраняем данные для active learning
        if use_active_learning and ml_probs:
            self._save_active_learning_data(text, ml_probs, llm_labels)
        
        return {
            "labels": llm_labels if llm_labels else {label: 0 for label in LABELS},
            "method": "llm",
            "heuristic_score": h_score,
            "ml_probabilities": ml_probs,
            "llm_labels": llm_labels,
            "confidence": "medium"
        }
    
    async def _classify_with_llm(self, text: str) -> Optional[Dict[str, int]]:
        """Классифицирует текст с помощью LLM.
        
        Returns:
            Словарь с метками или None при ошибке
        """
        import re
        import json
        
        prompt = f"""Определи, относится ли текст к категориям:
- ads (реклама)
- crypto (криптовалюты, блокчейн, токены)
- scam (мошенничество, обман, схема заработка)
- casino (казино, ставки, азартные игры)

Ответ СТРОГО в JSON формате:
{{"ads":0,"crypto":0,"scam":0,"casino":0}}

Текст:
{text[:1000]}
"""
        
        try:
            results = await classify_parallel(prompt)
            
            # Пытаемся извлечь JSON из результатов
            if results and len(results) > 0:
                for result in results:
                    reason = result.get("reason", "")
                    
                    # Ищем JSON паттерн
                    json_match = re.search(r'\{[^}]*"ads"[^}]*"crypto"[^}]*"scam"[^}]*"casino"[^}]*\}', reason)
                    if json_match:
                        try:
                            labels_dict = json.loads(json_match.group(0))
                            # Валидируем и возвращаем
                            valid_labels = {}
                            for label in LABELS:
                                value = labels_dict.get(label, 0)
                                valid_labels[label] = 1 if value in (1, True, "1") else 0
                            return valid_labels
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.debug(f"Ошибка при вызове LLM: {e}")
        
        # Если не удалось получить метки от LLM, используем эвристику
        return self._fallback_labels(text)
    
    def _fallback_labels(self, text: str) -> Dict[str, int]:
        """Fallback классификация на основе эвристики.
        
        Простая эвристика на основе ключевых слов.
        """
        text_lower = text.lower()
        labels = {label: 0 for label in LABELS}
        
        # Простые правила для каждой категории
        if any(word in text_lower for word in ['реклам', 'реклам', 'акция', 'скидка', 'бонус', 'промокод']):
            labels["ads"] = 1
        
        if any(word in text_lower for word in ['крипт', 'биткоин', 'bitcoin', 'btc', 'токен', 'token', 'блокчейн']):
            labels["crypto"] = 1
        
        if any(word in text_lower for word in ['скам', 'схем', 'мошенни', 'обман', 'p2p', 'арбитраж']):
            labels["scam"] = 1
        
        if any(word in text_lower for word in ['казино', 'casino', 'ставки', 'bet', 'слоты']):
            labels["casino"] = 1
        
        return labels
    
    def _save_active_learning_data(self, text: str, ml_probs: Dict[str, float], llm_labels: Dict[str, int]):
        """Сохраняет данные для active learning (обучения модели).
        
        Args:
            text: Текст сообщения
            ml_probs: Вероятности от ML модели
            llm_labels: Метки от LLM
        """
        record = {
            "text": text,
            "ml_probs": ml_probs,
            "llm_labels": llm_labels,
            "timestamp": time.time()
        }
        
        try:
            with open(ACTIVE_LEARNING_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Ошибка сохранения active learning данных: {e}")
    
    def classify(self, text: str, use_active_learning: bool = True) -> Dict[str, any]:
        """Синхронная обертка для classify_async.
        
        Используйте classify_async в async контексте для лучшей производительности.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Event loop уже запущен. Используйте classify_async() вместо classify()")
            return loop.run_until_complete(self.classify_async(text, use_active_learning))
        except RuntimeError:
            return asyncio.run(self.classify_async(text, use_active_learning))

