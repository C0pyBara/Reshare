"""Финальный pipeline классификации спама: Heuristics -> ML -> LLM."""
import logging
from typing import Dict, Optional

from spam_rules import heuristic_spam_score
from ml_model import SpamMLClassifier
from spam_model import classify_parallel

logger = logging.getLogger(__name__)


class SpamPipeline:
    """Финальный pipeline классификации спама с тремя этапами."""
    
    def __init__(self):
        """Инициализирует pipeline."""
        self.heuristics_available = True
        self.ml_classifier = SpamMLClassifier()
        self.ml_available = self.ml_classifier.is_available()
        self.llm_available = True  # LLM всегда доступен через spam_model
        
        logger.info("SpamPipeline инициализирован:")
        logger.info(f"  Heuristics: {'✓' if self.heuristics_available else '✗'}")
        logger.info(f"  ML Classifier: {'✓' if self.ml_available else '✗'}")
        logger.info(f"  LLM: {'✓' if self.llm_available else '✗'}")
    
    async def classify_async(self, text: str) -> Dict[str, any]:
        """Классифицирует текст через pipeline: Heuristics -> ML -> LLM.
        
        Args:
            text: Текст для классификации
            
        Returns:
            Словарь с результатами:
            {
                "decision": "SPAM" | "NOT_SPAM",
                "method": "heuristics" | "ml" | "llm",
                "heuristic_score": float,
                "ml_probability": Optional[float],
                "llm_score": Optional[float],
                "confidence": "high" | "medium" | "low"
            }
        """
        if not text or not text.strip():
            return {
                "decision": "NOT_SPAM",
                "method": "heuristics",
                "heuristic_score": 0.0,
                "ml_probability": None,
                "llm_score": None,
                "confidence": "high"
            }
        
        # ЭТАП 1: Heuristics (быстрая проверка)
        h_score = heuristic_spam_score(text)
        
        # Явно не реклама - уверенное решение
        if h_score <= 2:
            return {
                "decision": "NOT_SPAM",
                "method": "heuristics",
                "heuristic_score": h_score,
                "ml_probability": None,
                "llm_score": None,
                "confidence": "high"
            }
        
        # Явно реклама - уверенное решение
        if h_score >= 8:
            return {
                "decision": "SPAM",
                "method": "heuristics",
                "heuristic_score": h_score,
                "ml_probability": None,
                "llm_score": None,
                "confidence": "high"
            }
        
        # ЭТАП 2: ML классификатор (если доступен)
        if self.ml_available:
            ml_prob = self.ml_classifier.predict(text)
            
            # ML уверен - не спам
            if ml_prob < 0.3:
                return {
                    "decision": "NOT_SPAM",
                    "method": "ml",
                    "heuristic_score": h_score,
                    "ml_probability": ml_prob,
                    "llm_score": None,
                    "confidence": "high"
                }
            
            # ML уверен - спам
            if ml_prob > 0.7:
                return {
                    "decision": "SPAM",
                    "method": "ml",
                    "heuristic_score": h_score,
                    "ml_probability": ml_prob,
                    "llm_score": None,
                    "confidence": "high"
                }
            
            # ML не уверен (0.3 <= ml_prob <= 0.7) - переходим к LLM
            logger.debug(f"ML не уверен (prob={ml_prob:.3f}), используем LLM")
        else:
            # ML недоступен - переходим к LLM
            ml_prob = None
            logger.debug("ML недоступен, используем LLM")
        
        # ЭТАП 3: LLM (только если предыдущие методы не уверены)
        logger.info("Использование LLM для классификации (ML не уверен или недоступен)")
        llm_results = None
        try:
            # Используем существующую функцию classify_parallel (async)
            from spam_model import classify_parallel
            llm_results = await classify_parallel(text)
        except Exception as e:
            logger.exception(f"Ошибка при вызове LLM: {e}")
            # При ошибке LLM используем ML или heuristics
            if ml_prob is not None:
                decision = "SPAM" if ml_prob > 0.5 else "NOT_SPAM"
                return {
                    "decision": decision,
                    "method": "ml",
                    "heuristic_score": h_score,
                    "ml_probability": ml_prob,
                    "llm_score": None,
                    "confidence": "medium"
                }
            else:
                # Используем heuristics как fallback
                decision = "SPAM" if h_score >= 5 else "NOT_SPAM"
                return {
                    "decision": decision,
                    "method": "heuristics",
                    "heuristic_score": h_score,
                    "ml_probability": None,
                    "llm_score": None,
                    "confidence": "medium"
                }
        
        # Обрабатываем результаты LLM
        if llm_results and len(llm_results) > 0:
            # Берем средний score из всех методов LLM
            scores = [r["score"] for r in llm_results if "score" in r]
            if scores:
                avg_llm_score = sum(scores) / len(scores)
                decision = "SPAM" if avg_llm_score >= 0.6 else "NOT_SPAM"
                return {
                    "decision": decision,
                    "method": "llm",
                    "heuristic_score": h_score,
                    "ml_probability": ml_prob,
                    "llm_score": avg_llm_score,
                    "confidence": "medium"
                }
        
        # Если LLM не вернул результаты, используем fallback
        if ml_prob is not None:
            decision = "SPAM" if ml_prob > 0.5 else "NOT_SPAM"
            return {
                "decision": decision,
                "method": "ml",
                "heuristic_score": h_score,
                "ml_probability": ml_prob,
                "llm_score": None,
                "confidence": "medium"
            }
        else:
            decision = "SPAM" if h_score >= 5 else "NOT_SPAM"
            return {
                "decision": decision,
                "method": "heuristics",
                "heuristic_score": h_score,
                "ml_probability": None,
                "llm_score": None,
                "confidence": "low"
            }
    
    def classify(self, text: str) -> Dict[str, any]:
        """Синхронная обертка для classify_async.
        
        Используйте classify_async в async контексте для лучшей производительности.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если loop уже запущен, это ошибка - используйте classify_async
                raise RuntimeError("Event loop уже запущен. Используйте classify_async() вместо classify()")
            return loop.run_until_complete(self.classify_async(text))
        except RuntimeError:
            # Нет event loop, создаем новый
            return asyncio.run(self.classify_async(text))

