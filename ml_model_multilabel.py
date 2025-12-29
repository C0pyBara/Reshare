"""Multilabel ML классификатор спама на основе TF-IDF + Meta Features + OneVsRest LR."""
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_FILE = MODELS_DIR / "spam_ml_multilabel.pkl"

LABELS = ["ads", "crypto", "scam", "casino"]


class SpamMLMultilabelClassifier:
    """Multilabel ML классификатор для определения категорий спама в текстах."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Инициализирует ML классификатор.
        
        Args:
            model_path: Путь к файлу модели. Если None, используется модель по умолчанию.
        """
        self.pipeline = None
        self.thresholds = None
        self.labels = LABELS
        self.model_path = model_path or MODEL_FILE
        self._load_model()
    
    def _load_model(self):
        """Загружает обученную модель."""
        if not self.model_path.exists():
            logger.warning(f"ML модель не найдена: {self.model_path}")
            logger.warning("Обучите модель: python train_ml_multilabel.py")
            return
        
        try:
            model_data = joblib.load(self.model_path)
            self.pipeline = model_data.get("pipeline")
            self.thresholds = model_data.get("thresholds", {label: 0.5 for label in LABELS})
            self.labels = model_data.get("labels", LABELS)
            logger.info(f"ML модель загружена: {self.model_path}")
        except Exception as e:
            logger.exception(f"Ошибка загрузки ML модели: {e}")
            self.pipeline = None
    
    def is_available(self) -> bool:
        """Проверяет, доступна ли модель."""
        return self.pipeline is not None
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """Предсказывает вероятности для каждой метки.
        
        Args:
            text: Текст для классификации
            
        Returns:
            Словарь с вероятностями для каждой метки.
            Если модель недоступна, возвращает нейтральные вероятности (0.5).
        """
        if not self.pipeline:
            return {label: 0.5 for label in self.labels}
        
        if not text or not text.strip():
            return {label: 0.0 for label in self.labels}
        
        try:
            # Получаем вероятности от модели
            # Для OneVsRestClassifier predict_proba возвращает список массивов
            proba_list = self.pipeline.predict_proba([text])
            
            # Для OneVsRestClassifier predict_proba возвращает список массивов
            # Каждый элемент списка - это массив (n_samples, 2) с [P(класс=0), P(класс=1)]
            
            probabilities = {}
            if isinstance(proba_list, (list, tuple)) and len(proba_list) == len(self.labels):
                # OneVsRest возвращает список из len(labels) массивов
                for i, label in enumerate(self.labels):
                    class_proba = proba_list[i]  # Массив (1, 2) для одного примера
                    if isinstance(class_proba, np.ndarray):
                        if class_proba.ndim == 2 and class_proba.shape[1] >= 2:
                            probabilities[label] = float(class_proba[0][1])  # P(label=1)
                        elif class_proba.ndim == 1 and len(class_proba) >= 2:
                            probabilities[label] = float(class_proba[1])  # P(label=1)
                        else:
                            probabilities[label] = 0.5
                    else:
                        probabilities[label] = 0.5
            else:
                # Fallback - если формат другой
                for i, label in enumerate(self.labels):
                    probabilities[label] = 0.5
            
            return probabilities
        except Exception as e:
            logger.exception(f"Ошибка предсказания ML модели: {e}")
            return {label: 0.5 for label in self.labels}
    
    def predict(self, text: str) -> Dict[str, int]:
        """Предсказывает метки с применением порогов.
        
        Args:
            text: Текст для классификации
            
        Returns:
            Словарь с бинарными метками (0 или 1) для каждой категории.
        """
        probabilities = self.predict_proba(text)
        predictions = {}
        
        for label in self.labels:
            threshold = self.thresholds.get(label, 0.5) if self.thresholds else 0.5
            predictions[label] = 1 if probabilities.get(label, 0.0) >= threshold else 0
        
        return predictions
    
    def needs_llm(self, probs: Dict[str, float], uncertain_low: float = 0.4, uncertain_high: float = 0.6) -> bool:
        """Определяет, нужно ли использовать LLM для классификации.
        
        LLM используется, если хотя бы одна вероятность находится в неопределенной зоне.
        
        Args:
            probs: Словарь с вероятностями для каждой метки
            uncertain_low: Нижний порог неопределенности
            uncertain_high: Верхний порог неопределенности
            
        Returns:
            True если нужен LLM, False иначе
        """
        return any(
            uncertain_low < probs.get(label, 0.5) < uncertain_high
            for label in self.labels
        )

