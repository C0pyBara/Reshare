"""ML классификатор спама на основе TF-IDF + Logistic Regression."""
import joblib
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_FILE = MODELS_DIR / "spam_ml.pkl"


class SpamMLClassifier:
    """ML классификатор для определения спама в текстах."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Инициализирует ML классификатор.
        
        Args:
            model_path: Путь к файлу модели. Если None, используется модель по умолчанию.
        """
        self.model = None
        self.model_path = model_path or MODEL_FILE
        self._load_model()
    
    def _load_model(self):
        """Загружает обученную модель."""
        if not self.model_path.exists():
            logger.warning(f"ML модель не найдена: {self.model_path}")
            logger.warning("Обучите модель: python train_ml.py")
            return
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"ML модель загружена: {self.model_path}")
        except Exception as e:
            logger.exception(f"Ошибка загрузки ML модели: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Проверяет, доступна ли модель."""
        return self.model is not None
    
    def predict(self, text: str) -> float:
        """Предсказывает вероятность того, что текст является спамом.
        
        Args:
            text: Текст для классификации
            
        Returns:
            Вероятность спама (0.0 - 1.0). Если модель недоступна, возвращает 0.5.
        """
        if not self.model:
            return 0.5  # Нейтральная вероятность если модель недоступна
        
        if not text or not text.strip():
            return 0.0  # Пустой текст = не спам
        
        try:
            # Получаем вероятность класса "спам" (label=1)
            proba = self.model.predict_proba([text])[0]
            # proba[0] = вероятность класса 0 (не спам)
            # proba[1] = вероятность класса 1 (спам)
            spam_probability = float(proba[1])
            return spam_probability
        except Exception as e:
            logger.exception(f"Ошибка предсказания ML модели: {e}")
            return 0.5  # Нейтральная вероятность при ошибке

