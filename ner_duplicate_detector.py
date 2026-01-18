"""
Модуль для обнаружения дубликатов постов на основе векторных эмбеддингов.
Использует sentence-transformers для создания семантических эмбеддингов
и FAISS для быстрого поиска похожих постов.
"""
import logging
import time
import warnings
from collections import defaultdict
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger("tg-analyzer.ner")

# Подавляем предупреждения от sentence-transformers о entailment label
warnings.filterwarnings("ignore", message=".*entailment.*")
warnings.filterwarnings("ignore", message=".*Failed to determine.*")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers не установлена. Детектор дубликатов будет отключен.")

try:
    import faiss  # type: ignore[import-not-found]
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS не установлена. Будет использован медленный поиск.")


@dataclass
class PostEmbedding:
    """Эмбеддинг поста."""
    embedding: np.ndarray  # Векторное представление текста
    media_hash: Optional[str]  # Хеш медиа (file_id или другой идентификатор)
    timestamp: float  # Время создания профиля
    message_id: int  # ID сообщения для логирования
    channel_id: int  # ID канала


class SemanticDuplicateDetector:
    """
    Детектор дубликатов постов на основе семантических эмбеддингов.
    Использует косинусное сходство для сравнения постов.
    """
    
    def __init__(
        self, 
        ttl_hours: int = 8, 
        similarity_threshold: float = 0.70,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Инициализация детектора дубликатов.
        
        Args:
            ttl_hours: Время жизни кэша в часах (по умолчанию 4 часа)
            similarity_threshold: Порог схожести для определения дубликата (0.0-1.0)
            model_name: Название модели для эмбеддингов
        """
        self.ttl_seconds = ttl_hours * 3600
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        
        # Кэш постов: {channel_id: [PostEmbedding, ...]}
        self.posts_cache: Dict[int, List[PostEmbedding]] = defaultdict(list)
        
        # FAISS индексы для быстрого поиска: {channel_id: faiss.Index}
        self.faiss_indices: Dict[int, Optional[faiss.Index]] = {}
        
        # Инициализация модели эмбеддингов
        self.model = None
        self.embedding_dim = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info("Инициализация модели эмбеддингов: %s...", model_name)
                # Подавляем предупреждения при загрузке модели
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = SentenceTransformer(model_name)
                # Получаем размерность эмбеддингов
                test_embedding = self.model.encode("test", convert_to_numpy=True, show_progress_bar=False)
                self.embedding_dim = len(test_embedding)
                logger.info("Модель эмбеддингов инициализирована успешно (размерность: %d)", self.embedding_dim)
            except Exception as e:
                logger.exception("Ошибка инициализации модели эмбеддингов: %s", e)
                self.model = None
        else:
            logger.warning("sentence-transformers не установлена. Детектор дубликатов отключен.")
    
    def get_media_hash(self, media) -> Optional[str]:
        """
        Получает хеш/идентификатор медиа для сравнения.
        В Telethon медиа имеет структуру: MessageMediaPhoto, MessageMediaDocument и т.д.
        
        Args:
            media: Объект медиа из Telegram (Telethon)
            
        Returns:
            Хеш медиа или None
        """
        if not media:
            return None
        
        try:
            # В Telethon структура медиа:
            # MessageMediaPhoto -> photo (Photo объект) -> id
            # MessageMediaDocument -> document (Document объект) -> id
            
            # Проверяем тип медиа по имени класса
            media_type = type(media).__name__
            
            # MessageMediaPhoto
            if media_type == 'MessageMediaPhoto' and hasattr(media, 'photo'):
                if hasattr(media.photo, 'id'):
                    return f"photo_{media.photo.id}"
                # Альтернативно используем access_hash если есть
                if hasattr(media.photo, 'access_hash'):
                    return f"photo_{getattr(media.photo, 'id', 0)}_{media.photo.access_hash}"
            
            # MessageMediaDocument
            if media_type == 'MessageMediaDocument' and hasattr(media, 'document'):
                if hasattr(media.document, 'id'):
                    return f"doc_{media.document.id}"
                # Альтернативно используем access_hash если есть
                if hasattr(media.document, 'access_hash'):
                    return f"doc_{getattr(media.document, 'id', 0)}_{media.document.access_hash}"
            
            # MessageMediaGeo, MessageMediaVenue и т.д. - используем тип
            if media_type.startswith('MessageMedia'):
                # Пытаемся найти любой ID в объекте
                for attr in ['id', 'photo_id', 'document_id', 'file_id']:
                    if hasattr(media, attr):
                        value = getattr(media, attr)
                        if value:
                            return f"{media_type.lower()}_{value}"
            
            # В крайнем случае используем строковое представление типа
            return f"media_{media_type}"
            
        except Exception as e:
            logger.debug("Ошибка при получении хеша медиа: %s", e)
            return None
    
    def create_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Создает векторное представление (эмбеддинг) текста.
        
        Args:
            text: Текст для создания эмбеддинга
            
        Returns:
            Вектор эмбеддинга или None
        """
        if not self.model:
            return None
        
        if not text or not text.strip():
            return None
        
        try:
            # Создаем эмбеддинг
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.exception("Ошибка при создании эмбеддинга: %s", e)
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами.
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
            
        Returns:
            Косинусное сходство (0.0 - 1.0)
        """
        # Для нормализованных векторов косинусное сходство = скалярное произведение
        return float(np.dot(vec1, vec2))
    
    def cleanup_old_posts(self, channel_id: int):
        """Удаляет старые посты из кэша (старше TTL) и пересоздает FAISS индекс."""
        current_time = time.time()
        old_count = len(self.posts_cache[channel_id])
        
        self.posts_cache[channel_id] = [
            post for post in self.posts_cache[channel_id]
            if current_time - post.timestamp < self.ttl_seconds
        ]
        
        new_count = len(self.posts_cache[channel_id])
        if old_count != new_count:
            logger.debug("Очищено %d старых постов из кэша канала %s", old_count - new_count, channel_id)
            # Пересоздаем FAISS индекс
            self._rebuild_faiss_index(channel_id)
    
    def _rebuild_faiss_index(self, channel_id: int):
        """Пересоздает FAISS индекс для канала."""
        if not HAS_FAISS or not self.embedding_dim:
            return
        
        posts = self.posts_cache[channel_id]
        if not posts:
            self.faiss_indices[channel_id] = None
            return
        
        try:
            # Создаем FAISS индекс для косинусного поиска
            # Используем InnerProduct для нормализованных векторов (эквивалентно косинусному сходству)
            index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Добавляем все эмбеддинги в индекс
            embeddings_matrix = np.vstack([post.embedding for post in posts])
            index.add(embeddings_matrix.astype('float32'))
            
            self.faiss_indices[channel_id] = index
            logger.debug("FAISS индекс пересоздан для канала %s (%d постов)", channel_id, len(posts))
        except Exception as e:
            logger.exception("Ошибка при создании FAISS индекса: %s", e)
            self.faiss_indices[channel_id] = None
    
    def _find_duplicate_slow(
        self, 
        current_embedding: np.ndarray,
        current_media_hash: Optional[str],
        channel_id: int
    ) -> Tuple[float, Optional[int]]:
        """
        Медленный поиск дубликата (без FAISS).
        Используется если FAISS не установлена.
        """
        max_similarity = 0.0
        duplicate_message_id = None
        
        for cached_post in self.posts_cache[channel_id]:
            # Вычисляем косинусное сходство
            similarity = self.cosine_similarity(current_embedding, cached_post.embedding)
            
            # Бонус за совпадение медиа
            if current_media_hash and cached_post.media_hash:
                if current_media_hash == cached_post.media_hash:
                    similarity = min(1.0, similarity + 0.1)  # Добавляем 0.1 за совпадение медиа
            
            if similarity > max_similarity:
                max_similarity = similarity
                duplicate_message_id = cached_post.message_id
            
            # Если нашли очень похожий пост, можно прервать поиск
            if similarity >= self.similarity_threshold:
                break
        
        return max_similarity, duplicate_message_id
    
    def _find_duplicate_faiss(
        self,
        current_embedding: np.ndarray,
        current_media_hash: Optional[str],
        channel_id: int
    ) -> Tuple[float, Optional[int]]:
        """
        Быстрый поиск дубликата с использованием FAISS.
        """
        index = self.faiss_indices.get(channel_id)
        if not index:
            # Если индекс не создан, используем медленный поиск
            return self._find_duplicate_slow(current_embedding, current_media_hash, channel_id)
        
        try:
            # Ищем топ-5 наиболее похожих постов
            k = min(5, len(self.posts_cache[channel_id]))
            if k == 0:
                return 0.0, None
            
            # Поиск в FAISS
            query_vector = current_embedding.reshape(1, -1).astype('float32')
            similarities, indices = index.search(query_vector, k)
            
            max_similarity = 0.0
            duplicate_message_id = None
            
            # Проверяем найденные посты
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < 0 or idx >= len(self.posts_cache[channel_id]):
                    continue
                
                cached_post = self.posts_cache[channel_id][idx]
                similarity_float = float(similarity)
                
                # Бонус за совпадение медиа
                if current_media_hash and cached_post.media_hash:
                    if current_media_hash == cached_post.media_hash:
                        similarity_float = min(1.0, similarity_float + 0.1)
                
                if similarity_float > max_similarity:
                    max_similarity = similarity_float
                    duplicate_message_id = cached_post.message_id
                
                # Если нашли очень похожий пост, можно прервать поиск
                if similarity_float >= self.similarity_threshold:
                    break
            
            return max_similarity, duplicate_message_id
            
        except Exception as e:
            logger.exception("Ошибка при поиске в FAISS: %s", e)
            # Fallback на медленный поиск
            return self._find_duplicate_slow(current_embedding, current_media_hash, channel_id)
    
    def is_duplicate(
        self, 
        text: str, 
        channel_id: int, 
        message_id: int,
        media=None
    ) -> Tuple[bool, float, Optional[int]]:
        """
        Проверяет, является ли пост дубликатом.
        
        Args:
            text: Текст поста
            channel_id: ID канала
            message_id: ID сообщения
            media: Медиа поста (опционально)
            
        Returns:
            Tuple[is_duplicate, similarity_score, duplicate_message_id]
            - is_duplicate: True если пост является дубликатом
            - similarity_score: Коэффициент схожести (0.0-1.0)
            - duplicate_message_id: ID сообщения-дубликата (если найден)
        """
        if not self.model:
            return False, 0.0, None
        
        # Очищаем старые посты
        self.cleanup_old_posts(channel_id)
        
        # Создаем эмбеддинг для текущего поста
        current_embedding = self.create_embedding(text)
        if current_embedding is None:
            logger.debug("Не удалось создать эмбеддинг для сообщения %s/%s", channel_id, message_id)
            return False, 0.0, None
        
        # Получаем хеш медиа
        current_media_hash = self.get_media_hash(media)
        
        # Ищем дубликат
        if HAS_FAISS and self.embedding_dim:
            max_similarity, duplicate_message_id = self._find_duplicate_faiss(
                current_embedding, current_media_hash, channel_id
            )
        else:
            max_similarity, duplicate_message_id = self._find_duplicate_slow(
                current_embedding, current_media_hash, channel_id
            )
        
        is_duplicate = max_similarity >= self.similarity_threshold
        
        if is_duplicate:
            # Находим пост-дубликат для логирования медиа
            media_match = False
            if duplicate_message_id:
                for post in self.posts_cache[channel_id]:
                    if post.message_id == duplicate_message_id:
                        media_match = (current_media_hash == post.media_hash) if current_media_hash and post.media_hash else False
                        break
            
            logger.info(
                "Найден дубликат: %s/%s похож на %s/%s (similarity=%.3f, media_match=%s)",
                channel_id, message_id, channel_id, duplicate_message_id, max_similarity, media_match
            )
        elif max_similarity > 0:
            logger.debug(
                "Сообщение %s/%s не является дубликатом (similarity=%.3f)",
                channel_id, message_id, max_similarity
            )
        
        # Добавляем текущий пост в кэш (даже если он дубликат)
        post_embedding = PostEmbedding(
            embedding=current_embedding,
            media_hash=current_media_hash,
            timestamp=time.time(),
            message_id=message_id,
            channel_id=channel_id
        )
        self.posts_cache[channel_id].append(post_embedding)
        
        # Обновляем FAISS индекс
        if HAS_FAISS and self.embedding_dim:
            self._rebuild_faiss_index(channel_id)
        
        return is_duplicate, max_similarity, duplicate_message_id


# Глобальный экземпляр детектора
_detector: Optional[SemanticDuplicateDetector] = None


def get_ner_detector(ttl_hours: int = 4, similarity_threshold: float = 0.85) -> SemanticDuplicateDetector:
    """
    Получает или создает глобальный экземпляр детектора дубликатов.
    
    Args:
        ttl_hours: Время жизни кэша в часах
        similarity_threshold: Порог схожести для определения дубликата
        
    Returns:
        Экземпляр SemanticDuplicateDetector
    """
    global _detector
    if _detector is None:
        _detector = SemanticDuplicateDetector(ttl_hours=ttl_hours, similarity_threshold=similarity_threshold)
    return _detector
