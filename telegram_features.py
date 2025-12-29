"""Telegram-специфичные фичи для ML классификатора."""
import re
from typing import List
import numpy as np


class MetaFeatureExtractor:
    """Извлекает мета-признаки специфичные для Telegram постов."""
    
    def fit(self, X, y=None):
        """Не требует обучения."""
        return self
    
    def transform(self, X) -> np.ndarray:
        """Извлекает мета-признаки из текстов.
        
        Returns:
            Массив формы (n_samples, n_features) с мета-признаками
        """
        features = []
        
        for text in X:
            text_str = str(text)
            text_lower = text_str.lower()
            
            # 1. Структурные признаки
            length = len(text_str)
            num_lines = text_str.count('\n') + 1
            caps_ratio = sum(1 for c in text_str if c.isupper()) / max(1, length)
            
            # 2. Признаки повторения символов
            repeated_chars = len(re.findall(r'(.)\1{2,}', text_str))  # 3+ повторяющихся символов
            
            # 3. Признаки ссылок
            url_count = len(re.findall(r'https?://|t\.me/', text_lower))
            short_domains = len(re.findall(r'\b(bit\.ly|t\.co|tinyurl|goo\.gl|ow\.ly)', text_lower))
            telegram_links = text_lower.count('t.me/')
            
            # 4. Маркетинговые признаки
            percent_pattern = len(re.findall(r'\d+%', text_str))  # Проценты
            currency_symbols = len(re.findall(r'[$€£₽₴₸]|\d+\s*(руб|р\.|USD|EUR)', text_lower))
            multiplier_pattern = len(re.findall(r'\b[x×]\s*\d+|\d+\s*[x×]\b', text_lower))  # x2, x10
            promo_code = len(re.findall(r'\b[A-Z0-9]{4,}\b', text_str))  # Промокоды (CAPS буквы+цифры)
            
            # 5. Эмодзи и специальные символы
            emoji_count = len(re.findall(r'[🔥💎💰🎰🎲🎯💵💸🚀📈📊⭐✨🎁🎉]', text_str))
            exclamation_count = text_str.count('!')
            question_count = text_str.count('?')
            ellipsis_count = text_str.count('...') + text_str.count('…')
            
            # 6. Ключевые слова (часто встречаются в спаме)
            spam_keywords = sum(1 for word in [
                'скидка', 'акция', 'бонус', 'промокод', 'бесплатно',
                'гарантия', 'быстро', 'сегодня', 'ограничено',
                'подпишись', 'переходи', 'забирай', 'жми'
            ] if word in text_lower)
            
            crypto_keywords = sum(1 for word in [
                'биткоин', 'bitcoin', 'btc', 'эфир', 'ethereum', 'eth',
                'токен', 'token', 'крипта', 'crypto', 'блокчейн', 'blockchain',
                'майнинг', 'mining', 'nft', 'defi', 'airdrop', 'листинг'
            ] if word in text_lower)
            
            scam_keywords = sum(1 for word in [
                'схема', 'гарантирован', 'без вложений', 'быстрый заработок',
                'пассивный доход', 'работа на дому', 'инвестиции', 'прибыль',
                'p2p', 'арбитраж', 'профит'
            ] if word in text_lower)
            
            casino_keywords = sum(1 for word in [
                'казино', 'casino', 'ставки', 'бет', 'bet', 'слоты', 'slots',
                'выигрыш', 'джекпот', 'jackpot', 'рулетка', 'roulette'
            ] if word in text_lower)
            
            # Собираем все признаки в вектор
            feature_vector = [
                length,              # 0: длина текста
                num_lines,           # 1: количество строк
                caps_ratio,          # 2: доля заглавных букв
                repeated_chars,      # 3: повторяющиеся символы
                url_count,           # 4: количество URL
                short_domains,       # 5: короткие домены
                telegram_links,      # 6: ссылки на Telegram
                percent_pattern,     # 7: проценты
                currency_symbols,    # 8: валюты
                multiplier_pattern,  # 9: множители (x2, x10)
                promo_code,          # 10: промокоды
                emoji_count,         # 11: эмодзи
                exclamation_count,   # 12: восклицательные знаки
                question_count,      # 13: вопросительные знаки
                ellipsis_count,      # 14: многоточия
                spam_keywords,       # 15: спам ключевые слова
                crypto_keywords,     # 16: крипто ключевые слова
                scam_keywords,       # 17: скам ключевые слова
                casino_keywords,     # 18: казино ключевые слова
            ]
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)

