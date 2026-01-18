import asyncio
import logging
import re
import sys
from asyncio import Queue
from collections import deque

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageEntityUrl, MessageEntityTextUrl, MessageMediaWebPage, MessageMediaEmpty

from config import (
    API_ID,
    API_HASH,
    SESSION_NAME,
    CHANNELS,
    CHECK_INTERVAL,
    WORKERS,
    QUEUE_MAXSIZE,
)

from classifier_multilabel import classify_multilabel
from data_logger import log_message_for_ml
from ner_duplicate_detector import get_ner_detector

logging.basicConfig(
    level=logging.INFO,
    format=(
        "\n[%(asctime)s] %(levelname)s | %(name)s\n"
        "  %(message)s"
    ),
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("tg-analyzer")

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

message_queue = Queue(maxsize=QUEUE_MAXSIZE)
last_ids = {}

PROCESSED_CACHE_SIZE = 10_000
processed_ids = deque(maxlen=PROCESSED_CACHE_SIZE)

TARGET_GROUP_ID = -1003172147499  # ID группы для публикации
SPAM_MONITOR_USER_ID = 534239907  # ID пользователя для мониторинга спама


def can_send_as_file(media):
    """
    Проверяет, можно ли отправить медиа как файл.
    Некоторые типы медиа (например, MessageMediaWebPage) нельзя отправить как файл.
    """
    if not media:
        return False
    
    # MessageMediaWebPage и MessageMediaEmpty нельзя отправить как файл
    if isinstance(media, (MessageMediaWebPage, MessageMediaEmpty)):
        return False
    
    # Остальные типы медиа (фото, документы, видео и т.д.) можно отправить
    return True


def utf16_len(text):
    """Подсчитывает длину строки в UTF-16 code units."""
    return len(text.encode('utf-16-le')) // 2


def utf16_to_python_pos(text, utf16_offset):
    """Конвертирует UTF-16 offset в позицию в Python строке."""
    if utf16_offset <= 0:
        return 0
    
    # Итерируемся по символам и подсчитываем UTF-16 единицы
    utf16_count = 0
    for i, char in enumerate(text):
        # Каждый символ занимает 1 или 2 UTF-16 code units (surrogate pairs)
        char_utf16_len = len(char.encode('utf-16-le')) // 2
        if utf16_count + char_utf16_len > utf16_offset:
            return i
        utf16_count += char_utf16_len
        if utf16_count >= utf16_offset:
            return i + 1
    
    return len(text)


def python_to_utf16_offset(text, python_pos):
    """Конвертирует позицию в Python строке в UTF-16 offset."""
    if python_pos >= len(text):
        return utf16_len(text)
    
    substring = text[:python_pos]
    return utf16_len(substring)


def remove_hyperlinks(text, entities):
    """
    Удаляет все гиперссылки из текста сообщения.
    Возвращает очищенный текст без ссылок.
    entities используют UTF-16 offsets, поэтому нужно правильно конвертировать.
    """
    if not entities:
        return text
    
    # Создаем список диапазонов ссылок в UTF-16 offsets
    link_ranges_utf16 = []
    for entity in entities:
        if isinstance(entity, (MessageEntityUrl, MessageEntityTextUrl)):
            link_ranges_utf16.append((entity.offset, entity.offset + entity.length))
    
    if not link_ranges_utf16:
        return text
    
    # Сортируем по начальной позиции
    link_ranges_utf16.sort(key=lambda x: x[0])
    
    # Удаляем перекрывающиеся диапазоны
    merged_ranges_utf16 = []
    for start, end in link_ranges_utf16:
        if merged_ranges_utf16 and start <= merged_ranges_utf16[-1][1]:
            # Перекрывается с предыдущим диапазоном
            merged_ranges_utf16[-1] = (merged_ranges_utf16[-1][0], max(merged_ranges_utf16[-1][1], end))
        else:
            merged_ranges_utf16.append((start, end))
    
    # Конвертируем UTF-16 offsets в позиции Python строки
    link_ranges_python = []
    for start_utf16, end_utf16 in merged_ranges_utf16:
        start_python = utf16_to_python_pos(text, start_utf16)
        end_python = utf16_to_python_pos(text, end_utf16)
        link_ranges_python.append((start_python, end_python))
    
    # Строим новый текст, исключая ссылки
    result = []
    last_pos = 0
    
    for start, end in link_ranges_python:
        # Добавляем текст до ссылки
        if start > last_pos:
            result.append(text[last_pos:start])
        last_pos = end
    
    # Добавляем оставшийся текст после последней ссылки
    if last_pos < len(text):
        result.append(text[last_pos:])
    
    return ''.join(result).strip()


def clean_text_artifacts(text):
    """
    Удаляет артефакты из текста после удаления ссылок:
    - одиночные символы-разделители (|, -, • и т.д.)
    - множественные пробелы
    - пустые строки
    - строки, состоящие только из разделителей
    """
    if not text:
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Разделители, которые нужно удалять если они стоят отдельно
    separators = ['|', '•', '-', '—', '–', '·', '▪', '▫']
    
    for line in lines:
        # Удаляем пробелы в начале и конце строки
        line = line.strip()
        
        # Пропускаем пустые строки
        if not line:
            continue
        
        # Пропускаем строки, состоящие только из разделителей и пробелов
        if all(c in separators + [' '] for c in line):
            continue
        
        # Удаляем одиночные разделители в начале и конце строки
        # Но оставляем их если они часть текста
        # Удаляем разделители, которые стоят отдельно (окружены пробелами или в начале/конце)
        
        # Удаляем разделители в начале строки (с пробелами после или без)
        line = re.sub(r'^[' + re.escape('|•-—–·▪▫') + r']+\s*', '', line)
        # Удаляем разделители в конце строки (с пробелами перед или без)
        line = re.sub(r'\s*[' + re.escape('|•-—–·▪▫') + r']+$', '', line)
        # Удаляем разделители, окруженные пробелами с обеих сторон
        line = re.sub(r'\s+[' + re.escape('|•-—–·▪▫') + r']+\s+', ' ', line)
        # Удаляем разделители с пробелом только слева (перед пробелом или концом строки)
        line = re.sub(r'\s+[' + re.escape('|•-—–·▪▫') + r']+(?=\s|$)', '', line)
        # Удаляем разделители с пробелом только справа (после пробела или в начале строки)
        # Используем простой подход: заменяем "пробел + разделители + пробел" на один пробел
        line = re.sub(r'\s[' + re.escape('|•-—–·▪▫') + r']+\s+', ' ', line)
        # Удаляем множественные пробелы
        line = re.sub(r'\s+', ' ', line)
        
        # Пропускаем строки, которые стали пустыми после очистки
        if not line.strip():
            continue
        
        cleaned_lines.append(line)
    
    # Объединяем строки обратно
    result = '\n'.join(cleaned_lines)
    
    # Удаляем множественные переносы строк (более 2 подряд)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    # Удаляем пробелы в начале и конце всего текста
    result = result.strip()
    
    return result


def remove_subscription_prompts(text):
    """
    Удаляет призывы к подписке в конце поста:
    - "Подписывайтесь на нас"
    - "в 👉"
    - "Подписаться на ... в"
    - "Мы в 👉"
    """
    if not text:
        return text
    
    lines = text.split('\n')
    
    # Паттерны для удаления призывов к подписке
    subscription_patterns = [
        r'^🛑?\s*[Пп]одписывайтесь\s+на\s+нас',
        r'^📲?\s*[Пп]одписаться\s+на\s+[^в]*\s+в\s*$',
        r'^📱?\s*[Мм]ы\s+в\s*👉\s*$',
        r'^в\s*👉\s*$',
        r'^👉\s*$',
        r'^📲\s*[Пп]одписаться',
        r'^📱\s*[Мм]ы\s+в',
        r'^🛑\s*[Пп]одписывайтесь',
    ]
    
    # Удаляем строки с призывами к подписке с конца
    # Идем с конца и удаляем призывы к подписке, пока не встретим обычную строку
    cleaned_lines = []
    i = len(lines) - 1
    
    while i >= 0:
        line = lines[i].strip()
        
        # Пропускаем пустые строки в конце
        if not line:
            i -= 1
            continue
        
        # Проверяем, является ли строка призывом к подписке
        is_subscription = False
        for pattern in subscription_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_subscription = True
                break
        
        if is_subscription:
            # Это призыв к подписке - пропускаем его
            i -= 1
        else:
            # Это обычная строка - останавливаемся и возвращаем все до этого места
            cleaned_lines = lines[:i+1]
            break
    
    # Если все строки были призывами к подписке, возвращаем пустой текст
    if not cleaned_lines:
        return ""
    
    return '\n'.join(cleaned_lines).strip()


async def initialize_channel_last_id(entity):
    """Инициализирует last_id для канала, получая текущий последний ID сообщения."""
    if entity.id not in last_ids:
        try:
            # Получаем последнее сообщение из канала
            last_msg = await client.get_messages(entity, limit=1)
            if last_msg and len(last_msg) > 0:
                initial_id = last_msg[0].id
                last_ids[entity.id] = initial_id
                logger.info(
                    "INIT | %s | установлен начальный last_id=%s (пропускаем старые посты)",
                    entity.username,
                    initial_id
                )
            else:
                # Если канал пустой, устанавливаем 0
                last_ids[entity.id] = 0
                logger.info("INIT | %s | канал пустой, last_id=0", entity.username)
        except Exception as e:
            logger.exception("Ошибка инициализации last_id для %s", entity.username)
            last_ids[entity.id] = 0


async def poll_channels():
    # Инициализируем last_id для всех каналов при первом запуске
    logger.info("Инициализация каналов (пропуск старых постов)...")
    for ch in CHANNELS:
        try:
            entity = await client.get_entity(ch)
            await initialize_channel_last_id(entity)
        except Exception:
            logger.exception("Ошибка инициализации канала %s", ch)
    
    logger.info("Начало мониторинга новых постов...")
    
    while True:
        for ch in CHANNELS:
            try:
                entity = await client.get_entity(ch)
                last_id = last_ids.get(entity.id, 0)
                
                # Получаем новые сообщения
                messages = await client.get_messages(entity, min_id=last_id, limit=10)
                
                if messages:
                    # Обновляем last_id на максимальный ID из полученных сообщений
                    new_last_id = max(msg.id for msg in messages)
                    last_ids[entity.id] = new_last_id
                    
                    # Добавляем сообщения в очередь
                    for msg in messages:
                        if msg.id not in processed_ids:
                            try:
                                message_queue.put_nowait((entity, msg))
                                processed_ids.append(msg.id)
                            except Exception:
                                logger.warning("Очередь переполнена, пропускаем сообщение %s/%s", ch, msg.id)
                
            except FloodWaitError as e:
                logger.warning("FloodWait: ждем %d секунд", e.seconds)
                await asyncio.sleep(e.seconds)
            except Exception:
                logger.exception("Ошибка при опросе канала %s", ch)
        
        await asyncio.sleep(CHECK_INTERVAL)


async def process_message(entity, msg):
    """Обрабатывает одно сообщение: классифицирует и пересылает если нужно."""
    channel = entity.username or str(entity.id)
    text = msg.message or ""
    
    # Пропускаем сообщения без текста (только медиа)
    if not text or not text.strip():
        logger.debug("Пропуск сообщения %s/%s: нет текста (только медиа)", channel, msg.id)
        return

    # Параллельная классификация: эвристика и BERT работают одновременно
    logger.info("Начало параллельной классификации для %s/%s", channel, msg.id)
    
    async def get_heuristic_result():
        """Получает мультиметочную оценку от эвристики."""
        try:
            from spam_rules_multilabel import heuristic_multilabel_score, heuristic_multilabel_predict
            scores = heuristic_multilabel_score(text)
            predictions = heuristic_multilabel_predict(text)
            
            # Вычисляем общую оценку (максимум из всех категорий)
            max_score = max(scores.values())
            is_spam = any(predictions.values())
            
            return {
                "method": "heuristics",
                "scores": scores,
                "predictions": predictions,
                "score": max_score,
                "is_spam": is_spam,
                "reason": "heuristics_multilabel"
            }
        except Exception as e:
            logger.exception("Ошибка эвристики")
            return {
                "method": "heuristics",
                "scores": {"ads": 0.0, "crypto": 0.0, "scam": 0.0, "casino": 0.0},
                "predictions": {"ads": 0, "crypto": 0, "scam": 0, "casino": 0},
                "score": 0.5,
                "is_spam": False,
                "reason": f"error: {str(e)}"
            }
    
    async def get_bert_result():
        """Получает мультиметочную оценку от BERT."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, classify_multilabel, text)
            
            # Вычисляем общую оценку (максимум из всех категорий)
            max_score = max(result["scores"].values())
            is_spam = any(result["predictions"].values())
            
            return {
                "method": "bert",
                "scores": result["scores"],
                "predictions": result["predictions"],
                "score": max_score,
                "is_spam": is_spam,
                "reason": "bert_multilabel",
                "methods_used": result["methods"]
            }
        except Exception as e:
            logger.exception("Ошибка BERT")
            # Fallback на эвристику
            try:
                from spam_rules_multilabel import heuristic_multilabel_score, heuristic_multilabel_predict
                scores = heuristic_multilabel_score(text)
                predictions = heuristic_multilabel_predict(text)
                max_score = max(scores.values())
                is_spam = any(predictions.values())
                return {
                    "method": "bert",
                    "scores": scores,
                    "predictions": predictions,
                    "score": max_score,
                    "is_spam": is_spam,
                    "reason": f"bert_error_fallback_heuristic: {str(e)}"
                }
            except Exception as e2:
                return {
                    "method": "bert",
                    "scores": {"ads": 0.0, "crypto": 0.0, "scam": 0.0, "casino": 0.0},
                    "predictions": {"ads": 0, "crypto": 0, "scam": 0, "casino": 0},
                    "score": 0.5,
                    "is_spam": False,
                    "reason": f"critical_error: {str(e2)}"
                }
    
    # Запускаем оба метода параллельно
    heuristic_task = asyncio.create_task(get_heuristic_result())
    bert_task = asyncio.create_task(get_bert_result())
    
    # Ждем завершения всех задач
    heuristic_result, bert_result = await asyncio.gather(
        heuristic_task, bert_task
    )
    
    # Логируем сообщение для сбора датасета (используем общий score)
    log_message_for_ml(text, heuristic_result["score"] * 10.0, channel, msg.id)
    
    # Объединяем результаты: берем среднее для каждой категории
    final_scores = {}
    final_predictions = {}
    for category in ["ads", "crypto", "scam", "casino"]:
        h_score = heuristic_result["scores"].get(category, 0.0)
        b_score = bert_result["scores"].get(category, 0.0)
        final_scores[category] = (h_score + b_score) / 2.0 if (h_score > 0 or b_score > 0) else 0.0
        # Предсказание: если хотя бы один метод предсказал 1, то итог = 1
        final_predictions[category] = 1 if (
            heuristic_result["predictions"].get(category, 0) == 1 or 
            bert_result["predictions"].get(category, 0) == 1
        ) else 0
    
    # Общая оценка и решение
    max_score = max(final_scores.values())
    is_spam = any(final_predictions.values())

    # Детальное логирование всех результатов
    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ МУЛЬТИМЕТОЧНОЙ КЛАССИФИКАЦИИ | %s/%s", channel, msg.id)
    
    # Логируем эвристику
    logger.info("  • Эвристика:")
    logger.info("      Оценки: ads=%.3f, crypto=%.3f, scam=%.3f, casino=%.3f",
                heuristic_result["scores"]["ads"],
                heuristic_result["scores"]["crypto"],
                heuristic_result["scores"]["scam"],
                heuristic_result["scores"]["casino"])
    logger.info("      Предсказания: ads=%d, crypto=%d, scam=%d, casino=%d",
                heuristic_result["predictions"]["ads"],
                heuristic_result["predictions"]["crypto"],
                heuristic_result["predictions"]["scam"],
                heuristic_result["predictions"]["casino"])
    
    # Логируем BERT
    bert_reason = bert_result.get("reason", "")
    if "fallback" in bert_reason or "error" in bert_reason:
        logger.info("  • BERT: (%s)", bert_reason)
    else:
        logger.info("  • BERT:")
        logger.info("      Оценки: ads=%.3f, crypto=%.3f, scam=%.3f, casino=%.3f",
                    bert_result["scores"]["ads"],
                    bert_result["scores"]["crypto"],
                    bert_result["scores"]["scam"],
                    bert_result["scores"]["casino"])
        logger.info("      Предсказания: ads=%d, crypto=%d, scam=%d, casino=%d",
                    bert_result["predictions"]["ads"],
                    bert_result["predictions"]["crypto"],
                    bert_result["predictions"]["scam"],
                    bert_result["predictions"]["casino"])
        if "methods_used" in bert_result:
            logger.info("      Методы: %s", bert_result["methods_used"])
    
    # Логируем итоговые результаты
    logger.info("  • ИТОГО (объединенные):")
    logger.info("      Оценки: ads=%.3f, crypto=%.3f, scam=%.3f, casino=%.3f",
                final_scores["ads"],
                final_scores["crypto"],
                final_scores["scam"],
                final_scores["casino"])
    logger.info("      Предсказания: ads=%d, crypto=%d, scam=%d, casino=%d",
                final_predictions["ads"],
                final_predictions["crypto"],
                final_predictions["scam"],
                final_predictions["casino"])
    logger.info("      Общая оценка: %.3f | %s", max_score, "СПАМ" if is_spam else "ОК")
    logger.info("=" * 60)

    # Если пост определен как спам - не постим его, но отправляем в личку для мониторинга
    if is_spam:
        logger.info("Пропуск сообщения %s/%s: определено как СПАМ", channel, msg.id)
        
        # Отправляем спам в личку для мониторинга
        try:
            monitor_user = await client.get_entity(SPAM_MONITOR_USER_ID)
            
            # Формируем сообщение с информацией о классификации
            spam_info_lines = [
                f"🚫 СПАМ обнаружен",
                f"",
                f"Канал: {channel}",
                f"ID сообщения: {msg.id}",
                f"",
                f"Итоговые оценки:",
                f"  • ads: {final_scores['ads']:.3f} ({'ДА' if final_predictions['ads'] else 'НЕТ'})",
                f"  • crypto: {final_scores['crypto']:.3f} ({'ДА' if final_predictions['crypto'] else 'НЕТ'})",
                f"  • scam: {final_scores['scam']:.3f} ({'ДА' if final_predictions['scam'] else 'НЕТ'})",
                f"  • casino: {final_scores['casino']:.3f} ({'ДА' if final_predictions['casino'] else 'НЕТ'})",
                f"",
                f"Общая оценка: {max_score:.3f}",
                f"",
                f"Оригинальный текст:",
                f"─" * 40,
            ]
            
            spam_info_text = "\n".join(spam_info_lines)
            full_message = f"{spam_info_text}\n{text}"
            
            # Отправляем текст (если он слишком длинный, разбиваем на части)
            MAX_MESSAGE_LENGTH = 4096
            if len(full_message) <= MAX_MESSAGE_LENGTH:
                # Отправляем медиа отдельно, если оно есть и можно отправить
                if msg.media and can_send_as_file(msg.media):
                    # Отправляем медиа с текстом
                    await client.send_message(
                        monitor_user,
                        full_message,
                        file=msg.media
                    )
                else:
                    # Отправляем только текст
                    await client.send_message(monitor_user, full_message)
            else:
                # Текст слишком длинный - отправляем информацию отдельно, затем текст
                await client.send_message(monitor_user, spam_info_text)
                # Отправляем текст частями
                text_part = f"Текст сообщения:\n{'─' * 40}\n{text}"
                if len(text_part) > MAX_MESSAGE_LENGTH:
                    # Разбиваем текст на части
                    chunks = [text[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]
                    for i, chunk in enumerate(chunks, 1):
                        await client.send_message(monitor_user, f"[Часть {i}/{len(chunks)}]\n{chunk}")
                else:
                    await client.send_message(monitor_user, text_part)
                
                # Отправляем медиа отдельно, если есть
                if msg.media and can_send_as_file(msg.media):
                    await client.send_message(monitor_user, file=msg.media)
            
            logger.info("Спам-сообщение отправлено в личку для мониторинга (ID: %s)", SPAM_MONITOR_USER_ID)
        except Exception as e:
            logger.exception("Ошибка при отправке спама в личку для мониторинга: %s", e)
        
        return

    # Получаем entities сообщения для удаления ссылок
    entities = msg.entities or []
    
    # Удаляем все гиперссылки из текста
    cleaned_text = remove_hyperlinks(text, entities)
    
    # Очищаем артефакты (одиночные разделители, пустые строки и т.д.)
    cleaned_text = clean_text_artifacts(cleaned_text)
    
    # Удаляем призывы к подписке в конце поста
    cleaned_text = remove_subscription_prompts(cleaned_text)
    
    # Проверка на дубликаты через NER
    try:
        ner_detector = get_ner_detector(ttl_hours=4, similarity_threshold=0.85)
        is_duplicate, similarity_score, duplicate_msg_id = ner_detector.is_duplicate(
            cleaned_text, entity.id, msg.id, media=msg.media
        )
        
        if is_duplicate:
            logger.info(
                "Пропуск сообщения %s/%s: обнаружен дубликат (similarity=%.2f, дубликат: %s/%s)",
                channel, msg.id, similarity_score, channel, duplicate_msg_id
            )
            return
        elif similarity_score > 0:
            logger.debug(
                "Сообщение %s/%s не является дубликатом (similarity=%.2f)",
                channel, msg.id, similarity_score
            )
    except Exception as e:
        logger.exception("Ошибка при проверке на дубликаты: %s", e)
        # Продолжаем обработку, если NER не работает
    
    # Добавляем в конец текст "Подписывайся" с ссылкой
    subscribe_text = "Подписывайся"
    subscribe_url = "https://t.me/+RpcJU9JMs9QwNTFi"
    
    # Формируем ссылку на источник (канал)
    source_text = "Источник"
    if entity.username:
        source_url = f"https://t.me/{entity.username}"
    else:
        # Если нет username, используем ID канала
        # Для каналов/групп ID начинается с -100, нужно убрать префикс
        channel_id = str(entity.id)
        if channel_id.startswith('-100'):
            channel_id = channel_id[4:]  # Убираем префикс -100
        else:
            channel_id = channel_id.lstrip('-')  # Убираем минус если есть
        source_url = f"https://t.me/c/{channel_id}/{msg.id}"
    
    # Формируем финальный текст
    if cleaned_text:
        final_text = f"{cleaned_text}\n\n{subscribe_text}\n{source_text}"
    else:
        final_text = f"{subscribe_text}\n{source_text}"
    
    # Функция-хелпер для создания entity ссылки
    def create_text_url_entity(text, link_text, url):
        """Создает MessageEntityTextUrl для ссылки в тексте."""
        start_python = text.find(link_text)
        start_utf16 = python_to_utf16_offset(text, start_python)
        length_utf16 = utf16_len(link_text)
        return MessageEntityTextUrl(offset=start_utf16, length=length_utf16, url=url)
    
    # Создаем entities для ссылок
    formatting_entities = [
        create_text_url_entity(final_text, subscribe_text, subscribe_url),
        create_text_url_entity(final_text, source_text, source_url),
    ]
    
    # Отправляем как новое сообщение в группу с форматированием и медиа
    try:
        target_group = await client.get_entity(TARGET_GROUP_ID)
        
        MAX_MEDIA_CAPTION_LENGTH = 1024
        
        # Проверяем, можно ли отправить медиа как файл
        has_sendable_media = msg.media and can_send_as_file(msg.media)
        
        # Если есть отправляемое медиа и текст слишком длинный для подписи
        if has_sendable_media and len(final_text) > MAX_MEDIA_CAPTION_LENGTH:
            # Отправляем медиа с короткой подписью (только "Подписывайся" и "Источник")
            short_caption = f"{subscribe_text}\n{source_text}"
            short_formatting_entities = [
                create_text_url_entity(short_caption, subscribe_text, subscribe_url),
                create_text_url_entity(short_caption, source_text, source_url),
            ]
            
            # Отправляем медиа с короткой подписью
            await client.send_message(
                target_group,
                short_caption,
                file=msg.media,
                formatting_entities=short_formatting_entities
            )
            
            # Отправляем полный текст отдельным сообщением
            await client.send_message(
                target_group,
                final_text,
                formatting_entities=formatting_entities
            )
            
            logger.info("Медиа отправлено с короткой подписью, полный текст отправлен отдельным сообщением")
        else:
            # Обычная отправка: текст + медиа (если есть и можно отправить) в одном сообщении
            send_kwargs = {
                'entity': target_group,
                'message': final_text,
                'formatting_entities': formatting_entities
            }
            
            # Добавляем медиа, если оно есть и можно отправить как файл
            if has_sendable_media:
                send_kwargs['file'] = msg.media
            
            await client.send_message(**send_kwargs)
            logger.info("Сообщение отправлено в группу %s (с медиа: %s)", TARGET_GROUP_ID, "да" if has_sendable_media else "нет")
    except Exception as e:
        logger.exception("Ошибка при отправке сообщения в группу %s: %s", TARGET_GROUP_ID, e)


async def worker():
    """Воркер для обработки сообщений из очереди."""
    while True:
        try:
            entity, msg = await message_queue.get()
            await process_message(entity, msg)
            message_queue.task_done()
        except Exception:
            logger.exception("Ошибка в воркере")


async def main():
    await client.start()
    logger.info("Бот запущен")
    
    # Запускаем воркеры
    workers = [asyncio.create_task(worker()) for _ in range(WORKERS)]
    
    # Запускаем опрос каналов
    poll_task = asyncio.create_task(poll_channels())
    
    # Ждем завершения (никогда не завершится, но это нормально)
    await asyncio.gather(poll_task, *workers)


if __name__ == "__main__":
    asyncio.run(main())
