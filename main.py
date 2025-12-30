import asyncio
import logging
import sys
from asyncio import Queue
from collections import deque

from telethon import TelegramClient, Button
from telethon.errors import FloodWaitError

from config import (
    API_ID,
    API_HASH,
    SESSION_NAME,
    CHANNELS,
    TARGET_GROUP,
    CHECK_INTERVAL,
    WORKERS,
    QUEUE_MAXSIZE,
)

from classifier_multilabel import classify_multilabel
from data_logger import log_message_for_ml

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
TARGET_ENTITY = None

PROCESSED_CACHE_SIZE = 10_000
processed_ids = deque(maxlen=PROCESSED_CACHE_SIZE)


async def resolve_target_entity():
    global TARGET_ENTITY
    if not TARGET_GROUP:
        logger.warning("TARGET_GROUP не задан")
        return

    try:
        try:
            TARGET_ENTITY = await client.get_entity(int(TARGET_GROUP))
        except ValueError:
            TARGET_ENTITY = await client.get_entity(TARGET_GROUP)

        logger.info("TARGET_GROUP resolved: %s", TARGET_ENTITY.id)
    except Exception:
        logger.exception("Не удалось разрешить TARGET_GROUP")


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

    if not TARGET_ENTITY:
        return

    # пересылаем оригинал
    await client.forward_messages(
        TARGET_ENTITY,
        msg,
        from_peer=entity
    )

    # кнопка "Открыть пост"
    buttons = None
    if entity.username:
        url = f"https://t.me/{entity.username}/{msg.id}"
        buttons = [Button.url("🔗 Открыть пост", url)]

    flag = "⚠️ ВОЗМОЖНО СПАМ" if is_spam else "✅ок"
    
    # Формируем информацию о категориях
    categories_info = []
    category_emojis = {
        "ads": "📢",
        "crypto": "₿",
        "scam": "⚠️",
        "casino": "🎰"
    }
    
    for category in ["ads", "crypto", "scam", "casino"]:
        if final_predictions[category] == 1:
            emoji = category_emojis.get(category, "•")
            score = final_scores[category]
            categories_info.append(f"{emoji} {category.upper()}: {score:.2f}")
    
    categories_text = "\n".join(categories_info) if categories_info else "Нет категорий"
    
    # Формируем информацию о методах
    evaluations = []
    
    # 1. Эвристика
    heuristic_spam_text = "🔴 СПАМ" if heuristic_result["is_spam"] else "🟢 НОРМ"
    evaluations.append(
        f"📊 Эвристика\n"
        f"  {heuristic_spam_text} | score={heuristic_result['score']:.3f}"
    )
    
    # 2. BERT
    bert_spam_text = "🔴 СПАМ" if bert_result["is_spam"] else "🟢 НОРМ"
    bert_reason = bert_result.get("reason", "")
    if "fallback" in bert_reason or "error" in bert_reason:
        evaluations.append(
            f"🤖 BERT ({bert_reason[:30]})\n"
            f"  {bert_spam_text} | score={bert_result['score']:.3f}"
        )
    else:
        evaluations.append(
            f"🤖 BERT\n"
            f"  {bert_spam_text} | score={bert_result['score']:.3f}"
        )
    
    evaluations_text = "\n\n".join(evaluations)
    
    # Подсчитываем количество активных методов
    active_methods_count = 2  # Эвристика и BERT
    
    message_text = (
        f"{flag}\n\n"
        f"📊 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ ({active_methods_count} методов):\n\n"
        f"{evaluations_text}\n\n"
        f"🏷️ КАТЕГОРИИ:\n{categories_text}\n\n"
        f"📝 Исходный текст:\n{text[:200]}{'...' if len(text) > 200 else ''}"
    )
    
    await client.send_message(
        TARGET_ENTITY,
        message_text,
        buttons=buttons
    )


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
    
    await resolve_target_entity()
    
    # Запускаем воркеры
    workers = [asyncio.create_task(worker()) for _ in range(WORKERS)]
    
    # Запускаем опрос каналов
    poll_task = asyncio.create_task(poll_channels())
    
    # Ждем завершения (никогда не завершится, но это нормально)
    await asyncio.gather(poll_task, *workers)


if __name__ == "__main__":
    asyncio.run(main())
