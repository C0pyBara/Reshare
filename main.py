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

from spam_model import classify_parallel

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

                logger.info(
                    "POLL | %s | last_id=%s",
                    entity.username,
                    last_id
                )

                async for msg in client.iter_messages(
                    entity,
                    min_id=last_id,
                    reverse=True
                ):
                    if not msg.message:
                        continue

                    msg_key = (entity.id, msg.id)

                    if msg_key in processed_ids:
                        logger.debug("DUPLICATE skip %s/%s", entity.username, msg.id)
                        continue

                    if message_queue.full():
                        logger.warning("QUEUE FULL — skipping msg %s/%s", entity.username, msg.id)
                        continue

                    processed_ids.append(msg_key)
                    await message_queue.put((entity, msg))
                    last_ids[entity.id] = msg.id

                    logger.info(
                        "QUEUE + | %s/%s | size=%s",
                        entity.username,
                        msg.id,
                        message_queue.qsize()
                    )

            except Exception:
                logger.exception("Ошибка polling канала %s", ch)

        logger.info("Polling sleep %s sec", CHECK_INTERVAL)
        await asyncio.sleep(CHECK_INTERVAL)


async def worker_loop(worker_id: int):
    logger.info("WORKER-%s started", worker_id)

    while True:
        entity, msg = await message_queue.get()

        try:
            logger.info(
                "WORKER-%s | PROCESS %s/%s",
                worker_id,
                entity.username,
                msg.id
            )

            await process_message(entity, msg)

        except FloodWaitError as e:
            logger.warning("WORKER-%s | FloodWait %s sec", worker_id, e.seconds)
            await asyncio.sleep(e.seconds + 1)
        except Exception:
            logger.exception(
                "WORKER-%s | ERROR msg_id=%s",
                worker_id,
                msg.id
            )
        finally:
            message_queue.task_done()


async def process_message(entity, msg):
    text = msg.message or ""
    channel = entity.username or entity.title or "unknown"

    # Получаем все оценки параллельно
    logger.info("Начало классификации для %s/%s", channel, msg.id)
    results = await classify_parallel(text)
    
    # Если результатов нет, используем минимальную эвристику напрямую
    if not results:
        logger.warning("⚠ Не получено оценок от основных методов, используем прямую эвристику для %s/%s", channel, msg.id)
        try:
            from spam_rules import heuristic_spam_score
            score = heuristic_spam_score(text)
            # Нормализуем score в диапазон 0-1 (эвристика возвращает обычно 0-30)
            normalized_score = min(1.0, score / 10.0) if score > 1.0 else score
            results = [{
                "method": "fallback",
                "score": normalized_score,
                "reason": "heuristic_emergency"
            }]
            logger.info("  ✓ fallback (emergency): score=%.3f, raw_score=%.1f", normalized_score, score)
        except Exception as e:
            logger.error("❌ Критическая ошибка: не удалось использовать даже эвристику: %s", str(e))
            # В крайнем случае используем нейтральную оценку
            results = [{
                "method": "fallback",
                "score": 0.5,
                "reason": f"critical_error: {str(e)}"
            }]

    # Вычисляем средний score
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    is_spam = avg_score >= 0.6

    # Детальное логирование всех результатов
    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ | %s/%s", channel, msg.id)
    logger.info("  Всего методов: %d", len(results))
    for result in sorted(results, key=lambda x: x["method"]):
        logger.info("  • %s: score=%.3f (%s)", result["method"], result["score"], result["reason"])
    logger.info("  Средний score: %.3f | Итог: %s", avg_score, "СПАМ" if is_spam else "ОК")
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

    flag = "⚠️ ВОЗМОЖНО СПАМ" if is_spam else "✅ вероятно ок"
    
    # Сортируем результаты по методу для единообразия
    results_sorted = sorted(results, key=lambda x: x["method"])
    
    # Формируем информацию о всех оценках
    evaluations = []
    method_names = {
        "llama_cli": "🤖 Qwen (llama-cli)",
        "llama_cpp": "🤖 Qwen (llama-cpp-python)",
        "transformers": "🤖 Qwen (transformers)",
        "fallback": "📊 Fallback (эвристика/BERT)"
    }
    
    for result in results_sorted:
        method = result["method"]
        score = result["score"]
        reason = result.get("reason", "")
        method_display = method_names.get(method, method)
        result_text = "🔴 СПАМ" if score >= 0.6 else "🟢 НОРМ"
        evaluations.append(f"{method_display}\n  {result_text} | score={score:.3f}")
    
    evaluations_text = "\n\n".join(evaluations)
    
    comment = (
        f"{flag}\n\n"
        f"📊 РЕЗУЛЬТАТЫ ВСЕХ МЕТОДОВ ({len(results)}):\n\n"
        f"{evaluations_text}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Средний score: {avg_score:.3f}\n"
        f"🎯 Итоговый вердикт: {'🔴 СПАМ' if is_spam else '🟢 НОРМ'}\n\n"
        f"📺 Канал: {channel}\n"
        f"🆔 ID: {msg.id}"
    )

    await client.send_message(
        TARGET_ENTITY,
        comment,
        buttons=buttons
    )


async def main():
    await client.start()
    logger.info("Telegram client started")

    await resolve_target_entity()

    for i in range(WORKERS):
        asyncio.create_task(worker_loop(i + 1))

    await poll_channels()


if __name__ == "__main__":
    asyncio.run(main())
