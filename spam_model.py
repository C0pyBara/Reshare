from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Tuple
import json
import re
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Пробуем использовать Qwen через разные методы
HAS_LLAMA_CPP = False
HAS_TRANSFORMERS = False
HAS_LLAMA_CLI = False

# 1. Проверяем llama-cpp-python (для GGUF через Python API)
if TYPE_CHECKING:
    from llama_cpp import Llama  # type: ignore[import-not-found]

try:
    from llama_cpp import Llama  # type: ignore[import-not-found]
    HAS_LLAMA_CPP = True
    logger.info("llama-cpp-python доступен")
except ImportError:
    pass

# 2. Проверяем transformers (для HuggingFace моделей)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    HAS_TRANSFORMERS = True
    # Используем print для гарантии вывода при импорте модуля (до настройки логирования)
    import sys
    print(f"[spam_model] transformers доступен, HAS_TRANSFORMERS={HAS_TRANSFORMERS}", file=sys.stderr)
    logger.info("transformers доступен")
except ImportError as e:
    import sys
    print(f"[spam_model] transformers ImportError: {e}, HAS_TRANSFORMERS останется False", file=sys.stderr)
except Exception as e:
    import sys
    print(f"[spam_model] transformers Exception ({type(e).__name__}): {e}, HAS_TRANSFORMERS останется False", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

# 3. Проверяем llama-cli (для GGUF через subprocess)
import os
BASE_DIR = Path(__file__).parent
if os.name == "nt":
    LLAMA_CLI_PATH = BASE_DIR / "llama.cpp" / "llama-cli.exe"
else:
    LLAMA_CLI_PATH = BASE_DIR / "llama.cpp" / "llama-cli"

if LLAMA_CLI_PATH.exists():
    HAS_LLAMA_CLI = True
    logger.info("llama-cli найден")
else:
    logger.debug("llama-cli не найден (путь: %s). Будет использован другой метод классификации.", LLAMA_CLI_PATH)

# Пути к моделям - Qwen3-8B-GGUF
GGUF_MODEL_PATH = BASE_DIR / "models" / "Qwen3-8B-Q4_K_M.gguf"
if not GGUF_MODEL_PATH.exists():
    GGUF_MODEL_PATH = BASE_DIR / "models" / "qwen3-8b-q4_k_m.gguf"
if not GGUF_MODEL_PATH.exists():
    GGUF_MODEL_PATH = BASE_DIR / "llama.cpp" / "models" / "Qwen3-8B-Q4_K_M.gguf"
if not GGUF_MODEL_PATH.exists():
    GGUF_MODEL_PATH = BASE_DIR / "llama.cpp" / "models" / "qwen3-8b-q4_k_m.gguf"

# Логируем доступность методов при загрузке модуля
if GGUF_MODEL_PATH.exists():
    logger.info("GGUF модель найдена: %s", GGUF_MODEL_PATH)
else:
    logger.warning("GGUF модель не найдена по пути: %s", GGUF_MODEL_PATH)

# Инициализация моделей (ленивая загрузка)
llm = None
qwen_tokenizer = None
qwen_model = None

# Имя модели для transformers (легкая версия для быстрой работы)
QWEN_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Маленькая модель для быстрой работы

SYSTEM_PROMPT = (
    "Ты классификатор Telegram-постов. "
    "Определи, является ли пост рекламой, спамом или низкокачественным контентом. "
    "score: 0.0 = точно не спам, 1.0 = точно спам. "
    "Ответь ТОЛЬКО валидным JSON без дополнительного текста:\n"
    '{"score": 0.5, "is_spam": false, "reason": "описание"}'
)


def _get_llm_gguf():
    """Ленивая инициализация GGUF модели через llama-cpp-python."""
    global llm
    if llm is None:
        if not GGUF_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {GGUF_MODEL_PATH}. "
                "Please download the model first."
            )
        # Настройки для Qwen3-8B
        llm = Llama(
            model_path=str(GGUF_MODEL_PATH),
            n_ctx=4096,  # Увеличено для Qwen3 (поддерживает до 32K, но для классификации достаточно)
            n_threads=8,
            n_gpu_layers=0,  # 0 = CPU only
            verbose=False,
        )
    return llm


def _get_qwen_hf():
    """Ленивая инициализация Qwen через transformers."""
    global qwen_tokenizer, qwen_model
    if qwen_tokenizer is None or qwen_model is None:
        logger.info("Загрузка Qwen модели через transformers: %s", QWEN_HF_MODEL)
        try:
            qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_HF_MODEL)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                QWEN_HF_MODEL,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            if not torch.cuda.is_available():
                qwen_model = qwen_model.to("cpu")
            logger.info("Qwen модель успешно загружена")
        except Exception as e:
            logger.exception("Ошибка загрузки Qwen модели: %s", str(e))
            raise
    return qwen_tokenizer, qwen_model


def _classify_with_llama_cli(text: str) -> tuple[float, str]:
    """Классификация через llama-cli (subprocess)."""
    try:
        from llm_qwen import qwen_classify
        result = qwen_classify(text)
        reason = str(result.get("reason", "llama_cli"))
        
        # Проверяем, есть ли критическая ошибка в reason
        if "llm_error:" in reason.lower() or "not found" in reason.lower():
            logger.warning("llama-cli недоступен: %s", reason)
            return 0.0, f"llama_cli_error: {reason}"
        
        score = float(result.get("score", 0.0))
        is_spam = result.get("is_spam", False)
        
        # Если модель вернула валидный результат, используем его
        if score >= 0.0 and score <= 1.0:
            return max(0.0, min(1.0, score)), f"llama_cli: {reason}"
        else:
            # Нормализуем score если он вне диапазона
            normalized_score = max(0.0, min(1.0, score))
            return normalized_score, f"llama_cli: {reason}"
    except Exception as e:
        logger.exception("Ошибка llama-cli: %s", str(e))
        return 0.0, f"llama_cli_error: {str(e)}"


def _classify_with_llama_cpp(text: str) -> tuple[float, str]:
    """Классификация через llama-cpp-python (GGUF)."""
    try:
        model = _get_llm_gguf()
        
        # Форматируем промпт для Qwen3
        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{text[:1000]}<|im_end|>
<|im_start|>assistant
"""
        
        # Параметры для Qwen3 согласно документации
        output = model(
            prompt,
            max_tokens=200,
            temperature=0.6,  # Рекомендуемая для Qwen3
            top_k=20,
            top_p=0.95,
            min_p=0.0,
            repeat_penalty=1.5,  # presence penalty
            stop=["<|im_end|>"],
        )

        if isinstance(output, dict):
            if "choices" in output and len(output["choices"]) > 0:
                content = output["choices"][0].get("text", "")
            else:
                content = str(output)
        else:
            content = str(output)

        json_match = re.search(r"\{[\s\S]*?\}", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = float(data.get("score", 0.0))
                reason = str(data.get("reason", "no_reason"))
                # Нормализуем score и возвращаем с префиксом метода
                normalized_score = max(0.0, min(1.0, score))
                return normalized_score, f"llama_cpp: {reason}"
            except json.JSONDecodeError as e:
                logger.warning("Не удалось распарсить JSON от llama_cpp: %s", content[:200])
                return 0.0, f"llama_cpp_error: invalid_json"
        logger.warning("llama_cpp не вернул JSON. Ответ: %s", content[:200])
        return 0.0, "llama_cpp_error: no_json"
    except Exception as e:
        logger.exception("Ошибка llama-cpp-python")
        return 0.0, f"llama_cpp_error: {str(e)}"


def _classify_with_transformers(text: str) -> tuple[float, str]:
    """Классификация через transformers (HuggingFace Qwen)."""
    try:
        tokenizer, model = _get_qwen_hf()
        
        # Формируем промпт для Qwen
        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{text[:1000]}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Перемещаем на правильное устройство
        if not torch.cuda.is_available():
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        
        with torch.no_grad():
            # Параметры для Qwen3 (рекомендуемые для non-thinking mode)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,  # Рекомендуемая для non-thinking mode
                top_k=20,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.5,  # Важно для Qwen3
            )
        
        # Правильно извлекаем только сгенерированную часть
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Пытаемся найти JSON в ответе
        json_match = re.search(r"\{[\s\S]*?\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = float(data.get("score", 0.0))
                reason = str(data.get("reason", "no_reason"))
                # Ограничиваем score в диапазоне 0-1
                score = max(0.0, min(1.0, score))
                logger.info("Qwen (transformers) вернул: score=%.3f, reason=%s", score, reason)
                return score, f"transformers: {reason}"
            except json.JSONDecodeError as e:
                logger.warning("Не удалось распарсить JSON от Qwen: %s", response[:200])
                return 0.0, "transformers_error: invalid_json"
        else:
            logger.warning("Qwen не вернул JSON. Ответ: %s", response[:200])
            return 0.0, "transformers_error: no_json"
    except Exception as e:
        logger.exception("Ошибка transformers при классификации")
        return 0.0, f"transformers_error: {str(e)}"


async def _classify_parallel(text: str) -> List[Dict[str, any]]:
    """Запускает все доступные методы классификации параллельно."""
    global HAS_LLAMA_CLI, HAS_LLAMA_CPP, HAS_TRANSFORMERS, GGUF_MODEL_PATH, LLAMA_CLI_PATH
    
    # Перепроверяем доступность transformers на месте - возможно, модуль был импортирован до настройки окружения
    if not HAS_TRANSFORMERS:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
            HAS_TRANSFORMERS = True
            logger.info("transformers стал доступен после повторной проверки")
        except Exception as e:
            logger.debug("transformers все еще недоступен: %s", str(e))
    
    results = []
    task_list = []
    method_map = {}
    executor = ThreadPoolExecutor(max_workers=4)
    
    # Вспомогательная функция для обертки синхронных функций
    async def run_in_executor(func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)
    
    # Логируем какие методы будут запущены
    methods_to_run = []
    
    # Собираем задачи для параллельного выполнения
    # Логируем доступность методов для диагностики
    logger.info("=== ПРОВЕРКА ДОСТУПНОСТИ МЕТОДОВ ===")
    logger.info("HAS_LLAMA_CLI=%s, HAS_LLAMA_CPP=%s, HAS_TRANSFORMERS=%s", 
                 HAS_LLAMA_CLI, HAS_LLAMA_CPP, HAS_TRANSFORMERS)
    logger.info("GGUF_MODEL_PATH.exists()=%s, путь=%s", GGUF_MODEL_PATH.exists(), GGUF_MODEL_PATH)
    logger.info("LLAMA_CLI_PATH.exists()=%s, путь=%s", LLAMA_CLI_PATH.exists(), LLAMA_CLI_PATH)
    
    if HAS_LLAMA_CLI and GGUF_MODEL_PATH.exists():
        task = asyncio.create_task(run_in_executor(_classify_with_llama_cli, text))
        task_list.append(task)
        method_map[task] = "llama_cli"
        methods_to_run.append("llama_cli")
        logger.info("✓ Добавлен метод: llama_cli")
    elif HAS_LLAMA_CLI:
        logger.warning("✗ llama_cli доступен, но модель не найдена: %s", GGUF_MODEL_PATH)
    elif GGUF_MODEL_PATH.exists():
        logger.warning("✗ Модель найдена, но llama_cli недоступен: %s", LLAMA_CLI_PATH)
    
    if HAS_LLAMA_CPP and GGUF_MODEL_PATH.exists():
        task = asyncio.create_task(run_in_executor(_classify_with_llama_cpp, text))
        task_list.append(task)
        method_map[task] = "llama_cpp"
        methods_to_run.append("llama_cpp")
        logger.info("✓ Добавлен метод: llama_cpp")
    elif HAS_LLAMA_CPP:
        logger.warning("✗ llama_cpp доступен, но модель не найдена: %s", GGUF_MODEL_PATH)
    
    if HAS_TRANSFORMERS:
        task = asyncio.create_task(run_in_executor(_classify_with_transformers, text))
        task_list.append(task)
        method_map[task] = "transformers"
        methods_to_run.append("transformers")
        logger.info("✓ Добавлен метод: transformers (HAS_TRANSFORMERS=%s)", HAS_TRANSFORMERS)
    else:
        logger.warning("✗ transformers недоступен (HAS_TRANSFORMERS=%s)", HAS_TRANSFORMERS)
    
    # Fallback (эвристика) ВСЕГДА запускаем - гарантированный метод
    # Используем прямую эвристику как fallback, не зависим от импорта classifier
    task = asyncio.create_task(run_in_executor(_classify_with_fallback, text))
    task_list.append(task)
    method_map[task] = "fallback"
    methods_to_run.append("fallback")
    
    logger.info("Параллельный запуск методов классификации: %s", ", ".join(methods_to_run))
    
    # Выполняем все задачи параллельно
    if task_list:
        done, pending = await asyncio.wait(
            task_list,
            return_when=asyncio.ALL_COMPLETED,
            timeout=60.0  # Увеличен таймаут до 60 секунд для Qwen3
        )
        
        failed_methods = []
        # Собираем результаты
        for task in done:
            try:
                method_name = method_map.get(task, "unknown")
                score, reason = await task
                
                # Проверяем валидность результата
                # Принимаем результат если:
                # 1. score >= 0.0 (валидный диапазон)
                # 2. reason НЕ содержит критических ошибок (начинается с _error: или содержит llm_error:)
                # Критические ошибки - это когда метод вообще не смог выполниться
                # НЕ критические - это когда метод выполнился, но вернул score=0.0 или ошибку парсинга
                is_critical_error = (
                    reason.lower().startswith(("llama_cli_error:", "llama_cpp_error:", "transformers_error:")) or
                    "llm_error:" in reason.lower() or
                    "not found" in reason.lower() or
                    "critical_error" in reason.lower()
                )
                
                # Если score валидный и это не критическая ошибка - принимаем результат
                # Даже score=0.0 это валидный результат (модель считает, что это не спам)
                if score >= 0.0 and not is_critical_error:
                    results.append({
                        "method": method_name,
                        "score": score,
                        "reason": reason
                    })
                    logger.info("  ✓ %s: score=%.3f, reason=%s", method_name, score, reason)
                else:
                    failed_methods.append(f"{method_name} (score=%.3f, reason=%s)" % (score, reason))
                    logger.warning("  ✗ %s: отклонен - score=%.3f, reason=%s", method_name, score, reason)
            except Exception as e:
                method_name = method_map.get(task, "unknown")
                failed_methods.append(f"{method_name} (исключение: {str(e)})")
                logger.warning("  ✗ %s: исключение - %s", method_name, str(e))
        
        # Отменяем оставшиеся задачи
        for task in pending:
            method_name = method_map.get(task, "unknown")
            failed_methods.append(f"{method_name} (таймаут)")
            logger.warning("  ✗ %s: таймаут (60 сек)", method_name)
            task.cancel()
        
        if failed_methods:
            logger.info("Не сработавшие методы: %s", ", ".join(failed_methods))
        
        executor.shutdown(wait=False)
    
    logger.info("Итого успешных результатов: %d из %d методов", len(results), len(methods_to_run))
    return results


def _classify_with_fallback(text: str) -> tuple[float, str]:
    """Fallback классификация - всегда использует эвристику напрямую."""
    try:
        # Сначала пробуем через classifier (может включать BERT и другие методы)
        from classifier import classify_text
        result = classify_text(text)
        score = float(result.get("score", 0.0))
        reason = str(result.get("reason", "fallback"))
        # Нормализуем score в диапазон 0-1 (если он был из эвристики)
        if score > 1.0:
            score = min(1.0, score / 10.0)  # эвристика возвращает 0-10
        return score, reason
    except Exception:
        # Если classifier недоступен, используем только эвристику напрямую
        try:
            from spam_rules import heuristic_spam_score
            score = heuristic_spam_score(text)
            # Нормализуем score в диапазон 0-1 (эвристика возвращает 0-10, обычно максимум ~30)
            normalized_score = min(1.0, score / 10.0) if score > 1.0 else score
            reason = "heuristic_direct"
            return normalized_score, reason
        except Exception as e:
            logger.exception("Критическая ошибка в fallback классификаторе")
            # В крайнем случае возвращаем нейтральную оценку
            return 0.5, f"fallback_critical_error: {str(e)}"


def classify(text: str) -> tuple[float, str]:
    """Классифицирует текст и возвращает (score, reason).
    
    УСТАРЕВШИЙ МЕТОД - используйте classify_parallel для получения всех оценок.
    Оставлен для обратной совместимости.
    """
    # Запускаем параллельную классификацию синхронно
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(_classify_parallel(text))
        if results:
            # Возвращаем первый успешный результат
            return results[0]["score"], results[0]["reason"]
    finally:
        loop.close()
    
    # Если ничего не получилось
    return 0.0, "no_results"


async def classify_parallel(text: str) -> List[Dict[str, any]]:
    """Классифицирует текст параллельно всеми доступными методами и возвращает все результаты.
    
    Returns:
        List[Dict]: Список результатов, каждый содержит:
            - method: название метода
            - score: оценка от 0 до 1
            - reason: причина/описание
    """
    return await _classify_parallel(text)

