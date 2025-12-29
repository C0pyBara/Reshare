"""Скрипт для multilabel разметки данных с использованием LLM для первичной разметки."""
import json
import asyncio
from pathlib import Path
from typing import Dict

from spam_model import classify_parallel

DATA_DIR = Path(__file__).parent / "data"
RAW_MESSAGES_FILE = DATA_DIR / "raw_messages.jsonl"
LABELED_CSV_FILE = DATA_DIR / "labeled_multilabel.csv"

LABELS = ["ads", "crypto", "scam", "casino"]


async def llm_label_text(text: str) -> Dict[str, int]:
    """Использует LLM для первичной разметки текста.
    
    Returns:
        Словарь с метками: {"ads": 0/1, "crypto": 0/1, "scam": 0/1, "casino": 0/1}
    """
    import re
    import json
    from spam_model import classify_parallel
    
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
        # Используем LLM для классификации
        results = await classify_parallel(prompt)
        
        # Ищем JSON в ответе
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
    except Exception:
        # Молча игнорируем ошибки LLM
        pass
    
    # Если LLM не сработал, возвращаем нули
    return {label: 0 for label in LABELS}


def manual_label_text(text: str, llm_labels: Dict[str, int] = None) -> Dict[str, int]:
    """Интерактивная разметка текста человеком.
    
    Args:
        text: Текст для разметки
        llm_labels: Предложенные метки от LLM (опционально)
        
    Returns:
        Словарь с метками
    """
    if llm_labels:
        print(f"\n🤖 LLM предложил: {', '.join(k for k, v in llm_labels.items() if v == 1) or 'нет меток'}")
    
    labels = {}
    for label in LABELS:
        default = "1" if (llm_labels and llm_labels.get(label, 0) == 1) else "0"
        user_input = input(f"  {label:8s} (0/1, Enter={default}): ").strip()
        labels[label] = 1 if (user_input or default) == "1" else 0
    
    return labels


def main():
    """Интерактивная multilabel разметка данных."""
    if not RAW_MESSAGES_FILE.exists():
        print(f"Файл {RAW_MESSAGES_FILE} не найден!")
        print("Запустите бота, чтобы начать собирать данные.")
        return
    
    rows = []
    processed = 0
    skipped = 0
    
    print("=" * 70)
    print("MULTILABEL РАЗМЕТКА ДАННЫХ ДЛЯ ML МОДЕЛИ")
    print("=" * 70)
    print(f"Читаем из: {RAW_MESSAGES_FILE}")
    print(f"Сохраняем в: {LABELED_CSV_FILE}")
    print(f"\nМетки: {', '.join(LABELS)}")
    print("\nИнструкция:")
    print("  1 - относится к категории")
    print("  0 - не относится")
    print("  Enter - использовать предложенное значение (от LLM или 0)")
    print("  q - завершить разметку")
    print("=" * 70)
    print()
    
    use_llm = input("Использовать LLM для первичной разметки? (y/n, default=n): ").strip().lower() == 'y'
    
    with open(RAW_MESSAGES_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                text = obj.get("text", "").strip()
                
                if not text:
                    continue
                
                heuristic_score = obj.get("heuristic_score", 0)
                channel = obj.get("channel", "unknown")
                
                print(f"\n[{line_num}] Канал: {channel}")
                print(f"Heuristic score: {heuristic_score:.1f}")
                print("-" * 70)
                print("ТЕКСТ:")
                # Ограничиваем длину для удобства
                display_text = text[:500] + "..." if len(text) > 500 else text
                print(display_text)
                print("-" * 70)
                
                # Получаем метки от LLM если нужно
                llm_labels = None
                if use_llm:
                    print("🤖 Запрос к LLM...")
                    try:
                        llm_labels = asyncio.run(llm_label_text(text))
                    except Exception as e:
                        print(f"⚠ Ошибка LLM: {e}")
                
                # Ручная разметка
                labels = manual_label_text(text, llm_labels)
                
                # Проверяем, что хотя бы одна метка установлена или пользователь хочет сохранить
                has_labels = any(labels.values())
                
                label_input = input("\nСохранить? (Enter=да, n=нет, q=quit): ").strip().lower()
                
                if label_input == 'q':
                    print("\nЗавершение разметки...")
                    break
                
                if label_input == 'n':
                    skipped += 1
                    print("⊘ Пропущено")
                    continue
                
                # Сохраняем
                text_escaped = text.replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                row = [text_escaped] + [labels[label] for label in LABELS]
                rows.append(row)
                processed += 1
                
                active_labels = ', '.join(label for label in LABELS if labels[label] == 1)
                print(f"✓ Сохранено: {active_labels or 'нет меток'}")
                
            except json.JSONDecodeError:
                print(f"⚠ Ошибка парсинга JSON на строке {line_num}")
                continue
            except KeyboardInterrupt:
                print("\n\nПрервано пользователем")
                break
            except Exception as e:
                print(f"⚠ Ошибка на строке {line_num}: {e}")
                continue
    
    if not rows:
        print("\nНет размеченных данных для сохранения.")
        return
    
    # Сохраняем в CSV
    with open(LABELED_CSV_FILE, "w", encoding="utf-8") as f:
        # Заголовок
        f.write("text," + ",".join(LABELS) + "\n")
        for row in rows:
            text = row[0]
            labels = row[1:]
            f.write(f'"{text}",{",".join(map(str, labels))}\n')
    
    print("\n" + "=" * 70)
    print(f"✓ Сохранено {len(rows)} размеченных примеров")
    print(f"⊘ Пропущено {skipped} примеров")
    print(f"Файл: {LABELED_CSV_FILE}")
    print("=" * 70)
    print("\nТеперь можно обучить модель:")
    print("  python train_ml_multilabel.py")


if __name__ == "__main__":
    main()

