"""Скрипт для ручной разметки данных для обучения ML модели."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_MESSAGES_FILE = DATA_DIR / "raw_messages.jsonl"
LABELED_CSV_FILE = DATA_DIR / "labeled.csv"

def main():
    """Интерактивная разметка данных."""
    if not RAW_MESSAGES_FILE.exists():
        print(f"Файл {RAW_MESSAGES_FILE} не найден!")
        print("Запустите бота, чтобы начать собирать данные.")
        return
    
    rows = []
    processed = 0
    skipped = 0
    
    print("=" * 60)
    print("РАЗМЕТКА ДАННЫХ ДЛЯ ML МОДЕЛИ")
    print("=" * 60)
    print(f"Читаем из: {RAW_MESSAGES_FILE}")
    print(f"Сохраняем в: {LABELED_CSV_FILE}")
    print("\nИнструкция:")
    print("  1 - реклама/спам")
    print("  0 - не реклама")
    print("  Enter или любой другой символ - пропустить")
    print("  q - завершить разметку")
    print("=" * 60)
    print()
    
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
                print("-" * 60)
                print("ТЕКСТ:")
                # Ограничиваем длину для удобства
                display_text = text[:500] + "..." if len(text) > 500 else text
                print(display_text)
                print("-" * 60)
                
                label_input = input("Реклама/Спам? (1/0, Enter=skip, q=quit): ").strip()
                
                if label_input.lower() == 'q':
                    print("\nЗавершение разметки...")
                    break
                
                if label_input in {"0", "1"}:
                    label = int(label_input)
                    # Экранируем кавычки и переносы строк в тексте
                    text_escaped = text.replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                    rows.append((text_escaped, label))
                    processed += 1
                    print(f"✓ Размечено как: {'СПАМ' if label == 1 else 'НЕ СПАМ'}")
                else:
                    skipped += 1
                    print("⊘ Пропущено")
                
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
        f.write("text,label\n")
        for text, label in rows:
            f.write(f'"{text}",{label}\n')
    
    print("\n" + "=" * 60)
    print(f"✓ Сохранено {len(rows)} размеченных примеров")
    print(f"⊘ Пропущено {skipped} примеров")
    print(f"Файл: {LABELED_CSV_FILE}")
    print("=" * 60)
    print("\nТеперь можно обучить модель:")
    print("  python train_ml.py")


if __name__ == "__main__":
    main()

