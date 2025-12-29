"""Обучение ML классификатора для определения спама в Telegram постах."""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
LABELED_CSV_FILE = DATA_DIR / "labeled.csv"
MODEL_FILE = MODELS_DIR / "spam_ml.pkl"

MODELS_DIR.mkdir(exist_ok=True)


def main():
    """Обучает ML модель для классификации спама."""
    print("=" * 60)
    print("ОБУЧЕНИЕ ML КЛАССИФИКАТОРА СПАМА")
    print("=" * 60)
    
    if not LABELED_CSV_FILE.exists():
        print(f"❌ Файл {LABELED_CSV_FILE} не найден!")
        print("Сначала разметьте данные:")
        print("  python label_data.py")
        return
    
    # Загружаем данные
    print(f"\n📂 Загрузка данных из {LABELED_CSV_FILE}...")
    try:
        df = pd.read_csv(LABELED_CSV_FILE)
        print(f"✓ Загружено {len(df)} примеров")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Проверяем баланс классов
    label_counts = df["label"].value_counts()
    print(f"\n📊 Распределение классов:")
    print(f"  Спам (1): {label_counts.get(1, 0)}")
    print(f"  Не спам (0): {label_counts.get(0, 0)}")
    
    if len(label_counts) < 2:
        print("❌ Нужны примеры обоих классов (0 и 1)!")
        return
    
    if len(df) < 50:
        print("⚠ Мало данных для обучения. Рекомендуется минимум 300-500 примеров.")
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )
    
    print(f"\n🔀 Разделение данных:")
    print(f"  Train: {len(X_train)} примеров")
    print(f"  Test: {len(X_test)} примеров")
    
    # Создаем pipeline
    print("\n🔧 Создание pipeline (TF-IDF + Logistic Regression)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),  # Униграммы и биграммы
            min_df=3,  # Минимум 3 вхождения слова
            max_df=0.9,  # Исключаем слишком частые слова
            max_features=10000  # Ограничение количества признаков
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # Балансируем классы
            random_state=42
        ))
    ])
    
    # Обучаем модель
    print("\n🚀 Обучение модели...")
    pipeline.fit(X_train, y_train)
    print("✓ Обучение завершено")
    
    # Оцениваем на тестовой выборке
    print("\n📈 Оценка качества модели:")
    y_pred = pipeline.predict(X_test)
    
    print("\n" + classification_report(y_test, y_pred, target_names=["Не спам", "Спам"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print(f"                Предсказано")
    print(f"              Не спам  Спам")
    print(f"Реально Не спам   {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"        Спам       {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    # Проверяем метрики
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    
    print(f"\n🎯 Ключевые метрики (класс 'Спам'):")
    print(f"  F1-score: {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    
    if f1 < 0.8:
        print("\n⚠ F1-score < 0.8. Рекомендуется собрать больше данных или улучшить разметку.")
    else:
        print("\n✓ F1-score >= 0.8. Модель готова к использованию!")
    
    # Сохраняем модель
    print(f"\n💾 Сохранение модели в {MODEL_FILE}...")
    joblib.dump(pipeline, MODEL_FILE)
    print("✓ Модель сохранена")
    
    print("\n" + "=" * 60)
    print("✅ Обучение завершено!")
    print(f"Модель: {MODEL_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

