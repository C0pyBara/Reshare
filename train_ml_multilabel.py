"""Обучение multilabel ML классификатора для определения категорий спама."""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from telegram_features import MetaFeatureExtractor

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
LABELED_CSV_FILE = DATA_DIR / "labeled_multilabel.csv"
MODEL_FILE = MODELS_DIR / "spam_ml_multilabel.pkl"

MODELS_DIR.mkdir(exist_ok=True)

LABELS = ["ads", "crypto", "scam", "casino"]

# Пороги для каждого класса (можно настроить)
THRESHOLDS = {
    "ads": 0.4,
    "crypto": 0.4,
    "scam": 0.3,  # Ниже порог для scam (важнее recall)
    "casino": 0.4
}


def main():
    """Обучает multilabel ML модель для классификации спама."""
    print("=" * 70)
    print("ОБУЧЕНИЕ MULTILABEL ML КЛАССИФИКАТОРА")
    print("=" * 70)
    
    if not LABELED_CSV_FILE.exists():
        print(f"❌ Файл {LABELED_CSV_FILE} не найден!")
        print("Сначала разметьте данные:")
        print("  python label_data_ml.py")
        return
    
    # Загружаем данные
    print(f"\n📂 Загрузка данных из {LABELED_CSV_FILE}...")
    try:
        df = pd.read_csv(LABELED_CSV_FILE)
        print(f"✓ Загружено {len(df)} примеров")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Проверяем наличие всех колонок
    required_cols = ["text"] + LABELS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Отсутствуют колонки: {missing_cols}")
        return
    
    # Проверяем баланс классов
    print(f"\n📊 Распределение меток:")
    for label in LABELS:
        count = df[label].sum()
        pct = (count / len(df)) * 100
        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
    
    if len(df) < 100:
        print("⚠ Мало данных для обучения. Рекомендуется минимум 300-500 примеров.")
    
    # Подготавливаем данные
    X = df["text"].fillna("")
    y = df[LABELS].values
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=None  # Для multilabel stratify сложнее, пропускаем
    )
    
    print(f"\n🔀 Разделение данных:")
    print(f"  Train: {len(X_train)} примеров")
    print(f"  Test: {len(X_test)} примеров")
    
    # Создаем pipeline с FeatureUnion (TF-IDF + мета-признаки)
    print("\n🔧 Создание pipeline (TF-IDF + Meta Features + OneVsRest LR)...")
    
    # Feature Union: объединяем текстовые и мета-признаки
    feature_union = FeatureUnion([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=10000
        )),
        ("meta", Pipeline([
            ("meta_extractor", MetaFeatureExtractor()),
            ("scaler", StandardScaler())
        ]))
    ])
    
    # OneVsRest для multilabel классификации
    pipeline = Pipeline([
        ("features", feature_union),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        ))
    ])
    
    # Обучаем модель
    print("\n🚀 Обучение модели...")
    pipeline.fit(X_train, y_train)
    print("✓ Обучение завершено")
    
    # Предсказываем вероятности
    # Для OneVsRestClassifier predict_proba возвращает список массивов
    # Каждый элемент списка - это массив [P(0), P(1)] для соответствующей метки
    proba_list = pipeline.predict_proba(X_test)
    
    # Преобразуем в удобный формат: массив (n_samples, n_labels) с вероятностями P(1)
    n_samples = len(X_test)
    y_pred_proba = np.zeros((n_samples, len(LABELS)))
    
    if isinstance(proba_list, list) and len(proba_list) == len(LABELS):
        for i in range(len(LABELS)):
            class_proba = proba_list[i]  # Массив (n_samples, 2)
            if isinstance(class_proba, np.ndarray) and class_proba.ndim == 2:
                y_pred_proba[:, i] = class_proba[:, 1]  # P(label=1)
            else:
                y_pred_proba[:, i] = 0.5  # Fallback
    else:
        # Fallback - предполагаем что это уже правильный формат
        y_pred_proba = np.array(proba_list)
    
    # Применяем пороги для каждого класса
    y_pred = np.zeros_like(y_test)
    for i, label in enumerate(LABELS):
        threshold = THRESHOLDS[label]
        y_pred[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
    
    # Оцениваем на тестовой выборке
    print("\n📈 Оценка качества модели:")
    print("\n" + classification_report(
        y_test, y_pred, 
        target_names=LABELS,
        zero_division=0
    ))
    
    # Micro и Macro F1
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"\n🎯 Общие метрики:")
    print(f"  Micro F1: {micro_f1:.3f}")
    print(f"  Macro F1: {macro_f1:.3f}")
    
    # F1 по каждому классу
    print(f"\n📊 F1-score по классам:")
    for i, label in enumerate(LABELS):
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        print(f"  {label:8s}: {f1:.3f}")
    
    # Проверяем production-метрики
    from sklearn.metrics import precision_score, recall_score
    
    print(f"\n🎯 Production метрики:")
    production_metrics = {
        "ads": {"recall": 0.90, "precision": 0.85},
        "crypto": {"recall": 0.85, "precision": 0.80},
        "scam": {"recall": 0.95, "precision": 0.75},  # Для scam recall важнее
        "casino": {"recall": 0.90, "precision": 0.85}
    }
    
    all_ok = True
    for i, label in enumerate(LABELS):
        recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
        precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
        target_recall = production_metrics[label]["recall"]
        target_precision = production_metrics[label]["precision"]
        
        recall_ok = recall >= target_recall
        precision_ok = precision >= target_precision
        
        status = "✓" if (recall_ok and precision_ok) else "⚠"
        if not (recall_ok and precision_ok):
            all_ok = False
        
        print(f"  {status} {label:8s}: recall={recall:.3f} (≥{target_recall}), precision={precision:.3f} (≥{target_precision})")
    
    if all_ok:
        print("\n✅ Все метрики соответствуют production требованиям!")
    else:
        print("\n⚠ Некоторые метрики ниже целевых. Рекомендуется:")
        print("  - Собрать больше данных")
        print("  - Настроить пороги (THRESHOLDS)")
        print("  - Улучшить разметку данных")
    
    # Сохраняем модель и пороги
    print(f"\n💾 Сохранение модели в {MODEL_FILE}...")
    model_data = {
        "pipeline": pipeline,
        "thresholds": THRESHOLDS,
        "labels": LABELS
    }
    joblib.dump(model_data, MODEL_FILE)
    print("✓ Модель сохранена")
    
    print("\n" + "=" * 70)
    print("✅ Обучение завершено!")
    print(f"Модель: {MODEL_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

