import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from Bio import SeqIO
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Настройки
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)


# ======================
# 1. Загрузка и обработка данных
# ======================

def load_rb1_sequence(fasta_file=r"D:\Education\Phyton\RB1_analiz\ data\raw\RB1.fna"):
    """Загружает последовательность RB1 из FASTA"""
    try:
        record = next(SeqIO.parse(fasta_file, "fasta"))
        sequence = str(record.seq)
        print(f"Загружена последовательность RB1 длиной {len(sequence)} нуклеотидов")
        return sequence
    except Exception as e:
        print(f"Ошибка загрузки FASTA: {e}")
        print("Создаю тестовую последовательность...")
        return "ATGC" * 50000  # Тестовая последовательность


def load_clinvar_data(csv_file="RB1_mutations.csv"):
    """Загружает и обрабатывает данные из ClinVar"""
    try:
        df = pd.read_csv(csv_file, sep='\t', quotechar='"', on_bad_lines='skip')

        # Проверка колонок
        if 'Name' not in df.columns or 'Germline classification' not in df.columns:
            raise ValueError("Файл CSV должен содержать колонки 'Name' и 'Germline classification'")

        # Фильтрация патогенных мутаций RB1
        pathogenic = df[
            (df['Germline classification'].isin(['Pathogenic', 'Likely pathogenic'])) &
            (df['Name'].str.contains('RB1', na=False))
            ].copy()

        # Извлечение позиций мутаций
        pathogenic['Position'] = pathogenic['Name'].str.extract(r'c\.(\d+)').astype(int)

        print(f"Первые 5 мутаций:\n{pathogenic[['Name', 'Position']].head()}")
        return pathogenic['Position'].unique()

    except Exception as e:
        print(f"Ошибка загрузки ClinVar: {str(e)}")
        return np.array([100, 500, 1000])  # Тестовые мутации


def encode_sequences(sequences, seq_length=500):
    """Кодирует ДНК в числовой формат (A=0, T=1, G=2, C=3)"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    encoded = np.zeros((len(sequences), seq_length), dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j in range(min(len(seq), seq_length)):
            encoded[i, j] = mapping.get(seq[j], 0)
    return encoded


def create_dataset(sequence, mutation_positions, window_size=500, n_samples=1000):
    """Создает сбалансированный датасет"""
    # Создаем карту мутаций
    mutation_map = np.zeros(len(sequence), dtype=np.int8)
    for pos in mutation_positions:
        if 0 < pos <= len(sequence):
            mutation_map[pos - 1] = 1  # 1-based в 0-based

    # Находим позиции для выборки
    positive_pos = np.where(mutation_map == 1)[0]
    negative_pos = np.where(mutation_map == 0)[0]

    # Балансируем классы
    n_each = min(n_samples // 2, len(positive_pos), len(negative_pos) // 10)
    pos_samples = np.random.choice(positive_pos, n_each, replace=False)
    neg_samples = np.random.choice(negative_pos, n_each, replace=False)

    # Собираем фрагменты
    X, y = [], []
    for pos in pos_samples:
        start = max(0, pos - window_size // 2)
        end = start + window_size
        fragment = sequence[start:end]
        if len(fragment) == window_size:
            X.append(fragment)
            y.append(1)

    for pos in neg_samples:
        start = max(0, pos - window_size // 2)
        end = start + window_size
        fragment = sequence[start:end]
        if len(fragment) == window_size:
            X.append(fragment)
            y.append(0)

    return X, np.array(y)


# ======================
# 2. Модель классификации
# ======================

class RB1MutationClassifier:
    def __init__(self, seq_length=500):
        self.seq_length = seq_length
        self.model = self._build_model()

    def _build_model(self):
        """Оптимизированная архитектура CNN"""
        model = models.Sequential([
            layers.Input(shape=(self.seq_length,)),
            layers.Embedding(input_dim=4, output_dim=8),
            layers.Conv1D(32, kernel_size=15, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(64, kernel_size=7, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        return model

    def train(self, X, y, epochs=30, batch_size=32):
        """Обучение с мониторингом"""
        # Балансировка весов классов
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weights = dict(enumerate(class_weights))

        # Коллбэки
        callbacks_list = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]

        # Обучение
        history = self.model.fit(
            X, y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )

        # Визуализация
        self._plot_metrics(history)
        return history

    def _plot_metrics(self, history):
        """Визуализация метрик обучения"""
        plt.figure(figsize=(15, 5))

        # Точность и AUC
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Точность (обучение)')
        plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
        plt.plot(history.history['auc'], label='AUC (обучение)')
        plt.plot(history.history['val_auc'], label='AUC (валидация)')
        plt.title('Точность и AUC')
        plt.xlabel('Эпоха')
        plt.legend()

        # Precision-Recall
        plt.subplot(1, 3, 2)
        plt.plot(history.history['precision'], label='Precision')
        plt.plot(history.history['recall'], label='Recall')
        plt.plot(history.history['val_precision'], label='Val Precision')
        plt.plot(history.history['val_recall'], label='Val Recall')
        plt.title('Precision и Recall')
        plt.xlabel('Эпоха')
        plt.legend()

        # Потери
        plt.subplot(1, 3, 3)
        plt.plot(history.history['loss'], label='Потери (обучение)')
        plt.plot(history.history['val_loss'], label='Потери (валидация)')
        plt.title('Потери')
        plt.xlabel('Эпоха')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(self, X_test, y_test):
        """Оценка модели на тестовых данных"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, results))

    def predict(self, sequence):
        """Предсказание для новой последовательности"""
        encoded = encode_sequences([sequence], self.seq_length)
        proba = self.model.predict(encoded, verbose=0)[0][0]
        return {
            "prediction": "Мутация" if proba > 0.5 else "Норма",
            "confidence": float(proba if proba > 0.5 else 1 - proba),
            "probability": float(proba)
        }


# ======================
# 3. Запуск обучения
# ======================

if __name__ == "__main__":
    print("=" * 60)
    print("Классификатор патогенных мутаций RB1")
    print("=" * 60)

    # 1. Загрузка данных
    print("\nЗагрузка данных...")
    rb1_sequence = load_rb1_sequence()
    mutation_positions = load_clinvar_data()

    # 2. Подготовка датасета
    print("\nСоздание датасета...")
    X, y = create_dataset(rb1_sequence, mutation_positions, n_samples=2000)
    X_encoded = encode_sequences(X)

    print("\nСтатистика датасета:")
    print(f"- Всего примеров: {len(y)}")
    print(f"- С мутациями: {sum(y)} ({sum(y) / len(y) * 100:.1f}%)")
    print(f"- Без мутаций: {len(y) - sum(y)} ({(1 - sum(y) / len(y)) * 100:.1f}%)")

    # 3. Обучение модели
    print("\nСоздание модели...")
    classifier = RB1MutationClassifier()

    print("\nОбучение модели...")
    history = classifier.train(X_encoded, y, epochs=50)

    # 4. Тестирование
    test_cases = [
        rb1_sequence[10000:10500],  # Нормальная область
        rb1_sequence[mutation_positions[0] - 250:mutation_positions[0] + 250]  # Область с мутацией
    ]

    print("\nТестирование модели:")
    for i, test_seq in enumerate(test_cases):
        result = classifier.predict(test_seq)
        print(f"\nТест {i + 1}:")
        print(f"- Длина: {len(test_seq)} нуклеотидов")
        print(f"- Предсказание: {result['prediction']}")
        print(f"- Вероятность: {result['probability']:.4f}")
        print(f"- Достоверность: {result['confidence']:.2f}")