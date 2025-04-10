import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from Bio import SeqIO
import matplotlib.pyplot as plt

# Отключаем ненужные предупреждения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# ======================
# 1. Загрузка и обработка данных
# ======================

def load_rb1_sequence(fasta_file="RB1.fna"):
    """Загружает последовательность RB1 из FASTA"""
    try:
        record = next(SeqIO.parse(fasta_file, "fasta"))
        sequence = str(record.seq)
        print(f"Загружена последовательность RB1 длиной {len(sequence)} нуклеотидов")
        return sequence
    except Exception as e:
        print(f"Ошибка загрузки FASTA: {e}")
        print("Используется тестовая последовательность")
        return "ATGC" * 50000  # Заглушка длиной 200k нуклеотидов


def load_clinvar_data(csv_file="RB1_mutations.csv"):
    """Загружает и обрабатывает данные из сложного CSV ClinVar"""
    try:
        # Чтение файла с учетом двойных кавычек
        df = pd.read_csv(csv_file, sep='\t', quotechar='"', on_bad_lines='skip')

        # Проверка необходимых колонок
        required_cols = ['Name', 'Germline classification', 'Molecular consequence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")

        # Фильтрация патогенных мутаций RB1
        pathogenic = df[
            (df['Germline classification'].isin(['Pathogenic', 'Likely pathogenic'])) &
            (df['Molecular consequence'].str.contains('frameshift|nonsense|splice', na=False, case=False)) &
            (df['Name'].str.contains('RB1', na=False))
            ].copy()

        # Извлечение позиций из названия варианта (например: c.9_42dup → 9)
        pathogenic['Position'] = pathogenic['Name'].str.extract(r'c\.(\d+)').astype(int)

        return pathogenic['Position'].unique()  # Уникальные позиции

    except Exception as e:
        print(f"Ошибка загрузки ClinVar: {str(e)}")
        return np.array([100, 200, 300])  # Тестовые мутации


def encode_sequences(sequences, seq_length=2000):
    """Кодирует ДНК-последовательности в числовой формат"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    encoded = np.zeros((len(sequences), seq_length), dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j in range(min(len(seq), seq_length)):
            encoded[i, j] = mapping.get(seq[j], 0)
    return encoded


def create_dataset(sequence, mutation_positions, window_size=2000):
    """Создает сбалансированный датасет"""
    # Карта мутаций
    mutation_map = np.zeros(len(sequence), dtype=np.int8)
    for pos in mutation_positions:
        if 0 < pos <= len(sequence):
            mutation_map[pos - 1] = 1  # 1-based → 0-based

    # Балансировка классов
    positive_indices = np.where(mutation_map == 1)[0]
    negative_indices = np.where(mutation_map == 0)[0]

    n_samples = min(500, len(positive_indices))  # Ограничиваем количество примеров
    selected_pos = np.random.choice(positive_indices, n_samples, replace=False)
    selected_neg = np.random.choice(negative_indices, n_samples, replace=False)

    # Создаем фрагменты
    X, y = [], []
    for pos in selected_pos:
        start = max(0, pos - window_size // 2)
        end = start + window_size
        fragment = sequence[start:end]
        if len(fragment) == window_size:  # Проверяем длину
            X.append(fragment)
            y.append(1)

    for pos in selected_neg:
        start = max(0, pos - window_size // 2)
        end = start + window_size
        fragment = sequence[start:end]
        if len(fragment) == window_size:
            X.append(fragment)
            y.append(0)

    return X, np.array(y)


# ======================
# 2. Модель и обучение
# ======================

class RB1MutationClassifier:
    def __init__(self, seq_length=2000):
        self.seq_length = seq_length
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.seq_length,)),
            layers.Embedding(input_dim=4, output_dim=32),
            layers.Conv1D(64, kernel_size=20, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        return model

    def train(self, X, y, epochs=10, batch_size=32):
        history = self.model.fit(
            X, y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callbacks.EarlyStopping(patience=3)],
            verbose=1
        )
        self._plot_metrics(history)
        return history

    def _plot_metrics(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Точность')
        plt.plot(history.history['val_accuracy'], label='Валидация')
        plt.title('Точность модели')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Потери')
        plt.plot(history.history['val_loss'], label='Валидация')
        plt.title('Потери модели')
        plt.legend()

        plt.show()

    def predict(self, sequence):
        """Предсказывает наличие мутации"""
        encoded = encode_sequences([sequence], self.seq_length)
        proba = self.model.predict(encoded, verbose=0)[0][0]
        return {
            "prediction": "Мутация" if proba > 0.5 else "Норма",
            "confidence": float(proba if proba > 0.5 else 1 - proba),
            "probability": float(proba)
        }


# ======================
# 3. Запуск программы
# ======================

if __name__ == "__main__":
    print("=" * 60)
    print("Классификатор патогенных мутаций RB1")
    print("=" * 60)

    # 1. Загрузка данных
    print("\nЗагрузка данных...")
    rb1_sequence = load_rb1_sequence()
    mutation_positions = load_clinvar_data()

    print(f"Найдено патогенных мутаций: {len(mutation_positions)}")

    # 2. Подготовка датасета
    print("\nСоздание датасета...")
    X, y = create_dataset(rb1_sequence, mutation_positions)
    X_encoded = encode_sequences(X)

    print(f"\nСтатистика датасета:")
    print(f"- Всего примеров: {len(y)}")
    print(f"- С мутациями: {sum(y)} ({(sum(y) / len(y)) * 100:.1f}%)")
    print(f"- Без мутаций: {len(y) - sum(y)} ({(1 - sum(y) / len(y)) * 100:.1f}%)")

    # 3. Обучение модели
    print("\nСоздание модели...")
    classifier = RB1MutationClassifier()

    print("\nОбучение модели...")
    history = classifier.train(X_encoded, y)

    # 4. Тестирование
    test_seq = rb1_sequence[10000:12000]  # Тестовый фрагмент
    result = classifier.predict(test_seq)

    print("\nРезультат теста:")
    print(f"- Предсказание: {result['prediction']}")
    print(f"- Достоверность: {result['confidence']:.2f}")
    print(f"- Вероятность: {result['probability']:.4f}")