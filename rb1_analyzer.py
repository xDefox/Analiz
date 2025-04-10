import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from Bio import SeqIO
import os


# ======================
# 1. Вспомогательные функции
# ======================
def load_rb1_sequence(fasta_file="RB1.fna"):
    """Загружает последовательность RB1 из FASTA-файла"""
    try:
        record = next(SeqIO.parse(fasta_file, "fasta"))
        return str(record.seq)
    except Exception as e:
        print(f"Ошибка загрузки FASTA: {e}. Используется тестовая последовательность.")
        return "ATGGCTCCCT" + "A" * 1990  # Заглушка


def encode_sequence(sequence, seq_length=2000):
    """Кодирует ДНК-последовательность в числовой формат"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    encoded = np.zeros(seq_length, dtype=np.int32)
    for i, nuc in enumerate(sequence[:seq_length]):
        encoded[i] = mapping.get(nuc, 0)
    return encoded


# ======================
# 2. Основной класс
# ======================
class RB1CancerRiskPredictor:
    def __init__(self, seq_length=2000):
        self.seq_length = seq_length
        self.model = self._build_model()

    def _build_model(self):
        """Создает архитектуру нейросети"""
        model = models.Sequential([
            layers.Input(shape=(self.seq_length,)),
            layers.Embedding(input_dim=4, output_dim=32),
            layers.Conv1D(32, kernel_size=10, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, epochs=5):
        """Обучение модели"""
        self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        self.model.save("rb1_model.keras")

    def predict_risk(self, sequence):
        """Предсказывает риск для новой последовательности"""
        encoded = encode_sequence(sequence)
        prediction = self.model.predict(np.array([encoded]))
        return ["Низкий", "Средний", "Высокий"][np.argmax(prediction)]


# ======================
# 3. Подготовка данных и запуск
# ======================
if __name__ == "__main__":
    print("Инициализация модели...")

    # 1. Загрузка последовательности RB1
    rb1_seq = load_rb1_sequence()

    # 2. Создание синтетического датасета (замените на реальные данные)
    X_train = np.array([encode_sequence(rb1_seq)] * 100)  # 100 примеров одной последовательности
    y_train = np.random.randint(0, 3, size=100)  # Случайные метки (0-2)

    # 3. Обучение
    predictor = RB1CancerRiskPredictor()
    print("Обучение модели...")
    predictor.train(X_train, y_train)

    # 4. Тестирование
    test_seq = rb1_seq[:500] + "T" * 1500  # Тестовая последовательность
    print(f"Прогноз риска: {predictor.predict_risk(test_seq)}")