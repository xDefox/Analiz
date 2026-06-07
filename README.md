# RB1 Mutation Classifier

Классификатор патогенных мутаций гена RB1 с использованием свёрточной нейронной сети.

## Описание

Проект анализирует последовательность гена RB1 и предсказывает наличие патогенных мутаций на основе данных из базы ClinVar. Модель обучается на фрагментах ДНК длиной 500 нуклеотидов и классифицирует их как «мутация» или «норма».

## Стек

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Biopython (SeqIO)
- Matplotlib
- scikit-learn

## Установка

```bash
pip install numpy pandas tensorflow biopython matplotlib scikit-learn
