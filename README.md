# Сравнительный анализ тональности заголовков новостей об ИИ

Тема проекта: **сравнительный анализ тональности заголовков новостей об искусственном интеллекте на сайтах РБК, VC.ru и РИА Новости за январь–май 2026 года**.

Проект собран под шаблон: **вопрос → метод → теория → результаты → выводы → ссылка на данные и код**.

## Что внутри

- `data/ai_headlines_jan_may_2026.csv` — основной корпус заголовков.
- `scripts/analyze_tonality.py` — рабочий Python-скрипт.
- `results/` — уже готовые таблицы, графики и итоговый текст.
- `docs/project_report.docx` — краткий письменный вариант проекта.
- `slides/ai_tonality_presentation.pptx` — презентация на защиту.
- `requirements.txt` — зависимости.

## Как запустить

```bash
pip install -r requirements.txt
python scripts/analyze_tonality.py --input data/ai_headlines_jan_may_2026.csv --out results
```

## Что делает код

1. Загружает корпус и проверяет структуру таблицы.
2. Нормализует даты и названия сайтов.
3. Удаляет дубли и ограничивает период январём–маем 2026 года.
4. Обучает модель `TF-IDF + Logistic Regression` на размеченных заголовках.
5. Считает `5-fold cross-validation`.
6. Строит итоговые таблицы и диаграммы.
7. Сохраняет расширенный корпус с автоматическим предсказанием.

## Основные результаты

- Всего заголовков: **180**
- Позитивная тональность: **79**
- Нейтральная тональность: **63**
- Негативная тональность: **38**
- Точность модели по `5-fold cross-validation`: **96.1%**

## Файлы результатов

- `results/overall_sentiment_counts.csv`
- `results/overall_sentiment_percent.csv`
- `results/sentiment_by_site_counts.csv`
- `results/sentiment_by_site_percent.csv`
- `results/model_metrics.csv`
- `results/model_confusion_matrix.csv`
- `results/corpus_with_predictions.csv`
- `results/overall_sentiment.png`
- `results/sentiment_by_site_counts.png`
- `results/sentiment_by_site_percent.png`
- `results/top_words_by_sentiment.png`

## Структура корпуса

Столбцы таблицы:

`id`, `site`, `date`, `title`, `url`, `keyword`, `sentiment_manual`, `sentiment_auto`, `sentiment_final`, `comment`
