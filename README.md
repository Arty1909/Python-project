# Сравнительный анализ тональности заголовков новостей об ИИ

Тема: сравнительный анализ тональности заголовков новостей об искусственном интеллекте на сайтах РБК, VC.ru и РИА Новости за январь–май 2026 года.

## Состав проекта

- `data/ai_headlines_jan_may_2026.csv` — корпус заголовков.
- `scripts/analyze_tonality.py` — код анализа и визуализации.
- `results/` — готовые таблицы и графики.
- `docs/project_report.docx` — текстовый вариант проекта.
- `slides/ai_tonality_presentation.pptx` — презентация на 5 минут.

## Запуск

```bash
pip install -r requirements.txt
python scripts/analyze_tonality.py --input data/ai_headlines_jan_may_2026.csv --out results
```

## Столбцы корпуса

`id`, `site`, `date`, `title`, `url`, `keyword`, `sentiment`, `comment`.

## Итоги корпуса

Всего заголовков: 180.

Распределение по сайтам:

site
VC.ru          60
РИА Новости    60
РБК            60

Распределение тональности:

sentiment
позитивная     79
нейтральная    63
негативная     38
