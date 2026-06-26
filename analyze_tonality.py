from __future__ import annotations

import argparse
from pathlib import Path
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

LABELS = ["позитивная", "нейтральная", "негативная"]
REQUIRED_COLUMNS = {"id", "site", "date", "title", "url", "keyword", "sentiment_manual"}

STOPWORDS = {
    "и", "в", "на", "с", "для", "по", "о", "об", "к", "до", "за", "от", "из", "как", "что", "это", "или", "у",
    "не", "под", "над", "при", "года", "году", "2026", "ии", "нейросеть", "нейросети", "искусственный", "интеллект",
    "россии", "россия", "может", "новые", "главные", "стало", "стали", "самые", "новостей", "новости", "рынка",
    "данные", "который", "которая", "после", "надо", "ли", "ru", "рбк", "ria", "vc", "новость", "год", "экспертов", "эксперты", "разбор", "известно", "началу", "месяца", "июня", "мая", "марта", "апреля", "января", "февраля", "сентября", "октября", "ноября", "декабря", "почему", "где", "какие", "том", "так", "без", "уже", "все", "эта", "этот", "этом", "искусственного", "интеллекта"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ai_headlines_jan_may_2026.csv")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sentiment" in df.columns and "sentiment_manual" not in df.columns:
        df = df.rename(columns={"sentiment": "sentiment_manual"})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Не хватает столбцов: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "site", "title"]).copy()
    df["title"] = df["title"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["site"] = df["site"].replace({"RBC": "РБК", "RIA": "РИА Новости", "VC": "VC.ru"})
    df = df[(df["date"] >= "2026-01-01") & (df["date"] <= "2026-05-31")].copy()
    df = df.drop_duplicates(subset=["site", "date", "title"]).sort_values(["site", "date", "title"]).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)
    return df


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
    ])


def add_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame]:
    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(model, df["title"], df["sentiment_manual"], cv=cv)
    df = df.copy()
    df["sentiment_auto"] = pred
    df["sentiment_final"] = df["sentiment_manual"]
    accuracy = accuracy_score(df["sentiment_manual"], pred)
    report = classification_report(df["sentiment_manual"], pred, output_dict=True, zero_division=0)
    metrics = []
    for label in LABELS:
        metrics.append({
            "class": label,
            "precision": round(report[label]["precision"], 3),
            "recall": round(report[label]["recall"], 3),
            "f1": round(report[label]["f1-score"], 3),
            "support": int(report[label]["support"]),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.loc[len(metrics_df)] = {"class": "accuracy", "precision": "", "recall": "", "f1": round(accuracy, 3), "support": len(df)}
    cm = confusion_matrix(df["sentiment_manual"], pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"manual_{x}" for x in LABELS], columns=[f"auto_{x}" for x in LABELS]).reset_index().rename(columns={"index": "class"})
    return df, accuracy, metrics_df, cm_df


def make_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overall_counts = df["sentiment_final"].value_counts().reindex(LABELS).fillna(0).astype(int).rename_axis("sentiment").reset_index(name="count")
    overall_pct = overall_counts.copy()
    overall_pct["percent"] = (overall_pct["count"] / overall_pct["count"].sum() * 100).round(1)
    site_counts = pd.crosstab(df["site"], df["sentiment_final"]).reindex(columns=LABELS).reset_index()
    site_pct = pd.crosstab(df["site"], df["sentiment_final"], normalize="index").mul(100).reindex(columns=LABELS).round(1).reset_index()
    return {
        "overall_counts": overall_counts,
        "overall_pct": overall_pct,
        "site_counts": site_counts,
        "site_pct": site_pct,
    }


def tokenise(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z\-]{3,}", text.lower())


def top_words_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sentiment, group in df.groupby("sentiment_final"):
        counter = Counter()
        for title in group["title"]:
            counter.update(word for word in tokenise(title) if word not in STOPWORDS)
        for word, freq in counter.most_common(15):
            rows.append({"sentiment": sentiment, "word": word, "freq": freq})
    return pd.DataFrame(rows)


def save_tables(df: pd.DataFrame, accuracy: float, metrics_df: pd.DataFrame, cm_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = make_tables(df)
    tables["overall_counts"].to_csv(out_dir / "overall_sentiment_counts.csv", index=False, encoding="utf-8-sig")
    tables["overall_pct"].to_csv(out_dir / "overall_sentiment_percent.csv", index=False, encoding="utf-8-sig")
    tables["site_counts"].to_csv(out_dir / "sentiment_by_site_counts.csv", index=False, encoding="utf-8-sig")
    tables["site_pct"].to_csv(out_dir / "sentiment_by_site_percent.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "model_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    top_words_by_sentiment(df).to_csv(out_dir / "top_words_by_sentiment.csv", index=False, encoding="utf-8-sig")
    enriched = df.copy()
    enriched["date"] = enriched["date"].dt.date.astype(str)
    enriched[["id", "site", "date", "title", "url", "keyword", "sentiment_manual", "sentiment_auto", "sentiment_final", "comment"]].to_csv(
        out_dir / "corpus_with_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary = [
        f"Всего заголовков: {len(df)}",
        f"Точность 5-fold cross-validation: {accuracy:.1%}",
        "",
        "Распределение по сайтам:",
        make_tables(df)["site_counts"].to_string(index=False),
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")


def plot_overall_counts(overall_counts: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(overall_counts["sentiment"], overall_counts["count"])
    plt.title("Общее распределение тональности")
    plt.xlabel("Тональность")
    plt.ylabel("Количество заголовков")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_sentiment.png", dpi=200)
    plt.close()


def plot_site_counts(site_counts: pd.DataFrame, out_dir: Path) -> None:
    plot_df = site_counts.set_index("site")[LABELS]
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Тональность по сайтам: абсолютные значения")
    ax.set_xlabel("Сайт")
    ax.set_ylabel("Количество")
    ax.legend(title="Тональность")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_by_site_counts.png", dpi=200)
    plt.close()


def plot_site_percent(site_pct: pd.DataFrame, out_dir: Path) -> None:
    plot_df = site_pct.set_index("site")[LABELS]
    ax = plot_df.plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_title("Тональность по сайтам: доли, %")
    ax.set_xlabel("Сайт")
    ax.set_ylabel("Процент")
    ax.legend(title="Тональность", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_by_site_percent.png", dpi=200)
    plt.close()


def plot_top_words(df_words: pd.DataFrame, out_dir: Path) -> None:
    top = df_words.groupby("sentiment").head(5).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    y = []
    labels = []
    widths = []
    for idx, (_, row) in enumerate(top.iterrows()):
        y.append(idx)
        labels.append(f"{row['sentiment']}: {row['word']}")
        widths.append(row["freq"])
    ax.barh(y, widths)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Частотные маркеры по типам тональности")
    ax.set_xlabel("Частота")
    plt.tight_layout()
    plt.savefig(out_dir / "top_words_by_sentiment.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out)
    df = load_data(input_path)
    df, accuracy, metrics_df, cm_df = add_predictions(df)
    save_tables(df, accuracy, metrics_df, cm_df, out_dir)
    tables = make_tables(df)
    plot_overall_counts(tables["overall_counts"], out_dir)
    plot_site_counts(tables["site_counts"], out_dir)
    plot_site_percent(tables["site_pct"], out_dir)
    plot_top_words(top_words_by_sentiment(df), out_dir)
    print(f"Всего заголовков: {len(df)}")
    print(f"Точность 5-fold cross-validation: {accuracy:.1%}")
    print(f"Результаты сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
