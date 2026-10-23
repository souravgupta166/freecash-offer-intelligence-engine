
"""Offer Intelligence Engine — Streamlit NLP dashboard.

Development dataset: UCI Recipe Reviews and User Feedback.
This prototype does not contain Freecash or Trustpilot review data.
Sentiment labels are derived from star ratings, not human annotations.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "processed" / "uci_recipe_reviews_labeled.csv"
MODEL_REPORT = ROOT / "reports" / "sentiment_baseline_evaluation.json"

SENTIMENTS = ["negative", "neutral", "positive"]

st.set_page_config(
    page_title="Offer Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data
def load_reviews(path: str, modified_time: float) -> pd.DataFrame:
    """Load and validate the prepared review dataset."""
    df = pd.read_csv(path)

    required = {"text", "stars", "proxy_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["proxy_sentiment"] = df["proxy_sentiment"].fillna("unlabeled")
    df["text_length"] = df["text"].str.len()

    return df


@st.cache_data
def load_model_report(path: str, modified_time: float) -> dict:
    """Load the saved baseline evaluation report."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Offer Intelligence Engine")
st.caption(
    "Review analytics  ·  NLP prototype  ·  Model evaluation"
)

st.warning(
    "**DEVELOPMENT DATA:** This dashboard uses the UCI Recipe Reviews "
    "and User Feedback dataset. It does not contain Freecash or "
    "Trustpilot reviews. Sentiment labels are rating-derived proxies, "
    "not human-annotated text sentiment."
)

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info("Run the data ingestion and labeling scripts first.")
    st.stop()

try:
    reviews = load_reviews(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load the review dataset: {exc}")
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.title("Explore Reviews")
st.sidebar.caption("Filter the development dataset")

available_sentiments = [
    sentiment
    for sentiment in SENTIMENTS
    if sentiment in reviews["proxy_sentiment"].unique()
]

selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=available_sentiments,
    default=available_sentiments,
)

available_ratings = sorted(
    reviews["stars"].dropna().unique().tolist()
)

selected_ratings = st.sidebar.multiselect(
    "Star rating",
    options=available_ratings,
    default=available_ratings,
    format_func=lambda value: f"{int(value)} star"
    if float(value).is_integer()
    else str(value),
)

search_text = st.sidebar.text_input(
    "Search review text",
    placeholder="e.g. easy, delicious, bland",
)

max_available_length = int(reviews["text_length"].max())

min_length = st.sidebar.slider(
    "Minimum review length",
    min_value=0,
    max_value=max(10, min(500, max_available_length)),
    value=0,
    step=5,
)

# ---------------------------------------------------------
# Filter data
# ---------------------------------------------------------

filtered = reviews[
    reviews["proxy_sentiment"].isin(selected_sentiments)
    & reviews["stars"].isin(selected_ratings)
    & (reviews["text_length"] >= min_length)
].copy()

if search_text.strip():
    filtered = filtered[
        filtered["text"].str.contains(
            search_text.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ].copy()

rated_reviews = reviews[reviews["stars"].between(1, 5)].copy()
unrated_count = int((reviews["stars"] == 0).sum())

# ---------------------------------------------------------
# Overview KPIs
# ---------------------------------------------------------

st.subheader("Dataset Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Reviews matching filters", f"{len(filtered):,}")
k2.metric(
    "Positive",
    f"{(filtered['proxy_sentiment'] == 'positive').sum():,}",
)
k3.metric(
    "Negative",
    f"{(filtered['proxy_sentiment'] == 'negative').sum():,}",
)
k4.metric(
    "Neutral",
    f"{(filtered['proxy_sentiment'] == 'neutral').sum():,}",
)

with st.expander("Dataset provenance and quality"):
    p1, p2, p3 = st.columns(3)
    p1.metric("All processed reviews", f"{len(reviews):,}")
    p2.metric("Rated reviews", f"{len(rated_reviews):,}")
    p3.metric("Unrated reviews excluded from labeling", f"{unrated_count:,}")

    st.markdown(
        """
        **Source:** [UCI Recipe Reviews and User Feedback](https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset)

        **License:** CC BY 4.0. Attribution is required.

        **Label rules:** 1–2 stars = negative, 3 = neutral,
        4–5 = positive. A rating of 0 means no rating and is excluded
        from rating-based sentiment training.

        These rules create proxy labels. They do not guarantee that
        the written review expresses the same sentiment as its rating.
        """
    )

st.divider()


# ---------------------------------------------------------
# Distribution charts
# ---------------------------------------------------------

st.subheader("Review Distribution")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown("#### Sentiment")

    sentiment_counts = (
        filtered["proxy_sentiment"]
        .value_counts()
        .reindex(SENTIMENTS, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    if len(filtered):
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Reviews",
            color="Sentiment",
            category_orders={"Sentiment": SENTIMENTS},
            title="Reviews by proxy sentiment",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reviews match the selected filters.")

with chart_right:
    st.markdown("#### Original star ratings")

    rating_counts = (
        filtered["stars"]
        .value_counts()
        .sort_index()
        .rename_axis("Star rating")
        .reset_index(name="Reviews")
    )

    if not rating_counts.empty:
        fig = px.bar(
            rating_counts,
            x="Star rating",
            y="Reviews",
            title="Reviews by original rating",
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ratings match the selected filters.")

st.divider()


# ---------------------------------------------------------
# Review explorer
# ---------------------------------------------------------

st.subheader("Review Explorer")
st.caption(
    "Search and inspect the underlying review text and its rating-derived label."
)

display_columns = [
    "text",
    "stars",
    "proxy_sentiment",
    "text_length",
]

if filtered.empty:
    st.info("No reviews match your filters. Adjust the sidebar selections.")
else:
    st.dataframe(
        filtered[display_columns].sort_values(
            "text_length",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.download_button(
        "Download filtered reviews as CSV",
        data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="filtered_review_analysis.csv",
        mime="text/csv",
    )

st.divider()


# ---------------------------------------------------------
# Review Intelligence: recurring words and phrases
# ---------------------------------------------------------

st.subheader("Review Intelligence")
st.write(
    "Explore recurring terms and phrases within each sentiment group. "
    "These are descriptive patterns, not verified causes of sentiment."
)

theme_controls, theme_visual = st.columns([1, 2])

with theme_controls:
    theme_sentiment = st.selectbox(
        "Analyze sentiment",
        options=SENTIMENTS,
        index=0,
        key="theme_sentiment",
    )

    phrase_type = st.radio(
        "Term type",
        options=["Single words", "Two-word phrases"],
        index=1,
        key="theme_phrase_type",
    )

    top_n = st.slider(
        "Number of terms",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
        key="theme_top_n",
    )

theme_reviews = filtered.loc[
    filtered["proxy_sentiment"] == theme_sentiment,
    "text",
].dropna()

with theme_visual:
    st.markdown(f"#### Frequent terms · {theme_sentiment.title()}")
    st.caption(
        f"Analyzing {len(theme_reviews):,} reviews matching your filters."
    )

    if len(theme_reviews) < 2:
        st.info("Select filters that include at least two reviews.")
    else:
        ngram_range = (
            (1, 1)
            if phrase_type == "Single words"
            else (2, 2)
        )

        vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=ngram_range,
            min_df=2,
            max_features=5000,
        )

        try:
            term_matrix = vectorizer.fit_transform(theme_reviews)
            term_counts = term_matrix.sum(axis=0).A1
            terms = vectorizer.get_feature_names_out()

            term_table = pd.DataFrame({
                "Term": terms,
                "Occurrences": term_counts,
            }).sort_values(
                ["Occurrences", "Term"],
                ascending=[False, True],
            ).head(top_n)

            if term_table.empty:
                st.info(
                    "No repeated terms found. Try single words or "
                    "broaden the filters."
                )
            else:
                chart_data = term_table.sort_values("Occurrences")

                fig = px.bar(
                    chart_data,
                    x="Occurrences",
                    y="Term",
                    orientation="h",
                    title=f"Most frequent terms in {theme_sentiment} reviews",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    term_table,
                    use_container_width=True,
                    hide_index=True,
                )

        except ValueError:
            st.info(
                "Not enough repeated vocabulary for this selection. "
                "Try a different sentiment or broader filters."
            )

st.divider()


# ---------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------

st.subheader("Sentiment Model Evaluation")

if MODEL_REPORT.exists():
    try:
        report = load_model_report(
            str(MODEL_REPORT),
            MODEL_REPORT.stat().st_mtime,
        )
        metrics = report.get("tfidf_logistic_regression", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Test accuracy",
            f"{metrics.get('accuracy', 0):.1%}",
        )
        m2.metric(
            "Macro F1",
            f"{metrics.get('macro_f1', 0):.3f}",
        )
        m3.metric(
            "Balanced accuracy",
            f"{metrics.get('balanced_accuracy', 0):.1%}",
        )
        m4.metric(
            "Weighted F1",
            f"{metrics.get('weighted_f1', 0):.3f}",
        )

        st.caption(
            f"Training rows: {report.get('train_rows', 0):,} · "
            f"Test rows: {report.get('test_rows', 0):,} · "
            f"Random seed: {report.get('random_state', 'N/A')}"
        )

        st.info(
            "These metrics come from the original stratified 80/20 "
            "experiment. They are not metrics from the later "
            "60/20/20 threshold-tuning experiment."
        )

        with st.expander("View per-class evaluation metrics"):
            class_report = metrics.get("classification_report", {})
            rows = [
                {"Class": label, **class_report[label]}
                for label in SENTIMENTS
                if label in class_report
            ]

            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index("Class").round(3),
                    use_container_width=True,
                )

            matrix = metrics.get("confusion_matrix")
            label_order = report.get(
                "confusion_matrix_label_order",
                SENTIMENTS,
            )

            if matrix:
                st.markdown("**Confusion matrix**")
                st.caption("Rows = actual class; columns = predicted class.")
                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=label_order,
                        columns=label_order,
                    ),
                    use_container_width=True,
                )

    except (ValueError, OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read model evaluation report: {exc}")
else:
    st.info(
        "Model evaluation report not found. Run "
        "`python src/processing/train_sentiment_baseline.py` first."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()
st.caption(
    "Prototype status: UCI recipe-review development data only. "
    "Freecash and Trustpilot data collection requires appropriate authorization."
)
