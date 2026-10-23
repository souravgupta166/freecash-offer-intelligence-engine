# Offer Intelligence Engine

An NLP and machine-learning prototype for exploring review sentiment, complaint patterns, text characteristics, and model performance through an interactive Streamlit dashboard.

**Important:** This development version uses the UCI Recipe Reviews and User Feedback dataset. It does not contain Freecash or Trustpilot reviews. Results from this dataset must not be presented as findings about either platform.

## Live Demo

Add the deployed Streamlit URL here once the application has been published.

## Project Overview

The project demonstrates a review-intelligence workflow:

1. Ingest and validate a licensed development dataset.
2. Prepare review text and rating-derived sentiment labels.
3. Train and evaluate a TF-IDF and Logistic Regression classifier.
4. Compare model configurations and inspect classification errors.
5. Present review distributions, text patterns, evaluation metrics, and review examples in Streamlit.
6. Maintain source provenance and access checks for future data collection.

## Dataset and Attribution

**Dataset:** UCI Recipe Reviews and User Feedback Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset)  
**DOI:** [10.24432/C5FG95](https://doi.org/10.24432/C5FG95)  
**License:** CC BY 4.0, as identified by the dataset record.

The data represents recipe reviews, not reward offers or earning-platform experiences. The project uses it as a substitute development dataset to test review-analysis methods without implying access to unauthorized platform data.

## Methodology

### 1. Data preparation

The ingestion pipeline selects relevant review fields, normalizes types, handles missing text and ratings, removes duplicate records, and saves processed datasets with provenance metadata.

### 2. Proxy sentiment labels

Sentiment labels are derived from star ratings:

| Star rating | Assigned label |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |
| 0 or unrated | Excluded from supervised labeling |

These labels are **rating-derived proxies**, not manually annotated sentiment. A review's rating may not perfectly match the sentiment expressed in its text.

### 3. Text classification

The baseline classifier uses:

- TF-IDF to convert text into numerical features.
- Unigrams and bigrams to represent individual words and two-word sequences.
- Logistic Regression with class weighting to address class imbalance.
- A stratified train/test split for model evaluation.

The TF-IDF vocabulary is fitted on training data through a scikit-learn pipeline to prevent test-data leakage.

### 4. Baseline and model comparison

The project compares the classifier against a majority-class dummy baseline and compares unigram-only features with unigram-plus-bigram features.

The dataset is highly imbalanced, with positive labels accounting for approximately 93.9% of the labeled examples. Accuracy alone is therefore insufficient to assess model quality. Balanced accuracy, macro F1, per-class precision and recall, and the confusion matrix provide additional evidence.

### 5. Threshold and error analysis

Experiments examine alternative classification thresholds, misclassified reviews, and performance across review-length groups. Threshold changes are evaluated rather than assumed to improve results. The current threshold experiment did not improve overall macro F1, so the default argmax decision rule was retained.

The project also explores selective prediction using review-length cutoffs. No reliable cutoff was established; the current evidence does not justify claiming that short reviews should automatically be excluded.

## Reported Evaluation Results

The baseline experiment used a stratified 80/20 train/test split with a fixed random seed.

| Metric | Majority baseline | TF-IDF + Logistic Regression |
|---|---:|---:|
| Accuracy | 93.92% | 92.56% |
| Balanced accuracy | 33.33% | 68.54% |
| Macro F1 | 32.29% | 61.60% |
| Weighted F1 | 90.98% | 93.40% |

The majority baseline achieves high accuracy by predicting the dominant positive class. Its low balanced accuracy and macro F1 show why those metrics matter for this imbalanced problem.

The classifier improves balanced accuracy and macro F1 relative to that baseline, but minority-class performance remains limited. These results describe this particular dataset and experiment; they are not evidence of model performance on Freecash or Trustpilot reviews.

## Dashboard Features

- Filter reviews by proxy sentiment and star rating.
- Search review text and filter by minimum text length.
- Explore sentiment and rating distributions.
- Inspect individual reviews.
- Explore frequent words and bigrams.
- View model metrics, per-class results, and confusion matrix.
- Export filtered review data as CSV.
- Inspect data provenance and limitations.

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   ├── metadata/
│   └── processed/
├── models/                 # Generated model artifacts are ignored
├── reports/
├── src/
│   ├── collectors/
│   └── processing/
└── tests/
```

## Run Locally

Python 3.14.7 was used during development. Install a compatible Python version and create a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard expects the processed CSV and baseline evaluation report to be present at the paths configured in `app.py`.

To run the tests:

```powershell
python -m pytest tests -v
```

## Data Collection and Responsible Use

Freecash and Trustpilot are registered as sources that are not approved for automated collection in the current project configuration. Website reachability does not establish permission to collect data.

Automated collection from these services must not be enabled without the required authorization and compliance review. Prefer an authorized API or a dataset whose license and terms permit the intended use. Record source, retrieval time, license or permission basis, and transformations for collected data.

## Limitations

- The development dataset is about recipes, not reward offers.
- Sentiment labels are proxies derived from ratings.
- The class distribution is strongly imbalanced.
- Minority-class predictions are less reliable than the positive-class predictions.
- Results depend on the dataset, split, preprocessing, and evaluation protocol.
- Model performance has not been validated on actual reward-platform reviews.
- A successful local test suite does not guarantee deployment readiness.

## Future Work

- Obtain appropriately authorized reward-offer and review data.
- Develop a domain-specific taxonomy for reward completion, tracking delays, eligibility, account issues, and payout complaints.
- Create a manually reviewed evaluation sample.
- Compare classical NLP baselines with stronger text models.
- Add reproducible experiment configuration and model-version tracking.
- Expand automated tests to cover dashboard loading and data-quality checks.

## License and Attribution

The dataset is attributed to the UCI Machine Learning Repository and its stated CC BY 4.0 license. The license for this project's original source code should be specified separately by the repository owner. Dataset licensing does not automatically determine the license for the code.
