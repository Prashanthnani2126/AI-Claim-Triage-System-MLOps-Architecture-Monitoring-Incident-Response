# NCI Engineering and Evaluating AI Systems CA

This project classifies customer interaction messages into the dependent label hierarchy `Type 2`, `Type 3`, and `Type 4`. It refactors the starter prototype into a clean, reproducible ML engineering submission with training, evaluation, saved artefacts, batch inference, and architecture documentation.

## Folder Structure

```text
src/
  config.py
  data_loader.py
  preprocessing.py
  features.py
  train.py
  predict.py
  evaluation.py
  utils.py
  models/
    __init__.py
    base_model.py
    logistic_regression_model.py
    random_forest_model.py
data/
  AppGallery.csv
  Purchasing.csv
  train.csv
  new_messages.csv
artifacts/
  model.joblib
  vectorizer.joblib
  label_encoders.joblib
  metadata.json
outputs/
  metrics.json
  classification_report.csv
  predictions.csv
  error_analysis.csv
docs/
  training_pipeline_architecture.md
  inference_monitoring_architecture.md
```

The original dataset files remain in `data/AppGallery.csv` and `data/Purchasing.csv`. `data/train.csv` is the combined training file used by the reproducible pipeline.

## Installation

Run from the project root:

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py
```

Training validates the required columns, handles missing text, removes rows with missing labels, applies shared text preprocessing, creates TF-IDF features, trains Logistic Regression and Random Forest multi-output classifiers, and selects the best model by dependent hierarchical accuracy.

## Prediction

```bash
python src/predict.py --input data/new_messages.csv --output outputs/predictions.csv
```

The prediction script loads the same saved vectorizer, label encoders, and model used during training. The batch output includes `row_id`, input text, predicted `Type 2`, predicted `Type 3`, predicted `Type 4`, confidence, model name, and prediction timestamp.

## Outputs

- `artifacts/model.joblib`: selected trained model.
- `artifacts/vectorizer.joblib`: fitted TF-IDF vectorizer.
- `artifacts/label_encoders.joblib`: fitted label encoders for `Type 2`, `Type 3`, and `Type 4`.
- `artifacts/metadata.json`: reproducibility metadata.
- `outputs/metrics.json`: model comparison metrics and dependent hierarchical accuracy.
- `outputs/classification_report.csv`: per-label classification reports with precision, recall, F1-score, macro average, and weighted average.
- `outputs/predictions.csv`: test-set predictions from training or batch inference predictions when `predict.py` is run with that output path.
- `outputs/error_analysis.csv`: rows where the selected model did not achieve full hierarchical credit.

## Dependent Hierarchical Accuracy

The labels are not treated as three unrelated accuracy values. A row receives credit only when the hierarchy is respected:

- If `Type 2` is wrong, the row score is `0`, even when `Type 3` or `Type 4` match.
- If `Type 2` is correct but `Type 3` is wrong, the row score is `1/3`, even when `Type 4` matches.
- If `Type 2` and `Type 3` are correct but `Type 4` is wrong, the row score is `2/3`.
- If all three labels are correct, the row score is `1`.

The final dependent hierarchical accuracy is the mean of these row scores.

## Evidence of Execution

The following commands were run successfully from the project root:

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py --input data/new_messages.csv --output outputs/predictions.csv
```

The run saved the model artefacts, metrics, classification report, predictions, and error analysis files in the required folders.

## Verification Summary

Commands executed during verification:

- `pip install -r requirements.txt`
- `python src/train.py`
- `python src/predict.py --input data/new_messages.csv --output outputs/predictions.csv`

Generated artifact files:

- `artifacts/model.joblib`
- `artifacts/vectorizer.joblib`
- `artifacts/label_encoders.joblib`
- `artifacts/metadata.json`

Generated output files:

- `outputs/metrics.json`
- `outputs/classification_report.csv`
- `outputs/predictions.csv`
- `outputs/error_analysis.csv`

The dependent hierarchical accuracy rule was tested with explicit cases for `0`, `1/3`, `2/3`, and `1` row scores.

## Architecture Diagrams

Mermaid diagrams are included in:

- `docs/training_pipeline_architecture.md`
- `docs/inference_monitoring_architecture.md`
