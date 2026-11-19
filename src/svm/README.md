# Java/Weka SVM workflow

Each operation has its own Java entry point:

- `ConvertToArff.java` converts the preprocessed JSONL reviews into ARFF.
- `Train.java` trains the final model on the full ARFF dataset and saves it.
- `Evaluate.java` loads the saved model's settings and runs stratified 10-fold CV.
- `SvmData.java` and `SvmClassifierFactory.java` contain shared data and classifier setup.

Python is not required.

## Setup

Run once from this folder in PowerShell:

```powershell
.\setup.ps1
```

This downloads Weka 3.8.6 dependencies and installs/copies the Weka LibLINEAR package. A JDK with `java` and `javac` on `PATH` is required.

## Run

```powershell
.\run.bat convert "D:\DataMining\Preprocessing\cleaned_minimal.jsonl" .\amazon_reviews_svm.arff
.\run.bat train .\amazon_reviews_svm.arff .\weka_svm.model
.\run.bat evaluate .\amazon_reviews_svm.arff .\weka_svm.model .\weka_cross_validation_results.json
```

`run.bat` compiles the Java sources into `.build` and dispatches to the requested entry point. The default Java heap is 8 GB; edit `-Xmx8g` in `run.bat` if needed.

Evaluation reads the saved model to reuse its Weka settings. It retrains a fresh classifier and text filter inside each fold, which is required for valid cross-validation and prevents vocabulary/IDF leakage from the full-data model.

## Model configuration

- The converter writes `review_text` as an ARFF string attribute and `sentiment_class` as a nominal class. `title` and `text` are combined.
- Weka `StringToWordVector` uses lowercase unigrams and bigrams, word counts, log TF, IDF, minimum term frequency 2, and up to 50,000 terms.
- Weka LibLINEAR uses the primal L2-loss linear SVM solver, `C=1`, and balanced class costs.
- Evaluation uses stratified 10-fold CV with seed 42.
- Each fold and the aggregate JSON report include accuracy, macro-averaged precision,
  macro-averaged recall, MAE, RMSE, and macro-averaged F1 score. MAE/RMSE use the
  ordered sentiment values Negative=0, Neutral=1, Positive=2, so confusing adjacent
  sentiment classes has a smaller error than confusing the two extremes.

## Outputs

- `amazon_reviews_svm.arff`: converted dataset.
- `weka_svm.model`: serialized final Weka `FilteredClassifier`.
- `weka_cross_validation_results.json`: fold metrics, aggregate metrics, confusion matrix, and runtime measurements.
