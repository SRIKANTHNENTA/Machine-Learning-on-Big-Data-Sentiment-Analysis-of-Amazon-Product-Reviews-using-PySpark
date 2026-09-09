# Machine-Learning-on-Big-Data-Sentiment-Analysis-of-Amazon-Product-Reviews-using-PySpark
# Amazon Review Sentiment Analysis with PySpark

This project applies natural language processing and machine learning to Amazon product reviews. It combines Apache Spark for large-scale data loading and cleaning with SpaCy, NLTK, Gensim, and scikit-learn for text preparation, TF-IDF feature extraction, model training, and evaluation.

The main notebook compares four classifiers for review sentiment or rating classification:

- Random Forest
- Logistic Regression
- XGBoost
- Linear Support Vector Classifier (LinearSVC)

The project is implemented in:

- [Machine Learning on Big Data- Sentiment Analysis of Amazon Product Reviews using PySpark.ipynb](Machine%20Learning%20on%20Big%20Data-%20Sentiment%20Analysis%20of%20Amazon%20Product%20Reviews%20using%20PySpark.ipynb)

## Project Goals

- Load and clean a large Amazon reviews dataset with PySpark.
- Explore review length and frequent terms using sampled data.
- Normalise review text with tokenisation, stop-word removal, and lemmatisation.
- Convert review text into TF-IDF features.
- Train and compare several classification algorithms.
- Tune model hyperparameters with cross-validation.
- Evaluate predictions using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

## Workflow

1. Start a Spark session configured for large text data.
2. Load the CSV dataset and inspect its schema and sample rows.
3. Standardise the key columns as `review_text` and `star_rating`.
4. Remove null records and keep valid ratings from 1 to 5.
5. Sample up to 100,000 rows for exploratory visualisation.
6. Clean review text by removing URLs, HTML, mentions, hashtags, stop words, and non-alphabetic tokens, then lemmatise the remaining text.
7. Build a Gensim TF-IDF corpus and a scikit-learn TF-IDF matrix with up to 5,000 features.
8. Split the data into training and test sets.
9. Train the four baseline classifiers.
10. Tune model parameters with `GridSearchCV` and `RandomizedSearchCV`.
11. Compare the final models using reports, charts, and confusion matrices.

## Dataset

The notebook expects a CSV file named `AMAZON REVIEWS.csv`. In the supplied project folder, it is located beside the notebook.

The notebook currently loads the file using the Google Colab path:

```python
df_spark = spark.read.csv("/content/AMAZON REVIEWS.csv", header=True, inferSchema=True)
```

When running locally, replace that path with the local file path. For example:

```python
df_spark = spark.read.csv("AMAZON REVIEWS.csv", header=True, inferSchema=True)
```

The input should contain at least these fields:

| Original column | Purpose |
| --- | --- |
| `reviewText` | Review text used as the model input |
| `rating` | Numeric review rating used to create the target label |

Other metadata columns can remain in the file. The notebook drops rows containing null values and filters ratings to the inclusive range 1 to 5.

The dataset is large, so it should be handled responsibly and should not be committed to a public repository without checking its licensing and distribution terms.

## Requirements

- Python 3.9 or later
- Java runtime compatible with the installed PySpark version
- Jupyter Notebook or Google Colab
- Apache Spark / PySpark
- NumPy
- Pandas
- SciPy
- scikit-learn
- XGBoost
- SpaCy
- NLTK
- Gensim
- Matplotlib
- Seaborn
- WordCloud

## Installation

Create and activate a virtual environment, then install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyspark pandas numpy scipy scikit-learn xgboost spacy gensim nltk matplotlib seaborn wordcloud jupyter
```

Download the language resources used by the notebook:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The notebook contains package installation cells for Colab. When running locally, installing dependencies once in the environment is preferable to running those cells repeatedly.

## Running the Notebook

1. Place `AMAZON REVIEWS.csv` in the same directory as the notebook, or update the CSV path in the data-loading cell.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open the project notebook.
4. Run the cells from top to bottom. The notebook defines variables in earlier cells that are required by later modelling and visualisation cells.
5. Review the printed classification reports, model comparison chart, and confusion matrices.

For Google Colab, upload the notebook and dataset, mount Google Drive if needed, and update the dataset path to match the location of the uploaded CSV file.

## Reported Results

The notebook reports the following approximate baseline accuracies on its test split:

| Model | Accuracy |
| --- | ---: |
| Random Forest | 87.55% |
| Logistic Regression | 86.85% |
| XGBoost | 87.31% |
| LinearSVC | 88.82% |

LinearSVC achieved the highest reported accuracy and showed better negative-class recall than the other baseline models. The reports also indicate class imbalance: positive reviews are identified more reliably than negative reviews. For that reason, precision, recall, F1-score, and confusion matrices should be considered alongside accuracy.

These figures are notebook results, not guaranteed benchmarks. They may change with the dataset version, train/test split, dependency versions, preprocessing choices, and tuned parameters.

## Limitations and Future Improvements

- Make the target definition explicit and consistently distinguish rating prediction from binary sentiment classification.
- Use a fixed, documented train/test split and record the class distribution.
- Report macro-averaged metrics and per-class results as primary measures when classes are imbalanced.
- Avoid converting more data than necessary from Spark to Pandas for very large datasets.
- Consolidate repeated training and evaluation cells into reusable functions or pipelines.
- Save the fitted vectoriser and best model for later inference.
- Add a separate inference example for classifying new reviews.
- Update deprecated or version-sensitive model arguments, including XGBoost compatibility settings.

## License and Data Notice

No software license or dataset license is specified in the notebook. Add an appropriate license before distributing the project, and verify that the Amazon review dataset may legally be used and redistributed in your intended context.
