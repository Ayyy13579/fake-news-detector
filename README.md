# 📰 Fake News Detector

A machine-learning pipeline that classifies news headlines and articles as **Fake** or **Real**, with an interactive command-line tool for single, batch, and multi-model predictions.

Built with Python, NLTK, and scikit-learn.

---

## ✨ Features

- **Full NLP preprocessing**: lowercasing, URL/number/punctuation removal, tokenisation, stop-word removal, and POS-aware lemmatisation
- **Hybrid TF-IDF features**: word n-grams (1–2) capture *vocabulary*; character n-grams (3–5) capture *writing style*; the two are stacked into one feature matrix
- **Three models compared**: Logistic Regression, Multinomial Naive Bayes, Random Forest. The best one (by F1) is saved automatically
- **Evaluation report**: accuracy, precision, recall, F1, confusion matrix, and the top words that push a prediction toward Fake or Real
- **Interactive CLI** with confidence scores and a probability breakdown for every prediction

---

## 🧠 How it works

```
Raw text
  → Preprocessing (clean → tokenise → remove stop-words → lemmatise)
  → TF-IDF  (word 1–2 grams  +  char 3–5 grams, stacked)
  → Logistic Regression / Naive Bayes / Random Forest
  → Evaluation + plots → best model saved with joblib
```

**Why two kinds of n-grams?** Fake news often differs from real news in *how* it is written (sensational phrasing, odd word endings) as much as *what* it says. Character n-grams pick up the style; word n-grams pick up the topic and vocabulary.

---

## 📁 Project structure

```
fake-news-detector/
├── fakenews_train.py        # preprocessing, training, evaluation, saving the model
├── fakenews_cli.py          # interactive prediction menu
├── fakenews_dataset.csv     # dataset (columns: text, label) – see "Dataset"
├── requirements.txt
├── README.md
├── .gitignore
└── (generated, git-ignored) *.pkl model files, fakenews_test_predictions.csv, outputs/
```

---

## 🚀 Getting started

```bash
# 1. Clone
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the models
python fakenews_train.py

# 5. Run the CLI
python fakenews_cli.py
```

NLTK data (stopwords, WordNet, tokenizer, POS tagger) is downloaded automatically on first run.

---

## 🖥️ CLI usage

```
══════════════════════════════════════════════════
   FAKE NEWS DETECTION SYSTEM
══════════════════════════════════════════════════
  1.  Check a single headline / article
  2.  Batch prediction from CSV file
  3.  Compare all models on one sentence
  4.  Exit
```

| Option | What it does |
|---|---|
| **1. Single check** | Enter a headline or article; get the label, confidence, and a Fake/Real probability bar |
| **2. Batch mode** | Provide a CSV with a `text` column; get a results CSV with prediction and confidence per row |
| **3. Compare models** | Retrains all three models in memory and shows each model's verdict side by side |

**Example output**

```
  Prediction  : 🛑 FAKE NEWS
  Confidence  : 91.4%

  Probability breakdown:
    🛑 Fake   91.4%  ████████████████████████████████████
    ✅ Real    8.6%  ███
```

---

## 📊 Results

> Replace this table with your own numbers after running `fakenews_train.py`.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | – | – | – | – |
| Naive Bayes | – | – | – | – |
| Random Forest | – | – | – | – |

Training also produces `outputs/fakenews_analysis.png` with a model comparison chart, confusion matrix, and top Fake/Real indicator words. Add it here:

```markdown
![Analysis](outputs/fakenews_analysis.png)
```

---

## 📂 Dataset

The scripts expect a CSV with two columns:

| Column | Description |
|---|---|
| `text` | Headline or article text |
| `label` | `Fake` or `Real` |

Describe your data source here (name, size, class balance, link). If the dataset is too large or its licence forbids redistribution, don't commit it. Link to it instead and tell users where to place the file.

---

## ⚠️ Limitations

- The model learns patterns in its **training data**; it does not fact-check claims. A well-written false statement can be classified as Real.
- Performance depends heavily on the dataset and may not generalise to other news sources, topics, or time periods.
- Treat predictions as a signal, not a verdict.

---

## 🔮 Future improvements

- Try transformer models (e.g. DistilBERT) and compare against the TF-IDF baseline
- Add cross-validation and hyperparameter tuning
- Wrap the model in a small web app (Flask / Streamlit)
- Add unit tests for the preprocessing function

---

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · NLTK · SciPy · matplotlib · seaborn · joblib

---

## 📄 License

MIT. See `LICENSE`.
