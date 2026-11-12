"""
fakenews_train.py
------------------
Fake news detection system.

Pipeline:
  Raw text  →  Preprocessing (clean / tokenize / lemmatize)
            →  TF-IDF (word + char n-grams, combined)
            →  Logistic Regression  (primary)
            →  Naive Bayes          (comparison)
            →  Random Forest        (comparison / bonus)
            →  Evaluation + confusion matrix + feature importance
            →  Save best model + vectorizer with joblib
"""

import re
import os
import string
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from scipy.sparse import hstack

warnings.filterwarnings("ignore")

# ── NLTK setup ──────────────────────────────────────────────────────────────
for pkg in ["stopwords", "wordnet", "omw-1.4",
            "punkt", "punkt_tab", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()
OUT_DIR     = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1.  PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def _wordnet_pos(tag: str) -> str:
    """Map Penn Treebank POS tag to WordNet POS for accurate lemmatization."""
    if tag.startswith("J"): return "a"
    if tag.startswith("V"): return "v"
    if tag.startswith("R"): return "r"
    return "n"


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline:
      lowercase → strip URLs/numbers/punctuation → tokenise →
      remove stopwords → POS-aware lemmatise → rejoin.

    Fake news text often contains ALL-CAPS, exclamation spam, and
    sensationalist vocabulary — lowercasing and cleaning reveal the
    underlying token patterns the model learns from.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"\d+", " ", text)                       # remove numbers
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep letters only
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    tagged  = pos_tag(tokens)
    lemmas  = [LEMMATIZER.lemmatize(w, _wordnet_pos(t)) for w, t in tagged]

    return " ".join(lemmas)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    # strip surrounding quotes left by CSV formatting
    df["text"]  = df["text"].str.strip('"').str.strip("'")
    print(f"  Loaded {len(df)} samples  →  {df['label'].value_counts().to_dict()}")
    df["clean"] = df["text"].apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  (word TF-IDF  +  char TF-IDF, stacked)
# ════════════════════════════════════════════════════════════════════════════

def build_features(X_train, X_test):
    """
    Combine two TF-IDF representations:
      • Word-level (1–2 grams): captures vocabulary differences
        ("aliens", "suppressed", "conspiracy" vs factual vocab)
      • Char-level (3–5 grams): captures sensationalist writing style
        patterns independent of individual words

    Returns sparse matrices for train/test plus both fitted vectorizers.
    """
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8_000,
        sublinear_tf=True,
        min_df=2,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5_000,
        sublinear_tf=True,
        min_df=2,
    )

    X_train_word = word_vec.fit_transform(X_train)
    X_test_word  = word_vec.transform(X_test)

    X_train_char = char_vec.fit_transform(X_train)
    X_test_char  = char_vec.transform(X_test)

    X_train_combined = hstack([X_train_word, X_train_char])
    X_test_combined  = hstack([X_test_word,  X_test_char])

    return X_train_combined, X_test_combined, word_vec, char_vec


# ════════════════════════════════════════════════════════════════════════════
# 3.  MODEL TRAINING  (Logistic Regression, Naive Bayes, Random Forest)
# ════════════════════════════════════════════════════════════════════════════

def train_models(X_train, X_test, y_train, y_test):
    """Train three classifiers, return results dict keyed by model name."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.3),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "model":  clf,
            "y_pred": y_pred,
            "acc":    accuracy_score(y_test, y_pred),
            "prec":   precision_score(y_test, y_pred, pos_label="Fake", zero_division=0),
            "rec":    recall_score(y_test, y_pred,    pos_label="Fake", zero_division=0),
            "f1":     f1_score(y_test, y_pred,         pos_label="Fake", zero_division=0),
        }
        print(f"  {name:22s}  acc={results[name]['acc']:.4f}  "
              f"f1={results[name]['f1']:.4f}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION  (reports + confusion matrices + feature importance)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_best(results: dict, y_test, best_name: str):
    """Print full classification report for the best model."""
    res    = results[best_name]
    y_pred = res["y_pred"]

    print(f"\n{'='*55}")
    print(f"  BEST MODEL: {best_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {res['acc']:.4f}")
    print(f"  Precision: {res['prec']:.4f}")
    print(f"  Recall   : {res['rec']:.4f}")
    print(f"  F1-score : {res['f1']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")


def plot_all(results: dict, y_test, word_vec, best_name: str):
    """
    One figure, four panels:
      1. Model comparison bar chart
      2. Confusion matrix (best model)
      3. Top Fake-news feature words
      4. Top Real-news feature words
    """
    labels   = sorted(set(y_test))
    fig      = plt.figure(figsize=(16, 12))
    gs       = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: model comparison ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    names    = list(results.keys())
    metrics  = ["acc", "prec", "rec", "f1"]
    mlabels  = ["Accuracy", "Precision", "Recall", "F1"]
    x        = np.arange(len(names))
    width    = 0.18
    colors   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, (m, lbl, col) in enumerate(zip(metrics, mlabels, colors)):
        vals = [results[n][m] for n in names]
        ax1.bar(x + i * width, vals, width, label=lbl, color=col, alpha=0.85)

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Model Comparison", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: confusion matrix ────────────────────────────────────────
    ax2  = fig.add_subplot(gs[0, 1])
    best = results[best_name]
    cm   = confusion_matrix(y_test, best["y_pred"], labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax2, linewidths=0.5)
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("Actual",    fontsize=11)
    ax2.set_title(f"Confusion Matrix ({best_name})", fontsize=12, fontweight="bold")

    # ── Panels 3 & 4: top features per class (Logistic Regression coefs) ─
    lr_result = results.get("Logistic Regression", results[best_name])
    clf       = lr_result["model"]

    if hasattr(clf, "coef_"):
        coef      = clf.coef_[0]
        feat_names= word_vec.get_feature_names_out()
        top_n     = 20

        # Word-vec feature names only (char features appended after, so slice)
        word_n = len(feat_names)

        # Top FAKE indicators (most positive coef → pushes toward "Fake")
        top_fake_idx  = np.argsort(coef[:word_n])[-top_n:][::-1]
        top_real_idx  = np.argsort(coef[:word_n])[:top_n]

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.barh(range(top_n),
                 coef[:word_n][top_fake_idx][::-1],
                 color="#C44E52", alpha=0.85)
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels(feat_names[top_fake_idx][::-1], fontsize=8)
        ax3.set_title("Top Words → Fake News", fontsize=12, fontweight="bold")
        ax3.set_xlabel("LR Coefficient", fontsize=10)
        ax3.axvline(0, color="black", linewidth=0.5)
        ax3.grid(axis="x", alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.barh(range(top_n),
                 np.abs(coef[:word_n][top_real_idx])[::-1],
                 color="#55A868", alpha=0.85)
        ax4.set_yticks(range(top_n))
        ax4.set_yticklabels(feat_names[top_real_idx][::-1], fontsize=8)
        ax4.set_title("Top Words → Real News", fontsize=12, fontweight="bold")
        ax4.set_xlabel("|LR Coefficient|", fontsize=10)
        ax4.axvline(0, color="black", linewidth=0.5)
        ax4.grid(axis="x", alpha=0.3)

    out = os.path.join(OUT_DIR, "fakenews_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved → {out}")


# ════════════════════════════════════════════════════════════════════════════
# 5.  SAVE ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

def save_artifacts(best_model, word_vec, char_vec, label_list):
    paths = {
        "model":      "fakenews_model.pkl",
        "word_vec":   "fakenews_word_vec.pkl",
        "char_vec":   "fakenews_char_vec.pkl",
        "labels":     "fakenews_labels.pkl",
    }
    joblib.dump(best_model, paths["model"])
    joblib.dump(word_vec,   paths["word_vec"])
    joblib.dump(char_vec,   paths["char_vec"])
    joblib.dump(label_list, paths["labels"])

    for k, p in paths.items():
        print(f"  Saved {k:12s} → {p}")
    return paths


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n── Loading dataset ─────────────────────────────────")
    df = load_data("fakenews_dataset.csv")

    X = df["clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    print("\n── Building features ───────────────────────────────")
    X_tr_feat, X_te_feat, word_vec, char_vec = build_features(X_train, X_test)
    print(f"  Combined feature matrix: {X_tr_feat.shape[1]} features")

    print("\n── Training models ─────────────────────────────────")
    results = train_models(X_tr_feat, X_te_feat, y_train, y_test)

    best_name = max(results, key=lambda k: results[k]["f1"])
    evaluate_best(results, y_test, best_name)

    print("\n── Saving visualizations ───────────────────────────")
    plot_all(results, y_test, word_vec, best_name)

    print("\n── Saving model artifacts ──────────────────────────")
    save_artifacts(results[best_name]["model"], word_vec, char_vec,
                   sorted(y.unique()))

    # Export test predictions to CSV
    test_df = pd.DataFrame({
        "text":      X_test.values,
        "actual":    y_test.values,
        "predicted": results[best_name]["y_pred"],
    })
    out_csv = os.path.join(OUT_DIR, "fakenews_test_predictions.csv")
    test_df.to_csv(out_csv, index=False)
    print(f"  Test predictions saved → {out_csv}")

    print(f"\n✅ Done. Best model: {best_name}  "
          f"(Accuracy {results[best_name]['acc']:.1%}  "
          f"F1 {results[best_name]['f1']:.1%})")
