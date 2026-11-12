"""
fakenews_cli.py
----------------
CLI prediction system for the fake news detection model.

Menu:
  1. Check a single news headline / article
  2. Batch prediction from CSV file
  3. Compare all three models on a sentence
  4. Exit
"""

import os
import sys
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack

from fakenews_train import preprocess

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

MODEL_PATH    = os.path.join(BASE_DIR, "fakenews_model.pkl")
WORD_VEC_PATH = os.path.join(BASE_DIR, "fakenews_word_vec.pkl")
CHAR_VEC_PATH = os.path.join(BASE_DIR, "fakenews_char_vec.pkl")
LABELS_PATH   = os.path.join(BASE_DIR, "fakenews_labels.pkl")


# ── Load artifacts ────────────────────────────────────────────────────────────
def load_artifacts():
    for p in [MODEL_PATH, WORD_VEC_PATH, CHAR_VEC_PATH]:
        if not os.path.exists(p):
            print(f"Model file not found: {p}")
            print("Run fakenews_train.py first.")
            sys.exit(1)
    return (
        joblib.load(MODEL_PATH),
        joblib.load(WORD_VEC_PATH),
        joblib.load(CHAR_VEC_PATH),
        joblib.load(LABELS_PATH),
    )


# ── Core prediction ───────────────────────────────────────────────────────────
def predict_one(text: str, model, word_vec, char_vec):
    """
    Returns dict with label, confidence, and full probability breakdown.
    Shows both 'Fake' and 'Real' probabilities so the user can judge
    how confident the model is — not just the binary call.
    """
    clean = preprocess(text)
    if not clean:
        return {"error": "Could not extract features from this text."}

    word_feat = word_vec.transform([clean])
    char_feat = char_vec.transform([clean])
    combined  = hstack([word_feat, char_feat])

    proba     = model.predict_proba(combined)[0]
    classes   = model.classes_
    prob_dict = dict(zip(classes, proba))

    predicted  = classes[np.argmax(proba)]
    confidence = np.max(proba)

    return {
        "text":       text,
        "predicted":  predicted,
        "confidence": confidence,
        "probs":      prob_dict,
    }


def print_result(r: dict, show_bar: bool = True):
    if "error" in r:
        print(f"\n  ⚠  {r['error']}")
        return

    label = r["predicted"]
    conf  = r["confidence"] * 100
    icon  = "🛑 FAKE NEWS" if label == "Fake" else "✅ REAL NEWS"

    print(f"\n  Prediction  : {icon}")
    print(f"  Confidence  : {conf:.1f}%")

    if show_bar:
        print("\n  Probability breakdown:")
        for cls, p in sorted(r["probs"].items(), key=lambda x: -x[1]):
            bar  = "█" * int(p * 40)
            sym  = "🛑" if cls == "Fake" else "✅"
            print(f"    {sym} {cls:5s}  {p*100:5.1f}%  {bar}")


# ── Menu option 1: single text ────────────────────────────────────────────────
def single_mode(model, word_vec, char_vec):
    text = input("\n  Enter news headline or article text:\n  > ").strip()
    if not text:
        print("  Empty input, please try again.")
        return
    result = predict_one(text, model, word_vec, char_vec)
    print_result(result)


# ── Menu option 2: batch prediction from CSV ──────────────────────────────────
def batch_mode(model, word_vec, char_vec):
    """
    Reads CSV with a 'text' column, predicts each row,
    writes results + confidence scores to CSV.
    """
    path = input("\n  Enter path to CSV file (must have a 'text' column):\n  > ").strip()
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return

    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.strip().lower() == "text"), None)
    if col is None:
        print("  CSV must contain a column named 'text'.")
        return

    sentences = df[col].astype(str).tolist()
    print(f"\n  Processing {len(sentences)} sentences...")

    records = []
    for s in sentences:
        r = predict_one(s, model, word_vec, char_vec)
        if "error" in r:
            records.append({"text": s, "prediction": "ERROR",
                             "confidence_%": 0, "fake_%": 0, "real_%": 0})
        else:
            records.append({
                "text":          r["text"],
                "prediction":    r["predicted"],
                "confidence_%":  round(r["confidence"] * 100, 2),
                "fake_%":        round(r["probs"].get("Fake", 0) * 100, 2),
                "real_%":        round(r["probs"].get("Real", 0) * 100, 2),
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fakenews_batch_predictions.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    # Print a quick summary table
    out_df = pd.DataFrame(records)
    print("\n  Preview (first 10 rows):")
    print(out_df.head(10).to_string(index=False))

    fakes = (out_df["prediction"] == "Fake").sum()
    reals = (out_df["prediction"] == "Real").sum()
    print(f"\n  Summary: {fakes} Fake  |  {reals} Real  |  {len(out_df)} Total")


# ── Menu option 3: compare all models on a single sentence ───────────────────
def compare_mode():
    """
    Bonus: loads all three saved models (if present) and shows
    each one's prediction + confidence for the same input text.
    """
    # Re-load all three individually saved models for comparison
    # (training script saves only the best; here we retrain all three quickly
    #  or we just show what's available — we'll load from training results)
    print("\n  This option re-trains all three models in memory for comparison.")
    print("  Loading dataset...")

    try:
        from fakenews_train import (
            load_data, build_features, train_models
        )
        from sklearn.model_selection import train_test_split

        df = load_data("fakenews_dataset.csv")
        X = df["clean"]; y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_tr, X_te, wv, cv = build_features(X_train, X_test)
        all_models = train_models(X_tr, X_te, y_train, y_test)
    except Exception as e:
        print(f"  Could not load models: {e}")
        return

    text = input("\n  Enter news text to compare across models:\n  > ").strip()
    if not text:
        print("  Empty input.")
        return

    clean    = preprocess(text)
    w_feat   = wv.transform([clean])
    c_feat   = cv.transform([clean])
    combined = hstack([w_feat, c_feat])

    print(f"\n  {'Model':<22}  {'Prediction':<8}  {'Confidence':>10}")
    print("  " + "─" * 46)
    for name, res in all_models.items():
        clf    = res["model"]
        proba  = clf.predict_proba(combined)[0]
        pred   = clf.classes_[np.argmax(proba)]
        conf   = np.max(proba) * 100
        icon   = "🛑" if pred == "Fake" else "✅"
        print(f"  {name:<22}  {icon} {pred:<6}    {conf:>8.1f}%")


# ── Main menu ─────────────────────────────────────────────────────────────────
def main():
    model, word_vec, char_vec, labels = load_artifacts()

    while True:
        print("\n" + "═" * 50)
        print("   FAKE NEWS DETECTION SYSTEM")
        print("═" * 50)
        print("  1.  Check a single headline / article")
        print("  2.  Batch prediction from CSV file")
        print("  3.  Compare all models on one sentence")
        print("  4.  Exit")

        choice = input("\n  Select (1–4): ").strip()

        if   choice == "1": single_mode(model, word_vec, char_vec)
        elif choice == "2": batch_mode(model, word_vec, char_vec)
        elif choice == "3": compare_mode()
        elif choice == "4": print("  Goodbye!"); break
        else:               print("  Invalid choice. Please enter 1–4.")


if __name__ == "__main__":
    main()
