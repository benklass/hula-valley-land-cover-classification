"""
Hula Valley Land Cover ML (Python)
- Input: CSV exported from GEE (features + label + rand)
- Models: Random Forest, SVM (RBF), ANN (MLP)
- Evaluation: confusion matrix, OA, producer/user accuracy, F1, AUC (OvR)
- Outputs: plots for appendix (confusion matrix + ROC curves)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


# -------------------------
# 1) LOAD DATA
# -------------------------
csv_path = Path("Hula_S2_DynamicWorld_samples.csv")
df = pd.read_csv(csv_path)

bands = ['B2','B3','B4','B8','B11','B12','NDVI','NDWI','MNDWI','NDBI']
target = 'label'

# Use the same split field exported from GEE
# train: rand < 0.7, test: rand >= 0.7
train_df = df[df['rand'] < 0.7].copy()
test_df  = df[df['rand'] >= 0.7].copy()

X_train = train_df[bands].values
y_train = train_df[target].astype(int).values
X_test  = test_df[bands].values
y_test  = test_df[target].astype(int).values

classes = np.unique(df[target].astype(int).values)
n_classes = len(classes)

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Classes:", classes)


# -------------------------
# 2) HELPERS: producer/user accuracy from confusion matrix
# -------------------------
def producer_user_accuracy(cm: np.ndarray):
    """
    Producer's accuracy (recall per class) = diag / column sum (reference total)
    User's accuracy (precision per class)  = diag / row sum (predicted total)
    """
    diag = np.diag(cm).astype(float)
    col_sum = cm.sum(axis=0).astype(float)
    row_sum = cm.sum(axis=1).astype(float)

    producer = np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum != 0)
    user = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum != 0)
    return producer, user


def evaluate_model(name, model, X_train, y_train, X_test, y_test, classes):
    """
    Trains model, prints metrics, saves confusion matrix plot, computes multiclass AUC (OvR).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    oa = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    producer, user = producer_user_accuracy(cm)

    print("\n" + "="*60)
    print(f"{name}")
    print("="*60)
    print("Overall accuracy:", round(oa, 4))
    print("F1 (macro):", round(f1_macro, 4))
    print("\nClassification report:\n", classification_report(y_test, y_pred, labels=classes))

    print("Producer's accuracy (per class):", np.round(producer, 4))
    print("User's accuracy (per class):", np.round(user, 4))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title(f"Confusion Matrix: {name}")
    fig.tight_layout()
    fig.savefig(f"confusion_{name.replace(' ', '_').lower()}.png", dpi=200)
    plt.close(fig)

    # AUC (One-vs-Rest)
    # Need scores/probabilities for each class
    y_test_bin = label_binarize(y_test, classes=classes)

    # Try predict_proba first; fallback to decision_function
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    else:
        # Some pipelines wrap estimators; handle that
        try:
            scores = model.decision_function(X_test)
        except Exception:
            scores = None

    auc_macro = None
    if scores is not None:
        # roc_auc_score supports multiclass='ovr' with probability-like scores
        auc_macro = roc_auc_score(y_test_bin, scores, average="macro", multi_class="ovr")
        print("AUC (macro, OvR):", round(auc_macro, 4))
    else:
        print("AUC not computed: model provides no probability/decision scores.")

    return model, cm, auc_macro


# -------------------------
# 3) DEFINE MODELS
# -------------------------

# Model 1: Tree-based (Random Forest)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

# Model 2: SVM (RBF) - use scaling + probability for AUC
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))
])

# Model 3: ANN (MLP) - use scaling
ann = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=42
    ))
])


# -------------------------
# 4) TRAIN + EVALUATE
# -------------------------
trained_rf, rf_cm, rf_auc = evaluate_model("Random Forest", rf, X_train, y_train, X_test, y_test, classes)
trained_svm, svm_cm, svm_auc = evaluate_model("SVM (RBF)", svm, X_train, y_train, X_test, y_test, classes)
trained_ann, ann_cm, ann_auc = evaluate_model("ANN (MLP)", ann, X_train, y_train, X_test, y_test, classes)


# -------------------------
# 5) OPTIONAL: ROC CURVES (OvR) for ONE model (example: ANN)
# -------------------------
def plot_roc_ovr(name, model, X_test, y_test, classes):
    y_bin = label_binarize(y_test, classes=classes)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    else:
        # pipeline case: try pull decision_function
        scores = model.decision_function(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        ax.plot(fpr, tpr, label=f"Class {c}")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (OvR): {name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"roc_{name.replace(' ', '_').lower()}.png", dpi=200)
    plt.close(fig)

plot_roc_ovr("ANN (MLP)", trained_ann, X_test, y_test, classes)

print("\nSaved figures: confusion_*.png and roc_*.png")

