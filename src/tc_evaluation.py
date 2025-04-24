import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def evaluate_classification(gs: pd.DataFrame, pred: pd.DataFrame) -> dict:
    y_true = gs['label']
    y_pred = pred['label']
    labels = sorted(set(y_true) | set(y_pred))  # Ensure consistency of labels

    # Compute overall scores for precision, recall, and F1
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')

    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # Detailed report per class
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    # Assemble result
    result = {
        "per_class": {
            label: {
                "precision": report[str(label)]["precision"],
                "recall": report[str(label)]["recall"],
                "f1": report[str(label)]["f1-score"],
                "support": report[str(label)]["support"]
            }
            for label in labels
        },
        "overall": {
            "precision_macro": precision_macro,
            "precision_micro": precision_micro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
            "recall_weighted": recall_weighted,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
        }
    }

    return result
