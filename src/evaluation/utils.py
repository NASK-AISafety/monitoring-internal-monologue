import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from typing import Optional, List

def calculate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    binarize_output: bool = False,
    calculate_auc: bool = True,
    labels_ids=(0, 1),
    positive_ids:Optional[List[int]] = None,
):
    if scores.ndim == 1:
        scores = np.expand_dims(scores, axis=1)
    if scores.shape[1] == 1:
        scores = scores[:, 0]
        preds = (scores > 0.5).astype(int)
    elif scores.shape[1] == 2:
        scores = scores[:, 1]
        preds = (scores > 0.5).astype(int)
    else:
        preds = np.argmax(scores, axis=1).astype(int)
    if binarize_output:
        preds = preds >= 1
    if positive_ids is not None:
        preds = np.isin(preds, positive_ids).astype(int)
        scores = np.sum(scores[:, positive_ids], axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=labels_ids).ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
    scores_dict = {
        "accuracy": accuracy,
        "f1": f1,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "negative_support": int(tn + fp),
        "positive_support": int(tp + fn),
    }
    if calculate_auc:
        roc_auc = roc_auc_score(labels, scores, labels=labels_ids)
        ap = average_precision_score(labels, scores)
        scores_dict["roc_auc"] = roc_auc
        scores_dict["average_precision"] = ap
    return scores_dict

def calculate_scores_multidataset(
    scores: np.ndarray,
    labels: np.ndarray,
    calculate_auc: bool = True,
    labels_ids=(0, 1),
):
    preds = np.argmax(scores, axis=1).astype(int)
    preds = (preds % 2) == 1
    scores = np.sum(scores[:, 1::2], axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=labels_ids).ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
    scores_dict = {
        "accuracy": accuracy,
        "f1": f1,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "negative_support": int(tn + fp),
        "positive_support": int(tp + fn),
    }
    if calculate_auc:
        roc_auc = roc_auc_score(labels, scores, labels=labels_ids)
        ap = average_precision_score(labels, scores)
        scores_dict["roc_auc"] = roc_auc
        scores_dict["average_precision"] = ap
    return scores_dict

def calculate_scores_multiclass(
    scores: np.ndarray,
    labels: np.ndarray,
    calculate_auc: bool = True,
    **kwargs,
):
    preds = np.argmax(scores, axis=1).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0, average="macro")
    scores_dict = {
        "accuracy": accuracy,
        "f1": f1,
    }
    if calculate_auc:
        if scores.shape[1] == 2:
            roc_auc = roc_auc_score(labels, scores[:, 1])
            ap = average_precision_score(labels, scores[:, 1])
        else:
            roc_auc = roc_auc_score(labels, scores, multi_class="ovr")
            ap = average_precision_score(labels, scores)
        scores_dict["roc_auc"] = roc_auc
        scores_dict["average_precision"] = ap
    return scores_dict
