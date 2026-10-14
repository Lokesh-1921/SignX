"""Metrics without hidden external dependencies."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("expected logits [N,C] and targets [N]")
    predictions = logits.argmax(dim=1)
    correct = predictions.eq(targets)
    class_count = logits.shape[1]
    f1_values: list[torch.Tensor] = []
    precision_values: list[torch.Tensor] = []
    recall_values: list[torch.Tensor] = []
    for class_index in range(class_count):
        predicted = predictions == class_index
        actual = targets == class_index
        tp = (predicted & actual).sum().float()
        precision = tp / predicted.sum().clamp_min(1)
        recall = tp / actual.sum().clamp_min(1)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(2 * precision * recall / (precision + recall).clamp_min(1e-12))
    maximum_k = min(5, class_count)
    topk = logits.topk(maximum_k, dim=1).indices
    return {
        "accuracy": correct.float().mean().item(),
        "top1_accuracy": correct.float().mean().item(),
        "top5_accuracy": topk.eq(targets[:, None]).any(dim=1).float().mean().item(),
        "precision_macro": torch.stack(precision_values).mean().item(),
        "recall_macro": torch.stack(recall_values).mean().item(),
        "f1_macro": torch.stack(f1_values).mean().item(),
    }


def word_error_rate(prediction: Sequence[str], reference: Sequence[str]) -> float:
    """Levenshtein word error rate for continuous-recognition evaluation."""

    if not reference:
        return 0.0 if not prediction else 1.0
    prior = list(range(len(reference) + 1))
    for pred_index, word in enumerate(prediction, start=1):
        current = [pred_index]
        for ref_index, ref_word in enumerate(reference, start=1):
            current.append(min(prior[ref_index] + 1, current[ref_index - 1] + 1, prior[ref_index - 1] + (word != ref_word)))
        prior = current
    return prior[-1] / len(reference)
