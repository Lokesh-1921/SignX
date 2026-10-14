import pytest
import torch

from evaluation.metrics import classification_metrics, word_error_rate


def test_classification_metrics_and_top5():
    metrics = classification_metrics(torch.tensor([[4.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 1]))
    assert metrics["accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_word_error_rate():
    assert word_error_rate("YOU COME".split(), "YOU COME".split()) == 0.0
    assert word_error_rate("YOU GO".split(), "YOU COME".split()) == pytest.approx(0.5)
