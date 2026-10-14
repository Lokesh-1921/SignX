# Evaluation

The current classification evaluator reports accuracy, top-5 accuracy, macro precision, macro recall, and macro F1. `word_error_rate` is ready for the continuous-recognition stage. Translation will add BLEU, ROUGE, chrF, and semantic similarity only when a sign-to-text decoder and human-grounded translations exist.

Evaluate held-out signers, held-out regions, and held-out datasets separately. Accuracy from the synthetic smoke test is not a benchmark and must never be reported as sign-language performance. Error-analysis reports should include confidence, lighting/background, occlusion, signer, regional metadata where consented, and motion speed—not assumptions about the signer.
