# v2.8 Draft Single Model Export

This is the current best balanced single-model draft for MCP use.

Files:
- `best_model.pth`: recommended checkpoint
- `final_model.pth`: final post-train checkpoint
- `deploy_config.json`: deployment-facing config
- `labels.json`: class order
- `training_config.json`: original saved training config
- `training_history.json`: original training history

Load with:
- model: `tf_efficientnet_b0`
- num_classes: `6`
- input size: `336`
- resize: `int(336 * 1.143)` then center crop `336`
- normalize mean: `[0.485, 0.456, 0.406]`
- normalize std: `[0.229, 0.224, 0.225]`

Class order:
['illustration', 'painting_physical', 'real_photo', 'rendered_2d', 'rendered_3d', 'screenshot']

Best holdout metrics:
- harmonic_f1: 0.8654
- accuracy: 0.8743

Recommended default checkpoint: `best_model.pth`
