# Deep Learning Interpretability Suite (DLIS)

DLIS aims to provide a user interface for those to understand the hidden workings behind neural networks.

## Workflow

1. Upload your model.
2. Choose method.
3. $$
   $$

# React Components

- Choose method
- Upload model
- Run button
- GradCam Component
- Activation Atlas Component

# Design
For the MVP, we have decided to focus on CNNs and regular NNs.

## CNNs
Here, we consider the following user scenarios:
* Classification
  * ImageNet (top10, top5, top1)
    * Show different probabilties for the top 10/5/1 classes
    * GradCam for top1
      * Can show a pretty graph of model to help the user pick out which
      layer to gradcam on.
  * Binary (thresholds)
    * Image, Segmentation
      * Threshold bar
* Segmentation
  * Softmax/Sigmoid
  * Binary (Thresholds if sigmoid)
  * t-SNE for features?

## NNs
* Tabular
  * SHAP values and graphs?
  * t-SNE
