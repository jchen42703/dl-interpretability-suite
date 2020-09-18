# Deep Learning Interpretability Suite (DLIS)

DLIS aims to provide a user interface for those to understand the hidden workings behind neural networks.

## Workflow
1. Upload your model.
2. Choose method.
3. $$

## Assumptions
* Preprocessing must be specified.
  * Library provides preprocessing for standard `torchvision` models with `preprocess_img`
## Main classes
* `BaseMethod`: includes `model`, `input_image`, `device`, and `**params` as arguments
  * This is the abstract class that other interpretability methods will be called after
  * Methods:
    * `visualize`
      * has a save option
      * Need `visualize_img` function and `save_img` function
    * `run`
      * Runs the method (i.e. gradcam; gathers gradients, backprop, normalizes)
