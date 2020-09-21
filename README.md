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

# Python

```
class Method():
    def __init__(model, **kwargs):
        self.model

    @abc.abstractmethod
    def calculate(self, **kwargs):
        return
```
