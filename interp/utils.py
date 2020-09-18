import lucent.optvis.objectives as objectives
import lucent.optvis.transform as transform
import torch


class ModuleHook(object):
    """
    Simple class for a torch module hook for storing the model and model output.
    This is mainly used for storing outputted feature maps.
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        """
        This method is called every time the gradient is calculated or during
        forward pass.
        """
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()

@torch.no_grad()
def get_layer(model, layer, X):
    """
    Gets the list of features (model outputs)

    Args:
        model (torch.nn.Module): torch model
        layer (str): layer name to extract from model
        X (torch.Tensor): input
    Returns:
        hook.features (list): Model outputs
    """
    hook = ModuleHook(getattr(model, layer))
    model(X)
    hook.close()
    return hook.features

def normalize_upsample(img):
    normalize = (
        transform.preprocess_inceptionv1()
        if model._get_name() == "InceptionV1"
        else transform.normalize()
    )
    transforms = transform.standard_transforms.copy() + [
        normalize,
        torch.nn.Upsample(size=224, mode="bilinear", align_corners=True),
    ]
    transforms_f = transform.compose(transforms)
    return transforms_f(img)
