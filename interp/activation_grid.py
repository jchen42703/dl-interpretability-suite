from itertools import product
import sys
from PIL import Image
from functools import partial

import numpy as np
import torch
import torchvision

from lucent.modelzoo import *
from lucent.misc.io import show
import lucent.optvis.objectives as objectives
import lucent.optvis.param as param
import lucent.optvis.render as render
from lucent.misc.channel_reducer import ChannelReducer

from .utils import ModuleHook, get_layer, normalize_upsample


class ActivationGrid(object):
    """
    Class for:
    1. computing the activation grid
    2. visualizing + saving the activation grid
    """
    def __init__(self, img, model, layer_name, cell_image_size=60, n_groups=6,
                 n_steps=1024, batch_size=64):
        self.img = img
        self.model = model
        self.layer_name = layer_name
        self.cell_image_size = cell_image_size
        self.n_groups = n_groups
        self.batch_size = batch_size

        self.objective = objectives.Objective(self.objective_func)

    def compute_activations(self, img):
        """
        For getting the activation maps.
        """
        # First wee need, to normalize and resize the image
        img = torch.tensor(np.transpose(img, [2, 0, 1])).to(device)
        # shape: (1, 3, original height of img, original width of img)
        img = img.unsqueeze(0)
        # shape: (1, 3, 224, 224)
        img = normalize_upsample(img)
        # Here we compute the activations of the layer `layer` using `img` as input
        # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
        acts = get_layer(model, layer, img)[0]
        # shape: (layer_height, layer_width, layer_channels)
        acts = acts.permute(1, 2, 0)
        # shape: (layer_height*layer_width, layer_channels)
        acts = acts.view(-1, acts.shape[-1])
        acts_np = acts.cpu().numpy()
        nb_cells = acts.shape[0]
        return (acts_np, nb_cells)

    def NMF(self, acts_np):
        """
        Negative matrix factorization `NMF` is used to reduce the number
        of channels to n_groups. This will be used as the following:
        - Each cell image in the grid is decomposed into a sum of
        (n_groups+1) images.
        - First, each cell has its own set of parameters this is what is called
        `cells_params` (see below).
        - At the same time, we have a group of images of size 'n_groups',
        which also have their own image parametrized by `groups_params`.
        - The resulting image for a given cell in the grid is the sum of its
        own image (parametrized by `cells_params`) plus a weighted sum of the
        images of the group.
            - Each image from the group is weighted by:
                `groups[cell_index, group_idx]`.
        Basically, this is a way of having the possibility to make cells
        with similar activations have a similar image, because cells with
        similar activations will have a similar weighting for the elements of
        the group.
        """
        if self.n_groups > 0:
            reducer = ChannelReducer(self.n_groups, "NMF")
            groups = reducer.fit_transform(acts_np)
            groups /= groups.max(0)
        else:
            groups = np.zeros([])
        # shape: (layer_height*layer_width, n_groups)
        groups = torch.from_numpy(groups)
        return groups

    def image_f(self, groups_image_f, cells_image_f):
        """
        First, we need to construct the images of the grid from the
        parameterizations
        """
        groups_images = groups_image_f()
        cells_images = cells_image_f()
        X = []
        for i in range(nb_cells):
            x = 0.7 * cells_images[i] + 0.5 * sum(
                groups[i, j] * groups_images[j] for j in range(n_groups)
            )
            X.append(x)
        X = torch.stack(X)
        return X

    def sample(self, image_f, batch_size):
        """
        After constructing the cells images, we sample randomly a mini-batch of
        cells from the grid. This is to prevent memory overflow, especially if
        the grid is large.
        """
        def f():
            X = image_f()
            inds = torch.randint(0, len(X), size=(batch_size,))
            inputs = X[inds]
            # HACK to store indices of the mini-batch, because we need them
            # in objective func. Might be better ways to do that
            sample.inds = inds
            return inputs

        return f

    def objective_func(self, model):
        """
        Objective function for the activation grid.
        """
        # shape: (batch_size, layer_channels, cell_layer_height, cell_layer_width)
        pred = model(layer)
        # use the sampled indices from `sample` to get the corresponding targets
        target = acts[sample.inds].to(pred.device)
        # shape: (batch_size, layer_channels, 1, 1)
        target = target.view(target.shape[0], target.shape[1], 1, 1)
        dot = (pred * target).sum(dim=1).mean()
        return -dot

    def param_f(self, groups_params, cells_params):
        """
        ???
        """
        # We optimize the parametrizations of both the groups and the cells
        params = list(groups_params) + list(cells_params)
        return params, image_f_sampled

    def render_and_save_grid(self):
        """
        1. Computes activation.
        2. runs NMF
        3. Parameterization
        4. Randomly sample a bunch of cells to prevent memory overflow
        5. render_vis with the desired objective function
        """
        acts_np, nb_cells = self.compute_activations(self.img)
        groups = self.NMF(acts_np)
        # Parametrization of the images of the groups (we have 'n_groups' groups)
        groups_params, groups_image_f = param.fft_image(
            [self.n_groups, 3, self.cell_image_size, self.cell_image_size]
        )
        # Parametrization of the images of each cell in the grid (we have 'layer_height*layer_width' cells)
        cells_params, cells_image_f = param.fft_image(
            [nb_cells, 3, self.cell_image_size, self.cell_image_size]
        )
        # make sure the images are between 0 and 1
        image_f = param.to_valid_rgb(partial(self.image_f,
                                             groups_image_f=groups_image_f,
                                             cells_image_f=cells_image_f),
                                             decorrelate=True)
        image_f_sampled = sample(image_f, batch_size=batch_size)

        results = render.render_vis(
            self.model,
            self.objective,
            self.param_f,
            thresholds=(self.n_steps,),
            show_image=False,
            progress=True,
            fixed_image_size=self.cell_image_size,
        )
        # shape: (layer_height*layer_width, 3, grid_image_size, grid_image_size)
        imgs = image_f()
        imgs = imgs.cpu().data
        imgs = imgs[:, :, 2:-2, 2:-2]
        # turn imgs into a a grid
        grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(nb_cells)),
                                           padding=0)
        grid = grid.permute(1, 2, 0)
        grid = grid.numpy()
        render.show(grid)
