import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.transforms import ToPILImage

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def save_activation(module, input, output):
            self.activations = output.detach()

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_backward_hook(save_gradient)

    def generate_cam(self, input_image, target_class=None):
        input_image.requires_grad_()
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()