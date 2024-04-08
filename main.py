from torchinfo import summary
from torchview import draw_graph
import torch
from torchvision import models

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
summary(model, (1, 3, 224, 224), depth=3)
