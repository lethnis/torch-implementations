from models import DenseNet
from torchinfo import summary
from torchview import draw_graph
import torch

model: DenseNet = DenseNet([4, 4, 4], 16, 0.5, False, 3, 10, 16)

x = torch.randn(1, 3, 128, 128)

summary(model, input_data=x, depth=2)
draw_graph(model, x, expand_nested=True, save_graph=True, depth=3)
