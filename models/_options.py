from typing import Literal

ResNet_options = Literal[
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]

DenseNet_options = Literal[
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "DenseNet264",
]

MobileNetv3_options = Literal[
    "Small",
    "Large",
]

EfficientNet_options = Literal["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
