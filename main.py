from models import ResNet

model: ResNet = ResNet.from_options("ResNet18")


def func(model: ResNet):
    print(type(model))


func(model)
