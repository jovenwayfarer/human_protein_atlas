import torch.nn as nn
import timm

class Net(nn.Module):
    def __init__(self, name = 'eca_nfnet_l1', num_classes=19):
        super(Net, self).__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.model(x)

        return out