from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
#from torch.autograd import Function


class ResNet(nn.Module):
    def __init__(self, option='resnet101', pret=True, class_size=211):
        super(ResNet, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        self.linear1 = nn.Linear(self.dim, class_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        x = self.linear1(x)
        return x