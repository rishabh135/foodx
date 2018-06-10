import pretrainedmodels
import torch.nn as nn


# print(pretrainedmodels.model_names)
# print(pretrainedmodels.pretrained_settings[model_name])
#
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# print(list(model.children())[-2:])

class FineTuneModel(nn.Module):
    def __init__(self, original_model):
        super(FineTuneModel, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 211)
        )
        self.fix_params()

    def fix_params(self):
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def train_params(self):
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x):
        y = self.features(x)
        z = y
        y = self.classifier(y.view(-1, 2048))
        return y,z


if __name__ == "__main__":
    model_name = "resnet152"
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = FineTuneModel(model)
    model.train_params()
    print(list(filter(lambda p: p.requires_grad, model.parameters())))