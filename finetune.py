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
        #
        # self.features = nn.Sequential(*list(original_model.children())[:-1])
        # self.h_dim = original_model.last_linear.in_features
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.h_dim, 211)
        # )
        # self.fix_params()
        self.model = original_model
        dim_feats = original_model.last_linear.in_features
        self.model.last_linear = nn.Linear(dim_feats, 211)

    def fix_params(self):
        # Freeze those weights
        for p in self.model.parameters():
            p.requires_grad = False

    def train_params(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        # y = self.features(x)
        # y = self.classifier(y.view(-1, self.h_dim))
        self.model(x)
        return y


if __name__ == "__main__":
    for model_name in pretrainedmodels.model_names:
    # model_name = "nasnetalarge"  # "resnet152"
        if "imagenet" in pretrainedmodels.pretrained_settings[model_name]:
            print(model_name, pretrainedmodels.pretrained_settings[model_name]["imagenet"]["input_size"])
    # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # print(list(model.children())[-3:])
    # model = FineTuneModel(model)
    # model.train_params()
    # print(list(filter(lambda p: p.requires_grad, model.parameters())))
