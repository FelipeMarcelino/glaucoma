import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from torchvision import transforms


class MultiInputModel(nn.Module):
    """docstring for MultiInputModel."""

    def __init__(self, features_img, input_tab, output_tab, in_features, n_classes):
        super(MultiInputModel, self).__init__()
        self.features_img = features_img
        self.tab_mlp = nn.Sequential(
            nn.Linear(input_tab, output_tab), nn.ReLU(inplace=True)
        )
        self.concat_mlp = nn.Linear(in_features + output_tab, n_classes)

    def forward(self, img, tab):
        output_img = self.features_img(img)
        output_tab = self.tab_mlp(tab)

        output_img_tab = torch.cat((output_img.squeeze(), output_tab), dim=1)

        output = self.concat_mlp(output_img_tab)

        return output


def init_transforms(input_size: int):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    preprocessing_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # pre-processing val
    preprocessing_val = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    preprocessing_tab = transforms.Compose([transforms.ToTensor()])

    return preprocessing_train, preprocessing_val, preprocessing_tab


def init_optimizer(model, feature_extract, debug):

    params_to_update = model.parameters()

    if debug < 2:
        print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if debug < 2:
                    print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                if debug < 2:
                    print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    return optimizer


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def init_model(
    model_name: str,
    pretrained: bool,
    feature_extract: bool,
    n_classes: int,
    multi_input: bool,
    output_tab: int,
    ft_size: int,
):

    if model_name == "regnet":
        model = models.regnet_y_800mf(pretrained)
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            model = MultiInputModel(
                features, ft_size, output_tab, in_features, n_classes
            )
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_classes)
        input_size = 224

    if model_name == "mobile":
        model = models.mobilenet_v3_large(pretrained)
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            model = MultiInputModel(
                features, ft_size, output_tab, in_features, n_classes
            )
        else:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        input_size = 224

    if model_name == "efficient":
        model = models.mobilenet_v3_large(pretrained)
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            model = MultiInputModel(
                features, ft_size, output_tab, in_features, n_classes
            )
        else:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        input_size = 224

    if model_name == "shuffle":
        model = models.shufflenet_v2_x1_5()
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            model = MultiInputModel(
                features, ft_size, output_tab, in_features, n_classes
            )
        else:
            num_ftrs = model.fc.in_features
            model.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        input_size = 224

    return model, input_size
