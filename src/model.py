import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
from torchvision import models, transforms

from torchvision import transforms


class MultiInputModel(nn.Module):
    """docstring for MultiInputModel."""

    def __init__(
        self,
        features_img_1,
        features_img_2,
        input_tab,
        output_tab,
        in_features,
    ):
        super(MultiInputModel, self).__init__()
        self.features_img_1 = features_img_1
        self.features_img_2 = features_img_2
        self.tab_mlp = nn.Sequential(
            nn.Linear(input_tab, output_tab), nn.ReLU(inplace=True)
        )
        self.mul_img_features = 1 if self.features_img_2 is None else 2
        self.concat_mlp = nn.Linear(self.mul_img_features * in_features + output_tab, 1)

    def forward(self, img_1, img_2, tab):
        output_img = self.features_img_1(img_1)
        output_tab = self.tab_mlp(tab)

        if self.mul_img_features == 1:
            output_img_tab = torch.cat((output_img.squeeze(), output_tab), dim=1)
        else:
            output_img_2 = self.features_img_2(img_2)
            output_img_tab = torch.cat(
                (output_img.squeeze(), output_img_2.squeeze(), output_tab), dim=1
            )

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
    multi_input: bool,
    double_img: bool,
    output_tab: int,
    ft_size: int,
):

    if model_name == "regnet":
        model = models.regnet_y_800mf(pretrained)
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features, features_2, ft_size, output_tab, in_features
            )
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)
        input_size = 224

    if model_name == "mobile":
        model = models.mobilenet_v3_large(pretrained)
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features, features_2, ft_size, output_tab, in_features, 1
            )
        else:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, 1)
        input_size = 224

    if model_name == "efficient":
        model = models.mobilenet_v3_large(pretrained)
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, 1)
        input_size = 224

    if model_name == "shuffle":
        model = models.shufflenet_v2_x1_5()
        set_parameter_requires_grad(model, feature_extract)

        if multi_input:
            features = nn.Sequential(*list(model.children()))[:-1]
            in_features = 784
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features, features_2, ft_size, output_tab, in_features
            )
        else:
            num_ftrs = model.fc.in_features
            model.classifier[-1] = nn.Linear(num_ftrs, 1)
        input_size = 224

    return model, input_size
