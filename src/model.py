import sys
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# from pytorch_forecasting.optim import Ranger
from torch_optimizer import Ranger, RAdam
from torchvision import models, transforms as T
from torchinfo import summary

from torchvision import transforms


class MultiInputModel(nn.Module):
    """docstring for MultiInputModel."""

    def __init__(
        self,
        features_img_1,
        features_img_2,
        preprocessing_train,
        preprocessing_val,
        input_tab,
        output_tab,
        in_features,
    ):
        super(MultiInputModel, self).__init__()
        self.features_img_1 = features_img_1
        self.features_img_2 = features_img_2
        self.preprocessing_train = preprocessing_train
        self.preprocessing_val = preprocessing_val

        if output_tab:
            self.tab_mlp = nn.Sequential(
                nn.Linear(input_tab, int(input_tab / 2)),
                nn.ReLU(inplace=True),
                nn.Linear(int(input_tab / 2), output_tab),
                nn.ReLU(inplace=True),
            )
        else:
            self.tab_mlp = None
            output_tab = 0

        self.mul_img_features = 1 if self.features_img_2 is None else 2

        if output_tab or self.features_img_2:
            self.concat_mlp = nn.Sequential(
                nn.Linear(
                    self.mul_img_features * in_features + output_tab,
                    int((self.mul_img_features * in_features + output_tab) / 2),
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    int((self.mul_img_features * in_features + output_tab) / 2),
                    int((self.mul_img_features * in_features + output_tab) / 4),
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    int((self.mul_img_features * in_features + output_tab) / 4),
                    1,
                ),
            )
        else:
            self.concat_mlp = None

    def forward(self, img_1, img_2, tab):
        if self.training:
            trans_img1 = self.preprocessing_train(img_1)
        else:
            trans_img1 = self.preprocessing_val(img_1)

        output_img = self.features_img_1(trans_img1)

        if self.features_img_2:
            if self.training:
                trans_img2 = self.preprocessing_train(img_2)
            else:
                trans_img2 = self.preprocessing_val(img_2)

            output_img_2 = self.features_img_2(trans_img2)
            output_img_b = torch.cat(
                (output_img.squeeze(), output_img_2.squeeze()), dim=1
            )
        else:
            output_img_b = output_img

        if self.tab_mlp:
            output_tab = self.tab_mlp(tab)
            output_img_c = torch.cat((output_img_b.squeeze(), output_tab), dim=1)
        else:
            output_img_c = output_img_b

        if self.concat_mlp:
            output = self.concat_mlp(output_img_c)
        else:
            output = output_img_c

        return output


def init_transforms(input_size: int):

    # preprocessing_train = transforms.Compose(
    #     [
    #         transforms.ToPILImage(),
    #         transforms.RandomResizedCrop(input_size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]
    # )

    # # pre-processing val
    # preprocessing_val = transforms.Compose(
    #     [
    #         transforms.ToPILImage(),
    #         transforms.Resize(input_size),
    #         transforms.CenterCrop(input_size),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]
    # )

    preprocessing_train = nn.Sequential(
        T.RandomResizedCrop(input_size),
        T.RandomHorizontalFlip(),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    preprocessing_val = nn.Sequential(
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    return preprocessing_train, preprocessing_val


def init_optimizer(model, debug, optim_selected, lr):

    params_to_update = model.parameters()

    # Observe that all parameters are being optimized
    if optim_selected == "ranger":
        optimizer = Ranger(params_to_update, lr=lr)
    elif optim_selected == "adam":
        optimizer = optim.Adam(params_to_update, lr=lr)
    elif optim_selected == "radam":
        optimizer = RAdam(params_to_update, lr=lr)
    elif optim_selected == "sgd":
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)
    else:
        print("Optimizer not available")
        sys.exit(1)

    return optimizer


def init_model(
    model_name: str,
    pretrained: bool,
    double_img: bool,
    output_tab: int,
    ft_size: int,
):

    if model_name == "regnetx" or model_name == "regnet":
        if "regnet" == model_name:
            backbone = models.regnet_y_800mf(pretrained)
            in_features = 784
        else:
            backbone = models.regnet_x_800mf(pretrained)
            in_features = 672

        input_size = 224
        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            features = nn.Sequential(*list(backbone.children()))[:-1]
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "regnet16x" or model_name == "regnet16":
        if "regnet16" == model_name:
            backbone = models.regnet_y_1_6gf(pretrained)
            in_features = 888
        else:
            backbone = models.regnet_x_1_6gf(pretrained)
            in_features = 912

        input_size = 224
        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            features = nn.Sequential(*list(backbone.children()))[:-1]
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "regnet32x" or model_name == "regnet32":
        if "regnet32" == model_name:
            backbone = models.regnet_y_3_2gf(pretrained)
            in_features = 1512
        else:
            backbone = models.regnet_x_3_2gf(pretrained)
            in_features = 1008

        input_size = 224
        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            features = nn.Sequential(*list(backbone.children()))[:-1]
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "mobile":
        backbone = models.mobilenet_v3_large(pretrained)

        in_features = 960
        input_size = 224

        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            features = nn.Sequential(*list(backbone.children()))[:-1]
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:

            num_ftrs = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(num_ftrs, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "resnet":
        backbone = models.resnet50(pretrained)

        in_features = 2048
        input_size = 224

        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            features = nn.Sequential(*list(backbone.children()))[:-1]
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "efficient":
        backbone = models.efficientnet_b0(pretrained)

        in_features = 1280
        input_size = 224
        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            features = nn.Sequential(*list(backbone.children()))[:-1]
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(num_ftrs, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "shuffle":
        backbone = models.shufflenet_v2_x1_5(pretrained)
        in_features = 1024
        input_size = 224

        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            features = nn.Sequential(*list(backbone.children()))[:-1]
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "inception":
        backbone = models.inception_v3(pretrained)

        in_features = 2048
        input_size = 299
        preprocessing_train, preprocessing_val = init_transforms(input_size)

        if output_tab or double_img:
            backbone.fc = nn.Identity()
            backbone.aux_logits = False
            # features = nn.Sequential(*list(model.children()))[:-1]
            # in_features_aux = 768
            features = backbone
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            num_ftrs = backbone.fc.in_features
            num_ftrs_aux = backbone.AuxLogits.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, 1)
            backbone.AuxLogits.fc = nn.Linear(num_ftrs_aux, 1)

            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    if model_name == "vit":
        if double_img or output_tab:
            num_classes = 0
        else:
            num_classes = 1

        input_size = 224
        preprocessing_train, preprocessing_val = init_transforms(input_size)

        backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=num_classes
        )

        in_features = 768

        if output_tab or double_img:
            features = model
            if double_img:
                features_2 = copy.deepcopy(features)
            else:
                features_2 = None
            model = MultiInputModel(
                features,
                features_2,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )
        else:
            model = MultiInputModel(
                backbone,
                None,
                preprocessing_train,
                preprocessing_val,
                ft_size,
                output_tab,
                in_features,
            )

    return model, input_size
