import pandas as pd
import torch
import numpy as np
import shap
import pickle

CHANNELS = 3


def shap_values(
    model,
    photos1_train,
    photos2_train,
    ft_numerical_train,
    photos1_val,
    photos2_val,
    ft_numerical_val,
    input_size,
):
    """
    Return shap_values_photo1, shap_values_photo2, shap_values_nuermical
    """

    explainer = shap.DeepExplainer(
        model,
        [
            photos1_train,
            photos2_train,
            ft_numerical_train,
        ],
    )
    (
        shap_values_photo1,
        shap_values_photo2,
        shap_values_numerical,
    ) = explainer.shap_values([photos1_val, photos2_val, ft_numerical_val])

    return (
        shap_values_photo1.reshape(-1, input_size, input_size, CHANNELS),
        shap_values_photo2.reshape(-1, input_size, input_size, CHANNELS),
        shap_values_numerical,
    )


def get_samples_from_dataloader_balanced(dataloader, size):
    """
    Return photos1, photos2, ft_numerical, labels
    """

    list_of_photos_1 = []
    list_of_photos_2 = []
    list_of_features = []
    list_of_labels = []

    positive_class = 0
    negative_class = 0

    for photos1, photos2, numericalft, labels in dataloader:
        for i in range(len(labels)):
            if int(labels[i].numpy()) == 0 and negative_class >= (size / 2):
                continue
            elif int(labels[i].numpy()) == 0 and negative_class < (size / 2):
                negative_class += 1

            if int(labels[i].numpy()) == 1 and positive_class >= (size / 2):
                continue
            elif int(labels[i].numpy()) == 1 and positive_class < (size / 2):
                positive_class += 1

            list_of_photos_1.append(photos1[i])
            list_of_photos_2.append(photos2[i])
            list_of_features.append(numericalft[i])
            list_of_labels.append(labels[i].numpy())

        if len(list_of_labels) >= size:
            break

    photos1 = torch.stack(list_of_photos_1)
    photos2 = torch.stack(list_of_photos_2)
    ft_numerical = torch.stack(list_of_features)
    labels = np.array(list_of_labels)

    return (photos1, photos2, ft_numerical, labels)


def get_samples_from_dataloader(dataloader, size):
    """
    Return photos1, photos2, ft_numerical, labels
    """

    list_of_photos_1 = []
    list_of_photos_2 = []
    list_of_features = []
    list_of_labels = []

    for photos1, photos2, numericalft, labels in dataloader:
        for i in range(len(labels)):
            list_of_photos_1.append(photos1[i])
            list_of_photos_2.append(photos2[i])
            list_of_features.append(numericalft[i])
            list_of_labels.append(labels[i].numpy())

            if len(list_of_labels) >= size:
                break

        if len(list_of_labels) >= size:
            break

    photos1 = torch.stack(list_of_photos_1)
    photos2 = torch.stack(list_of_photos_2)
    ft_numerical = torch.stack(list_of_features)
    labels = np.array(list_of_labels)

    return (photos1, photos2, ft_numerical, labels)


def get_sigmoid_pred(
    model, train_loader, val_loader, output_tab, double_img, model_id, model_folder
):

    pred_list_train = []
    true_list_train = []
    pred_list_val = []
    true_list_val = []

    for imgs_photo_1, imgs_photo_2, ft_numerical, labels in train_loader:

        if double_img and not output_tab:
            outputs = model(imgs_photo_1, imgs_photo_2, None)
        elif double_img and output_tab:
            outputs = model(imgs_photo_1, imgs_photo_2, ft_numerical)
        elif not double_img and output_tab:
            outputs = model(imgs_photo_1, None, ft_numerical)
        else:
            outputs = model(imgs_photo_1)

        sigmoid_output = torch.sigmoid(outputs)

        pred_list_train.extend(
            sigmoid_output.data.cpu().detach().numpy().squeeze().tolist()
        )
        true_list_train.extend(labels.data.cpu().detach().numpy().tolist())

    for imgs_photo_1, imgs_photo_2, ft_numerical, labels in val_loader:

        if double_img and not output_tab:
            outputs = model(imgs_photo_1, imgs_photo_2, None)
        elif double_img and output_tab:
            outputs = model(imgs_photo_1, imgs_photo_2, ft_numerical)
        elif not double_img and output_tab:
            outputs = model(imgs_photo_1, None, ft_numerical)
        else:
            outputs = model(imgs_photo_1)

        sigmoid_output = torch.sigmoid(outputs)

        pred_list_val.extend(
            sigmoid_output.data.cpu().detach().numpy().squeeze().tolist()
        )
        true_list_val.extend(labels.data.cpu().detach().numpy().tolist())

    train_df_pred = pd.DataFrame(
        list(zip(pred_list_train, true_list_train)), columns=["pred_sigmoid", "true"]
    )
    train_df_pred["dataset"] = "train"

    val_df_pred = pd.DataFrame(
        list(zip(pred_list_val, true_list_val)), columns=["pred_sigmoid", "true"]
    )
    val_df_pred["dataset"] = "val"

    df_pred = pd.concat([train_df_pred, val_df_pred])
    df_pred["model_id"] = model_id

    df_pred.to_csv(model_folder + str(model_id) + "/" + "pred.csv", index=False)


def get_shap_values(
    model,
    train_dataloader,
    val_dataloader,
    oos_dataloader,
    balanced,
    size,
    input_size,
    path,
):

    print(size)
    if balanced:
        (
            photos1_train,
            photos2_train,
            ft_numerical_train,
            _,
        ) = get_samples_from_dataloader_balanced(train_dataloader, size)
    else:
        (
            photos1_train,
            photos2_train,
            ft_numerical_train,
            _,
        ) = get_samples_from_dataloader(train_dataloader, size)

    (
        photos1_val,
        photos2_val,
        ft_numerical_val,
        _,
    ) = get_samples_from_dataloader(val_dataloader, len(val_dataloader))

    print(photos1_train.shape)
    print(photos2_train.shape)
    print(ft_numerical_train.shape)
    print(photos1_val.shape)
    print(photos2_val.shape)
    print(ft_numerical_val.shape)
    (
        shap_values_photo1_val,
        shap_values_photo2_val,
        shap_values_numerical_val,
    ) = shap_values(
        model,
        photos1_train,
        photos2_train,
        ft_numerical_train,
        photos1_val,
        photos2_val,
        ft_numerical_val,
        input_size,
    )

    shap_dict_val = {}

    shap_dict_val["shap_values_photo1_val"] = shap_values_photo1_val
    shap_dict_val["shap_values_photo2_val"] = shap_values_photo2_val
    shap_dict_val["shap_values_numerical_val"] = shap_values_numerical_val

    name_file = (
        "balanced_shap_values_dict_val.pkl" if balanced else "shap_values_dict_val.pkl"
    )

    with open(path + name_file, "wb") as handle:
        pickle.dump(shap_dict_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if oos_dataloader:
        (
            photos1_oos,
            photos2_oos,
            ft_numerical_oos,
            _,
        ) = get_samples_from_dataloader(val_dataloader, len(oos_dataloader))

        (
            shap_values_photo1_oos,
            shap_values_photo2_oos,
            shap_values_numerical_oos,
        ) = shap_values(
            model,
            photos1_train,
            photos2_train,
            ft_numerical_train,
            photos1_oos,
            photos2_oos,
            ft_numerical_oos,
            input_size,
        )

        shap_dict_oos = {}

        shap_dict_oos["shap_values_photo1_oos"] = shap_values_photo1_oos
        shap_dict_oos["shap_values_photo2_oos"] = shap_values_photo2_oos
        shap_dict_oos["shap_values_numerical_oos"] = shap_values_numerical_oos

        name_file = (
            "balanced_shap_values_dict_oos.pkl"
            if balanced
            else "shap_values_dict_oos.pkl"
        )

        with open(path + name_file, "wb") as handle:
            pickle.dump(shap_dict_oos, handle, protocol=pickle.HIGHEST_PROTOCOL)
