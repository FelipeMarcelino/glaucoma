import time
import copy
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from dataset import init_dataloader
from model import init_model, init_optimizer, init_transforms


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    num_epochs=25,
    is_inception=False,
    multi_input=False,
    patient=10,
):
    since = time.time()

    val_acc_history = []
    val_auc_history = []
    val_sensitivity_history = []
    val_specificity_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0

    trigger_time = 0
    last_loss = np.inf

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            pred_epoch = []
            true_epoch = []

            # Iterate over data.
            for imgs_photo_1, imgs_photo_2, ft_numerical, labels in dataloaders[phase]:
                imgs_photo_1 = imgs_photo_1.to(device)
                imgs_photo_2 = imgs_photo_2.to(device)
                ft_numerical = ft_numerical.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                labels = labels.unsqueeze(1).float()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == "train":
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        if multi_input:
                            outputs, aux_outputs = model(
                                imgs_photo_1, imgs_photo_1, ft_numerical
                            )
                        else:
                            outputs, aux_outputs = model(imgs_photo_1)

                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        if multi_input:
                            outputs = model(imgs_photo_1, imgs_photo_2, ft_numerical)
                        else:
                            outputs = model(imgs_photo_1)

                        loss = criterion(outputs, labels)

                        sigmoid_outputs = torch.sigmoid(outputs)
                        preds = (sigmoid_outputs > 0.5).float()

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * imgs_photo_1.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pred_epoch.extend(list(preds.cpu().detach().numpy()))
                true_epoch.extend(list(labels.data.cpu().detach().numpy()))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_auc = roc_auc_score(true_epoch, pred_epoch)
            tn, fp, fn, tp = confusion_matrix(true_epoch, pred_epoch).ravel()

            # Calculate specificity
            specificity = tn / (tn + fp)

            # Calculate sensitivity
            sensitivity = tp / (tp + fn)

            print(
                "{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}, SP: {:.4f}, SN: {:.4f}".format(
                    phase, epoch_loss, epoch_acc, epoch_auc, specificity, sensitivity
                )
            )

            if epoch_loss > last_loss:
                trigger_time += 1
            else:
                trigger_time = 0

            if trigger_time >= patient:
                break

            last_loss = epoch_loss

            # deep copy the model
            if phase == "val" and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc.cpu().detach().numpy())
                val_auc_history.append(epoch_auc)
                val_sensitivity_history.append(sensitivity)
                val_specificity_history.append(specificity)

            if phase == "train":
                if epoch_loss > last_loss:
                    trigger_time += 1
                else:
                    trigger_time = 0

                if trigger_time >= patient:
                    break

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Auc: {:4f}".format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return (
        model,
        val_acc_history,
        val_auc_history,
        val_sensitivity_history,
        val_specificity_history,
    )


def pre_train(
    train,
    val,
    preprocessing_train,
    preprocessing_val,
    preprocessing_tab,
    batch_size,
    model,
    feature_extract,
    device,
    debug,
    ft_columns,
    double_img,
    optim,
    lr,
):

    dataloader_train = init_dataloader(
        train,
        preprocessing_train,
        preprocessing_tab,
        batch_size,
        ft_columns,
        double_img,
    )
    dataloader_val = init_dataloader(
        val, preprocessing_val, preprocessing_tab, batch_size, ft_columns, double_img
    )

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)

    optimizer = init_optimizer(model, feature_extract, debug, optim, lr)

    return model, optimizer, criterion, dataloader_train, dataloader_val
