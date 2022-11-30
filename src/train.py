import time
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from dataset import init_dataloader
from model import init_optimizer
from pytorch_memlab import profile

num_accum_grad = 2


@profile
def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    double_img,
    output_tab,
    is_inception,
    early_start=100,
    num_epochs=100,
    patient=10,
    autocast=False,
    cudnn_bench=False,
):
    since = time.time()

    val_acc_history = []
    val_auc_history = []
    val_sensitivity_history = []
    val_specificity_history = []
    val_loss_history = []
    train_auc_history = []
    train_acc_history = []
    train_sensitivity_history = []
    train_specificity_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    trigger_time = 0
    last_loss = np.inf

    scaler = torch.cuda.amp.GradScaler(enabled=autocast)

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    torch.backends.cudnn.benchmark = cudnn_bench

    for epoch in range(num_epochs):
        start_epoch_time = time.time()
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
            for iter_batch, (
                imgs_photo_1,
                imgs_photo_2,
                ft_numerical,
                labels,
            ) in enumerate(dataloaders[phase]):

                imgs_photo_1 = imgs_photo_1.to(
                    device,
                    non_blocking=True,
                )
                imgs_photo_2 = imgs_photo_2.to(
                    device,
                    non_blocking=True,
                )
                ft_numerical = ft_numerical.to(
                    device,
                    non_blocking=True,
                )
                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                # zero the parameter gradients
                labels = labels.unsqueeze(1).float()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    with torch.autocast(
                        device_type=device_type,
                        dtype=torch.float16,
                        enabled=autocast,
                    ):

                        outputs = model(imgs_photo_1, imgs_photo_2, ft_numerical)

                        loss = criterion(outputs, labels)

                    sigmoid_outputs = torch.sigmoid(outputs)
                    preds = (sigmoid_outputs > 0.5).float()

                    # backward + optimize only if in training phase
                    if phase == "train":
                        # loss.backward()
                        # optimizer.step()
                        scaler.scale(loss).backward()

                        if (iter_batch + 1) % num_accum_grad == 0 or (
                            iter_batch + 1
                        ) == len(dataloaders[phase]):
                            scaler.step(optimizer)
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()

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

            # deep copy the model
            if phase == "val" and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc.cpu().detach().numpy())
                val_auc_history.append(epoch_auc)
                val_sensitivity_history.append(sensitivity)
                val_specificity_history.append(specificity)
                val_loss_history.append(running_loss)

            if phase == "train":
                train_acc_history.append(epoch_acc.cpu().detach().numpy())
                train_auc_history.append(epoch_auc)
                train_sensitivity_history.append(sensitivity)
                train_specificity_history.append(specificity)
                train_loss_history.append(running_loss)

            if phase == "val" and epoch >= early_start:
                if epoch_loss > last_loss:
                    trigger_time += 1
                else:
                    trigger_time = 0

                last_loss = epoch_loss

                if trigger_time >= patient:
                    break

        time_elapsed_epoch = time.time() - start_epoch_time

        print(f"Epoch in {int(time_elapsed_epoch)}/s")
        print("")

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
        val_loss_history,
        val_acc_history,
        val_auc_history,
        val_sensitivity_history,
        val_specificity_history,
        train_loss_history,
        train_acc_history,
        train_auc_history,
        train_sensitivity_history,
        train_specificity_history,
    )


def pre_train(
    train,
    val,
    batch_size,
    model,
    device,
    debug,
    ft_columns,
    double_img,
    optim,
    lr,
):

    dataloader_train = init_dataloader(
        train,
        batch_size,
        ft_columns,
        double_img,
        device,
    )

    dataloader_val = init_dataloader(
        val,
        batch_size,
        ft_columns,
        double_img,
        device,
    )

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)

    optimizer = init_optimizer(model, debug, optim, lr)

    return model, optimizer, criterion, dataloader_train, dataloader_val
