import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import tqdm
from torch.cuda.amp import GradScaler
from datetime import datetime
import logging
import os
from scipy.spatial.distance import cdist
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from typing import Union


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred, gt):
        gt = gt.squeeze(1)
        gt = gt.long()
        ce_loss = self.ce_loss(pred, gt)
        return ce_loss


class DiceLoss(nn.Module):
    def __init__(self, sigmoid: bool = False):
        super().__init__()
        self.beginning_layer = nn.Sigmoid() if sigmoid else nn.Identity()

    def dice_loss(self, pred: torch.Tensor, gt: torch.tensor):
        intersection = torch.sum((pred * gt))
        epsilon = 1e-8
        union = (pred + gt).sum() + epsilon
        return 1 - ((2 * intersection) / union)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = self.beginning_layer(pred)
        dice_loss = 0
        for cl in range(pred.shape[1]):
            dice_loss += self.dice_loss(pred[:, cl, :, :], gt[:, cl, :, :].float())

        dice_loss /= pred.shape[1]
        return dice_loss


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 3, beta: float = 4,
                       needs_sigmoid: bool = True, weights: Union[torch.Tensor, None] = None,
                       verbose: bool = False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.needs_sigmoid = needs_sigmoid
        self.weights = weights
        self.verbose = verbose

    def focal_loss_heatmap(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Computes focal loss for heatmap prediction. Based on https://arxiv.org/pdf/1808.01244.pdf (see page 5)

        :param pred: prediction tensor
        :param gt: ground thruth tensor
        :param alpha: alpha hyperparameter of loss
        :param beta: beta hyperparameter of loss for the negative weight (the higher the less impact have negative samples)
        :param weights: class weights tensor
        :param verbose:
        :return:
        """

        if self.weights is not None:
            weights = self.weights.to(pred.device)
            try:
                weights = weights.view(1, pred.shape[1], 1, 1)
            except RuntimeError:
                raise ValueError("Bitte sicherstellen, dass in der Config nur die class weights für Klassen angegeben sind, "
                                 "auf die auch trainiert wird!")



        # convert to probabilities
        if self.needs_sigmoid:
            pred_sigmoid = torch.sigmoid(pred)  # enable during training
        else:
            pred_sigmoid = pred

        # Finding where the ground truth is positive (i.e., = 1)
        pos_inds = gt.eq(1).float()

        # Finding where the ground truth is negative (i.e., < 1)
        neg_inds = gt.lt(1).float()

        # TODO: compare num of pos indices to num of neg indices and tackle the class imbalance!!
        # Initialize loss to 0
        loss = 0

        # Calculate the negative weights
        neg_weights = torch.pow(1 - gt, self.beta)

        # Compute positive loss
        # WICHTIG: Ich habe 1e-7 verwendet, weil ab 1e-8 und der Verwendung von Mixed Precision für floating point numbers
        # die 1nur 16bit verwenden, dies nicht mehr berechnet werden kann und einfach als 0 ausgegeben wird
        # Trade off zwischen genaugigkeit und effizienz sehe ich effizienz vor, weil wir mit 1e-7 trotzdem noch echt gut
        # approximieren können und den loss nicht wirklich verzerren

        try:
            pos_loss = torch.pow(1 - pred_sigmoid, self.alpha) * torch.log(pred_sigmoid + 1e-12) * pos_inds
        except RuntimeError:
            raise ValueError("Bitte prüfen ob Model Scale passt in der config file!")

        # Compute negative loss
        neg_loss = neg_weights * torch.pow(pred_sigmoid, self.alpha) * torch.log(1 - pred_sigmoid + 1e-7) * neg_inds

        # Count the number of positive and negative samples
        num_pos = pos_inds.float().sum()

        if self.weights is not None:
            pos_loss = pos_loss * self.weights
            neg_loss = neg_loss * self.weights

        if self.verbose:
            for i in range(pos_loss.shape[1]):
                try:
                    print(f"Positive loss label {i}: {-pos_loss[i].sum()}")
                    print(f"Negative loss label {i}: {-neg_loss[i].sum()}")
                except IndexError:
                    print(f"Negative loss label {i}: {-neg_loss[i].sum()}")

        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

        if torch.isnan(loss):
            print("is nan")

        return loss

    def forward(self, pred: torch.Tensor, gt: torch. Tensor):
        return self.focal_loss_heatmap(pred, gt)


class SegmentationModelUtils:
    @staticmethod
    def dice_score(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.3):

        def comp_dice_score(pred, gt):
            pred = pred.gt(thresh).float()
            intersection = torch.sum((pred * gt))
            epsilon = 1e-8
            union = (pred + gt).sum() + epsilon
            return ((2 * intersection) / union)

        dice = 0
        for cl in range(pred.shape[0]):
            dice += comp_dice_score(pred[cl], gt[cl])

        dice /= len(pred)

        return dice

    @staticmethod
    def focal_loss_segmentation(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 3, beta: float = 4,
                                weights: Union[torch.Tensor, None] = None, verbose: bool = False) -> torch.Tensor:
        """
        Computes focal loss for heatmap prediction. Based on https://arxiv.org/pdf/1808.01244.pdf (see page 5)

        :param pred: prediction tensor
        :param gt: ground thruth tensor
        :param alpha: alpha hyperparameter of loss
        :param beta: beta hyperparameter of loss for the negative weight (the higher the less impact have negative samples)
        :param weights: class weights tensor
        :param verbose:
        :return:
        """

        if weights is not None:
            weights = weights.to(pred.device)
            try:
                weights = weights.view(1, pred.shape[1], 1, 1)
            except RuntimeError:
                raise ValueError(
                    "Bitte sicherstelle, dass in der Config nur die class weights für Klassen angegeben sind, "
                    "auf die auch trainiert wird!")

        # convert to probabilities
        pred_sigmoid = torch.sigmoid(pred)  # enable during training

        # Finding where the ground truth is positive (i.e., = 1)
        pos_inds = gt.eq(1).float()

        # Finding where the ground truth is negative (i.e., < 1)
        neg_inds = gt.lt(1).float()

        # TODO: compare num of pos indices to num of neg indices and tackle the class imbalance!!
        # Initialize loss to 0
        loss = 0

        # Calculate the negative weights
        neg_weights = torch.pow(1 - gt, beta)

        # Compute positive loss
        # WICHTIG: Ich habe 1e-7 verwendet, weil ab 1e-8 und der Verwendung von Mixed Precision für floating point numbers
        # die 1nur 16bit verwenden, dies nicht mehr berechnet werden kann und einfach als 0 ausgegeben wird
        # Trade off zwischen genaugigkeit und effizienz sehe ich effizienz vor, weil wir mit 1e-7 trotzdem noch echt gut
        # approximieren können und den loss nicht wirklich verzerren

        try:
            # TODO: aktuell sind predictions für center points oft so bei 0.1 - 0.2 --> hier sollte das modell noch sicherer werden
            pos_loss = torch.pow(1 - pred_sigmoid, alpha) * torch.log(pred_sigmoid + 1e-12) * pos_inds
        except RuntimeError:
            raise ValueError("Bitte prüfen ob Model Scale passt in der config file!")

        # Compute negative loss
        neg_loss = neg_weights * torch.pow(pred_sigmoid, alpha) * torch.log(1 - pred_sigmoid + 1e-7) * neg_inds

        # Count the number of positive and negative samples
        num_pos = pos_inds.float().sum()

        if weights is not None:
            pos_loss = pos_loss * weights
            neg_loss = neg_loss * weights

        if verbose:
            for i in range(pos_loss.shape[1]):
                try:
                    print(f"Positive loss label {i}: {-pos_loss[i].sum()}")
                    print(f"Negative loss label {i}: {-neg_loss[i].sum()}")
                except IndexError:
                    print(f"Negative loss label {i}: {-neg_loss[i].sum()}")

        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

        if torch.isnan(loss):
            print("is nan")

        return loss

    @staticmethod
    def prediction_score_heatmap(center_points_true: np.ndarray, center_points_pred: np.ndarray, label_true: np.ndarray,
                                 label_pred: np.ndarray, threshold: float = 50) -> Tuple[float, float]:
        """
        Calculate accuracy and precision metrics based on the location and label prediction.

        :param center_points_true: Array containing true center points of the objects.
        :param center_points_pred: Array containing predicted center points of the objects.
        :param label_true: Array containing true labels of the objects.
        :param label_pred: Array containing predicted labels of the objects.
        :param threshold: The maximum distance between the true center point and the predicted one for a prediction to be considered correct.
        :return: The accuracy and precision of the predictions.
        """

        # Initialize the list to store the results
        result = []

        # Copy the predicted center points and labels to avoid modifying the original ones
        center_points_pred_adjusted = np.copy(center_points_pred)
        label_pred_adjusted = np.copy(label_pred)

        # Iterate through each true center point and label
        for cp_true, label_true in zip(center_points_true, label_true):
            # Compute the distances between the true center point and all predicted ones
            distances = cdist(cp_true.reshape(1, -1), center_points_pred_adjusted)
            # Find the index of the minimum distance
            min_index = np.argmin(distances)

            # If the minimum distance is less than or equal to the threshold and the corresponding predicted label is correct
            # consider the prediction correct (True), otherwise consider it incorrect (False)
            if distances[0][min_index] <= threshold and label_true == label_pred_adjusted[min_index]:
                result.append(True)
            else:
                result.append(False)

            # Remove the predicted center point and label that were just used
            center_points_pred_adjusted = np.delete(center_points_pred_adjusted, min_index, axis=0)
            label_pred_adjusted = np.delete(label_pred_adjusted, min_index, axis=0)

        # Calculate the precision as the number of correct predictions divided by the total number of predictions
        precision = sum(result) / len(center_points_pred)

        # Calculate the accuracy as the number of correct predictions divided by the total number of true objects
        accuracy = sum(result) / len(result)

        return accuracy, precision

    @staticmethod
    def fit(run_name: str, model: nn.Module, epochs: int, train_loader: torch.utils.data.dataloader.DataLoader,
            val_loader: torch.utils.data.dataloader.DataLoader, optimizer: torch.optim, learning_rate: float,
            early_stopping: bool, lr_scheduler: torch.optim.lr_scheduler, loggable_params: dict, **kwargs) -> list:
        """
          Fits the segmentation model with defined optimizer, learning rate scheduler, early stopper  and learning rate

          :param model: PyTorch model which should be trained, possibly on gpu
          :param epochs: Number of epochs
          :param train_loader: DataLoader for training data, possibly on gpu
          :param val_loader: DataLoader for validation data, possible on gpu
          :param optimizer: PyTorch optimizer to optimize the loss
          :param learning_rate: Learning rate for the optimizer
          :param early_stopping: True if early stopping should be checked
          :param lr_scheduler: PyTorch learning rate scheduler to adjust the learning rate
          :param kwargs: additional keyword arguments for the learning rate scheduler
          :return: history of training
          :rtype: list
      """

        try:
            if run_name == "":
                run_name = np.random.randint(0, 1000000000)
            id_full = datetime.now().strftime("%Y-%m-%d %H-%M") + str(run_name)
            logging.basicConfig(
                filename='./logging/run.log',
                filemode='w',
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%d.%m.%y %H:%M:%S',
                level=logging.DEBUG
            )
            writer = SummaryWriter(log_dir=f'logging/runs/{id_full}')
            writer.add_hparams({}, loggable_params)
            scaler = GradScaler()
            optimizer = optimizer(model.parameters(), lr=learning_rate)

            # set up learning rate scheduler if desired
            if lr_scheduler:
                lrs = lr_scheduler(optimizer, **kwargs)

            # set up early stopping if desired
            if early_stopping:
                early_stopper = EarlyStopper()

            # set up list to log data of training
            history = []
            min_val_loss = float('inf')
            min_train_loss = 2

            # set model to train mode to activate layers specific for training, such as the dropout layer
            model.train()

            for epoch in tqdm.tqdm(range(epochs)):
                train_losses = []
                model.train()
                # zero the parameter gradients
                optimizer.zero_grad()

                for num, batch in enumerate(train_loader):
                    print(f"New batch [{num}]")
                    optimizer.zero_grad()
                    loss = model.training_step(batch)
                    # scaler.scale(loss).backward()
                    loss.backward()
                    # scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # scaler.step(optimizer)
                    # scaler.update()
                    optimizer.step()
                    train_losses.append(loss.detach())
                    print(f"Batch{num}:  Total loss:{loss}")

                result = model.evaluate(val_loader)
                result["train_loss"] = torch.stack(train_losses).mean().item()

                if lr_scheduler:
                    old_lr = optimizer.param_groups[0]['lr']
                    # lrs.step(metrics=result["val_loss"])
                    lrs.step()
                    if optimizer.param_groups[0]['lr'] != old_lr:
                        logging.info(
                            f"Learning rate decay in epoch: {epoch}. New learning rate: {optimizer.param_groups[0]['lr']}"
                        )

                if early_stopping:
                    if early_stopper.early_stop(result["val_loss"]):
                        logging.info(f"Early stop in epoch: {epoch}")
                        break

                history.append(result)

                model.epoch_end_val(epoch, result)
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                print(f"Train loss epoch: {result['train_loss']}")
                print(f"Validation loss epoch: {result['val_loss']}")

                writer.add_scalar("Train loss", result["train_loss"], epoch)
                writer.add_scalar("Val loss", result["val_loss"], epoch)
                writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

                # save best model
                if result["val_loss"] < min_val_loss:
                    rounded_loss = np.round(result["val_loss"], 4)
                    save_path = "logging/saved_model/" + id_full
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                    torch.save(model, save_path + '/best_model.pt')
                    torch.save(model, save_path + f'/best_model_val_loss{rounded_loss}.pt')
                    logging.info(f"Saved new best model in epoch: {epoch}. Validation loss of: {rounded_loss}")
                    min_val_loss = result["val_loss"]

                if result["train_loss"] < min_train_loss:
                    rounded_loss = np.round(result["train_loss"], 4)
                    torch.save(model, save_path + '/best_model_train.pt')
                    logging.info(f"Saved new best train model in epoch: {epoch}. Train loss of: {rounded_loss}")
                    min_train_loss = result["train_loss"]

            return history

        except KeyboardInterrupt:
            save_path = "logging/saved_model/" + id_full
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(model, save_path + '/keyboard_interrupt_model.pt')


    @staticmethod
    def accuracy(outputs, labels):
        # Get the predicted class by finding the max probability
        _, preds = torch.max(outputs, dim=1)

        # Compare with true labels
        corrects = torch.sum(preds == labels).item()
        # Calculate accuracy
        acc = corrects / labels.size(0)
        return acc



class EarlyStopper:
    def __init__(self, min_delta=0, patience_steps=5):
        self.min_delta = min_delta
        self.patience_steps = patience_steps
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, new_val_loss):
        """
        Returns True if criterion for early stop is met
        :param new_val_loss: validation loss of epoch
        :param float:
        :return: True if early stop should be done, False otherwise
        :rtype: bool
        """
        if self.min_validation_loss > new_val_loss:
            self.min_validation_loss = new_val_loss
            self.counter = 0
        elif self.min_validation_loss + self.min_delta <= new_val_loss:
            if self.counter >= self.patience_steps:
                return True
            else:
                self.counter += 1
        else:
            return False
