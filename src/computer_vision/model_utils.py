from abc import ABC, abstractmethod
import torch


class ModelBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def training_step(self, batch):
        """
        Training step which computes the loss and accuracy of a train batch
        :param batch: batch of pytorch dataloader
        :type batch: torch.utils.data.DataLoader
        :return: loss, accuracy and f1_score of batch
        :rtype: tuple[torch.tensor,...]
        """
        images, mask = batch
        mask_pred = self(images)

        train_loss = self.loss(mask_pred, mask)

        return train_loss

    def validation_step(self, batch):
        self.eval()
        with torch.no_grad():
            images, mask = batch
            mask_pred = self(images)
            val_loss = self.loss(mask_pred, mask)

            return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """
        Returns the epoch losses after computing the mean loss and accuracy of the test batches

        :param outputs: List of test step outputs
        :type outputs: list
        :return: epoch loss and epoch accuracy
        :rtype: dict
        """
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean().item()

        return {"val_loss": epoch_loss}

    def evaluate(self, dl):
        outputs = [self.validation_step(batch) for batch in dl]
        return self.validation_epoch_end(outputs)

    def epoch_end_val(self, epoch, results):
        """
        Prints validation epoch summary after every epoch

        :param epoch: epoch number
        :type epoch: int
        :param results: results from the evaluate method
        :type results: dictionary
        :return: None
        """

        print(
            f"Epoch:[{epoch}]: |validation loss: {results['val_loss']}|"
        )
