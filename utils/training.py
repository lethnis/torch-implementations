import os
import time
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics import ConfusionMatrix

from mlxtend.plotting import plot_confusion_matrix

from .writer import Writer


class Trainer:

    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        device: str | None = None,
    ):
        """Helper class for training different models.

        Args:
            train_dataset (torch.utils.data.Dataset): train dataset
            test_dataset (torch.utils.data.Dataset): test dataset
            batch_size (int, optional): batch size. Defaults to 32.
            device (str | None, optional): name of device where calculations will happen. Defaults to None.

        Attributes:
            train_dataloader (DataLoader): dataloader generator for training the model.
            test_dataloader (DataLoader): dataloader generator for evaluating the model.
            targets (tensor): tensor of target classes of testing dataset. Needed to plot confusion matrix.
            classes (list): list of classes.
            device (str): 'cpu' or 'cuda' device where training will happen.
        """
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

        self.targets = test_dataset.targets
        self.classes = train_dataset.classes

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

    def train(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy_fn: torchmetrics.Accuracy,
        writer: Writer,
        epochs: int = 10,
    ):
        """Main training loop. Train and test model, record loss and accuracy,
        print progress to the command line.

        Args:
            model (torch.nn.Module): model for training
            loss_fn (torch.nn.Module): loss function
            optimizer (torch.optim.Optimizer): optimizer
            accuracy_fn (torchmetrics.Accuracy): accuracy fuction
            writer (Writer): object to track progress and save it to csv
            epochs (int): number of epochs for training
        """
        for epoch in range(epochs):

            start = time.time()

            # initialize progress bar
            bar = tqdm(ncols=120, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            # run training
            train_loss, train_acc = self.train_step(model, self.train_dataloader, loss_fn, optimizer, accuracy_fn, bar)

            # reinitialize progress bar for testing
            bar = tqdm(ncols=120, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            # run testing
            test_loss, test_acc = self.test_step(model, self.test_dataloader, loss_fn, accuracy_fn, bar)

            print(
                f"Epoch {epoch+1}/{epochs} | time: {time.time()-start:.2f}sec. | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
            )

            # add losses and metrics to the writer
            writer.update(train_loss=train_loss, train_accuracy=train_acc, test_loss=test_loss, test_accuracy=test_acc)

        # save training info
        writer.plot_pairs()
        writer.save_to_csv()
        self.plot_confusion_matrix(model, writer.project_path)
        print(f"Info saved to {writer.project_path}")

    def train_step(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy_fn: torchmetrics.Accuracy,
        bar: tqdm,
    ):
        """Main loop of the training step per one epoch.

        Args:
            model (torch.nn.Module): model to train
            train_dataloader (torch.utils.data.DataLoader): data that will be fed to the model
            loss_fn (torch.nn.Module): loss function
            optimizer (torch.optim.Optimizer): optimizer
            accuracy_fn (torchmetrics.Accuracy): accuracy function
            bar (tqdm): tqdm bar object for printing progress

        Returns:
            tuple(float, float): train loss and train accuracy for printing and recording to csv
        """
        train_loss, train_acc = 0, 0
        model.to(self.device)
        model.train()

        # for tqdm bar we need to specify iterable object and number of iterations
        bar.iterable = train_dataloader
        bar.total = len(train_dataloader)

        for x, y in bar:
            x, y = x.to(self.device), y.to(self.device)

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # depending on the task get accuracy
            if accuracy_fn.__class__.__qualname__ == "BinaryAccuracy":
                train_acc = accuracy_fn(torch.sigmoid(y_pred), y)
            elif accuracy_fn.__class__.__qualname__ == "MulticlassAccuracy":
                train_acc = accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update messages on progress bar
            bar.set_postfix_str(f"train_loss: {loss:.4f} train_acc: {train_acc:.4f}")

        # recalculate overall loss and accuracy
        train_loss /= len(train_dataloader)
        train_acc = accuracy_fn.compute().item()

        return train_loss, train_acc

    def test_step(
        self,
        model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        accuracy_fn: torchmetrics.Accuracy,
        bar: tqdm,
    ):
        """Main test loop for one epoch

        Args:
            model (torch.nn.Module): model for testing
            test_dataloader (torch.utils.data.DataLoader): data that will be fed to the model
            loss_fn (torch.nn.Module): loss function
            accuracy_fn (torchmetrics.Accuracy): accuracy function
            bar (tqdm): tqdm object for printing progress

        Returns:
            tuple(float, float): tuple of test loss and test accuracy
        """

        test_loss, test_acc = 0, 0
        model.to(self.device)
        model.eval()

        # we need to reset total number of iterations and iterable object
        bar.reset(len(test_dataloader))
        bar.iterable = test_dataloader

        with torch.inference_mode():
            for x, y in bar:
                x, y = x.to(self.device), y.to(self.device)

                y_pred = model(x)
                test_loss += loss_fn(y_pred, y).item()

                # get accuracy based on the task
                if accuracy_fn.__class__.__qualname__ == "BinaryAccuracy":
                    test_acc = accuracy_fn(torch.sigmoid(y_pred), y)
                elif accuracy_fn.__class__.__qualname__ == "MulticlassAccuracy":
                    test_acc = accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)

        # recalculate overall loss and accuracy
        test_loss /= len(test_dataloader)
        test_acc = accuracy_fn.compute().item()

        return test_loss, test_acc

    def plot_confusion_matrix(self, model: torch.nn.Module, project_path: str):
        """Method plots and saves confusion matrix and normalized confusion matrix.

        Args:
            model (torch.nn.Module): model to test predictions.
            project_path (str): path to where matrixes will be saved.
        """

        y_preds = []
        model.eval()

        with torch.inference_mode():
            for x, y in self.test_dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_pred = torch.softmax(model(x), dim=1).argmax(dim=1)

                y_preds.append(y_pred.cpu())

        # transform list to tensor
        y_preds = torch.cat(y_preds)

        matrix = ConfusionMatrix(num_classes=len(self.classes), task="multiclass")
        matrix = matrix(preds=y_preds, target=self.targets)

        fig, ax = plot_confusion_matrix(
            matrix.numpy(), class_names=self.classes, figsize=(10, 10), show_absolute=False, show_normed=True
        )
        fig.tight_layout()
        fig.savefig(os.path.join(project_path, "conf_norm.png"))

        fig, ax = plot_confusion_matrix(
            matrix.numpy(), class_names=self.classes, figsize=(10, 10), show_absolute=True, show_normed=False
        )
        fig.tight_layout()
        fig.savefig(os.path.join(project_path, "conf_abs.png"))

        plt.close(fig)
