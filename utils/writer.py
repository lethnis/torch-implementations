import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt


class Writer:
    def __init__(self, project_name: str = "train"):
        """A writer object to track training history.

        Args:
            project_name (str, optional): name of the project, where information will be saved. Defaults to "train".

        Attributes:
            history (dict): dictionary with losses and metrics.
            project_path (str): relative path in runs/... May vary from project name, if folder with that name already existed.
        """

        self.history = defaultdict(list)

        project_name = os.path.normpath(project_name)
        project_name = os.path.basename(project_name)
        project_name = self.check_project_name(project_name)

        self.project_path = os.path.join("runs", project_name)
        os.makedirs(self.project_path)

    def update(self, **kwargs):
        """Update history dict. In arguments expected 'train_accuracy=train_accuracy' etc."""
        for k, v in kwargs.items():
            self.history[k].append(v)

    def plot_pairs(self):
        """Saves plots to the project path. Plots will contain accuracy/loss curves"""

        keys = ["loss", "acc", "accuracy"]

        for key in keys:
            # try to plot at least train progress
            if self.history.get("train_" + key, None) is not None:

                plt.figure(figsize=(5, 5))
                plt.plot(self.history.get("train_" + key), label="train", linewidth=2)

                # try to plot val progress if available
                if self.history.get("val_" + key, None) is not None:
                    plt.plot(self.history.get("val_" + key), label="val", linewidth=2)

                plt.ylabel(key)
                plt.xlabel("epoch")
                plt.legend()
                plt.grid()
                plt.xticks(range(len(self.history.get("train_" + key))))
                plt.tight_layout()
                plt.savefig(os.path.join(self.project_path, key + ".png"))
                plt.close()

    def check_project_name(self, project_name: str):
        """Function to check if project with that name already exists to prevent overwriting.

        Args:
            project_name (str): name of the project where will be saved training data.

        Returns:
            project_name (str): new or unchanged name of the project to save data.
        """
        new_path = os.path.join("runs", project_name)

        # if project with what name does not exist everything is ok
        if not os.path.exists(new_path):
            return project_name

        # if that path exists and ends with '_' we add '2' to the name
        elif project_name.endswith("_"):
            return self.check_project_name(project_name + "2")

        # if name is like 'name_21' it can be splitted by '_'. Increase last number by 1
        elif len(project_name.split("_")) > 1:
            try:
                # split the name
                new_name = project_name.split("_")
                # take last value e.g. '21' and increase by 1
                final_digits = int(new_name[-1]) + 1
                # get the name back with new final digits
                new_name = "_".join(new_name[:-1]) + "_" + str(final_digits)
                return self.check_project_name(new_name)

            except ValueError as e:
                print(f"Couldn't make new project name for {project_name} by increasing last digits. Reason: {e}")

        # just add '_2' to the project name if nothing else worked
        return self.check_project_name(project_name + "_2")

    def save_to_csv(self):
        """Save history to a csv file in project directory."""

        with open(os.path.join(self.project_path, "history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.history.keys())
            writer.writerows(zip(*self.history.values()))

    def is_best(self, monitor: str) -> bool:
        """Check history to decide if on current epoch model has best accuracy or loss.

        Args:
            monitor (str): parameter to decide if model is getting better.

        Returns:
            bool: True if model is getting better else False.
        """

        # loss should be at lowest overall to return True
        if monitor == "val_loss":
            if self.history.get("val_loss", None) is not None and len(self.history["val_loss"]) > 1:
                if self.history["val_loss"][-1] < min(self.history["val_loss"][:-1]):
                    return True

        # accuracy should be at highest overall to return True
        elif monitor == "val_acc":
            if self.history.get("val_acc", None) is not None and len(self.history["val_acc"]) > 1:
                if self.history["val_acc"][-1] > max(self.history["val_acc"][:-1]):
                    return True

        elif monitor == "val_accuracy":
            if self.history.get("val_accuracy", None) is not None and len(self.history["val_accuracy"]) > 1:
                if self.history["val_accuracy"][-1] > max(self.history["val_accuracy"][:-1]):
                    return True

        return False
