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
        print(f"Saving project to the {self.project_path}")

    def update(self, **kwargs):
        """Update history dict. In arguments write 'train_accuracy=train_accuracy' etc."""
        for k, v in kwargs.items():
            self.history[k].append(v)

    def get_pairs(self):
        """Helper function to extract keys from history."""
        keys = self.history.keys()
        keys = [key.split("_")[-1] for key in keys]
        keys = set([key for key in keys if keys.count(key) == 2])
        return keys

    def plot_pairs(self):
        """Saves plots to the project path. Plots will contain accuracy/loss curves"""
        keys = self.get_pairs()

        for key in keys:
            plt.figure(figsize=(5, 5))
            plt.plot(self.history.get("train_" + key), label="train", linewidth=2)
            plt.plot(self.history.get("test_" + key), label="test", linewidth=2)
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
        with open(os.path.join(self.project_path, "history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.history.keys())
            writer.writerows(zip(*self.history.values()))
