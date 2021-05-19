from __future__ import print_function, division
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from loss.loss_metric import Meter
from torch.optim import Adam
from Dataset.get_dataloader import get_dataloader


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        dataset: torch.utils.data.Dataset,
        criterion: nn.Module,
        lr: float,
        accumulation_steps: int,
        transform,
        batch_size: int,
        data_path: str,
        num_epochs: int,
        warm_up_epochs: int,
        display_plot: bool = False,
    ):
        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.warm_up_epochs = warm_up_epochs
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.warm_up_with_cosine_lr = lambda epoch: (
            epoch + 1
        ) / self.warm_up_epochs if epoch <= self.warm_up_epochs else (
            1 - epoch / self.num_epochs) ^ 0.9

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.warm_up_with_cosine_lr)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "validation"]

        self.dataloaders1 = {
            phase: get_dataloader(dataset=dataset,
                                  phase=phase,
                                  transform=transform,
                                  data_path=data_path,
                                  batch_size=batch_size,
                                  num_workers=4)
            for phase in self.phases
        }
        self.dataloaders2 = {
            phase: get_dataloader(dataset=dataset,
                                  phase=phase,
                                  transform=None,
                                  data_path=data_path,
                                  batch_size=batch_size,
                                  num_workers=4)
            for phase in self.phases
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.sens_scores = {phase: [] for phase in self.phases}
        self.spec_scores = {phase: [] for phase in self.phases}
        self.accu_scores = {phase: [] for phase in self.phases}
        #self.haus_scores = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images: torch.Tensor,
                                  targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits1, logits2 = self.net(images)
        loss = self.criterion(logits1, logits2, targets)
        return loss, logits1

    def _do_epoch(self, epoch: int, phase: str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        if phase == "train":
            dataloader = self.dataloaders1[phase]
        else:
            dataloader = self.dataloaders2[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        print(epoch_loss)
        epoch_dice, epoch_iou, epoch_sens, epoch_spec, epoch_accu = meter.get_metrics(
        )
        print(epoch_accu)
        print(epoch_dice)
        print(epoch_iou)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)
        self.sens_scores[phase].append(epoch_sens)
        self.spec_scores[phase].append(epoch_spec)
        self.accu_scores[phase].append(epoch_accu)
        return epoch_loss

    def run(self):
        for epoch in range(self.num_epochs):
            if epoch == 84:
                break
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "validation")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()

            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "best_model.pth")
            torch.save(self.net.state_dict(),
                       str(epoch + 16) + "_epoch_model.pth")
        self._save_train_history()

    def _plot_train_history(self):
        data = [
            self.losses, self.dice_scores, self.jaccard_scores,
            self.sens_scores, self.spec_scores
        ]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['validation'][-1]}
            """,
            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['validation'][-1]} 
            """,
            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['validation'][-1]}
            """,
            f"""
            train sensitivity score {self.sens_scores['train'][-1]}
            val sensitivity score {self.sens_scores['validation'][-1]}
            """,
            f"""
            train specificity score {self.spec_scores['train'][-1]}
            val specificity score {self.spec_scores['validation'][-1]}
            """,
        ]

        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(5, 1, figsize=(8, 30))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['validation'], c=colors[0], label="validation")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

    def load_predtrain_model(self, state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(), f"last_epoch_model.pth")

        logs_ = [
            self.losses, self.dice_scores, self.jaccard_scores,
            self.sens_scores, self.spec_scores
        ]
        log_names_ = [
            "_loss", "_dice", "_jaccard", "_sensitivity", "_specificity"
        ]
        logs = [
            logs_[i][key] for i in list(range(len(logs_))) for key in logs_[i]
        ]
        log_names = [
            key + log_names_[i] for i in list(range(len(logs_)))
            for key in logs_[i]
        ]
        pd.DataFrame(dict(zip(log_names, logs))).to_csv("train_log.csv",
                                                        index=False)
