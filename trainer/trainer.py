import os
import csv
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger
from models import DeepCrysTet
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        train_loader,
        valid_loader=None,
        test_loader=None,
        n_epochs=None,
        es=None,
        save_dirs=None,
        amp=None,
        feature_dim=None,
        is_regression=True,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.es = es
        self.save_dirs = save_dirs
        self.logger = getLogger(__name__)
        self.writer = SummaryWriter(log_dir=save_dirs["tensorboard"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp
        self.feature_dim = feature_dim
        if amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.is_regression = is_regression

    def train(self):
        best_metric, metric_type = (float("inf"), "mae") if self.is_regression else (0, "acc")
        with open(os.path.join(self.save_dirs["log"], "info.csv"), mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", f"train_{metric_type}", f"vaild_{metric_type}"])
        for epoch in range(1, self.n_epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            valid_loss, valid_acc = self._valid_epoch(epoch)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            if self.is_regression:
                self.logger.info(f"Epoch {epoch}: " f"Train: MAE {train_loss:.4f}")
                self.logger.info(f"Epoch {epoch}: " f"Valid: MAE {valid_loss:.4f}")
                with open(os.path.join(self.save_dirs["log"], "info.csv"), mode="a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, valid_loss])
                save_model = True if valid_loss < best_metric else False
            else:
                self.logger.info(
                    f"Epoch {epoch}: " f"Train: Loss {train_loss:.4f}, ACC {train_acc:.4f}"
                )
                self.logger.info(
                    f"Epoch {epoch}: " f"Valid: Loss {valid_loss:.4f}, ACC {valid_acc:.4f}, "
                )
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Acc/valid", valid_acc, epoch)
                with open(os.path.join(self.save_dirs["log"], "info.csv"), mode="a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_acc, valid_acc])
                save_model = True if valid_acc > best_metric else False

            save_info = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "loss": valid_loss,
                "acc": valid_acc,
            }
            torch.save(
                save_info,
                os.path.join(self.save_dirs["model"], "deepcrystest_latest.pt"),
            )
            if save_model:
                torch.save(
                    save_info,
                    os.path.join(self.save_dirs["model"], "deepcrystest_best.pt"),
                )
                best_metric = valid_loss if self.is_regression else valid_acc
            if self.es is not None:
                if self.es(valid_loss):
                    break

    def test(self):
        model = DeepCrysTet(self.feature_dim, self.is_regression).to(self.device)
        checkpoint = torch.load(os.path.join(self.save_dirs["model"], "deepcrystest_best.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        test_loss = 0
        preds = []
        targets = []
        with torch.no_grad():
            for batch in self.test_loader:
                X, target = batch
                targets.extend(target.tolist())
                X = self._to_device(X)
                target = target.to(self.device)
                output = model(X)
                if self.is_regression:
                    output = torch.squeeze(output)
                else:
                    pred = torch.argmax(output, dim=1)
                    preds.extend(pred.tolist())
                loss = self.criterion(output, target)
                test_loss += loss.item()
        test_loss = test_loss / len(self.test_loader)
        if self.is_regression:
            self.logger.info(f"Test: MAE {test_loss:.4f}")
        else:
            test_accuracy = accuracy_score(targets, preds)
            self.logger.info(f"Test: MAE {test_loss:.4f}, ACC {test_accuracy:.4f}")

    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        preds = []
        targets = []
        for batch in tqdm(self.train_loader):
            X, target = batch
            targets.extend(target.tolist())
            X = self._to_device(X)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            if self.amp:
                with torch.cuda.amp.autocast():
                    output = self.model(X)
                    if self.is_regression:
                        output = torch.squeeze(output)
                    else:
                        pred = torch.argmax(output, dim=1)
                        preds.extend(pred.tolist())
                    loss = self.criterion(output, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(X)
                output = torch.squeeze(output)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(self.train_loader)
        if self.is_regression:
            return train_loss, None
        else:
            train_accuracy = accuracy_score(targets, preds)
            return train_loss, train_accuracy

    def _valid_epoch(self, epoch):
        self.model.eval()
        valid_loss = 0
        preds = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(self.valid_loader):
                X, target = batch
                targets.extend(target.tolist())
                X = self._to_device(X)
                target = target.to(self.device)
                output = self.model(X)
                if self.is_regression:
                    output = torch.squeeze(output)
                else:
                    pred = torch.argmax(output, dim=1)
                    preds.extend(pred.tolist())
                loss = self.criterion(output, target)
                valid_loss += loss.item()
        valid_loss = valid_loss / len(self.test_loader)
        if self.is_regression:
            return valid_loss, None
        else:
            valid_accuracy = accuracy_score(targets, preds)
            return valid_loss, valid_accuracy

    def _to_device(self, X):
        centers, features, corners, edges, crystal_idx = X
        centers, features, corners, edges = (
            centers.to(self.device, non_blocking=True),
            features.to(self.device, non_blocking=True),
            corners.to(self.device, non_blocking=True),
            edges.to(self.device, non_blocking=True),
        )
        return (centers, features, corners, edges, crystal_idx)
