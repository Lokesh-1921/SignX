"""Portable supervised-training loop for the isolated-sign baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import classification_metrics
from .checkpoints import CheckpointManager
from .device import select_device
from .experiment_tracker import ExperimentTracker
from .scheduler import build_scheduler


class Trainer:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, config: dict[str, Any], tracker: ExperimentTracker
    ) -> None:
        self.device_info = select_device()
        self.device = self.device_info.device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.config = config
        self.tracker = tracker
        self.criterion = nn.CrossEntropyLoss()
        train_config = config["training"]
        self.amp_enabled = bool(train_config.get("amp", False) and self.device_info.amp_supported)
        self.grad_scaler = torch.amp.GradScaler(self.device_info.kind, enabled=self.amp_enabled)
        self.gradient_clip_norm = float(train_config.get("gradient_clip_norm", 0.0))
        self.start_epoch = 1
        self.best_val_accuracy = float("-inf")

        # Learning rate scheduler
        total_epochs = int(train_config["epochs"])
        scheduler_config = train_config.get("scheduler", {})
        self.scheduler = build_scheduler(self.optimizer, scheduler_config, total_epochs) if scheduler_config else None

    def resume(self, checkpoint_path: str | Path) -> None:
        payload = CheckpointManager.load(checkpoint_path, self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.best_val_accuracy = float(payload.get("best_val_accuracy", float("-inf")))
        if self.scheduler is not None and "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.tracker.log("resumed", checkpoint=str(checkpoint_path), next_epoch=self.start_epoch)

    def fit(self, train_loader: DataLoader[Any], val_loader: DataLoader[Any]) -> dict[str, float]:
        train_config = self.config["training"]
        checkpoint_root = Path(self.config.get("experiment", {}).get("checkpoint_dir", "checkpoints"))
        checkpoint_dir = checkpoint_root / self.tracker.experiment_id
        manager = CheckpointManager(checkpoint_dir)
        if train_config.get("resume"):
            self.resume(str(train_config["resume"]))
        for epoch in range(self.start_epoch, int(train_config["epochs"]) + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            summary = {f"train_{key}": value for key, value in train_metrics.items()} | {f"val_{key}": value for key, value in val_metrics.items()}
            summary["learning_rate"] = current_lr
            self.tracker.log("epoch", epoch=epoch, **summary)

            # TensorBoard logging
            self.tracker.log_scalars(
                {f"train/{k}": v for k, v in train_metrics.items()}
                | {f"val/{k}": v for k, v in val_metrics.items()}
                | {"train/learning_rate": current_lr},
                global_step=epoch,
            )

            payload: dict[str, Any] = {
                "epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "best_val_accuracy": self.best_val_accuracy, "config": self.config,
            }
            if self.scheduler is not None:
                payload["scheduler"] = self.scheduler.state_dict()

            manager.save("latest.pt", payload)
            if epoch % int(train_config.get("checkpoint_every", 1)) == 0:
                manager.save(f"epoch_{epoch:03d}.pt", payload)
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                payload["best_val_accuracy"] = self.best_val_accuracy
                manager.save("best.pt", payload)

            # Step the scheduler after each epoch
            if self.scheduler is not None:
                self.scheduler.step()

        summary_out = {"best_val_accuracy": self.best_val_accuracy, "checkpoint_dir": str(checkpoint_dir)}
        self.tracker.register(summary_out)
        return summary_out

    def _run_epoch(self, loader: DataLoader[Any], training: bool) -> dict[str, float]:
        self.model.train(training)
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0
        for batch in loader:
            frames = batch["frames"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            pose = batch.get("pose")
            if isinstance(pose, torch.Tensor):
                pose = pose.to(self.device, non_blocking=True)
            with torch.set_grad_enabled(training), torch.autocast(device_type=self.device_info.kind, enabled=self.amp_enabled):
                output = self.model(frames, attention_mask, pose)
                loss = self.criterion(output["logits"], labels)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                if self.gradient_clip_norm > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            total_loss += loss.detach().item() * labels.shape[0]
            all_logits.append(output["logits"].detach().cpu())
            all_labels.append(labels.detach().cpu())
        metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_labels))
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics
