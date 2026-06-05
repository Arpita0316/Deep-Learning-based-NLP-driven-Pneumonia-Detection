"""
src/trainer.py — Training loop, validation, and metrics utilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score
)
from typing import Optional


class Trainer:
    """
    General-purpose trainer for PneumoniaNet.

    Args:
        model       : PneumoniaNet instance
        optimizer   : torch.optim.Optimizer
        criterion   : loss function
        scheduler   : optional LR scheduler
        device      : 'cuda' or 'cpu'
        use_amp     : use mixed-precision training
        grad_clip   : max gradient norm (0 = disabled)
    """

    def __init__(self, model, optimizer, criterion, scheduler=None,
                 device='cpu', use_amp=True, grad_clip=1.0):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device    = device
        self.use_amp   = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip
        self.scaler    = GradScaler(enabled=self.use_amp)
        self.history   = {
            'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  [],
            'val_auc':    []
        }

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0., 0, 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += preds.eq(labels).sum().item()
            total      += images.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, loader, num_classes: int = 2):
        self.model.eval()
        total_loss, correct, total = 0., 0, 0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

            probs  = F.softmax(outputs, dim=1)
            preds  = probs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct    += preds.eq(labels).sum().item()
            total      += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # For AUC: use positive class prob (binary) or all probs (multi)
            if num_classes == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.cpu().numpy())

        loss_avg = total_loss / total
        acc      = correct / total

        try:
            if num_classes == 2:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                auc = roc_auc_score(
                    all_labels, all_probs,
                    multi_class='ovr', average='macro'
                )
        except Exception:
            auc = 0.

        return loss_avg, acc, auc, np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def fit(self, train_loader, val_loader, epochs: int,
            save_path: Optional[str] = None, num_classes: int = 2,
            verbose: bool = True):
        """
        Run the full training loop.

        Returns: training history dict.
        """
        best_auc = 0.

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            vl_loss, vl_acc, vl_auc, _, _, _ = self.validate(val_loader, num_classes)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)
            self.history['val_auc'].append(vl_auc)

            saved = ''
            if vl_auc > best_auc and save_path:
                best_auc = vl_auc
                torch.save(self.model.state_dict(), save_path)
                saved = ' [saved]'

            if verbose:
                print(
                    f'Epoch {epoch:03d}/{epochs} | '
                    f'Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | '
                    f'Val Loss {vl_loss:.4f} Acc {vl_acc:.4f} AUC {vl_auc:.4f}'
                    f'{saved}'
                )

        return self.history


# ── Standalone metric helpers ──────────────────────────────────────────────────

def print_metrics(labels, preds, probs, class_names, num_classes=2):
    print('\n── Classification Report ─────────────────────────────────────')
    print(classification_report(labels, preds, target_names=class_names))

    if num_classes == 2:
        auc = roc_auc_score(labels, probs)
        ap  = average_precision_score(labels, probs)
        print(f'ROC-AUC : {auc:.4f}')
        print(f'Avg Prec: {ap:.4f}')
    else:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        print(f'Macro ROC-AUC: {auc:.4f}')

    print('─────────────────────────────────────────────────────────────\n')
    return auc
