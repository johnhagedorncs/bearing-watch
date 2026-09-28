"""Dense autoencoder over the feature vector; reconstruction error is the score.

The idea is standard: train only on the healthy window, so the network learns to
reconstruct healthy feature vectors and nothing else. A degraded snapshot is
off the healthy manifold, reconstructs badly, and gets a high error.

What is worth being explicit about is why an autoencoder is not obviously the
right answer here. The healthy window is a few hundred snapshots of a stationary
process, and the feature vector is under a hundred dimensions. That is a regime
where a linear method has every advantage, and the PCA baseline in this package
frequently matches or beats this model. That result is reported rather than
buried -- see docs/RESULTS.md. The autoencoder is here because it is the thing
people expect and because having it lets the comparison be made honestly, not
because the comparison was assumed to go its way.

torch is imported lazily so that everything else in the package -- parsing,
features, PCA and kurtosis baselines, serving a PCA run folder -- works in an
environment without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AutoencoderConfig
from .base import AnomalyModel, ModelError

__all__ = ["AutoencoderModel", "torch_available"]

_WEIGHTS = "model.pt"
_META = "model_meta.json"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModelError(
            "the autoencoder needs PyTorch, which is not installed.\n"
            "  Install it:            pip install 'bearingwatch[dev]' torch\n"
            "  Or use a baseline:     bearingwatch train --model pca\n"
            "  The PCA baseline is frequently competitive on this problem; see "
            "docs/RESULTS.md before assuming the autoencoder is the one you want."
        ) from exc
    return torch


class AutoencoderModel(AnomalyModel):
    """Symmetric dense autoencoder. Score = mean squared reconstruction error."""

    kind = "autoencoder"

    def __init__(self, cfg: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AutoencoderConfig()
        self._net: Any = None

    # ----------------------------------------------------------------- fit

    def _build(self, n_features: int) -> Any:
        torch = _require_torch()
        nn = torch.nn

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        if self.cfg.activation not in activations:
            raise ModelError(
                f"unknown activation {self.cfg.activation!r}; known: {sorted(activations)}"
            )
        act = activations[self.cfg.activation]

        dims = [n_features, *self.cfg.hidden_dims, self.cfg.latent_dim]
        encoder: list[Any] = []
        for i in range(len(dims) - 1):
            encoder.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder.append(act())
                if self.cfg.dropout > 0:
                    encoder.append(nn.Dropout(self.cfg.dropout))

        decoder: list[Any] = []
        rdims = list(reversed(dims))
        for i in range(len(rdims) - 1):
            decoder.append(nn.Linear(rdims[i], rdims[i + 1]))
            if i < len(rdims) - 2:
                decoder.append(act())
                if self.cfg.dropout > 0:
                    decoder.append(nn.Dropout(self.cfg.dropout))

        return nn.Sequential(*encoder, *decoder)

    def _fit(self, x: np.ndarray) -> None:
        torch = _require_torch()
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)

        n_samples, n_features = x.shape
        self._net = self._build(n_features)

        # Split the healthy window for early stopping. Shuffling inside the
        # healthy window is legitimate -- it is stationary by declaration. The
        # chronological discipline that matters is that NOTHING after the
        # healthy window is visible to training at all.
        perm = rng.permutation(n_samples)
        n_val = max(1, int(round(self.cfg.val_fraction * n_samples)))
        n_val = min(n_val, n_samples - 1)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        xt = torch.tensor(x, dtype=torch.float32)
        x_train, x_val = xt[train_idx], xt[val_idx]

        opt = torch.optim.Adam(
            self._net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")
        best_state: dict[str, Any] | None = None
        bad_epochs = 0
        batch_size = max(1, min(self.cfg.batch_size, x_train.shape[0]))
        history: list[float] = []

        for _epoch in range(self.cfg.epochs):
            self._net.train()
            order = torch.randperm(x_train.shape[0])
            for start in range(0, x_train.shape[0], batch_size):
                batch = x_train[order[start : start + batch_size]]
                opt.zero_grad()
                loss = loss_fn(self._net(batch), batch)
                loss.backward()
                opt.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self._net(x_val), x_val).item())
            history.append(val_loss)

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()

        with torch.no_grad():
            train_scores = self._score(x)

        self.train_metrics_ = {
            "best_val_mse": best_val,
            "epochs_run": float(len(history)),
            "early_stopped": float(len(history) < self.cfg.epochs),
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
            "n_train": float(len(train_idx)),
            "n_val": float(len(val_idx)),
            "n_parameters": float(sum(p.numel() for p in self._net.parameters())),
        }

    # --------------------------------------------------------------- score

    def _score(self, x: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        self._net.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32)
            recon = self._net(xt)
            err = torch.mean((recon - xt) ** 2, dim=1)
        return err.numpy().astype(np.float64)

    # ----------------------------------------------------------------- io

    def save(self, directory: str | Path) -> list[Path]:
        torch = _require_torch()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        weights_path = directory / _WEIGHTS
        torch.save(self._net.state_dict(), weights_path)

        meta_path = directory / _META
        payload = {
            "kind": self.kind,
            "config": {
                "hidden_dims": list(self.cfg.hidden_dims),
                "latent_dim": self.cfg.latent_dim,
                "dropout": self.cfg.dropout,
                "activation": self.cfg.activation,
                "epochs": self.cfg.epochs,
                "batch_size": self.cfg.batch_size,
                "learning_rate": self.cfg.learning_rate,
                "weight_decay": self.cfg.weight_decay,
                "val_fraction": self.cfg.val_fraction,
                "patience": self.cfg.patience,
                "seed": self.cfg.seed,
            },
            "feature_names": self.feature_names_,
            "train_metrics": self.train_metrics_,
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

        return [weights_path, meta_path]

    @classmethod
    def load(cls, directory: str | Path) -> AutoencoderModel:
        torch = _require_torch()
        directory = Path(directory)
        meta_path = directory / _META
        weights_path = directory / _WEIGHTS
        for path in (meta_path, weights_path):
            if not path.is_file():
                raise ModelError(
                    f"missing {path.name} in {directory}. Inference loads a RUN FOLDER, "
                    "not a loose model file -- a bare weights file has no feature names, "
                    "no scaler and no config, and pairing it with the wrong scaler "
                    "produces plausible numbers rather than an error."
                )

        with meta_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        model = cls(AutoencoderConfig(**payload["config"]))
        model.feature_names_ = list(payload["feature_names"])
        model.train_metrics_ = dict(payload.get("train_metrics", {}))
        model._net = model._build(len(model.feature_names_))
        model._net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model._net.eval()
        model.fitted_ = True
        return model
