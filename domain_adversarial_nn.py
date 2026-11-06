"""Domain-adversarial neural network (DANN, Ganin et al. 2016) for
cross-database magnetic-moment transfer.

The other models in this study are tree-based (GBM, XGBoost, random
forest) and have no mechanism targeting domain shift. The DANN has a shared
feature encoder, a regression task head, and a domain-classifier head
connected through a gradient-reversal layer (GRL). It is trained so that
the encoder representation predicts the target but not the database a
compound's features came from (a domain-invariant representation).

Evaluation protocol, comparable to the GBM 4-condition table: two DANN
models are trained, each in the standard unsupervised domain-adaptation
setup (labeled source domain for the task loss; unlabeled target domain
used only for the domain-adversarial alignment loss; target labels are
never used in training). A held-out 20% test split of each database is
excluded from all training (task loss and alignment loss), so the internal
and zero-shot evaluations are free of leakage:

  Run "mp_source": task loss on MP_train only; domain-adversarial loss
  on (MP_train features, JARVIS_train features); no JARVIS labels are used.
  Evaluated on MP_test ("mp_internal") and JARVIS_test ("mp_to_jarvis",
  zero-shot).

  Run "jarvis_source": task loss on JARVIS_train only; domain-adversarial
  loss on (JARVIS_train features, MP_train features). Evaluated on
  JARVIS_test ("jarvis_internal") and MP_test ("jarvis_to_mp", zero-shot).

This gives the full 4-condition table from 2 trained models, with the same
structure as the GBM baseline, plus the correlated-electron subgroup
MAE/target-std diagnosis.

Output: data/domain_adversarial_results.json.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(0)
torch.set_num_threads(4)

RNG = 0
import os
PROJ = Path(os.environ.get("MAGMOM_ROOT", str(Path(__file__).resolve().parent)))
DATA = PROJ / "data"
CORRELATED_ELECTRON_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def has_correlated_electron_element(formula):
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return False
    return len(els & CORRELATED_ELECTRON_ELEMENTS) > 0


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DANN(nn.Module):
    def __init__(self, n_in=132, hidden=128, enc_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, enc_dim), nn.ReLU(),
        )
        self.task_head = nn.Sequential(
            nn.Linear(enc_dim, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(enc_dim, 32), nn.ReLU(), nn.Linear(32, 2),
        )

    def forward(self, x, lambd=0.0):
        z = self.encoder(x)
        y = self.task_head(z).squeeze(-1)
        d = self.domain_head(grad_reverse(z, lambd))
        return y, d


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "mae": None, "r2": None}
    return {"n": int(len(y_true)), "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def subgroup_mae_std(y_true, y_pred, corr_mask):
    out = {}
    for name, mask in [("correlated_electron", corr_mask), ("non_correlated", ~corr_mask)]:
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) < 2:
            out[name] = None
            continue
        mae = float(mean_absolute_error(yt, yp))
        std = float(np.std(yt))
        out[name] = {"n": int(len(yt)), "mae": mae, "target_std": std,
                      "mae_over_target_std": mae / std if std > 0 else None}
    return out


def train_dann(Xs_train, ys_train, Xt_train_unlabeled, n_epochs=120, batch_size=512, lr=1e-3):
    """Xs_train: source features+labels (task loss). Xt_train_unlabeled:
    target features only, used for domain-adversarial alignment."""
    model = DANN(n_in=Xs_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    task_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.CrossEntropyLoss()

    Xs = torch.tensor(Xs_train, dtype=torch.float32)
    ys = torch.tensor(ys_train, dtype=torch.float32)
    Xt = torch.tensor(Xt_train_unlabeled, dtype=torch.float32)

    n_s, n_t = len(Xs), len(Xt)
    steps_per_epoch = max(1, n_s // batch_size)
    total_steps = steps_per_epoch * n_epochs
    step = 0

    for epoch in range(n_epochs):
        perm_s = torch.randperm(n_s)
        perm_t = torch.randperm(n_t)
        for b in range(steps_per_epoch):
            p = step / total_steps
            lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            idx_s = perm_s[b * batch_size:(b + 1) * batch_size]
            idx_t = perm_t[(b * batch_size) % n_t: (b * batch_size) % n_t + batch_size]
            if len(idx_t) < 2:
                idx_t = perm_t[:batch_size]

            xb_s, yb_s = Xs[idx_s], ys[idx_s]
            xb_t = Xt[idx_t]

            opt.zero_grad()
            pred_s, dom_s = model(xb_s, lambd)
            _, dom_t = model(xb_t, lambd)

            loss_task = task_loss_fn(pred_s, yb_s)
            dom_labels_s = torch.zeros(len(xb_s), dtype=torch.long)
            dom_labels_t = torch.ones(len(xb_t), dtype=torch.long)
            loss_domain = domain_loss_fn(dom_s, dom_labels_s) + domain_loss_fn(dom_t, dom_labels_t)

            loss = loss_task + loss_domain
            loss.backward()
            opt.step()
            step += 1
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            log(f"    epoch {epoch}/{n_epochs} lambd={lambd:.3f} task_loss={loss_task.item():.5f} domain_loss={loss_domain.item():.5f}")
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        pred, _ = model(xb, lambd=0.0)
    return pred.numpy()


def main():
    t0 = time.time()
    mp_meta = pd.read_csv(DATA / "mp_meta_agg.csv")
    jv_meta = pd.read_csv(DATA / "jarvis_meta_agg.csv")
    Xmp_df = pd.read_parquet(DATA / "featurized_mp.parquet")
    Xjv_df = pd.read_parquet(DATA / "featurized_jarvis.parquet")
    assert len(mp_meta) == len(Xmp_df) and len(jv_meta) == len(Xjv_df)
    assert list(Xmp_df.columns) == list(Xjv_df.columns)

    mp_meta = mp_meta.copy(); jv_meta = jv_meta.copy()
    mp_meta["correlated_electron"] = mp_meta["reduced_formula"].apply(has_correlated_electron_element)
    jv_meta["correlated_electron"] = jv_meta["reduced_formula"].apply(has_correlated_electron_element)

    ymp = mp_meta["mag_per_atom"].to_numpy(dtype=float)
    yjv = jv_meta["mag_per_atom"].to_numpy(dtype=float)

    idx_mp = np.arange(len(Xmp_df))
    idx_jv = np.arange(len(Xjv_df))
    mp_tr_idx, mp_te_idx = train_test_split(idx_mp, test_size=0.2, random_state=RNG)
    jv_tr_idx, jv_te_idx = train_test_split(idx_jv, test_size=0.2, random_state=RNG)
    log(f"MP train={len(mp_tr_idx)} test={len(mp_te_idx)}; JARVIS train={len(jv_tr_idx)} test={len(jv_te_idx)}")

    results = {}

    for source_name, target_name, X_src_df, y_src, idx_src_tr, idx_src_te, src_meta, \
            X_tgt_df, y_tgt, idx_tgt_tr, idx_tgt_te, tgt_meta in [
        ("mp_source", "jarvis_target", Xmp_df, ymp, mp_tr_idx, mp_te_idx, mp_meta,
         Xjv_df, yjv, jv_tr_idx, jv_te_idx, jv_meta),
        ("jarvis_source", "mp_target", Xjv_df, yjv, jv_tr_idx, jv_te_idx, jv_meta,
         Xmp_df, ymp, mp_tr_idx, mp_te_idx, mp_meta),
    ]:
        log(f"=== Training DANN: source={source_name}, target={target_name} ===")
        Xsrc_tr = X_src_df.iloc[idx_src_tr].to_numpy(dtype=float)
        ysrc_tr = y_src[idx_src_tr]
        Xtgt_tr_unlabeled = X_tgt_df.iloc[idx_tgt_tr].to_numpy(dtype=float)

        scaler = StandardScaler().fit(np.concatenate([Xsrc_tr, Xtgt_tr_unlabeled], axis=0))
        Xsrc_tr_s = scaler.transform(Xsrc_tr)
        Xtgt_tr_s = scaler.transform(Xtgt_tr_unlabeled)

        model = train_dann(Xsrc_tr_s, ysrc_tr, Xtgt_tr_s, n_epochs=120, batch_size=512, lr=1e-3)

        Xsrc_te = scaler.transform(X_src_df.iloc[idx_src_te].to_numpy(dtype=float))
        ysrc_te = y_src[idx_src_te]
        Xtgt_te = scaler.transform(X_tgt_df.iloc[idx_tgt_te].to_numpy(dtype=float))
        ytgt_te = y_tgt[idx_tgt_te]

        pred_src_te = predict(model, Xsrc_te)
        pred_tgt_te = predict(model, Xtgt_te)

        src_internal = eval_metrics(ysrc_te, pred_src_te)
        zero_shot = eval_metrics(ytgt_te, pred_tgt_te)
        log(f"  {source_name}-internal: {src_internal}")
        log(f"  {source_name} to {target_name} zero-shot: {zero_shot}")

        corr_mask_tgt = tgt_meta.iloc[idx_tgt_te]["correlated_electron"].to_numpy()
        subgroup = subgroup_mae_std(ytgt_te, pred_tgt_te, corr_mask_tgt)
        log(f"  zero-shot subgroup MAE/std: {subgroup}")

        results[f"{source_name}_internal"] = src_internal
        results[f"{source_name}_to_{target_name.replace('_target','')}_zero_shot"] = zero_shot
        results[f"{source_name}_to_{target_name.replace('_target','')}_subgroup"] = subgroup

        with open(DATA / "domain_adversarial_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"  checkpointed after {source_name} run ({time.time()-t0:.1f}s)")

    log(f"\nSaved data/domain_adversarial_results.json ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
