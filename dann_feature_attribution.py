"""Feature-attribution check of the domain-adversarial network (DANN).

Tests whether the gradient-reversal domain-adversarial objective
suppresses the Magpie features that carry the MP-vs-JARVIS systematic
offset, specifically features tied to transition-metal / correlated-electron
(d-valence) character.

Method: train a plain encoder+task-head network (same architecture as the
DANN encoder+task_head, no domain head, no gradient reversal) and the DANN
(mp_source run, same hyperparameters and training loop as
domain_adversarial_nn.py) on the same MP-train split. For each model,
compute input-gradient attribution (|dy/dx|, averaged in magnitude over the
MP held-out test set) for each of the 132 Magpie features. Features are
ranked by the point-biserial correlation of each raw feature with the binary
has_correlated_electron flag, computed independently of either model. The
script compares the suppression of DANN attribution relative to plain-model
attribution for the most correlated-electron-associated features against
the least associated (control) features.

Output: data/dann_feature_attribution.json.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, spearmanr

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


class PlainNet(nn.Module):
    """Same encoder+task_head architecture as the DANN, no domain head or gradient reversal."""
    def __init__(self, n_in=132, hidden=128, enc_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, enc_dim), nn.ReLU(),
        )
        self.task_head = nn.Sequential(
            nn.Linear(enc_dim, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.task_head(z).squeeze(-1)


def train_dann(Xs_train, ys_train, Xt_train_unlabeled, n_epochs=120, batch_size=512, lr=1e-3):
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
        if epoch % 40 == 0 or epoch == n_epochs - 1:
            log(f"    [DANN] epoch {epoch}/{n_epochs} lambd={lambd:.3f} task_loss={loss_task.item():.5f}")
    return model


def train_plain(X_train, y_train, n_epochs=120, batch_size=512, lr=1e-3):
    model = PlainNet(n_in=X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    n = len(X)
    steps_per_epoch = max(1, n // batch_size)
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        if epoch % 40 == 0 or epoch == n_epochs - 1:
            log(f"    [Plain] epoch {epoch}/{n_epochs} task_loss={loss.item():.5f}")
    return model


def input_gradient_attribution(model, X, is_dann):
    """Mean |dy/dx_j| over all rows in X, for each input feature j."""
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    if is_dann:
        y, _ = model(xb, lambd=0.0)
    else:
        y = model(xb)
    y.sum().backward()
    grad = xb.grad.detach().numpy()
    return np.mean(np.abs(grad), axis=0)


def main():
    t0 = time.time()
    mp_meta = pd.read_csv(DATA / "mp_meta_agg.csv")
    jv_meta = pd.read_csv(DATA / "jarvis_meta_agg.csv")
    Xmp_df = pd.read_parquet(DATA / "featurized_mp.parquet")
    Xjv_df = pd.read_parquet(DATA / "featurized_jarvis.parquet")
    assert len(mp_meta) == len(Xmp_df) and len(jv_meta) == len(Xjv_df)
    feature_names = list(Xmp_df.columns)
    assert feature_names == list(Xjv_df.columns)

    mp_meta = mp_meta.copy()
    mp_meta["correlated_electron"] = mp_meta["reduced_formula"].apply(has_correlated_electron_element)
    ymp = mp_meta["mag_per_atom"].to_numpy(dtype=float)
    corr_flag = mp_meta["correlated_electron"].to_numpy(dtype=float)

    idx_mp = np.arange(len(Xmp_df))
    idx_jv = np.arange(len(Xjv_df))
    mp_tr_idx, mp_te_idx = train_test_split(idx_mp, test_size=0.2, random_state=RNG)
    jv_tr_idx, jv_te_idx = train_test_split(idx_jv, test_size=0.2, random_state=RNG)
    log(f"MP train={len(mp_tr_idx)} test={len(mp_te_idx)}; JARVIS train={len(jv_tr_idx)} test={len(jv_te_idx)}")

    Xsrc_tr = Xmp_df.iloc[mp_tr_idx].to_numpy(dtype=float)
    ysrc_tr = ymp[mp_tr_idx]
    Xtgt_tr_unlabeled = Xjv_df.iloc[jv_tr_idx].to_numpy(dtype=float)

    scaler = StandardScaler().fit(np.concatenate([Xsrc_tr, Xtgt_tr_unlabeled], axis=0))
    Xsrc_tr_s = scaler.transform(Xsrc_tr)
    Xtgt_tr_s = scaler.transform(Xtgt_tr_unlabeled)
    Xsrc_te_s = scaler.transform(Xmp_df.iloc[mp_te_idx].to_numpy(dtype=float))

    log("Training plain (non-adversarial) baseline on MP_train only...")
    plain_model = train_plain(Xsrc_tr_s, ysrc_tr, n_epochs=120)

    log("Training DANN (mp_source run, same settings as domain_adversarial_nn.py)...")
    dann_model = train_dann(Xsrc_tr_s, ysrc_tr, Xtgt_tr_s, n_epochs=120)

    log("Computing input-gradient attribution for both models on MP_test...")
    attr_plain = input_gradient_attribution(plain_model, Xsrc_te_s, is_dann=False)
    attr_dann = input_gradient_attribution(dann_model, Xsrc_te_s, is_dann=True)

    # Model-independent measure of how much each raw feature carries the
    # correlated-electron signal
    Xmp_tr_raw = Xmp_df.iloc[mp_tr_idx].to_numpy(dtype=float)
    corr_flag_tr = corr_flag[mp_tr_idx]
    pbr = []
    for j in range(Xmp_tr_raw.shape[1]):
        col = Xmp_tr_raw[:, j]
        if np.std(col) < 1e-12:
            pbr.append(0.0)
            continue
        r, _ = pointbiserialr(corr_flag_tr, col)
        pbr.append(0.0 if np.isnan(r) else abs(r))
    pbr = np.array(pbr)

    # Relative suppression: shrinkage of each feature's attribution (DANN vs
    # plain), normalized to remove overall attribution-scale differences
    # between the two models
    attr_plain_norm = attr_plain / (attr_plain.sum() + 1e-12)
    attr_dann_norm = attr_dann / (attr_dann.sum() + 1e-12)
    suppression = attr_plain_norm - attr_dann_norm  # positive = suppressed under DANN

    rho_all, p_all = spearmanr(pbr, suppression)

    # Most correlated-electron-associated features vs. a control set of
    # features with low association
    order = np.argsort(-pbr)
    top_k = 15
    top_idx = order[:top_k]
    bottom_idx = order[-top_k:]
    mean_suppression_top = float(np.mean(suppression[top_idx]))
    mean_suppression_bottom = float(np.mean(suppression[bottom_idx]))
    mean_suppression_all = float(np.mean(suppression))

    out = {
        "method": "Input-gradient attribution (mean |dy/dx| over MP_test, "
                   "normalized to sum to 1 within each model), compared "
                   "between a plain (non-adversarial) encoder+task-head "
                   "network and the DANN (mp_source run), both "
                   "trained with the same architecture and hyperparameters "
                   "except the domain head + gradient-reversal term. "
                   "Feature-level correlated-electron association measured "
                   "a priori and independently of either model via "
                   "point-biserial correlation against the binary "
                   "has_correlated_electron_element flag.",
        "top15_correlated_electron_associated_features": [
            {"feature": feature_names[i], "pointbiserial_r_vs_corr_flag": float(pbr[i]),
             "attr_plain_norm": float(attr_plain_norm[i]), "attr_dann_norm": float(attr_dann_norm[i]),
             "suppression_plain_minus_dann": float(suppression[i])}
            for i in top_idx
        ],
        "bottom15_least_correlated_electron_associated_features": [
            {"feature": feature_names[i], "pointbiserial_r_vs_corr_flag": float(pbr[i]),
             "attr_plain_norm": float(attr_plain_norm[i]), "attr_dann_norm": float(attr_dann_norm[i]),
             "suppression_plain_minus_dann": float(suppression[i])}
            for i in bottom_idx
        ],
        "summary": {
            "spearman_rho_corrflag_assoc_vs_suppression": float(rho_all),
            "spearman_p": float(p_all),
            "mean_suppression_top15_corr_electron_features": mean_suppression_top,
            "mean_suppression_bottom15_control_features": mean_suppression_bottom,
            "mean_suppression_all_132_features": mean_suppression_all,
        },
    }

    # Verdict
    direction_supports = (mean_suppression_top > mean_suppression_bottom) and (rho_all > 0)
    out["verdict"] = (
        f"Spearman rho between a feature's a-priori correlated-electron "
        f"association and how much its attribution is suppressed under "
        f"DANN relative to a plain network is {rho_all:.3f} (p={p_all:.3f}). "
        f"Mean suppression for the top-15 most correlated-electron-associated "
        f"features is {mean_suppression_top:.5f} vs {mean_suppression_bottom:.5f} "
        f"for the bottom-15 (least-associated, control) features. "
        + ("This supports the manuscript's inference: the gradient-reversal "
           "objective disproportionately suppresses the features that "
           "carry correlated-electron/transition-metal chemistry, consistent "
           "with discarding the domain-distinguishing signal."
           if direction_supports else
           "This does not clearly support the manuscript's inference of "
           "differential suppression: the top correlated-electron-"
           "associated features are not suppressed more than the control "
           "set under DANN by this measure. The domain-invariance objective "
           "appears to suppress broadly across the feature set rather than "
           "selectively targeting correlated-electron features, "
           "or the suppression signal is present but this attribution method "
           "is too noisy at this scale to detect it.")
    )
    log(out["verdict"])

    with open(DATA / "dann_feature_attribution.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Saved data/dann_feature_attribution.json ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
