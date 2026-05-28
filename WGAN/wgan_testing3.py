"""
    python wgan_testing.py

Požiadavky:
    pip install torch scikit-learn scipy matplotlib seaborn pandas openpyxl
"""

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# 0.  KONFIGURÁCIA
# ═══════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

TARGET_COL = "Závažnosť priebehu ochorenia"
CLASS_NAMES = {1: "Domáce liečenie", 2: "Preloženie", 3: "Exitus"}

DROP_COLS = [
    "Poradie", "Meno", "Kód príjmu", "Kód prepustenia",
    "Dátum príjmu", "Dátum prepustenia",
    "Liečba", "Epikríza", "Terajšie ochorenie", "SVLZ správy",
    "Diagnózy", "Lieková anamnéza", "Mikrobiológia",
    "Návyková anamnéza", "Epidemiologická anamnéza",
    "Objektívny nález", "Osobná anamnéza", "Dôvod hospitalizácie",
    "HLN Dg.", "DRG výkony", "Unnamed: 23", "A04.7", "Typ vakcíny",
]

DROP_SUFFIXES = [" min", " max"]

STRING_CATEGORICAL = ["Pohlavie"]

BOOL_COLS = [
    "Fajčenie", "Alkohol", "Hypertenzia", "Diabetes mellitus",
    "Kardiovaskulárne ochorenia", "Chronické respiračné ochorenia",
    "Renálne ochorenia", "Pečeňové ochorenia", "Onkologické ochorenia",
    "Imunosupresia", "Vakcinácia", "Prekonal COVID-19",
    "MD652 | FABIFLU TABLETS", "MD656 IV-BECT 6MG (ivermectin)",
    "5042D | VEKLURY", "9547D | PAXLOVID", "LAGEVRIO",
    "00584 | PYRIDOXIN LÉČIVA INJ", "24836 | ACIDUM ASCORBICUM BBP",
    "24814 | CALCIFEROL BBP 7,5 MG/ML",
    "00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK",
    "00449 | EREVIT 300 MG/ML", "89145 | VITAMIN C-INJEKTOPAS",
    "92973 ALPHA D3", "02963 | PREDNISON 20 LÉČIVA",
    "00269 | PREDNISON 5 LÉČIVA", "84090 | DEXAMED 6",
    "1275C | DEXAMETAZÓN KRKA", "MD661 BIODEXONE-DEXAMETHASONE",
    "2410B HYDROCORTISONE", "3242C | OLUMIANT 4 MG",
    "Anakinra", "RoActemra", "34045 | POLYOXIDONIUM 6 MG",
    "87299 | IMUNOR", "56930 IMMODIN", "Isoprinosine, ",
    "3879d INOMED", "35715 Azithromycin", "45954 Ceftriaxon",
    "0471B MOLOXIN", "9819A MOXIFLOXACIN",
    "58730 CIPROFLOXACIN KABI 200", "58746 CIPROFLOXACINKABI 400",
    "05044 OZZION", "4147C OMEMYL", "89662 NOLPAZA",
    "39397 PANTOPRAZOL", "62916 SMECTA", "30639 REASEC",
    "84370 LAGOSA", "93105 DEGAN ", "94918 AMBROBENE",
    "24859 PENTOXYPHILLINUM", "8893 ACC INJEKT", "24949 CODEIN ",
    "26846 OXANTIL", "FRAXIPARIN", "CLEXANE", "FRAGMIN",
    "ASPIRIN", "ANOPYRIN",
]

PRIORITY_FEATURES = [
    "S-CRP first", "S-IL6 first", "S-FER first", "WBC first",
    "PLT first", "D-dimér HS first", "HGB first", "Ly abs first",
    "Vek", "NE/LY(NLR) first",
]

# Hyperparametre WGAN-GP
LATENT_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
N_EPOCHS   = 500
LR         = 1e-4
N_CRITIC   = 5
LAMBDA_GP  = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════
# 1.  POMOCNÉ FUNKCIE
# ═══════════════════════════════════════════════════════════

def _list_datasets(data_dir):
    paths = []
    for pat in ["*.xlsx", "*.xls"]:
        paths.extend(glob.glob(os.path.join(data_dir, pat)))
    return sorted(set(paths))


def _tag(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")


def _save(filename):
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"[Graf] Uložený: {filename}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# 2.  NAČÍTANIE A PREDSPRACOVANIE DÁT
# ═══════════════════════════════════════════════════════════

def load_and_preprocess(path):
    print(f"\n[Data] Načítavam: {os.path.basename(path)}")
    df = pd.read_excel(path)
    print(f"[Data] Pôvodný tvar: {df.shape[0]} riadkov × {df.shape[1]} stĺpcov")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    drop_sfx = [c for c in df.columns if any(c.endswith(s) for s in DROP_SUFFIXES)]
    df = df.drop(columns=drop_sfx, errors="ignore")

    for col in STRING_CATEGORICAL:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    y = None
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df[df[TARGET_COL].notna()].reset_index(drop=True)
        y = df[TARGET_COL].values.astype(int)
        dist = {CLASS_NAMES.get(k, str(k)): int(v)
                for k, v in zip(*np.unique(y, return_counts=True))}
        print(f"[Data] Distribúcia tried: {dist}")

    df = df.apply(pd.to_numeric, errors="coerce")
    thresh = int(len(df) * 0.4)
    before = df.shape[1]
    df = df.dropna(axis=1, thresh=thresh)
    removed = before - df.shape[1]
    if removed:
        print(f"[Data] Odstránených {removed} stĺpcov (>60% chýbajúcich)")
    df = df.fillna(df.median(numeric_only=True))

    feature_names = df.columns.tolist()
    col_min = df.min()
    col_max = df.max()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(df.values.astype(np.float32))

    print(f"[Data] Finálny tvar: {X.shape[0]} riadkov × {X.shape[1]} stĺpcov")
    return X, y, feature_names, scaler, col_min, col_max


# ═══════════════════════════════════════════════════════════
# 3.  ARCHITEKTÚRA WGAN-GP
# ═══════════════════════════════════════════════════════════

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake):
    alpha  = torch.rand(real.size(0), 1, device=DEVICE)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score  = critic(interp)
    grads  = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 4.  TRÉNOVANIE
# ═══════════════════════════════════════════════════════════

def train_wgan(X_train):
    data_dim = X_train.shape[1]
    loader   = DataLoader(
        TensorDataset(torch.FloatTensor(X_train).to(DEVICE)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    G = Generator(LATENT_DIM, data_dim, HIDDEN_DIM).to(DEVICE)
    C = Critic(data_dim, HIDDEN_DIM).to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=LR, betas=(0.0, 0.9))

    history = {"critic_loss": [], "gen_loss": [], "grad_norm": []}
    print(f"\n[Trénovanie] WGAN-GP | Epochy: {N_EPOCHS} | {DEVICE}")

    for epoch in range(N_EPOCHS):
        for real_batch, in loader:
            bs = real_batch.size(0)
            for _ in range(N_CRITIC):
                z    = torch.randn(bs, LATENT_DIM, device=DEVICE)
                fake = G(z).detach()
                gp   = gradient_penalty(C, real_batch, fake.requires_grad_(True))
                loss_C = -C(real_batch).mean() + C(fake).mean() + LAMBDA_GP * gp
                opt_C.zero_grad(); loss_C.backward(); opt_C.step()

            grad_norm = sum(p.grad.norm().item()
                           for p in C.parameters() if p.grad is not None)
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            loss_G = -C(G(z)).mean()
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        history["critic_loss"].append(loss_C.item())
        history["gen_loss"].append(loss_G.item())
        history["grad_norm"].append(grad_norm)

        if (epoch + 1) % 50 == 0:
            print(f"  Epocha {epoch+1:4d}/{N_EPOCHS} | "
                  f"Kritik: {loss_C.item():.4f} | "
                  f"Generátor: {loss_G.item():.4f} | "
                  f"Norma grad.: {grad_norm:.4f}")

    return G, C, history


# ═══════════════════════════════════════════════════════════
# 5.  GENEROVANIE SYNTETICKÝCH DÁT
# ═══════════════════════════════════════════════════════════

def generate_synthetic(G, n_samples, scaler, col_min, col_max, feature_names):
    G.eval()
    with torch.no_grad():
        z   = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        raw = G(z).cpu().numpy()

    synthetic = scaler.inverse_transform(raw)
    df_syn = pd.DataFrame(synthetic, columns=feature_names)

    for col in feature_names:
        if col in col_min.index:
            df_syn[col] = df_syn[col].clip(float(col_min[col]), float(col_max[col]))

    for col in feature_names:
        if col in BOOL_COLS:
            df_syn[col] = df_syn[col].round().clip(0, 1).astype(int)
        elif col == TARGET_COL:
            df_syn[col] = df_syn[col].round().clip(1, 3).astype(int)

    print(f"[Generovanie] {len(df_syn)} syntetických záznamov vygenerovaných")
    return df_syn.values


# ═══════════════════════════════════════════════════════════
# 6.  VYHODNOCOVACIE METRIKY
# ═══════════════════════════════════════════════════════════

def eval_statistical(X_real, X_syn, feature_names):
    print("\n[Vyhodnotenie] Štatistická fidelita")
    results = []
    for i, name in enumerate(feature_names):
        wd   = wasserstein_distance(X_real[:, i], X_syn[:, i])
        _, p = ks_2samp(X_real[:, i], X_syn[:, i])
        results.append({
            "príznak":     name,
            "wasserstein": round(wd, 4),
            "ks_p":        round(p, 4),
            "podobný":     "✓" if p > 0.05 else "✗",
        })

    df = pd.DataFrame(results)
    mean_wd = df["wasserstein"].mean()
    n_sim   = (df["podobný"] == "✓").sum()
    pct_sim = n_sim / len(df) * 100
    print(f"  Priemerná Wassersteinova vzd. : {mean_wd:.4f}")
    print(f"  Podobné distribúcie (p>0.05)  : {n_sim}/{len(df)} ({pct_sim:.1f}%)")
    return df, mean_wd, pct_sim


def eval_class_distribution(y_real, X_syn, feature_names):
    print("\n[Vyhodnotenie] Distribúcia tried závažnosti")
    if TARGET_COL not in feature_names or y_real is None:
        print("  Preskočené.")
        return {}

    target_idx = feature_names.index(TARGET_COL)
    y_syn = np.round(X_syn[:, target_idx]).clip(1, 3).astype(int)

    real_dist = pd.Series(y_real).value_counts(normalize=True).sort_index()
    syn_dist  = pd.Series(y_syn).value_counts(normalize=True).sort_index()

    print(f"  {'Trieda':<25} {'Reálne %':>10} {'Syntet. %':>11} {'Rozdiel':>9}")
    print(f"  {'─'*58}")
    diffs = {}
    for cls in sorted(CLASS_NAMES.keys()):
        r = real_dist.get(cls, 0) * 100
        s = syn_dist.get(cls, 0) * 100
        d = abs(r - s)
        print(f"  {CLASS_NAMES[cls]:<25} {r:>9.1f}% {s:>10.1f}% {d:>8.1f}%")
        diffs[cls] = {"real_pct": r, "syn_pct": s, "diff": d}

    mean_diff = np.mean([v["diff"] for v in diffs.values()])
    print(f"  Priemerný rozdiel tried: {mean_diff:.1f}%")
    return diffs


def eval_correlation(X_real, X_syn, feature_names):
    real_corr = pd.DataFrame(X_real, columns=feature_names).corr().fillna(0)
    syn_corr  = pd.DataFrame(X_syn,  columns=feature_names).corr().fillna(0)
    diff      = (real_corr - syn_corr).abs()
    mae       = diff.values[np.triu_indices_from(diff.values, k=1)].mean()
    print(f"\n[Vyhodnotenie] MAE korelačnej matice: {mae:.4f}")
    return real_corr, syn_corr, diff, mae


def eval_tstr(X_train_real, y_train, X_syn, X_test_real, y_test, feature_names):
    print("\n[Vyhodnotenie] Klinická utilita — TSTR")

    if y_train is None or y_test is None:
        print("  Preskočené — chýba cieľová premenná.")
        return {}

    feat_idx = [i for i, n in enumerate(feature_names) if n != TARGET_COL]
    Xtr = X_train_real[:, feat_idx]
    Xte = X_test_real[:, feat_idx]
    Xsy = X_syn[:, feat_idx]

    if TARGET_COL in feature_names:
        target_idx = feature_names.index(TARGET_COL)
        syn_labels = np.round(X_syn[:, target_idx]).clip(1, 3).astype(int)
    else:
        clf_lab = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_lab.fit(Xtr, y_train)
        syn_labels = clf_lab.predict(Xsy)

    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(Xtr, y_train)
    yp_trtr  = clf_trtr.predict(Xte)
    ypr_trtr = clf_trtr.predict_proba(Xte)

    unique_syn, counts_syn = np.unique(syn_labels, return_counts=True)
    if len(unique_syn) < 2:
        print(f"  TSTR preskočené — syntetické dáta obsahujú len {len(unique_syn)} triedu.")
        return {}

    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(Xsy, syn_labels)
    yp_tstr  = clf_tstr.predict(Xte)
    ypr_tstr = clf_tstr.predict_proba(Xte)

    def safe_auc(y_true, y_proba, clf):
        try:
            return roc_auc_score(
                y_true, y_proba,
                multi_class="ovr", average="macro",
                labels=clf.classes_
            )
        except Exception as e:
            print(f"  AUC: {e}")
            return float("nan")

    trtr_auc = safe_auc(y_test, ypr_trtr, clf_trtr)
    tstr_auc = safe_auc(y_test, ypr_tstr, clf_tstr)

    m = {
        "TRTR AUC":      round(trtr_auc, 4),
        "TSTR AUC":      round(tstr_auc, 4),
        "TRTR Accuracy": round(accuracy_score(y_test, yp_trtr), 4),
        "TSTR Accuracy": round(accuracy_score(y_test, yp_tstr), 4),
        "TRTR F1":       round(f1_score(y_test, yp_trtr, average="macro", zero_division=0), 4),
        "TSTR F1":       round(f1_score(y_test, yp_tstr, average="macro", zero_division=0), 4),
    }
    if not np.isnan(trtr_auc) and trtr_auc > 0 and not np.isnan(tstr_auc):
        m["Retencia AUC (%)"] = round(tstr_auc / trtr_auc * 100, 1)
    if m["TRTR F1"] > 0:
        m["Retencia F1 (%)"] = round(m["TSTR F1"] / m["TRTR F1"] * 100, 1)

    print(f"\n  {'Metrika':<20} {'TRTR (reálne)':>15} {'TSTR (syntet.)':>15}")
    print(f"  {'─'*52}")
    for metric in ["AUC", "Accuracy", "F1"]:
        print(f"  {metric:<20} {str(m.get(f'TRTR {metric}', 'N/A')):>15} "
              f"{str(m.get(f'TSTR {metric}', 'N/A')):>15}")
    print(f"\n  Retencia AUC: {m.get('Retencia AUC (%)', 'N/A')} % | "
          f"Retencia F1: {m.get('Retencia F1 (%)', 'N/A')} %")
    return m


def eval_privacy(X_real, X_syn, k=5):
    print("\n[Vyhodnotenie] Ochrana súkromia — NND Ratio")
    n   = min(500, len(X_real), len(X_syn))
    r   = X_real[np.random.choice(len(X_real), n, replace=False)]
    s   = X_syn [np.random.choice(len(X_syn),  n, replace=False)]

    d_sr = cdist(s, r)
    d_rr = cdist(r, r)
    np.fill_diagonal(d_rr, np.inf)

    nnd_sr = np.sort(d_sr, axis=1)[:, :k].mean(axis=1).mean()
    nnd_rr = np.sort(d_rr, axis=1)[:, :k].mean(axis=1).mean()
    ratio  = nnd_sr / nnd_rr

    verdict = "✓ Bezpečné" if ratio >= 1.0 else "⚠ Možná memorizácia"
    print(f"  NND syntetické→reálne : {nnd_sr:.4f}")
    print(f"  NND reálne→reálne     : {nnd_rr:.4f}")
    print(f"  Ratio (>1.0 = bezpeč.): {ratio:.4f}  {verdict}")
    return {"nnd_sr": nnd_sr, "nnd_rr": nnd_rr, "ratio": ratio}


# ═══════════════════════════════════════════════════════════
# 7.  GRAFY
# ═══════════════════════════════════════════════════════════

def plot_training(history, tag):
    # ── Obrázok 1: Strata kritika & Strata generátora ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"WGAN-GP Trénovacie krivky — {tag} (1/2)", fontweight="bold")

    axes[0].plot(history["critic_loss"], color="#e63946")
    axes[0].set_title("Strata kritika"); axes[0].set_xlabel("Epocha")

    axes[1].plot(history["gen_loss"], color="#457b9d")
    axes[1].set_title("Strata generátora"); axes[1].set_xlabel("Epocha")

    plt.tight_layout()
    _save(f"WGAN/wgan_training_curves_{tag}_1_losses.png")

    # ── Obrázok 2: Norma gradientu ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"WGAN-GP Trénovacie krivky — {tag} (2/2)", fontweight="bold")

    axes[0].plot(history["grad_norm"], color="#2a9d8f")
    axes[0].axhline(1.0, linestyle="--", color="gray", alpha=0.7, label="Ideál=1.0")
    axes[0].set_title("Norma gradientu\n(ideálne ≈ 1.0)")
    axes[0].set_xlabel("Epocha"); axes[0].legend()

    axes[1].set_visible(False)

    plt.tight_layout()
    _save(f"WGAN/wgan_training_curves_{tag}_2_gradnorm.png")


def plot_distributions(X_real, X_syn, feature_names, tag):
    """Clinical feature histograms — 2 per figure."""
    cols = [f for f in PRIORITY_FEATURES if f in feature_names]
    if len(cols) < 8:
        others = [f for f in feature_names if f not in cols
                  and f not in BOOL_COLS and f != TARGET_COL]
        cols += others[:8 - len(cols)]
    cols = cols[:8]
    idxs = [feature_names.index(c) for c in cols]

    chunks = [idxs[i:i+2] for i in range(0, len(idxs), 2)]

    for part, part_idxs in enumerate(chunks, start=1):
        if not part_idxs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f"Clinical Feature Distributions: Real vs. Synthetic — {tag} (Part {part})",
            fontweight="bold"
        )
        axes = axes.flatten()
        for k, idx in enumerate(part_idxs):
            axes[k].hist(X_real[:, idx], bins=30, alpha=0.6, color="#457b9d",
                         label="Real", density=True)
            axes[k].hist(X_syn[:, idx],  bins=30, alpha=0.6, color="#e63946",
                         label="Synthetic", density=True)
            axes[k].set_title(feature_names[idx], fontsize=8)
            axes[k].set_xlabel("Value")
            axes[k].set_ylabel("Density")
            axes[k].legend(fontsize=7)
        for j in range(len(part_idxs), 2):
            axes[j].set_visible(False)
        plt.tight_layout()
        _save(f"WGAN/wgan_distributions_{tag}_part{part}.png")


def plot_class_balance(y_real, X_syn, feature_names, tag):
    if TARGET_COL not in feature_names or y_real is None:
        return
    target_idx = feature_names.index(TARGET_COL)
    y_syn = np.round(X_syn[:, target_idx]).clip(1, 3).astype(int)

    classes = sorted(CLASS_NAMES.keys())
    labels  = [CLASS_NAMES[c] for c in classes]
    real_c  = [int(np.sum(y_real == c)) for c in classes]
    syn_c   = [int(np.sum(y_syn  == c)) for c in classes]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - 0.2, real_c, 0.35, label="Reálne",      color="#457b9d")
    b2 = ax.bar(x + 0.2, syn_c,  0.35, label="Syntetické",  color="#e63946")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_title(f"Distribúcia tried závažnosti — {tag}", fontweight="bold")
    ax.set_ylabel("Počet pacientov"); ax.legend()
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", fontsize=9)
    plt.tight_layout()
    _save(f"WGAN/wgan_class_balance_{tag}.png")


def plot_correlations(real_corr, syn_corr, diff, tag):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Korelačné matice — {tag}", fontweight="bold")
    for ax, data, title, cmap in zip(
        axes,
        [real_corr, syn_corr, diff],
        ["Reálne dáta", "Syntetické dáta", "Absolútny rozdiel"],
        ["coolwarm", "coolwarm", "Reds"]
    ):
        sns.heatmap(data, ax=ax, cmap=cmap, square=True, linewidths=0.1,
                    center=0 if cmap != "Reds" else None,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title)
        ax.set_xticklabels([]); ax.set_yticklabels([])
    plt.tight_layout()
    _save(f"WGAN/wgan_correlations_{tag}.png")


def plot_wasserstein(stats_df, tag):
    df_s   = stats_df.sort_values("wasserstein", ascending=False).head(20)
    colors = ["#e63946" if v > 1.0 else "#457b9d" for v in df_s["wasserstein"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(df_s["príznak"], df_s["wasserstein"], color=colors)
    ax.axvline(1.0, linestyle="--", color="gray", alpha=0.6, label="Prah 1.0")
    ax.set_title(f"Wassersteinova vzdialenosť — top 20 príznakov — {tag}",
                 fontweight="bold")
    ax.set_xlabel("Wassersteinova vzdialenosť (nižšia = lepšia)")
    ax.legend(); plt.tight_layout()
    _save(f"WGAN/wgan_wasserstein_{tag}.png")


def plot_pca(X_real, X_syn, y_real, feature_names, tag):
    n   = min(500, len(X_real), len(X_syn))
    pca = PCA(n_components=2)
    pca.fit(np.vstack([X_real[:n], X_syn[:n]]))
    r2  = pca.transform(X_real[:n])
    s2  = pca.transform(X_syn[:n])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"PCA — Reálne vs. Syntetické — {tag}", fontweight="bold")

    colors_cls = {1: "#2a9d8f", 2: "#f4a261", 3: "#e63946"}

    ax = axes[0]
    if y_real is not None:
        for cls, col in colors_cls.items():
            mask = (y_real[:n] == cls) if len(y_real) >= n else (y_real == cls)
            ax.scatter(r2[mask, 0], r2[mask, 1], alpha=0.5, s=15, color=col,
                       label=CLASS_NAMES.get(cls, str(cls)))
        ax.legend(fontsize=8, title="Závažnosť")
    else:
        ax.scatter(r2[:, 0], r2[:, 1], alpha=0.4, s=15, color="#457b9d")
    ax.set_title("Reálne dáta", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    ax = axes[1]
    ax.scatter(s2[:, 0], s2[:, 1], alpha=0.4, s=15, color="#457b9d",
               label="Syntetické")
    ax.scatter(r2[:, 0], r2[:, 1], alpha=0.2, s=10, color="#e63946",
               label="Reálne (referencia)")
    ax.set_title("Syntetické vs. Reálne", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(f"WGAN/wgan_pca_{tag}.png")


def plot_tstr(metrics, tag):
    if not metrics:
        return
    labels    = ["AUC", "Accuracy", "F1"]
    trtr_vals = [metrics.get(f"TRTR {l}", 0) for l in labels]
    tstr_vals = [metrics.get(f"TSTR {l}", 0) for l in labels]

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    b1  = ax.bar(x - 0.2, trtr_vals, 0.35, label="TRTR (reálne)",     color="#457b9d")
    b2  = ax.bar(x + 0.2, tstr_vals, 0.35, label="TSTR (syntetické)", color="#e63946")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"TSTR vs. TRTR — Klinická utilita (macro) — {tag}", fontweight="bold")
    ax.set_ylabel("Skóre"); ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not (isinstance(h, float) and np.isnan(h)):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _save(f"WGAN/wgan_tstr_{tag}.png")


# ═══════════════════════════════════════════════════════════
# 8.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 65)
    print("  WGAN-GP Pipeline — COVID-19 Syntetické medicínske dáta")
    print("=" * 65)
    print(f"  Zariadenie : {DEVICE}")
    print(f"  DATA_DIR   : {os.path.abspath(DATA_DIR)}")

    os.chdir(SCRIPT_DIR)
    os.makedirs("WGAN", exist_ok=True)
    print(f"[Info] Pracovný adresár: {os.getcwd()}")

    data_paths = _list_datasets(DATA_DIR)
    if not data_paths:
        print(f"\n[CHYBA] Nenašli sa xlsx súbory v: {os.path.abspath(DATA_DIR)}")
        print("  Uprav premennú DATA_DIR na začiatku skriptu.")
        print(f"  Príklad: DATA_DIR = r'C:\\Users\\pepin\\Desktop\\bakalarka_modely\\datasets'")
        return

    print(f"\n  Nájdené datasety ({len(data_paths)}):")
    for p in data_paths:
        print(f"    - {os.path.basename(p)}")

    all_summaries = []

    for data_path in data_paths:
        tag = _tag(data_path)
        print(f"\n\n{'='*65}")
        print(f"  Dataset: {os.path.basename(data_path)}")
        print(f"{'='*65}")

        try:
            X, y, feature_names, scaler, col_min, col_max = \
                load_and_preprocess(data_path)
        except Exception as e:
            print(f"[CHYBA] Načítanie zlyhalo: {e}")
            continue

        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            stratify = y if counts.min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        else:
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            y_train = y_test = None

        print(f"[Split] Trénovanie: {len(X_train)} | Test: {len(X_test)}")

        # Trénovanie
        G, C, history = train_wgan(X_train)
        plot_training(history, tag)                          # ← opravené volanie

        # Generovanie syntetických dát
        synthetic = generate_synthetic(
            G, n_samples=len(X_train),
            scaler=scaler, col_min=col_min, col_max=col_max,
            feature_names=feature_names
        )
        pd.DataFrame(synthetic, columns=feature_names) \
          .to_csv(f"WGAN/synthetic_covid_wgan_{tag}.csv", index=False)
        print(f"[Uložené] WGAN/synthetic_covid_wgan_{tag}.csv")

        X_train_orig = scaler.inverse_transform(X_train)
        X_test_orig  = scaler.inverse_transform(X_test)

        # Štatistická fidelita
        stats_df, mean_wd, pct_sim = eval_statistical(
            X_train_orig, synthetic, feature_names
        )
        plot_wasserstein(stats_df, tag)
        plot_distributions(X_train_orig, synthetic, feature_names, tag)

        # Distribúcia tried závažnosti
        class_stats = eval_class_distribution(y_train, synthetic, feature_names)
        plot_class_balance(y_train, synthetic, feature_names, tag)

        # Zachovanie korelácie
        real_corr, syn_corr, diff, mae = eval_correlation(
            X_train_orig, synthetic, feature_names
        )
        plot_correlations(real_corr, syn_corr, diff, tag)

        # TSTR klinická utilita
        tstr = eval_tstr(
            X_train_orig, y_train, synthetic,
            X_test_orig,  y_test,  feature_names
        )
        plot_tstr(tstr, tag)

        # PCA vizualizácia
        plot_pca(X_train_orig, synthetic, y_train, feature_names, tag)

        # Ochrana súkromia
        privacy = eval_privacy(X_train_orig, synthetic)

        # Súhrn vlny
        print(f"\n{'─'*65}")
        print(f"  SÚHRN — {tag}")
        print(f"{'─'*65}")
        print(f"  Priemerná Wassersteinova vzd. : {mean_wd:.4f}")
        print(f"  Podobné distribúcie (p>0.05)  : {pct_sim:.1f}%")
        print(f"  MAE korelácia                 : {mae:.4f}")
        if tstr:
            print(f"  TRTR AUC (bázová línia)       : {tstr.get('TRTR AUC', 'N/A')}")
            print(f"  TSTR AUC                      : {tstr.get('TSTR AUC', 'N/A')}")
            print(f"  TSTR F1 (macro)               : {tstr.get('TSTR F1', 'N/A')}")
            print(f"  Retencia AUC                  : {tstr.get('Retencia AUC (%)', 'N/A')} %")
            print(f"  Retencia F1                   : {tstr.get('Retencia F1 (%)', 'N/A')} %")
        print(f"  NND Ratio                     : {privacy['ratio']:.4f}  "
              f"({'✓ Bezpečné' if privacy['ratio'] >= 1.0 else '⚠ Riziko'})")
        print(f"{'─'*65}")

        all_summaries.append({
            "Dataset":          os.path.basename(data_path),
            "Wasserstein":      round(mean_wd, 4),
            "KS Podobné %":     round(pct_sim, 1),
            "MAE korelácia":    round(mae, 4),
            "TRTR AUC":         tstr.get("TRTR AUC", "N/A"),
            "TSTR AUC":         tstr.get("TSTR AUC", "N/A"),
            "TSTR F1":          tstr.get("TSTR F1", "N/A"),
            "Retencia AUC %":   tstr.get("Retencia AUC (%)", "N/A"),
            "Retencia F1 %":    tstr.get("Retencia F1 (%)", "N/A"),
            "NND Ratio":        round(privacy["ratio"], 4),
        })

    # Celkový súhrn všetkých vĺn
    if all_summaries:
        print(f"\n\n{'='*65}")
        print("  CELKOVÝ SÚHRN — VŠETKY VLNY")
        print(f"{'='*65}")
        df_sum = pd.DataFrame(all_summaries)
        print(df_sum.to_string(index=False))
        df_sum.to_csv("WGAN/wgan_summary_all_waves.csv", index=False)
        print(f"\n[Uložené] WGAN/wgan_summary_all_waves.csv")


if __name__ == "__main__":
    main()
