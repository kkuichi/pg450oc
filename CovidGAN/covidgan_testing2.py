"""
Install:
    pip install torch scikit-learn scipy matplotlib seaborn pandas openpyxl

Usage:
    python covidgan_testing.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os


# ─────────────────────────────────────────────────────────
# 0.  KONFIGURÁCIA
# ─────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR   = os.path.join(BASE_DIR, "datasets")
SCRIPT_DIR = os.path.dirname(__file__)

import glob
import re


def _list_dataset_files(data_dir):
    patterns = ["*.xlsx", "*.xls", "*.csv"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(data_dir, pat)))
    return sorted(set(paths))


def _sanitize_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"[^0-9A-Za-z_-]+", "_", name)
    return name.strip("_")


DATA_PATHS = _list_dataset_files(DATA_DIR)
if not DATA_PATHS:
    raise FileNotFoundError(f"No dataset files found in {DATA_DIR}")

SHEET_NAME  = 0

# Toto je podmienená premenná CovidGAN — trieda (label)
# CovidGAN generuje dáta podmienené závažnosťou
TARGET_COL  = "Závažnosť priebehu ochorenia"

CATEGORICAL_COLS = [
    "Pohlavie",
    "Vakcinácia",
    "Typ vakcíny",
    "Prekonal COVID-19",
    "Hypertenzia",
    "Diabetes mellitus",
    "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia",
    "Renálne ochorenia",
    "Imunosupresia",
    "Onkologické ochorenia",
    TARGET_COL,
]

DROP_COLS = [
    "Poradie", "Meno", "Unnamed: 23", "A04.7",
    "Kód príjmu", "Kód prepustenia", "DRG výkony",
    "Epikríza", "Terajšie ochorenie", "SVLZ správy",
    "Diagnózy", "Lieková anamnéza", "Mikrobiológia",
    "Návyková anamnéza", "Epidemiologická anamnéza",
    "Objektívny nález", "Osobná anamnéza",
    "Dôvod hospitalizácie", "HLN Dg.",
]

DROP_SUFFIXES = [" min", " max"]

# CovidGAN / ACGAN hyperparametre
NOISE_DIM   = 100       
HIDDEN_DIM  = 256
EPOCHS      = 500
BATCH_SIZE  = 64
LR_GEN      = 2e-4
LR_DIS      = 2e-4
LAMBDA_CLS  = 1.0       
N_SYNTHETIC = None      

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────
# 1.  NAČÍTANIE A PREDSPRACOVANIE DÁT
# ─────────────────────────────────────────────────────────

def load_data(path, sheet_name, drop_cols, drop_suffixes):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    bad = [c for c in df.columns if any(c.endswith(s) for s in drop_suffixes)]
    df  = df.drop(columns=bad, errors="ignore")
    df  = df.dropna(axis=1, thresh=int(len(df) * 0.4))

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    print(f"[Data] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def encode_and_scale(df, categorical_cols, target_col):
    """Zakóduje kategórie, samostatne zakóduje cieľovú premennú a škáluje príznaky do rozsahu [0,1].
    Vráti maticu príznakov, vektory štítkov, mená príznakov, škálovač a enkódery.
    """
    df_enc = df.copy()
    encoders = {}

    
    target_le = LabelEncoder()
    if target_col in df_enc.columns:
        labels = target_le.fit_transform(df_enc[target_col].astype(str))
        encoders[target_col] = target_le
    else:
        labels = np.zeros(len(df_enc), dtype=int)
        print(f"[Warning] Target column '{target_col}' not found.")

    for col in df_enc.columns:
        if col != target_col and not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le

    feature_names = df_enc.columns.tolist()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_enc.values.astype(np.float32))

    n_classes = len(np.unique(labels))
    print(f"[Data] Classes ({target_col}): {n_classes} → {list(target_le.classes_)}")
    print(f"[Data] Class distribution: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    return X, labels, feature_names, scaler, encoders, target_le


# ─────────────────────────────────────────────────────────
# 2.  COVIDGAN (ACGAN) — ARCHITEKTÚRA MODELU
# ─────────────────────────────────────────────────────────

class CovidGenerator(nn.Module):
    """Generátor podmienený triedou — spája one-hot vektor triedy s náhodným šumom.
    Umožňuje generovať záznamy pre žiadanú úroveň závažnosti.
    """
    def __init__(self, noise_dim, n_classes, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = noise_dim + n_classes      # šum + one-hot label

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),                           # výstup v rozsahu [0, 1]
        )

    def forward(self, noise, labels_onehot):
        x = torch.cat([noise, labels_onehot], dim=1)
        return self.net(x)


class CovidDiscriminator(nn.Module):
    """ACGAN diskriminátor s dvoma výstupmi:
    1. real_fake  -> sigmoid skóre (reálny vs. syntetický)
    2. class_pred -> softmax pre pravdepodobnosti triedy závažnosti
    """
    def __init__(self, input_dim, n_classes, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()                            # reálne / syntetické
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, n_classes),
            nn.Softmax(dim=1)                       # pravdepodobnosti tried
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.adv_head(shared), self.cls_head(shared)


def labels_to_onehot(labels, n_classes, device):
    onehot = torch.zeros(len(labels), n_classes, device=device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


# ─────────────────────────────────────────────────────────
# 3.  TRÉNOVANIE
# ─────────────────────────────────────────────────────────

def train_covidgan(X_train, y_train, n_classes, epochs, batch_size,
                   noise_dim, hidden_dim, lr_gen, lr_dis, lambda_cls, checkpoint_path):
    """Trénuje CovidGAN na zadaných tréningových dátach.

    Trénovanie optimalizuje dve straty súčasne: adversariálnu (reálny vs. falošný)
    a klasifikačnú (predikcia triedy). Podporuje obnovenie z checkpointu,
    periodické ukladanie checkpointov a zber histórie strat a skóre.
    """

    data_dim = X_train.shape[1]

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.LongTensor(y_train).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    G = CovidGenerator(noise_dim, n_classes, data_dim, hidden_dim).to(DEVICE)
    D = CovidDiscriminator(data_dim, n_classes, hidden_dim).to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_dis, betas=(0.5, 0.999))

    adv_loss = nn.BCELoss()
    cls_loss = nn.CrossEntropyLoss()

    history = {
        "gen_loss": [], "dis_loss": [],
        "dis_adv": [], "dis_cls": [],
        "gen_adv": [], "gen_cls": [],
        "dis_real": [], "dis_fake": [],
    }

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resumed training from epoch {start_epoch}")

    print(f"\n[CovidGAN] Training on {DEVICE} | Epochs: {epochs} | Classes: {n_classes}")

    for epoch in range(start_epoch, epochs):
        g_adv, g_cls_l, d_adv, d_cls_l = [], [], [], []
        d_real_scores, d_fake_scores = [], []

        for real_x, real_y in loader:
            bs = real_x.size(0)

            real_labels  = torch.ones(bs, 1, device=DEVICE) * 0.9  # vyhladzovanie štítkov
            fake_labels  = torch.zeros(bs, 1, device=DEVICE)

            # ── Aktualizácia diskriminátora ────────────
            noise  = torch.randn(bs, noise_dim, device=DEVICE)
            fake_y = torch.randint(0, n_classes, (bs,), device=DEVICE)
            fake_onehot = labels_to_onehot(fake_y, n_classes, DEVICE)
            fake_x = G(noise, fake_onehot).detach()

            real_rf, real_cls = D(real_x)
            fake_rf, fake_cls = D(fake_x)

            loss_D_adv = adv_loss(real_rf, real_labels) + adv_loss(fake_rf, fake_labels)
            loss_D_cls = cls_loss(real_cls, real_y) + cls_loss(fake_cls, fake_y)
            loss_D = loss_D_adv + lambda_cls * loss_D_cls

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            d_adv.append(loss_D_adv.item())
            d_cls_l.append(loss_D_cls.item())
            d_real_scores.append(real_rf.mean().item())
            d_fake_scores.append(fake_rf.mean().item())

            # ── Aktualizácia generátora ──────────────
            noise  = torch.randn(bs, noise_dim, device=DEVICE)
            fake_y = torch.randint(0, n_classes, (bs,), device=DEVICE)
            fake_onehot = labels_to_onehot(fake_y, n_classes, DEVICE)
            fake_x = G(noise, fake_onehot)

            fake_rf, fake_cls = D(fake_x)

            loss_G_adv = adv_loss(fake_rf, torch.ones(bs, 1, device=DEVICE))
            loss_G_cls = cls_loss(fake_cls, fake_y)
            loss_G = loss_G_adv + lambda_cls * loss_G_cls

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_adv.append(loss_G_adv.item())
            g_cls_l.append(loss_G_cls.item())

        history["gen_loss"].append(np.mean(g_adv) + lambda_cls * np.mean(g_cls_l))
        history["dis_loss"].append(np.mean(d_adv) + lambda_cls * np.mean(d_cls_l))
        history["gen_adv"].append(np.mean(g_adv))
        history["gen_cls"].append(np.mean(g_cls_l))
        history["dis_adv"].append(np.mean(d_adv))
        history["dis_cls"].append(np.mean(d_cls_l))
        history["dis_real"].append(np.mean(d_real_scores))
        history["dis_fake"].append(np.mean(d_fake_scores))

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | "
                  f"G: {history['gen_loss'][-1]:.4f} "
                  f"(adv {history['gen_adv'][-1]:.3f} + "
                  f"cls {history['gen_cls'][-1]:.3f}) | "
                  f"D: {history['dis_loss'][-1]:.4f} | "
                  f"D(real): {history['dis_real'][-1]:.3f} | "
                  f"D(fake): {history['dis_fake'][-1]:.3f}")

            # Ulož checkpoint každých 50 epôch
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'history': history
            }, checkpoint_path)

    # Ulož finálny checkpoint
    torch.save({
        'epoch': epochs - 1,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
        'history': history
    }, checkpoint_path)

    print("  CovidGAN training complete.\n")
    return G, D, history


# ─────────────────────────────────────────────────────────
# 4.  GENEROVANIE — podmienené & vyvážené
# ─────────────────────────────────────────────────────────

def generate_synthetic(G, n_samples, n_classes, noise_dim, scaler,
                       class_labels=None):
    """Vygeneruje syntetické záznamy.
    Ak je `class_labels` None -> náhodné triedy (nepodmienené).
    Ak je `class_labels` pole -> vygeneruje presne tieto triedy (podmienené).
    """
    G.eval()
    with torch.no_grad():
        if class_labels is None:
            labels = torch.randint(0, n_classes, (n_samples,), device=DEVICE)
        else:
            labels = torch.LongTensor(class_labels).to(DEVICE)

        noise   = torch.randn(n_samples, noise_dim, device=DEVICE)
        onehot  = labels_to_onehot(labels, n_classes, DEVICE)
        fake_x  = G(noise, onehot).cpu().numpy()
        fake_y  = labels.cpu().numpy()

    synthetic = scaler.inverse_transform(fake_x)
    print(f"[Generated] {len(synthetic)} synthetic samples")
    return synthetic, fake_y


def generate_balanced(G, n_per_class, n_classes, noise_dim, scaler):
    """Funkcia CovidGAN: vygeneruje rovnaký počet vzoriek pre každú triedu.
    Užitečné pre oversampling minoritných tried.
    """
    print(f"\n[Balanced Generation] {n_per_class} samples × {n_classes} classes "
          f"= {n_per_class * n_classes} total")
    labels = np.repeat(np.arange(n_classes), n_per_class)
    syn, syn_y = generate_synthetic(G, len(labels), n_classes, noise_dim,
                                    scaler, class_labels=labels)
    counts = dict(zip(*np.unique(syn_y, return_counts=True)))
    print(f"  Generated class distribution: {counts}")
    return syn, syn_y


# ─────────────────────────────────────────────────────────
# 5.  VYHODNOCOVANIE
# ─────────────────────────────────────────────────────────

"""
 Blok vyhodnocovania: súbor funkcií na hodnotenie kvality syntetických dát — štatistická podobnosť,
 kvalita podmienenej generácie, utility (TSTR), súkromie (NND), korelácie a vizualizácie; 
 každá funkcia vracia metriky a/alebo grafy pre diagnostiku a export výsledkov.
"""
def evaluate_statistical_similarity(X_real, X_syn, feature_names, categorical_cols):
    print("\n" + "="*55)
    print("  ŠTATISTICKÁ PODOBNOSŤ")
    print("="*55)

    num_results, cat_results = [], []
    for i, name in enumerate(feature_names):
        r, s = X_real[:, i], X_syn[:, i]
        if name in categorical_cols:
            rv = pd.Series(np.round(r).astype(int).astype(str)).value_counts(normalize=True)
            sv = pd.Series(np.round(s).astype(int).astype(str)).value_counts(normalize=True)
            tvd = sum(abs(rv.get(v, 0) - sv.get(v, 0))
                      for v in set(rv.index) | set(sv.index)) / 2
            cat_results.append({"feature": name,
                                 "total_variation_dist": round(tvd, 4),
                                 "similar": "✓" if tvd < 0.1 else "✗"})
        else:
            wd = wasserstein_distance(r, s)
            ks, p = ks_2samp(r, s)
            num_results.append({"feature": name,
                                 "wasserstein": round(wd, 4),
                                 "ks_stat": round(ks, 4),
                                 "ks_p": round(p, 4),
                                 "similar": "✓" if p > 0.05 else "✗"})

    df_num = pd.DataFrame(num_results)
    df_cat = pd.DataFrame(cat_results)

    mean_wd  = df_num["wasserstein"].mean() if not df_num.empty else float("nan")
    sim_num  = (df_num["similar"] == "✓").mean() * 100 if not df_num.empty else 0
    sim_cat  = (df_cat["similar"] == "✓").mean() * 100 if not df_cat.empty else 0

    print(f"\n  Numerické  — Priemerný Wasserstein: {mean_wd:.4f} | Podobné: {sim_num:.1f}%")
    print(f"  Kategórie — Podobné (TVD<0.1): {sim_cat:.1f}%")
    print("\n  Top 15 numeric features by Wasserstein distance:")
    print(df_num.sort_values("wasserstein", ascending=False).head(15).to_string(index=False))

    return df_num, df_cat, mean_wd


def evaluate_class_conditioning(X_real, y_real, X_syn, y_syn,
                                 feature_names, target_le):
    """Špecifické pre CovidGAN: overí kvalitu podmienenej generácie.
    Porovná priemery príznakov pre jednotlivé triedy medzi reálnymi a syntetickými dátami.
    """
    print("\n" + "="*55)
    print("  CLASS CONDITIONING QUALITY")
    print("="*55)

    classes  = np.unique(y_real)
    n_show   = min(5, len(feature_names))
    key_feats = feature_names[:n_show]

    print(f"\n  Feature means per severity class (first {n_show} features):")
    print(f"  {'Class':<12} {'Source':<12}", end="")
    for f in key_feats:
        print(f"  {f[:12]:<14}", end="")
    print()
    print("  " + "-" * (26 + 16 * n_show))

    class_stats = {}
    for cls in classes:
        label = target_le.classes_[cls] if cls < len(target_le.classes_) else str(cls)
        r_mask = y_real == cls
        s_mask = y_syn  == cls

        r_means = X_real[r_mask][:, :n_show].mean(axis=0) if r_mask.sum() > 0 else np.zeros(n_show)
        s_means = X_syn [s_mask][:, :n_show].mean(axis=0) if s_mask.sum() > 0 else np.zeros(n_show)

        print(f"  {label:<12} {'Real':<12}", end="")
        for v in r_means:
            print(f"  {v:>14.3f}", end="")
        print()
        print(f"  {'':<12} {'Synthetic':<12}", end="")
        for v in s_means:
            print(f"  {v:>14.3f}", end="")
        print()

        class_stats[cls] = {"real_means": r_means, "syn_means": s_means,
                             "n_real": r_mask.sum(), "n_syn": s_mask.sum()}

    return class_stats


def evaluate_tstr(X_real, y_real, X_syn, y_syn):
    print("\n" + "="*55)
    print("  TSTR — HODNOTENIE UŽITOČNOSTI")
    print("="*55)

    if y_real is None:
        print("  No target found, skipping."); return {}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    n_classes = len(np.unique(y_real))
    avg = "binary" if n_classes == 2 else "macro"

    def metrics(clf, Xt, yt):
        yp  = clf.predict(Xt)
        ypr = clf.predict_proba(Xt)
        try:
            auc = (roc_auc_score(yt, ypr[:, 1]) if n_classes == 2
                   else roc_auc_score(yt, ypr, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        return {"AUC":      round(auc, 4),
                "Accuracy": round(accuracy_score(yt, yp), 4),
                "F1":       round(f1_score(yt, yp, average=avg, zero_division=0), 4)}

    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_tr, y_tr)
    trtr = metrics(clf_trtr, X_te, y_te)

    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(X_syn, y_syn)
    tstr = metrics(clf_tstr, X_te, y_te)

    print(f"\n  {'Metric':<12} {'TRTR (real)':>14} {'TSTR (synthetic)':>18}")
    print(f"  {'-'*46}")
    for m in ["AUC", "Accuracy", "F1"]:
        print(f"  {m:<12} {trtr[m]:>14} {tstr[m]:>18}")

    return {"TRTR": trtr, "TSTR": tstr,
            "clf_trtr": clf_trtr, "clf_tstr": clf_tstr,
            "X_te": X_te, "y_te": y_te}


def evaluate_privacy(X_real, X_syn, k=5):
    print("\n" + "="*55)
    print("  SÚKROMIE — NEAREST NEIGHBOR DISTANCE")
    print("="*55)

    n  = min(500, len(X_real), len(X_syn))
    r  = X_real[np.random.choice(len(X_real), n, replace=False)]
    s  = X_syn [np.random.choice(len(X_syn),  n, replace=False)]

    d_sr = cdist(s, r); d_rr = cdist(r, r)
    np.fill_diagonal(d_rr, np.inf)

    nnd_sr = np.sort(d_sr, axis=1)[:, :k].mean(axis=1).mean()
    nnd_rr = np.sort(d_rr, axis=1)[:, :k].mean(axis=1).mean()
    ratio  = nnd_sr / nnd_rr

    print(f"\n  NND synthetic→real : {nnd_sr:.4f}")
    print(f"  NND real→real      : {nnd_rr:.4f}")
    print(f"  Ratio (>1.0 = safe): {ratio:.4f}")
    print(f"  Verdict: {'✓ Safe — no memorization' if ratio >= 1.0 else '⚠ Possible memorization'}")
    return {"nnd_syn_real": nnd_sr, "nnd_real_real": nnd_rr, "ratio": ratio}


def evaluate_correlation(X_real, X_syn, feature_names):
    real_corr = pd.DataFrame(X_real, columns=feature_names).corr()
    syn_corr  = pd.DataFrame(X_syn,  columns=feature_names).corr()
    diff = (real_corr - syn_corr).abs()
    mean_diff = diff.values[np.triu_indices_from(diff.values, k=1)].mean()
    print(f"\n[Correlation] Mean Absolute Difference: {mean_diff:.4f}  (lower = better)")
    return real_corr, syn_corr, diff


# ─────────────────────────────────────────────────────────
# 6.  GRAFY
# ─────────────────────────────────────────────────────────

def plot_training_curves(history, save_path="covidgan_training.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("CovidGAN (ACGAN) Training Curves", fontsize=14, fontweight="bold")

    axes[0, 0].plot(history["gen_loss"], color="#457b9d")
    axes[0, 0].set_title("Generator Total Loss")

    axes[0, 1].plot(history["dis_loss"], color="#e63946")
    axes[0, 1].set_title("Discriminator Total Loss")

    axes[0, 2].plot(history["dis_real"], color="#2a9d8f", label="D(real)")
    axes[0, 2].plot(history["dis_fake"], color="#e63946", label="D(fake)", alpha=0.7)
    axes[0, 2].axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Ideal = 0.5")
    axes[0, 2].set_title("Discriminator Scores")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(history["gen_adv"],  color="#457b9d", label="G adv")
    axes[1, 0].plot(history["gen_cls"],  color="#f4a261", label="G cls", alpha=0.8)
    axes[1, 0].set_title("Generator: Adversarial vs. Class Loss")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(history["dis_adv"],  color="#e63946", label="D adv")
    axes[1, 1].plot(history["dis_cls"],  color="#f4a261", label="D cls", alpha=0.8)
    axes[1, 1].set_title("Discriminator: Adversarial vs. Class Loss")
    axes[1, 1].legend(fontsize=8)

    # Pomer triedy ku adversariálnej strate (mal by sa stabilizovať)
    gen_ratio = [c / (a + 1e-8) for a, c in
                 zip(history["gen_adv"], history["gen_cls"])]
    axes[1, 2].plot(gen_ratio, color="#8338ec")
    axes[1, 2].axhline(1.0, linestyle="--", color="gray", alpha=0.5)
    axes[1, 2].set_title("Generator: Class/Adv Loss Ratio\n(→1.0 = balanced)")

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_class_distributions(X_real, y_real, X_syn, y_syn, target_le,
                              save_path="covidgan_class_dist.png"):
    """Špecifické pre CovidGAN: rozdelenie príznakov podľa tried.
    Zobrazuje, či sú jednotlivé triedy generované verne.
    """
    classes   = np.unique(y_real)
    n_classes = len(classes)
    n_feats   = min(4, X_real.shape[1])

    fig, axes = plt.subplots(n_classes, n_feats,
                             figsize=(n_feats * 4, n_classes * 3))
    if n_classes == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Per-Class Feature Distributions: Real vs. Synthetic (CovidGAN)",
                 fontsize=13, fontweight="bold")

    for ci, cls in enumerate(classes):
        label = target_le.classes_[cls] if cls < len(target_le.classes_) else str(cls)
        r_mask = y_real == cls
        s_mask = y_syn  == cls

        for fi in range(n_feats):
            ax = axes[ci, fi]
            if r_mask.sum() > 0:
                ax.hist(X_real[r_mask, fi], bins=20, alpha=0.6,
                        color="#457b9d", label="Real", density=True)
            if s_mask.sum() > 0:
                ax.hist(X_syn[s_mask, fi],  bins=20, alpha=0.6,
                        color="#e63946", label="Synthetic", density=True)
            if fi == 0:
                ax.set_ylabel(f"Class: {label}", fontsize=8, fontweight="bold")
            ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_class_balance(y_real, y_syn, y_balanced, target_le,
                       save_path="covidgan_class_balance.png"):
    """Zobrazí rozdelenie tried pred a po augmentácii CovidGAN-om."""
    classes = [target_le.classes_[i] if i < len(target_le.classes_) else str(i)
               for i in range(len(target_le.classes_))]

    def count(y):
        counts = pd.Series(y).value_counts().sort_index()
        return [counts.get(i, 0) for i in range(len(classes))]

    r_counts  = count(y_real)
    s_counts  = count(y_syn)
    b_counts  = count(y_balanced)

    x = np.arange(len(classes))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Class Balance: Real vs. Generated vs. Balanced (CovidGAN)",
                 fontsize=13, fontweight="bold")

    for ax, counts, title, color in zip(
        axes,
        [r_counts, s_counts, b_counts],
        ["Real Data", "Synthetic (random)", "Synthetic (balanced)"],
        ["#457b9d", "#e63946", "#2a9d8f"]
    ):
        bars = ax.bar(x, counts, color=color, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(bar.get_height())), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_pca_by_class(X_real, y_real, X_syn, y_syn, target_le,
                      save_path="covidgan_pca_classes.png"):
    """PCA farebne podľa triedy — reálne vs. syntetické vedľa seba."""
    n = min(500, len(X_real), len(X_syn))
    pca = PCA(n_components=2)
    pca.fit(np.vstack([X_real[:n], X_syn[:n]]))
    r2 = pca.transform(X_real[:n])
    s2 = pca.transform(X_syn[:n])

    classes = np.unique(y_real)
    palette = plt.cm.Set1(np.linspace(0, 0.8, len(classes)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA by Severity Class: Real vs. Synthetic (CovidGAN)",
                 fontsize=13, fontweight="bold")

    for ax, data, labels, title in zip(
        axes, [r2, s2], [y_real[:n], y_syn[:n]], ["Real Data", "Synthetic Data"]
    ):
        for ci, cls in enumerate(classes):
            mask = labels == cls
            lbl  = target_le.classes_[cls] if cls < len(target_le.classes_) else str(cls)
            ax.scatter(data[mask, 0], data[mask, 1],
                       alpha=0.5, s=15, color=palette[ci], label=lbl)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=7, title="Severity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_distributions(X_real, X_syn, feature_names, n_features=8,
                       save_path="covidgan_distributions.png"):
    """Histogramy klinických príznakov rozdelené do dvoch častí po 4."""
    idx_list = [i for i, n in enumerate(feature_names)
                if n not in CATEGORICAL_COLS][:n_features]

    for part, part_idxs in enumerate([idx_list[:4], idx_list[4:]], start=1):
        if not part_idxs:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"Feature Distributions: Real vs. Synthetic (CovidGAN) — Part {part}",
            fontsize=14, fontweight="bold"
        )
        axes = axes.flatten()
        for k, idx in enumerate(part_idxs):
            axes[k].hist(X_real[:, idx], bins=30, alpha=0.6,
                         color="#457b9d", label="Real", density=True)
            axes[k].hist(X_syn[:, idx],  bins=30, alpha=0.6,
                         color="#e63946", label="Synthetic", density=True)
            axes[k].set_title(feature_names[idx], fontsize=8)
            axes[k].set_xlabel("Value")
            axes[k].set_ylabel("Density")
            axes[k].legend(fontsize=7)
        for j in range(len(part_idxs), 4):
            axes[j].set_visible(False)
        plt.tight_layout()
        base, ext = os.path.splitext(save_path)
        out = f"{base}_part{part}{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {out}")
        plt.close()


def plot_confusion_matrices(tstr_results, target_le,
                             save_path="covidgan_confusion.png"):
    """Compare confusion matrices: TRTR vs. TSTR — unique CovidGAN diagnostic."""
    if not tstr_results or "clf_trtr" not in tstr_results:
        return

    clf_trtr = tstr_results["clf_trtr"]
    clf_tstr = tstr_results["clf_tstr"]
    X_te, y_te = tstr_results["X_te"], tstr_results["y_te"]
    class_names = list(target_le.classes_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix: TRTR vs. TSTR (CovidGAN)",
                 fontsize=13, fontweight="bold")

    for ax, clf, title in zip(
        axes, [clf_trtr, clf_tstr], ["TRTR (trained on real)", "TSTR (trained on synthetic)"]
    ):
        cm = confusion_matrix(y_te, clf.predict(X_te))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_tstr_bar(tstr_results, save_path="covidgan_tstr.png"):
    if not tstr_results:
        return
    labels  = ["AUC", "Accuracy", "F1"]
    trtr_v  = [tstr_results["TRTR"].get(m, 0) for m in labels]
    tstr_v  = [tstr_results["TSTR"].get(m, 0) for m in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - 0.2, trtr_v, 0.35, label="TRTR (real)",      color="#457b9d")
    b2 = ax.bar(x + 0.2, tstr_v, 0.35, label="TSTR (synthetic)", color="#e63946")
    ax.set_ylim(0, 1.1); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("TSTR vs. TRTR — Utility (CovidGAN)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score"); ax.legend()
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_wasserstein_bar(df_num_stats, save_path="covidgan_wasserstein.png"):
    df_s = df_num_stats.sort_values("wasserstein", ascending=False).head(20)
    colors = ["#e63946" if v > 0.5 else "#457b9d" for v in df_s["wasserstein"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(df_s["feature"], df_s["wasserstein"], color=colors)
    ax.axvline(0.5, linestyle="--", color="gray", alpha=0.6, label="Threshold 0.5")
    ax.set_title("Per-feature Wasserstein Distance (CovidGAN)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Wasserstein Distance (lower = better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_correlation_heatmaps(real_corr, syn_corr, diff,
                               save_path="covidgan_correlations.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Correlation Matrix Comparison (CovidGAN)",
                 fontsize=14, fontweight="bold")
    for ax, data, title, cmap in zip(
        axes,
        [real_corr, syn_corr, diff],
        ["Real Data", "Synthetic Data", "Absolute Difference"],
        ["coolwarm", "coolwarm", "Reds"]
    ):
        sns.heatmap(data, ax=ax, cmap=cmap, square=True, linewidths=0.2,
                    center=0 if cmap != "Reds" else None,
                    cbar_kws={"shrink": 0.8},
                    annot=len(real_corr) <= 10, fmt=".1f", annot_kws={"size": 7})
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────

"""
Hlavná dávková pipeline skriptu: pre každý dataset načíta a predspracuje dáta,
natrénuje CovidGAN (alebo obnoví z checkpointu), vygeneruje náhodné a vyvážené
syntetické vzorky, uloží CSV súbory, vykreslí diagnostické grafy a spustí všetky
vyhodnocovacie metriky (štatistika, TSTR, korelácie, súkromie). Výstupy sú uložené
do lokálneho priečinka CovidGAN pre ďalšiu analýzu.
"""

def main():
    print("=" * 60)
    print("  CovidGAN Testing Pipeline — Medical Synthetic Data")
    print("=" * 60)

    os.chdir(SCRIPT_DIR)
    print(f"[Info] Changed working directory to {os.getcwd()}")

    for data_path in DATA_PATHS:
        data_tag = _sanitize_name(data_path)
        print(f"\n\n=== Dataset: {os.path.basename(data_path)} ({data_tag}) ===")

        # ── Load & preprocess ──────────────────────
        df = load_data(data_path, SHEET_NAME, DROP_COLS, DROP_SUFFIXES)
        X, y, feature_names, scaler, encoders, target_le = encode_and_scale(
            df, CATEGORICAL_COLS, TARGET_COL
        )
        n_classes = len(np.unique(y))
        data_dim  = X.shape[1]

        # ── Train/test split ───────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")

        # ── Train CovidGAN ─────────────────────────
        checkpoint_path = os.path.join(SCRIPT_DIR, f"covidgan_checkpoint_{data_tag}.pth")
        G, D, history = train_covidgan(
            X_train, y_train, n_classes,
            EPOCHS, BATCH_SIZE, NOISE_DIM, HIDDEN_DIM,
            LR_GEN, LR_DIS, LAMBDA_CLS, checkpoint_path
        )

        # ── Generate: random labels ────────────────
        n_gen = N_SYNTHETIC or len(X_train)
        X_syn, y_syn = generate_synthetic(
            G, n_gen, n_classes, NOISE_DIM, scaler
        )

        # ── Generate: class-balanced ───────────────
        n_per_class = max(50, len(X_train) // n_classes)
        X_bal, y_bal = generate_balanced(
            G, n_per_class, n_classes, NOISE_DIM, scaler
        )

        # ── Save outputs ───────────────────────────
        df_syn = pd.DataFrame(X_syn, columns=feature_names)
        df_syn[TARGET_COL + "_class"] = y_syn
        syn_path = os.path.join(SCRIPT_DIR, f"synthetic_covid_covidgan_{data_tag}.csv")
        df_syn.to_csv(syn_path, index=False)

        df_bal = pd.DataFrame(X_bal, columns=feature_names)
        df_bal[TARGET_COL + "_class"] = y_bal
        bal_path = os.path.join(SCRIPT_DIR, f"synthetic_covid_covidgan_balanced_{data_tag}.csv")
        df_bal.to_csv(bal_path, index=False)
        print(f"[Saved] {syn_path}")
        print(f"[Saved] {bal_path}")

        # ── Plots ──────────────────────────────────
        plot_training_curves(history, save_path=f"covidgan_training_{data_tag}.png")
        plot_distributions(X_train, X_syn, feature_names, save_path=f"covidgan_distributions_{data_tag}.png")
        plot_class_distributions(X_train, y_train, X_syn, y_syn, target_le, save_path=f"covidgan_class_dist_{data_tag}.png")
        plot_class_balance(y_train, y_syn, y_bal, target_le, save_path=f"covidgan_class_balance_{data_tag}.png")
        plot_pca_by_class(X_train, y_train, X_syn, y_syn, target_le, save_path=f"covidgan_pca_{data_tag}.png")

        # ── Statistical similarity ─────────────────
        df_num, df_cat, mean_wd = evaluate_statistical_similarity(
            X_train, X_syn, feature_names, CATEGORICAL_COLS
        )
        plot_wasserstein_bar(df_num, save_path=f"covidgan_wasserstein_{data_tag}.png")

        # ── Class conditioning quality ─────────────
        class_stats = evaluate_class_conditioning(
            X_train, y_train, X_syn, y_syn, feature_names, target_le
        )

        # ── Correlation ────────────────────────────
        real_corr, syn_corr, diff = evaluate_correlation(X_train, X_syn, feature_names)
        plot_correlation_heatmaps(real_corr, syn_corr, diff, save_path=f"covidgan_correlations_{data_tag}.png")

        # ── TSTR utility ───────────────────────────
        tstr_results = evaluate_tstr(X_train, y_train, X_syn, y_syn)
        plot_tstr_bar(tstr_results, save_path=f"covidgan_tstr_{data_tag}.png")
        plot_confusion_matrices(tstr_results, target_le, save_path=f"covidgan_confusion_{data_tag}.png")

        # ── Privacy ────────────────────────────────
        privacy = evaluate_privacy(X_train, X_syn)

        # ── Final summary ──────────────────────────
        print("\n" + "=" * 60)
        print(f"  FINAL SUMMARY — CovidGAN ({data_tag})")
        print("=" * 60)
        print(f"  Synthetic samples (random)    : {len(X_syn)}")
        print(f"  Synthetic samples (balanced)  : {len(X_bal)}")
        print(f"  Mean Wasserstein Distance      : {mean_wd:.4f}")
        sim_n = (df_num["similar"] == "✓").sum() if not df_num.empty else 0
        sim_c = (df_cat["similar"] == "✓").sum() if not df_cat.empty else 0
        print(f"  Similar numeric features       : {sim_n}/{len(df_num)}")
        print(f"  Similar categorical features   : {sim_c}/{len(df_cat)}")
        if tstr_results:
            print(f"  TSTR AUC   : {tstr_results['TSTR']['AUC']}  "
                  f"(TRTR: {tstr_results['TRTR']['AUC']})")
            print(f"  TSTR F1    : {tstr_results['TSTR']['F1']}  "
                  f"(TRTR: {tstr_results['TRTR']['F1']})")
        print(f"  Privacy ratio                  : {privacy['ratio']:.4f}  (>1.0 = safe)")
        print("=" * 60)


if __name__ == "__main__":
    main()
