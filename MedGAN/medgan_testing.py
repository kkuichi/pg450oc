"""
    Encoder  →  komprimuje reálne záznamy do latentného priestoru
    Decoder  →  rekonštruuje záznamy z latentných vektorov
    Generator →  produkuje falošné latentné vektory zo šumu
    Discriminator → rozlišuje reálne vs. falošné (zakódované) záznamy

Použitie:
    python medgan_testing3.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 50
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
import re
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# 0.  KONFIGURÁCIA
# ═══════════════════════════════════════════════════════════

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

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

SHEET_NAME      = 0
TARGET_COL      = "Závažnosť priebehu ochorenia"

CATEGORICAL_COLS = [
    "Pohlavie", "Vakcinácia", "Typ vakcíny", "Prekonal COVID-19",
    "Hypertenzia", "Diabetes mellitus", "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia", "Renálne ochorenia", "Imunosupresia",
    "Onkologické ochorenia", TARGET_COL,
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

# Hyperparametre MedGANu
LATENT_DIM  = 128
HIDDEN_DIM  = 256
BATCH_SIZE  = 64
AE_EPOCHS   = 100
GAN_EPOCHS  = 500
LR_AE       = 1e-3
LR_GEN      = 2e-4
LR_DIS      = 2e-4
N_CRITIC    = 2
NOISE_STD   = 0.01
N_SYNTHETIC = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════
# 1.  NAČÍTANIE A PREDSPRACOVANIE DÁT
# ═══════════════════════════════════════════════════════════

def load_data(path, sheet_name, drop_cols, drop_suffixes, categorical_cols):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    bad_cols = [c for c in df.columns if any(c.endswith(s) for s in drop_suffixes)]
    df = df.drop(columns=bad_cols, errors="ignore")
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, thresh=int(len(df) * 0.4))
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")
    print(f"[Data] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def encode_and_scale(df, categorical_cols):
    df_enc = df.copy()
    encoders = {}
    for col in categorical_cols:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    feature_names = df_enc.columns.tolist()
    pre_scale_values = df_enc.values.astype(np.float32).copy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(pre_scale_values)
    return X, feature_names, scaler, encoders, pre_scale_values


# ═══════════════════════════════════════════════════════════
# 2.  KOMPONENTY MODELU
# ═══════════════════════════════════════════════════════════

"""
Definuje architektúru autoenkodéra, generátora a diskriminátora.
"""

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
        )
    def forward(self, z): return self.net(z)


class Generator(nn.Module):
    def __init__(self, noise_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim), nn.Tanh(),
        )
    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════
# 3.  TRÉNOVANIE
# ═══════════════════════════════════════════════════════════

"""
Predtrénuje autoenkodér a potom trénuje adversariálny MedGAN na latentných reprezentáciách.
"""

def pretrain_autoencoder(encoder, decoder, X_train, epochs, batch_size, lr):
    print(f"\n[Step 1] Pre-training Autoencoder for {epochs} epochs...")
    tensor = torch.FloatTensor(X_train).to(DEVICE)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()
    ae_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in loader:
            z = encoder(batch)
            recon = decoder(z)
            loss = criterion(recon, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        ae_losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  AE Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f}")
    print("  Autoencoder pre-training complete.\n")
    return ae_losses


def train_medgan(encoder, decoder, generator, discriminator,
                 X_train, epochs, batch_size, lr_gen, lr_dis,
                 latent_dim, n_critic, noise_std):
    print(f"[Step 2] Training MedGAN for {epochs} epochs...")
    tensor = torch.FloatTensor(X_train).to(DEVICE)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    opt_G = optim.Adam(generator.parameters(),     lr=lr_gen, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr_dis, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = {"gen_loss": [], "dis_loss": [], "dis_real": [], "dis_fake": []}
    for epoch in range(epochs):
        g_losses, d_losses, d_reals, d_fakes = [], [], [], []
        for (real_batch,) in loader:
            bs = real_batch.size(0)
            with torch.no_grad():
                real_latent = encoder(real_batch)
            for _ in range(n_critic):
                noise_r = torch.randn_like(real_latent) * noise_std
                noise_f_in = torch.randn(bs, latent_dim, device=DEVICE)
                fake_latent = generator(noise_f_in).detach()
                noise_f = torch.randn_like(fake_latent) * noise_std
                real_scores = discriminator(real_latent + noise_r)
                fake_scores = discriminator(fake_latent + noise_f)
                real_labels = torch.ones(bs, 1, device=DEVICE) * 0.9
                fake_labels = torch.zeros(bs, 1, device=DEVICE)
                loss_D = criterion(real_scores, real_labels) + criterion(fake_scores, fake_labels)
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()
                d_losses.append(loss_D.item())
                d_reals.append(real_scores.mean().item())
                d_fakes.append(fake_scores.mean().item())
            noise_g = torch.randn(bs, latent_dim, device=DEVICE)
            fake_latent = generator(noise_g)
            fake_scores = discriminator(fake_latent)
            loss_G = criterion(fake_scores, torch.ones(bs, 1, device=DEVICE))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            g_losses.append(loss_G.item())
        history["gen_loss"].append(np.mean(g_losses))
        history["dis_loss"].append(np.mean(d_losses))
        history["dis_real"].append(np.mean(d_reals))
        history["dis_fake"].append(np.mean(d_fakes))
        if (epoch + 1) % 50 == 0:
            print(f"  GAN Epoch {epoch+1:4d}/{epochs} | "
                  f"G: {history['gen_loss'][-1]:.4f} | D: {history['dis_loss'][-1]:.4f} | "
                  f"D(real): {history['dis_real'][-1]:.3f} | D(fake): {history['dis_fake'][-1]:.3f}")
    print("  MedGAN training complete.\n")
    return history


# ═══════════════════════════════════════════════════════════
# 4.  GENEROVANIE SYNTHETICKÝCH DÁT
# ═══════════════════════════════════════════════════════════

"""
Vytvorí syntetické vzorky z náhodného šumu, dekóduje ich späť do pôvodného dátového priestoru.
"""

def generate_synthetic(generator, decoder, n_samples, latent_dim, scaler):
    generator.eval(); decoder.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim, device=DEVICE)
        fake_latent  = generator(noise)
        fake_records = decoder(fake_latent).cpu().numpy()
    synthetic = scaler.inverse_transform(fake_records)
    print(f"[Generated] {len(synthetic)} synthetic samples")
    return synthetic


# ═══════════════════════════════════════════════════════════
# 5.  VYHODNOCOVANIE
# ═══════════════════════════════════════════════════════════

"""
Porovnáva syntetické a reálne dáta pomocou štatistík, rekonstrukčnej chyby, utility (TSTR), súkromia a korelácií.
"""

def evaluate_statistical_similarity(X_real, X_syn, feature_names, categorical_cols):
    print("\n" + "="*55)
    print("  STATISTICAL SIMILARITY")
    print("="*55)
    num_results, cat_results = [], []
    for i, name in enumerate(feature_names):
        is_cat = name in categorical_cols
        r_col = X_real[:, i]; s_col = X_syn[:, i]
        if is_cat:
            r_vals = np.round(r_col).astype(int).astype(str)
            s_vals = np.round(s_col).astype(int).astype(str)
            all_v  = set(r_vals) | set(s_vals)
            r_freq = pd.Series(r_vals).value_counts(normalize=True)
            s_freq = pd.Series(s_vals).value_counts(normalize=True)
            tvd = sum(abs(r_freq.get(v, 0) - s_freq.get(v, 0)) for v in all_v) / 2
            cat_results.append({"feature": name, "total_variation_dist": round(tvd, 4),
                                 "similar": "✓" if tvd < 0.1 else "✗"})
        else:
            wd = wasserstein_distance(r_col, s_col)
            ks, p = ks_2samp(r_col, s_col)
            num_results.append({"feature": name, "wasserstein": round(wd, 4),
                                 "ks_stat": round(ks, 4), "ks_p": round(p, 4),
                                 "similar": "✓" if p > 0.05 else "✗"})
    df_num = pd.DataFrame(num_results)
    df_cat = pd.DataFrame(cat_results)
    mean_wd  = df_num["wasserstein"].mean() if not df_num.empty else float("nan")
    sim_num  = (df_num["similar"] == "✓").mean() * 100 if not df_num.empty else 0
    sim_cat  = (df_cat["similar"] == "✓").mean() * 100 if not df_cat.empty else 0
    print(f"\n  Numeric  — Mean Wasserstein: {mean_wd:.4f} | Similar: {sim_num:.1f}%")
    print(f"  Categorical — Similar (TVD<0.1): {sim_cat:.1f}%")
    return df_num, df_cat, mean_wd


def evaluate_reconstruction(encoder, decoder, X_real):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(X_real).to(DEVICE)
        recon = decoder(encoder(tensor)).cpu().numpy()
    mse = np.mean((X_real - recon) ** 2)
    print(f"\n[Autoencoder] Reconstruction MSE: {mse:.6f}")
    return mse


def evaluate_tstr(X_train_real, y_train_real, X_syn, X_test_real, y_test_real):
    print("\n" + "="*55)
    print("  TSTR — UTILITY EVALUATION")
    print("="*55)
    if y_train_real is None or y_test_real is None:
        print("  No target found, skipping TSTR.")
        return {}
    y_tr = y_train_real.astype(int)
    y_te = y_test_real.astype(int)
    n_classes = len(np.unique(y_tr))
    avg = "binary" if n_classes == 2 else "macro"
    def metrics(clf, Xt, yt):
        yp  = clf.predict(Xt); ypr = clf.predict_proba(Xt)
        try:
            auc = (roc_auc_score(yt, ypr[:, 1]) if n_classes == 2
                   else roc_auc_score(yt, ypr, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        return {"AUC": round(auc, 4), "Accuracy": round(accuracy_score(yt, yp), 4),
                "F1":  round(f1_score(yt, yp, average=avg, zero_division=0), 4)}
    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_train_real, y_tr)
    trtr = metrics(clf_trtr, X_test_real, y_te)
    syn_labels = clf_trtr.predict(X_syn)
    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(X_syn, syn_labels)
    tstr = metrics(clf_tstr, X_test_real, y_te)
    print(f"\n  {'Metric':<12} {'TRTR (real)':>14} {'TSTR (synthetic)':>18}")
    print(f"  {'-'*46}")
    for m in ["AUC", "Accuracy", "F1"]:
        print(f"  {m:<12} {trtr[m]:>14} {tstr[m]:>18}")
    return {"TRTR": trtr, "TSTR": tstr}


def evaluate_privacy(X_real, X_syn, k=5):
    print("\n" + "="*55)
    print("  PRIVACY — NEAREST NEIGHBOR DISTANCE")
    print("="*55)
    n = min(500, len(X_real), len(X_syn))
    r = X_real[np.random.choice(len(X_real), n, replace=False)]
    s = X_syn [np.random.choice(len(X_syn),  n, replace=False)]
    d_sr = cdist(s, r, metric="euclidean")
    d_rr = cdist(r, r, metric="euclidean")
    np.fill_diagonal(d_rr, np.inf)
    nnd_sr = np.sort(d_sr, axis=1)[:, :k].mean(axis=1).mean()
    nnd_rr = np.sort(d_rr, axis=1)[:, :k].mean(axis=1).mean()
    ratio  = nnd_sr / nnd_rr
    print(f"\n  NND synthetic→real : {nnd_sr:.4f}")
    print(f"  NND real→real      : {nnd_rr:.4f}")
    print(f"  Ratio (>1.0 = safe): {ratio:.4f}")
    print(f"  Verdict: {'✓ Safe' if ratio >= 1.0 else '⚠ Possible memorization'}")
    return {"nnd_syn_real": nnd_sr, "nnd_real_real": nnd_rr, "ratio": ratio}


def evaluate_correlation(X_real, X_syn, feature_names):
    df_r = pd.DataFrame(X_real, columns=feature_names)
    df_s = pd.DataFrame(X_syn,  columns=feature_names)
    real_corr = df_r.corr().fillna(0)
    syn_corr  = df_s.corr().fillna(0)
    diff = (real_corr - syn_corr).abs()
    mean_diff = diff.values[np.triu_indices_from(diff.values, k=1)].mean()
    print(f"\n[Correlation] Mean Absolute Difference: {mean_diff:.4f}")
    return real_corr, syn_corr, diff


# ═══════════════════════════════════════════════════════════
# 6.  GRAFY
# ═══════════════════════════════════════════════════════════

def plot_training_curves(ae_losses, gan_history, save_path="medgan_training.png"):
    """
    Rozloženie 2x2:
      Riadok 1: Strata autoenkódera | Strata generátora
      Riadok 2: Strata diskriminátora | Skóre diskriminátora
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("MedGAN Training Curves", fontsize=14, fontweight="bold")

    axes[0, 0].plot(ae_losses, color="#2a9d8f")
    axes[0, 0].set_title("Autoencoder\nReconstruction Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(gan_history["gen_loss"], color="#457b9d")
    axes[0, 1].set_title("Generator Loss")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(gan_history["dis_loss"], color="#e63946")
    axes[1, 0].set_title("Discriminator Loss")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Loss")

    axes[1, 1].plot(gan_history["dis_real"], color="#2a9d8f", label="D(real)")
    axes[1, 1].plot(gan_history["dis_fake"], color="#e63946", label="D(fake)", alpha=0.7)
    axes[1, 1].axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Ideal = 0.5")
    axes[1, 1].set_title("Discriminator Scores\n(should converge to 0.5)")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_distributions(X_real, X_syn, feature_names, n_features=8,
                       save_path="medgan_distributions.png"):
    """Rozdelí numerické vlastnosti do viacerých grafov po dvoch."""
    num_idx = [i for i in range(len(feature_names))
               if feature_names[i] not in CATEGORICAL_COLS][:n_features]
    base_path = save_path.replace(".png", "")

    chunks = [num_idx[i:i+2] for i in range(0, len(num_idx), 2)]

    for part, part_idxs in enumerate(chunks, start=1):
        if not part_idxs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f"Feature Distributions: Real vs. Synthetic (MedGAN) — Part {part}",
            fontsize=14, fontweight="bold"
        )
        axes = axes.flatten()
        for k, idx in enumerate(part_idxs):
            axes[k].hist(X_real[:, idx], bins=30, alpha=0.6,
                         color="#457b9d", label="Real", density=True)
            axes[k].hist(X_syn[:, idx],  bins=30, alpha=0.6,
                         color="#e63946", label="Synthetic", density=True)
            axes[k].set_title(feature_names[idx], fontsize=8)
            axes[k].set_xlabel("Value"); axes[k].set_ylabel("Density")
            axes[k].legend(fontsize=7)
        for j in range(len(part_idxs), 2):
            axes[j].set_visible(False)
        plt.tight_layout()
        out = f"{base_path}_part{part}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {out}")
        plt.close()


def plot_reconstruction(encoder, decoder, X_real, feature_names, n=6,
                        save_path="medgan_reconstruction.png"):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(X_real[:500]).to(DEVICE)
        recon  = decoder(encoder(tensor)).cpu().numpy()
    num_idx = [i for i, name in enumerate(feature_names)
               if pd.api.types.is_float_dtype(X_real[:, i])][:n]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Autoencoder Reconstruction Quality\n(Original vs. Reconstructed)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()
    for k, idx in enumerate(num_idx):
        axes[k].hist(X_real[:500, idx], bins=25, alpha=0.6,
                     color="#457b9d", label="Original", density=True)
        axes[k].hist(recon[:, idx], bins=25, alpha=0.6,
                     color="#2a9d8f", label="Reconstructed", density=True)
        axes[k].set_title(feature_names[idx], fontsize=8)
        axes[k].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_pca(X_real, X_syn, save_path="medgan_pca.png"):
    n = min(500, len(X_real), len(X_syn))
    pca = PCA(n_components=2)
    pca.fit(np.vstack([X_real[:n], X_syn[:n]]))
    r2 = pca.transform(X_real[:n]); s2 = pca.transform(X_syn[:n])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r2[:, 0], r2[:, 1], alpha=0.4, s=15, color="#457b9d", label="Real")
    ax.scatter(s2[:, 0], s2[:, 1], alpha=0.4, s=15, color="#e63946", label="Synthetic")
    ax.set_title("PCA: Real vs. Synthetic (MedGAN)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}"); plt.close()


def plot_latent_space(encoder, X_real, X_syn_latent, save_path="medgan_latent_pca.png"):
    encoder.eval()
    with torch.no_grad():
        real_latent = encoder(torch.FloatTensor(X_real[:500]).to(DEVICE)).cpu().numpy()
    n = min(500, len(real_latent), len(X_syn_latent))
    pca = PCA(n_components=2)
    pca.fit(np.vstack([real_latent[:n], X_syn_latent[:n]]))
    r2 = pca.transform(real_latent[:n]); s2 = pca.transform(X_syn_latent[:n])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r2[:, 0], r2[:, 1], alpha=0.4, s=15, color="#457b9d", label="Real (encoded)")
    ax.scatter(s2[:, 0], s2[:, 1], alpha=0.4, s=15, color="#e63946", label="Fake (generated)")
    ax.set_title("Latent Space PCA (MedGAN)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}"); plt.close()


def plot_correlation_heatmaps(real_corr, syn_corr, diff,
                               save_path="medgan_correlations.png"):
    real_corr = real_corr.fillna(0); syn_corr = syn_corr.fillna(0); diff = diff.fillna(0)
    n = len(real_corr); figsize = max(12, n // 4); do_annot = n <= 15
    fig, axes = plt.subplots(1, 3, figsize=(figsize * 1.2, figsize))
    fig.suptitle("Correlation Matrix Comparison (MedGAN)", fontsize=14, fontweight="bold")
    for ax, data, title, cmap in zip(
        axes, [real_corr, syn_corr, diff],
        ["Real Data", "Synthetic Data", "Absolute Difference"],
        ["coolwarm", "coolwarm", "Reds"]
    ):
        sns.heatmap(data, ax=ax, cmap=cmap, square=True, linewidths=0.1,
                    center=0 if cmap != "Reds" else None, cbar_kws={"shrink": 0.8},
                    annot=do_annot, fmt=".1f", annot_kws={"size": 7})
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}"); plt.close()


def plot_wasserstein_bar(df_num_stats, save_path="medgan_wasserstein.png"):
    df_s = df_num_stats.sort_values("wasserstein", ascending=False).head(20)
    colors = ["#e63946" if v > 0.5 else "#457b9d" for v in df_s["wasserstein"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(df_s["feature"], df_s["wasserstein"], color=colors)
    ax.axvline(0.5, linestyle="--", color="gray", alpha=0.6, label="Threshold 0.5")
    ax.set_title("Per-feature Wasserstein Distance (MedGAN)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wasserstein Distance (lower = better)")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}"); plt.close()


def plot_tstr_bar(tstr_results, save_path="medgan_tstr.png"):
    if not tstr_results: return
    labels = ["AUC", "Accuracy", "F1"]
    trtr_v = [tstr_results["TRTR"].get(m, 0) for m in labels]
    tstr_v = [tstr_results["TSTR"].get(m, 0) for m in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - 0.2, trtr_v, 0.35, label="TRTR (real)",      color="#457b9d")
    b2 = ax.bar(x + 0.2, tstr_v, 0.35, label="TSTR (synthetic)", color="#e63946")
    ax.set_ylim(0, 1.1); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("TSTR vs. TRTR — Utility (MedGAN)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score"); ax.legend()
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}"); plt.close()


# ═══════════════════════════════════════════════════════════
# 7.  Main pipeline
# ═══════════════════════════════════════════════════════════

"""
Spúšťa celý postup pre každý dataset, delí dáta na train/test, trénuje model, generuje výstupy a ukladá výsledky
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 60)
    print("  MedGAN Testing Pipeline v3 — Medical Synthetic Data")
    print("=" * 60)

    os.chdir(SCRIPT_DIR)
    os.makedirs("MedGAN", exist_ok=True)
    print(f"[Info] Working directory: {os.getcwd()}")

    for data_path in DATA_PATHS:
        data_tag = _sanitize_name(data_path)
        print(f"\n\n=== Dataset: {os.path.basename(data_path)} ({data_tag}) ===")

        df = load_data(data_path, SHEET_NAME, DROP_COLS, DROP_SUFFIXES, CATEGORICAL_COLS)
        X, feature_names, scaler, encoders, pre_scale = encode_and_scale(df, CATEGORICAL_COLS)
        data_dim = X.shape[1]

        if TARGET_COL in feature_names:
            target_idx = feature_names.index(TARGET_COL)
            y = pre_scale[:, target_idx].astype(int)
            valid = ~np.isnan(y.astype(float))
            if not valid.all():
                X = X[valid]; y = y[valid]; pre_scale = pre_scale[valid]
        else:
            y = None

        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            stratify_y = y if counts.min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y)
        else:
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            y_train, y_test = None, None

        print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")

        encoder       = Encoder(data_dim, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
        decoder       = Decoder(LATENT_DIM, data_dim, HIDDEN_DIM).to(DEVICE)
        generator     = Generator(LATENT_DIM, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
        discriminator = Discriminator(LATENT_DIM, HIDDEN_DIM).to(DEVICE)

        ae_losses   = pretrain_autoencoder(encoder, decoder, X_train, AE_EPOCHS, BATCH_SIZE, LR_AE)
        gan_history = train_medgan(encoder, decoder, generator, discriminator,
                                   X_train, GAN_EPOCHS, BATCH_SIZE, LR_GEN, LR_DIS,
                                   LATENT_DIM, N_CRITIC, NOISE_STD)

        n_gen = N_SYNTHETIC or len(X_train)
        synthetic = generate_synthetic(generator, decoder, n_gen, LATENT_DIM, scaler)

        generator.eval()
        with torch.no_grad():
            noise = torch.randn(min(500, n_gen), LATENT_DIM, device=DEVICE)
            fake_latent = generator(noise).cpu().numpy()

        df_syn = pd.DataFrame(synthetic, columns=feature_names)
        out_path = f"MedGAN/synthetic_covid_medgan_{data_tag}.csv"
        df_syn.to_csv(out_path, index=False)
        print(f"[Saved] {out_path}")

        plot_training_curves(ae_losses, gan_history,
                             save_path=f"MedGAN/medgan_training_{data_tag}.png")
        plot_reconstruction(encoder, decoder, X_train, feature_names,
                            save_path=f"MedGAN/medgan_reconstruction_{data_tag}.png")
        plot_distributions(X_train, synthetic, feature_names,
                           save_path=f"MedGAN/medgan_distributions_{data_tag}.png")
        plot_pca(X_train, synthetic,
                 save_path=f"MedGAN/medgan_pca_{data_tag}.png")
        plot_latent_space(encoder, X_train, fake_latent,
                          save_path=f"MedGAN/medgan_latent_pca_{data_tag}.png")

        df_num, df_cat, mean_wd = evaluate_statistical_similarity(
            X_train, synthetic, feature_names, CATEGORICAL_COLS)
        plot_wasserstein_bar(df_num,
                             save_path=f"MedGAN/medgan_wasserstein_{data_tag}.png")

        ae_mse = evaluate_reconstruction(encoder, decoder, X_train)

        real_corr, syn_corr, diff = evaluate_correlation(X_train, synthetic, feature_names)
        plot_correlation_heatmaps(real_corr, syn_corr, diff,
                                  save_path=f"MedGAN/medgan_correlations_{data_tag}.png")

        tstr_results = evaluate_tstr(X_train, y_train, synthetic, X_test, y_test)
        plot_tstr_bar(tstr_results,
                      save_path=f"MedGAN/medgan_tstr_{data_tag}.png")

        privacy = evaluate_privacy(X_train, synthetic)

        print("\n" + "=" * 60)
        print(f"  FINAL SUMMARY — MedGAN ({data_tag})")
        print("=" * 60)
        print(f"  Synthetic samples generated   : {len(synthetic)}")
        print(f"  AE Reconstruction MSE         : {ae_mse:.6f}")
        print(f"  Mean Wasserstein Distance      : {mean_wd:.4f}")
        sim_n = (df_num["similar"] == "✓").sum() if not df_num.empty else 0
        sim_c = (df_cat["similar"] == "✓").sum() if not df_cat.empty else 0
        print(f"  Similar numeric features       : {sim_n}/{len(df_num)}")
        print(f"  Similar categorical features   : {sim_c}/{len(df_cat)}")
        if tstr_results:
            print(f"  TSTR AUC : {tstr_results['TSTR']['AUC']}  (TRTR: {tstr_results['TRTR']['AUC']})")
            print(f"  TSTR F1  : {tstr_results['TSTR']['F1']}  (TRTR: {tstr_results['TRTR']['F1']})")
        print(f"  Privacy ratio                  : {privacy['ratio']:.4f}  (>1.0 = safe)")
        print("=" * 60)


if __name__ == "__main__":
    main()
