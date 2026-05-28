"""

Install:
    pip install ctgan sdv scikit-learn scipy matplotlib seaborn pandas openpyxl

Usage:
    python ctgan_testing.py
"""

import os

import numpy as np
import pandas as pd
# používame non-interactive backend, aby sa zabránilo problémom s vykresľovaním pri spustení v termináli
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 50
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import cdist

# ─────────────────────────────────────────────────────────
# 0.  KONFIGURÁCIA 
# ─────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

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
    "Dôvod hospitalizácie", "SVLZ správy", "HLN Dg.",
]


DROP_SUFFIXES = [" min", " max"]

# CTGAN hyperparameters
EPOCHS          = 300
BATCH_SIZE      = 500
GEN_DIM         = (256, 256)    
DIS_DIM         = (256, 256)    
N_SYNTHETIC     = None          


# ─────────────────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────

def load_data(path, sheet_name, drop_cols, drop_suffixes):
    """Načíta súbor, odstráni nepotrebné stĺpce a vráti očistený DataFrame."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    # Odstráň explicitne uvedené stĺpce
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Odstráň stĺpce končiace na nevhodné prípony (napr. ' min', ' max')
    cols_to_drop = [c for c in df.columns
                    if any(c.endswith(s) for s in drop_suffixes)]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Pokús sa previesť nie-kategorické stĺpce na numerické; nevalidné hodnoty budú NaN
    for col in df.columns:
        if col not in CATEGORICAL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Odstráň stĺpce, ktoré majú viac než 60% chýbajúcich hodnôt
    thresh = len(df) * 0.4
    df = df.dropna(axis=1, thresh=thresh)

    # Doplň zostávajúce chýbajúce hodnoty
    for col in df.columns:
        # pre numerické stĺpce použi medián; inak použi zástupnú hodnotu
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")


    print(f"[Data] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df



def encode_categoricals(df, categorical_cols):
    """Zakóduje kategorické stĺpce číselnými štítkami pre použitie v sklearn modeloch."""
    df_enc = df.copy()
    encoders = {}
    for col in categorical_cols:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    return df_enc, encoders


# ─────────────────────────────────────────────────────────
# 2.  TRÉNOVANIE CTGAN
# ─────────────────────────────────────────────────────────

def train_ctgan(df, categorical_cols, epochs, batch_size, gen_dim, dis_dim):
    """Natrénuje CTGAN na zadanom DataFrame. CTGAN pracuje s miešanými typmi, odovzdajte názvy kategórie."""
    # Odovzdaj len tie kategórie, ktoré sú v DataFrame prítomné
    cats = [c for c in categorical_cols if c in df.columns]

    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=gen_dim,
        discriminator_dim=dis_dim,
        verbose=True,
    )

    print(f"\n[CTGAN] Training for {epochs} epochs...")
    print(f"[CTGAN] Categorical columns ({len(cats)}): {cats}\n")
    model.fit(df, cats)
    return model


# ─────────────────────────────────────────────────────────
# 3.  GENEROVANIE SYNTETICKÝCH DÁT
# ─────────────────────────────────────────────────────────

def generate_synthetic(model, n_samples, df_real):
    """Vygeneruje syntetické vzorky z natrénovaného modelu."""
    n = n_samples or len(df_real)
    synthetic = model.sample(n)
    print(f"[Generated] {len(synthetic)} synthetic samples")
    return synthetic


# ─────────────────────────────────────────────────────────
# 4.  VYHODNOCOVACIE METRIKY
# ─────────────────────────────────────────────────────────

def evaluate_statistical_similarity(df_real, df_syn, categorical_cols):
    """Porovná štatistickú podobnosť: Wasserstein a KS pre numerické; frekvenčné porovnanie pre kategorické."""
    print("\n" + "="*55)
    print("  STATISTICAL SIMILARITY")
    print("="*55)

    numeric_cols = [c for c in df_real.columns
                    if c not in categorical_cols and pd.api.types.is_numeric_dtype(df_real[c])]
    cat_cols     = [c for c in categorical_cols if c in df_real.columns]

    # ── Numerické: Wasserstein + KS ──────────────
    num_results = []
    for col in numeric_cols:
        wd  = wasserstein_distance(df_real[col], df_syn[col])
        ks, p = ks_2samp(df_real[col], df_syn[col])
        num_results.append({
            "feature": col,
            "wasserstein": round(wd, 4),
            "ks_stat": round(ks, 4),
            "ks_p": round(p, 4),
            "similar": "✓" if p > 0.05 else "✗"
        })

    df_num = pd.DataFrame(num_results)
    mean_wd = df_num["wasserstein"].mean()
    similar_pct = (df_num["similar"] == "✓").mean() * 100

    print(f"\n  Numeric columns: {len(numeric_cols)}")
    print(f"  Mean Wasserstein Distance : {mean_wd:.4f}  (lower = better)")
    print(f"  Similar distributions (KS p>0.05): {similar_pct:.1f}%")
    print(df_num.to_string(index=False))

    # ── Kategórie: rozdiel vo frekvenciách hodnôt ─
    print(f"\n  Categorical columns: {len(cat_cols)}")
    cat_results = []
    for col in cat_cols:
        real_freq = df_real[col].astype(str).value_counts(normalize=True)
        syn_freq  = df_syn[col].astype(str).value_counts(normalize=True)
        all_vals  = set(real_freq.index) | set(syn_freq.index)
        tvd = sum(abs(real_freq.get(v, 0) - syn_freq.get(v, 0)) for v in all_vals) / 2
        cat_results.append({"feature": col, "total_variation_dist": round(tvd, 4),
                             "similar": "✓" if tvd < 0.1 else "✗"})

    df_cat = pd.DataFrame(cat_results)
    print(df_cat.to_string(index=False))

    return df_num, df_cat, mean_wd


def evaluate_correlation(df_real, df_syn):
    """Porovná korelačné matice reálnych a syntetických numerických dát."""
    real_corr = df_real.select_dtypes(include=np.number).corr().fillna(0)
    syn_corr  = df_syn.select_dtypes(include=np.number).corr().fillna(0)
    diff = (real_corr - syn_corr).abs()
    mean_diff = diff.values[np.triu_indices_from(diff.values, k=1)].mean()
    print(f"\n[Correlation] Mean Absolute Difference: {mean_diff:.4f}  (lower = better)")
    return real_corr, syn_corr, diff


def evaluate_tstr(df_real, df_syn, target_col, categorical_cols):
    """TSTR: trénuj na syntetických dátach, testuj na reálnych — porovná metriky (AUC/Accuracy/F1) s TRTR."""
    print("\n" + "="*55)
    print("  TSTR — UTILITY EVALUATION")
    print("="*55)

    if target_col not in df_real.columns:
        print(f"  Target column '{target_col}' not found, skipping.")
        return {}

    # Zakóduj pre sklearn
    df_r_enc, _ = encode_categoricals(df_real, categorical_cols)
    df_s_enc, _ = encode_categoricals(df_syn,  categorical_cols)

    X_real = df_r_enc.drop(columns=[target_col])
    y_real = df_r_enc[target_col]
    X_syn  = df_s_enc.drop(columns=[target_col], errors="ignore")
    y_syn  = df_s_enc[target_col] if target_col in df_s_enc.columns else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )

    n_classes = y_real.nunique()
    avg = "binary" if n_classes == 2 else "macro"

    def get_metrics(clf, X_test, y_test):
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        try:
            auc = (roc_auc_score(y_test, y_proba[:, 1])
                   if n_classes == 2
                   else roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        return {
            "AUC":      round(auc, 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1":       round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        }

    # TRTR (baseline na reálnych dátach)
    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_tr, y_tr)
    trtr = get_metrics(clf_trtr, X_te, y_te)

    # TSTR
    if y_syn is not None:
        X_syn_aligned = X_syn[X_real.columns] if set(X_real.columns).issubset(X_syn.columns) else X_syn
        clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_tstr.fit(X_syn_aligned, y_syn)
        tstr = get_metrics(clf_tstr, X_te, y_te)
    else:
        tstr = {"AUC": float("nan"), "Accuracy": float("nan"), "F1": float("nan")}

    print(f"\n  {'Metric':<12} {'TRTR (real)':>14} {'TSTR (synthetic)':>18}")
    print(f"  {'-'*46}")
    for m in ["AUC", "Accuracy", "F1"]:
        print(f"  {m:<12} {trtr[m]:>14} {tstr[m]:>18}")

    return {"TRTR": trtr, "TSTR": tstr}


def evaluate_privacy(df_real, df_syn, k=5):
    """Nearest-Neighbor Distance — kontrola, či model nememoruje reálne záznamy."""
    print("\n" + "="*55)
    print("  PRIVACY — NEAREST NEIGHBOR DISTANCE")
    print("="*55)

    num_cols = df_real.select_dtypes(include=np.number).columns.tolist()
    n = min(500, len(df_real), len(df_syn))

    real_s = df_real[num_cols].dropna().values
    syn_s  = df_syn[num_cols].dropna().values
    real_s = real_s[np.random.choice(len(real_s), min(n, len(real_s)), replace=False)]
    syn_s  = syn_s [np.random.choice(len(syn_s),  min(n, len(syn_s)),  replace=False)]

    d_sr = cdist(syn_s,  real_s, metric="euclidean")
    d_rr = cdist(real_s, real_s, metric="euclidean")
    np.fill_diagonal(d_rr, np.inf)

    nnd_sr = np.sort(d_sr, axis=1)[:, :k].mean(axis=1).mean()
    nnd_rr = np.sort(d_rr, axis=1)[:, :k].mean(axis=1).mean()
    ratio  = nnd_sr / nnd_rr

    print(f"\n  NND synthetic→real : {nnd_sr:.4f}")
    print(f"  NND real->real      : {nnd_rr:.4f}")
    print(f"  Ratio (>1.0 = safe): {ratio:.4f}")
    verdict = "Safe — no memorization detected" if ratio >= 1.0 else "Possible memorization"
    print(f"  Privacy verdict    : {verdict}")
    return {"nnd_syn_real": nnd_sr, "nnd_real_real": nnd_rr, "ratio": ratio}


# ─────────────────────────────────────────────────────────
# 5.  GRAFY
# ─────────────────────────────────────────────────────────

def plot_loss_curve(model, save_path="ctgan_loss_curve.png"):
    """Vykreslí priebeh strát generátora a diskriminátora, ak sú dostupné."""
    try:
        loss_values = model.loss_values
        if loss_values is None or len(loss_values) == 0:
            print("[Plot] No loss history available from CTGAN.")
            return
        df_loss = pd.DataFrame(loss_values)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("CTGAN Training Loss", fontsize=14, fontweight="bold")
        if "Generator Loss" in df_loss.columns:
            axes[0].plot(df_loss["Generator Loss"], color="#457b9d")
            axes[0].set_title("Generator Loss"); axes[0].set_xlabel("Epoch")
        if "Discriminator Loss" in df_loss.columns:
            axes[1].plot(df_loss["Discriminator Loss"], color="#e63946")
            axes[1].set_title("Discriminator Loss"); axes[1].set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")
        plt.close()
    except Exception as e:
        print(f"[Plot] Loss curve not available: {e}")


def plot_distributions(df_real, df_syn, n_features=8,
                       save_path="ctgan_distributions.png"):
    """Prekryje histogramy najdôležitejších numerických príznakov (reálne vs. syntetické)."""
    num_cols = df_real.select_dtypes(include=np.number).columns.tolist()[:n_features]
    n = len(num_cols)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.suptitle("Feature Distributions: Real vs. Synthetic (CTGAN)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(df_real[col].dropna(), bins=30, alpha=0.6,
                     color="#457b9d", label="Real", density=True)
        axes[i].hist(df_syn[col].dropna(),  bins=30, alpha=0.6,
                     color="#e63946", label="Synthetic", density=True)
        axes[i].set_title(col, fontsize=8)
        axes[i].legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_categorical_distributions(df_real, df_syn, categorical_cols,
                                   save_path="ctgan_categoricals.png"):
    """Stĺpcové grafy porovnávajúce frekvencie kategórií (reálne vs. syntetické)."""
    cats = [c for c in categorical_cols if c in df_real.columns][:6]
    if not cats:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Categorical Distributions: Real vs. Synthetic (CTGAN)",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(cats):
        real_freq = df_real[col].astype(str).value_counts(normalize=True).sort_index()
        syn_freq  = df_syn[col].astype(str).value_counts(normalize=True).sort_index()
        all_cats  = sorted(set(real_freq.index) | set(syn_freq.index))
        x = np.arange(len(all_cats))
        axes[i].bar(x - 0.2, [real_freq.get(c, 0) for c in all_cats], 0.35,
                    label="Real", color="#457b9d", alpha=0.8)
        axes[i].bar(x + 0.2, [syn_freq.get(c, 0) for c in all_cats], 0.35,
                    label="Synthetic", color="#e63946", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(all_cats, rotation=30, fontsize=7, ha="right")
        axes[i].legend(fontsize=7)

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_correlation_heatmaps(real_corr, syn_corr, diff,
                               save_path="ctgan_correlations.png"):
    """Heatmapy korelácií reálnych/syntetických dát a ich absolútny rozdiel."""
    # Doplň NaN hodnoty (konštantné stĺpce) nulami pre vizualizáciu
    real_corr = real_corr.fillna(0)
    syn_corr = syn_corr.fillna(0)
    diff = diff.fillna(0)
    
    # Urči veľkosť obrázku podľa rozmerov matice
    n = len(real_corr)
    figsize = max(12, n // 4)  # Minimálne 12, zväčši s rastúcou veľkosťou matice
    
    fig, axes = plt.subplots(1, 3, figsize=(figsize * 1.2, figsize))
    fig.suptitle("Correlation Matrix Comparison (CTGAN)", fontsize=14, fontweight="bold")
    
    # Anotuj hodnoty iba pre malé matice
    do_annot = n <= 15
    
    for ax, data, title, cmap in zip(
        axes,
        [real_corr, syn_corr, diff],
        ["Real Data", "Synthetic Data", "Absolute Difference"],
        ["coolwarm", "coolwarm", "Reds"]
    ):
        sns.heatmap(data, ax=ax, cmap=cmap, square=True, linewidths=0.1,
                    center=0 if cmap != "Reds" else None,
                    cbar_kws={"shrink": 0.8},
                    annot=do_annot, fmt=".1f", annot_kws={"size": 7})
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_pca(df_real, df_syn, save_path="ctgan_pca.png"):
    """PCA scatter — porovnanie reálnych a syntetických dát vo 2D."""
    num_cols = df_real.select_dtypes(include=np.number).columns.tolist()
    n = min(500, len(df_real), len(df_syn))
    real_s = df_real[num_cols].dropna().values[:n]
    syn_s  = df_syn[num_cols].dropna().values[:n]

    pca = PCA(n_components=2)
    pca.fit(np.vstack([real_s, syn_s]))
    r2 = pca.transform(real_s)
    s2 = pca.transform(syn_s)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r2[:, 0], r2[:, 1], alpha=0.4, s=15, color="#457b9d", label="Real")
    ax.scatter(s2[:, 0], s2[:, 1], alpha=0.4, s=15, color="#e63946", label="Synthetic")
    ax.set_title("PCA: Real vs. Synthetic (CTGAN)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_tstr_bar(tstr_results, save_path="ctgan_tstr.png"):
    """Vykreslí stĺpcový graf porovnania metrík TRTR vs TSTR."""
    if not tstr_results:
        return
    labels = ["AUC", "Accuracy", "F1"]
    trtr_vals = [tstr_results["TRTR"].get(m, 0) for m in labels]
    tstr_vals = [tstr_results["TSTR"].get(m, 0) for m in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - 0.2, trtr_vals, 0.35, label="TRTR (real)",      color="#457b9d")
    b2 = ax.bar(x + 0.2, tstr_vals, 0.35, label="TSTR (synthetic)", color="#e63946")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("TSTR vs. TRTR — Utility (CTGAN)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score"); ax.legend()
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


def plot_wasserstein_bar(df_num_stats, save_path="ctgan_wasserstein.png"):
    """Stĺpcový graf Wassersteinových vzdialeností pre jednotlivé príznaky."""
    df_sorted = df_num_stats.sort_values("wasserstein", ascending=False).head(20)
    colors = ["#e63946" if v > 0.5 else "#457b9d" for v in df_sorted["wasserstein"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(df_sorted["feature"], df_sorted["wasserstein"], color=colors)
    ax.axvline(x=0.5, linestyle="--", color="gray", alpha=0.6, label="Threshold 0.5")
    ax.set_title("Per-feature Wasserstein Distance (CTGAN)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wasserstein Distance (lower = better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CTGAN Testing Pipeline — Medical Synthetic Data")
    print("=" * 60)

    for data_path in DATA_PATHS:
        data_tag = _sanitize_name(data_path)
        print(f"\n\n=== Dataset: {os.path.basename(data_path)} ({data_tag}) ===")

        # ── Load & clean data ──────────────────────
        df = load_data(data_path, SHEET_NAME, DROP_COLS, DROP_SUFFIXES)

        # ── Train/test split (hold out 20% for TSTR) ─
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        print(f"[Split] Train: {len(df_train)} | Test (held out): {len(df_test)}")

        # ── Train CTGAN ────────────────────────────
        model = train_ctgan(
            df_train, CATEGORICAL_COLS,
            EPOCHS, BATCH_SIZE, GEN_DIM, DIS_DIM
        )

        # ── Generate synthetic data ────────────────
        n_gen = N_SYNTHETIC or len(df_train)
        df_syn = generate_synthetic(model, n_gen, df_train)

        # ── Save synthetic data ────────────────────
        os.makedirs("CTGAN", exist_ok=True)
        out_csv = os.path.join("CTGAN", f"synthetic_covid_ctgan_{data_tag}.csv")
        df_syn.to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")

        # ── Plots ──────────────────────────────────
        plot_loss_curve(model, save_path=os.path.join("CTGAN", f"ctgan_loss_curve_{data_tag}.png"))
        plot_distributions(df_train, df_syn, save_path=os.path.join("CTGAN", f"ctgan_distributions_{data_tag}.png"))
        plot_categorical_distributions(df_train, df_syn, CATEGORICAL_COLS, save_path=os.path.join("CTGAN", f"ctgan_categoricals_{data_tag}.png"))
        plot_pca(df_train, df_syn, save_path=os.path.join("CTGAN", f"ctgan_pca_{data_tag}.png"))

        # ── Statistical similarity ─────────────────
        df_num_stats, df_cat_stats, mean_wd = evaluate_statistical_similarity(
            df_train, df_syn, CATEGORICAL_COLS
        )
        plot_wasserstein_bar(df_num_stats, save_path=os.path.join("CTGAN", f"ctgan_wasserstein_{data_tag}.png"))

        # ── Correlation comparison ─────────────────
        real_corr, syn_corr, diff = evaluate_correlation(df_train, df_syn)
        plot_correlation_heatmaps(real_corr, syn_corr, diff, save_path=os.path.join("CTGAN", f"ctgan_correlations_{data_tag}.png"))

        # ── TSTR utility evaluation ───────────────
        tstr_results = evaluate_tstr(df_train, df_syn, TARGET_COL, CATEGORICAL_COLS)
        plot_tstr_bar(tstr_results, save_path=os.path.join("CTGAN", f"ctgan_tstr_{data_tag}.png"))

        # ── Privacy evaluation ─────────────────────
        privacy = evaluate_privacy(df_train, df_syn)

        # ── Final summary ──────────────────────────
        print("\n" + "=" * 60)
        print(f"  FINAL SUMMARY — CTGAN ({data_tag})")
        print("=" * 60)
        print(f"  Synthetic samples generated : {len(df_syn)}")
        print(f"  Mean Wasserstein Distance   : {mean_wd:.4f}")
        similar_n = (df_num_stats["similar"] == "✓").sum()
        print(f"  Similar numeric features    : {similar_n}/{len(df_num_stats)}")
        similar_c = (df_cat_stats["similar"] == "✓").sum()
        print(f"  Similar categorical features: {similar_c}/{len(df_cat_stats)}")
        if tstr_results:
            print(f"  TSTR AUC  : {tstr_results['TSTR']['AUC']}  "
                  f"(TRTR: {tstr_results['TRTR']['AUC']})")
            print(f"  TSTR F1   : {tstr_results['TSTR']['F1']}  "
                  f"(TRTR: {tstr_results['TRTR']['F1']})")
        print(f"  Privacy ratio               : {privacy['ratio']:.4f}  (>1.0 = safe)")
        print("=" * 60)


if __name__ == "__main__":
    main()

