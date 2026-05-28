"""
Predpoklady:
  - Natrénované checkpointy CovidGAN v ../CovidGAN/
  - Rovnaký priečinok datasets: ../datasets/

Použitie:
    cd DigitalTwin
    python digital_twin.py
"""

import os
import glob
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              classification_report)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ─────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ─────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR     = os.path.join(BASE_DIR, "datasets")
COVIDGAN_DIR = os.path.join(BASE_DIR, "CovidGAN")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")

for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET_COL = "Závažnosť priebehu ochorenia"
CATEGORICAL_COLS = [
    "Pohlavie", "Vakcinácia", "Typ vakcíny", "Prekonal COVID-19",
    "Hypertenzia", "Diabetes mellitus", "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia", "Renálne ochorenia",
    "Imunosupresia", "Onkologické ochorenia", TARGET_COL,
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

NOISE_DIM  = 100
HIDDEN_DIM = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────
# 1.  ARCHITEKTÚRA COVIDGAN
# ─────────────────────────────────────────────────────────

class CovidGenerator(nn.Module):
    def __init__(self, noise_dim, n_classes, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = noise_dim + n_classes
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
            nn.Sigmoid(),
        )

    def forward(self, noise, labels_onehot):
        return self.net(torch.cat([noise, labels_onehot], dim=1))


def labels_to_onehot(labels, n_classes, device):
    onehot = torch.zeros(len(labels), n_classes, device=device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


# ─────────────────────────────────────────────────────────
# 2.  NAČÍTANIE DÁT
# ─────────────────────────────────────────────────────────

def _sanitize_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")


def load_raw(path):
    """
    Načíta a zakóduje dataset, ale NEŠKÁLUJE.
    Škálovanie sa odkladá až po rozdelení train/test, takže scaler
    nikdy nevidí testovacie dáta (oprava úniku dát).
    """
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path, sheet_name=0)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    bad = [c for c in df.columns if any(c.endswith(s) for s in DROP_SUFFIXES)]
    df  = df.drop(columns=bad, errors="ignore")
    df  = df.dropna(axis=1, thresh=int(len(df) * 0.4))

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    df_enc    = df.copy()
    target_le = LabelEncoder()

    if TARGET_COL in df_enc.columns:
        labels = target_le.fit_transform(df_enc[TARGET_COL].astype(str))
    else:
        labels = np.zeros(len(df_enc), dtype=int)

    for col in df_enc.columns:
        if col != TARGET_COL and not pd.api.types.is_numeric_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    feature_names = df_enc.columns.tolist()
    X_raw = df_enc.values.astype(np.float32)

    n_classes = len(np.unique(labels))
    print(f"  [Data] {X_raw.shape[0]} patients | {X_raw.shape[1]} features | "
          f"{n_classes} classes")
    print(f"  [Data] Class distribution: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    return X_raw, labels, feature_names, target_le


# ─────────────────────────────────────────────────────────
# 3.  NAČÍTANIE CHECKPOINTU
# ─────────────────────────────────────────────────────────

"""
Načíta natrénovaný model z disku a obnoví váhy generátora.
"""

def load_generator(checkpoint_path, data_dim, n_classes):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"  Run covidgan_testing.py first."
        )
    G = CovidGenerator(NOISE_DIM, n_classes, data_dim, HIDDEN_DIM).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    print(f"  [Checkpoint] Loaded epoch {ckpt.get('epoch', '?')} "
          f"from {os.path.basename(checkpoint_path)}")
    return G


# ─────────────────────────────────────────────────────────
# 4.  GENEROVANIE
#     Výstup generátora je v rozsahu [0,1] — rovnaký priestor ako MinMaxScaler
# ─────────────────────────────────────────────────────────

"""
Vytvára syntetické vzorky v škálovanom priestore a prevádza ich späť na pôvodné klinické hodnoty.
"""

def generate_scaled(G, class_label, n_samples, n_classes):
    """Vygeneruje vzorky v škálovanom priestore [0,1] — pripravené pre klasifikátor."""
    with torch.no_grad():
        labels = torch.full((n_samples,), class_label, dtype=torch.long, device=DEVICE)
        noise  = torch.randn(n_samples, NOISE_DIM, device=DEVICE)
        onehot = labels_to_onehot(labels, n_classes, DEVICE)
        return G(noise, onehot).cpu().numpy()


def generate_original_scale(G, class_label, n_samples, n_classes, scaler):
    """Vygeneruje záznamy a prevedie ich späť na klinické hodnoty."""
    scaled = generate_scaled(G, class_label, n_samples, n_classes)
    scaled = np.clip(scaled, 0.0, 1.0)   # prevent impossible clinical values
    return scaler.inverse_transform(scaled)


# ─────────────────────────────────────────────────────────
# 5.  EXPERIMENT A — KVALITATÍVNA TABUĽKA DVOJČA
# ─────────────────────────────────────────────────────────

"""
Porovnáva reálneho pacienta so syntetickými prototypmi pre každú triedu závažnosti.
"""

def experiment_a_qualitative(G, X_test_scaled, y_test, feature_names,
                               scaler, target_le, n_classes, data_tag):
    """
    Vyberie prvého reálneho pacienta z testovacej sady.
    Vygeneruje jeden syntetický prototyp pre každú závažnosť.
    Zobrazí výsledky v klinických hodnotách (inverzne transformované).
    """
    # Add this near the top of experiment_a_qualitative
    EXCLUDE_FROM_DISPLAY = [
        "SatO2 %", "Dátum príjmu", "Dátum prepustenia",
        "Vek", "Pohlavie", "Vakcinácia"
    ]

    numeric_cols = [
        f for f in feature_names
        if f not in CATEGORICAL_COLS
        and f not in EXCLUDE_FROM_DISPLAY   
    ]

    print("\n" + "─" * 60)
    print("  EXPERIMENT A — Qualitative Digital Twin")
    print("─" * 60)

    real_scaled = X_test_scaled[0]
    real_record = scaler.inverse_transform(real_scaled.reshape(1, -1))[0]
    real_label  = target_le.classes_[y_test[0]]
    print(f"  Real patient severity class: {real_label}")

    rows = {"Reálny pacient": real_record}
    for cls in range(n_classes):
        label = target_le.classes_[cls]
        twin  = generate_original_scale(G, cls, 1, n_classes, scaler)[0]
        rows[f"Digitálne dvojča — {label}"] = twin
        print(f"  Generated twin for class: {label}")

    df_twin = pd.DataFrame(rows, index=feature_names).T
    df_twin.index.name = "Záznam"

    numeric_cols = [f for f in feature_names if f not in CATEGORICAL_COLS]
    top_cols = (df_twin[numeric_cols].var().nlargest(10).index.tolist()
                if len(numeric_cols) > 10 else numeric_cols)
    df_display = df_twin[top_cols].round(2)

    df_twin.round(2).to_csv(os.path.join(TABLES_DIR, f"twin_full_{data_tag}.csv"))
    df_display.to_csv(os.path.join(TABLES_DIR, f"twin_display_{data_tag}.csv"))

    print(f"\n  Top clinical features:\n")
    print(df_display.to_string())

    # Teplotná mapa
    fig, ax = plt.subplots(figsize=(max(12, len(top_cols) * 1.2), 4))
    fig.suptitle(
        f"Digital Twin — Real Patient vs. Synthetic Counterparts\n({data_tag})",
        fontsize=13, fontweight="bold"
    )
    df_norm = ((df_display - df_display.min()) /
               (df_display.max() - df_display.min() + 1e-8))
    sns.heatmap(df_norm, ax=ax, cmap="YlOrRd", annot=df_display.values,
                fmt=".1f", linewidths=0.5, cbar=False, annot_kws={"size": 8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.add_patch(plt.Rectangle((0, 0), len(top_cols), 1,
                                fill=False, edgecolor="blue", lw=3))
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"twin_heatmap_{data_tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {fig_path}")

    return df_twin


# ─────────────────────────────────────────────────────────
# 6.  EXPERIMENT B — KVANTITATÍVNE DOPLNENIE
# ─────────────────────────────────────────────────────────

"""
Porovnáva TRTR, TSTR a augmentovanú trénovaciu množinu, zároveň opravuje únik dát a vyvažuje triedy.
"""

def experiment_b_augmentation(G, X_raw, y, n_classes, target_le, data_tag):
    print("\n" + "─" * 60)
    print("  EXPERIMENT B — TRTR / TSTR / Augmented")
    print("─" * 60)

    avg = "macro" if n_classes > 2 else "binary"

    # ── OPRAVA 1: najprv rozdelíme surové dáta, potom scaler učíme iba na trénovacej množine ──
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)   # fit + transform on train
    X_test  = scaler.transform(X_test_raw)         # transform only on test

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"  Class counts (train): {class_counts}")

    def clf_metrics(clf, X_te, y_te):
        yp  = clf.predict(X_te)
        ypr = clf.predict_proba(X_te)
        try:
            auc = (roc_auc_score(y_te, ypr[:, 1]) if n_classes == 2
                   else roc_auc_score(y_te, ypr, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        return {
            "AUC":      round(auc, 4),
            "F1":       round(f1_score(y_te, yp, average=avg, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_te, yp), 4),
        }

    # ── Základ TRTR ────────────────────────────────────
    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_train, y_train)
    trtr = clf_metrics(clf_trtr, X_test, y_test)
    print(f"\n  TRTR (real only)  — AUC: {trtr['AUC']:.4f} | F1: {trtr['F1']:.4f}")

    # ── OPRAVA 3: TSTR s pomerným vzorkovaním tried ─────
    # Vzorkovanie tried pomerne podľa skutočnej trénovacej distribúcie
    real_weights = np.array([class_counts.get(c, 0) for c in range(n_classes)],
                             dtype=float)
    real_weights /= real_weights.sum()

    n_gen = len(X_train)
    labels_prop = np.random.choice(n_classes, size=n_gen, p=real_weights)

    X_syn_list, y_syn_list = [], []
    for cls in range(n_classes):
        idx = np.where(labels_prop == cls)[0]
        if len(idx) == 0:
            continue
        X_syn_list.append(generate_scaled(G, cls, len(idx), n_classes))
        y_syn_list.append(np.full(len(idx), cls))

    X_syn = np.vstack(X_syn_list)
    y_syn = np.concatenate(y_syn_list)

    syn_dist = dict(zip(*np.unique(y_syn, return_counts=True)))
    print(f"\n  TSTR synthetic distribution (proportional): {syn_dist}")

    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(X_syn, y_syn)
    tstr = clf_metrics(clf_tstr, X_test, y_test)
    print(f"  TSTR (synth only) — AUC: {tstr['AUC']:.4f} | F1: {tstr['F1']:.4f}")

    # ── OPRAVA 2: úplné vyváženie tried pre augmentáciu ──────
    majority_cls   = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_cls]
    minority_cls   = [c for c in class_counts if c != majority_cls]

    print(f"\n  Augmenting to full balance — majority count: {majority_count}")

    aug_X_list = [X_train]
    aug_y_list = [y_train]
    for cls in minority_cls:
        n_needed = majority_count - class_counts.get(cls, 0)
        if n_needed <= 0:
            continue
        synth = generate_scaled(G, cls, n_needed, n_classes)
        aug_X_list.append(synth)
        aug_y_list.append(np.full(n_needed, cls))
        print(f"  Class {cls} ({target_le.classes_[cls]}): "
              f"added {n_needed} synthetic records "
              f"({class_counts.get(cls, 0)} → {majority_count})")

    X_aug = np.vstack(aug_X_list)
    y_aug = np.concatenate(aug_y_list)
    aug_dist = dict(zip(*np.unique(y_aug, return_counts=True)))
    print(f"  Augmented distribution: {aug_dist}")
    print(f"  Total: {len(X_train)} real + "
          f"{len(X_aug) - len(X_train)} synthetic = {len(X_aug)}")

    clf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_aug.fit(X_aug, y_aug)
    augm = clf_metrics(clf_aug, X_test, y_test)
    print(f"\n  Augmented         — AUC: {augm['AUC']:.4f} | F1: {augm['F1']:.4f}")

    print(f"\n  Δ AUC (aug vs TRTR): {augm['AUC'] - trtr['AUC']:+.4f}")
    print(f"  Δ F1  (aug vs TRTR): {augm['F1']  - trtr['F1']:+.4f}")

    # ── OPRAVA 4: rozpis F1 pre jednotlivé triedy ────────────────────
    print("\n" + "─" * 60)
    print("  PER-CLASS BREAKDOWN")
    print("─" * 60)

    class_names = list(target_le.classes_)

    print("\n  TRTR (real only):")
    trtr_report = classification_report(
        y_test, clf_trtr.predict(X_test),
        target_names=class_names, output_dict=True, zero_division=0
    )
    print(classification_report(
        y_test, clf_trtr.predict(X_test),
        target_names=class_names, zero_division=0
    ))

    print("\n  Augmented (real + synthetic minority):")
    aug_report = classification_report(
        y_test, clf_aug.predict(X_test),
        target_names=class_names, output_dict=True, zero_division=0
    )
    print(classification_report(
        y_test, clf_aug.predict(X_test),
        target_names=class_names, zero_division=0
    ))

    # Vytvorenie tabuľky porovnania pre jednotlivé triedy
    per_class_rows = []
    for cls_name in class_names:
        if cls_name in trtr_report and cls_name in aug_report:
            per_class_rows.append({
                "Class":          cls_name,
                "TRTR F1":        round(trtr_report[cls_name]["f1-score"], 4),
                "Augmented F1":   round(aug_report[cls_name]["f1-score"],  4),
                "Δ F1":           round(aug_report[cls_name]["f1-score"] -
                                        trtr_report[cls_name]["f1-score"], 4),
                "TRTR Recall":    round(trtr_report[cls_name]["recall"], 4),
                "Augmented Recall": round(aug_report[cls_name]["recall"], 4),
                "Δ Recall":       round(aug_report[cls_name]["recall"] -
                                        trtr_report[cls_name]["recall"], 4),
            })

    df_perclass = pd.DataFrame(per_class_rows)
    perclass_path = os.path.join(TABLES_DIR, f"perclass_results_{data_tag}.csv")
    df_perclass.to_csv(perclass_path, index=False)
    print(f"  [Saved] {perclass_path}")
    print(df_perclass.to_string(index=False))

    # ── Uloženie súhrnných výsledkov ──────────────────────────────
    df_results = pd.DataFrame({
        "Setup":    ["TRTR (real only)", "TSTR (synthetic only)",
                     "Augmented (real + balanced minority)"],
        "AUC":      [trtr["AUC"],  tstr["AUC"],  augm["AUC"]],
        "F1":       [trtr["F1"],   tstr["F1"],   augm["F1"]],
        "Accuracy": [trtr["Accuracy"], tstr["Accuracy"], augm["Accuracy"]],
    })
    path = os.path.join(TABLES_DIR, f"augmentation_results_{data_tag}.csv")
    df_results.to_csv(path, index=False)
    print(f"\n  [Saved] {path}")

    # ── Grafy ─────────────────────────────────────────────
    _plot_augmentation(trtr, tstr, augm, data_tag)
    _plot_perclass(df_perclass, data_tag)

    return {
        "TRTR": trtr, "TSTR": tstr, "Augmented": augm,
        "per_class": df_perclass, "scaler": scaler
    }


def _plot_augmentation(trtr, tstr, augm, data_tag):
    metrics = ["AUC", "F1", "Accuracy"]
    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"Digital Twin — TRTR / TSTR / Augmented\n({data_tag})",
        fontsize=13, fontweight="bold"
    )
    b1 = ax.bar(x - w, [trtr[m] for m in metrics], w,
                label="TRTR (real only)",            color="#457b9d", alpha=0.9)
    b2 = ax.bar(x,     [tstr[m] for m in metrics], w,
                label="TSTR (synthetic only)",       color="#e63946", alpha=0.9)
    b3 = ax.bar(x + w, [augm[m] for m in metrics], w,
                label="Augmented (real + minority)", color="#2a9d8f", alpha=0.9)

    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)

    for bar in list(b1) + list(b2) + list(b3):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"augmentation_results_{data_tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {fig_path}")


def _plot_perclass(df_perclass, data_tag):
    """Stĺpcový graf ukazujúci F1 pre jednotlivé triedy: TRTR vs Augmented."""
    classes = df_perclass["Class"].tolist()
    x = np.arange(len(classes))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Per-Class Improvement — TRTR vs. Augmented\n({data_tag})",
        fontsize=13, fontweight="bold"
    )

    for ax, metric, col_trtr, col_aug in zip(
        axes,
        ["F1", "Recall"],
        ["TRTR F1", "TRTR Recall"],
        ["Augmented F1", "Augmented Recall"]
    ):
        b1 = ax.bar(x - w/2, df_perclass[col_trtr],  w,
                    label="TRTR",      color="#457b9d", alpha=0.9)
        b2 = ax.bar(x + w/2, df_perclass[col_aug],   w,
                    label="Augmented", color="#2a9d8f", alpha=0.9)
        ax.set_ylim(0, 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"Per-Class {metric}", fontweight="bold")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)

        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"perclass_results_{data_tag}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {fig_path}")


# ─────────────────────────────────────────────────────────
# 7.  ZHRNUTIE PRE VŠETKY VLNY
# ─────────────────────────────────────────────────────────

def save_summary(all_results):
    rows = []
    for tag, res in all_results.items():
        rows.append({
            "Wave":          tag,
            "TRTR AUC":      res["TRTR"]["AUC"],
            "TSTR AUC":      res["TSTR"]["AUC"],
            "Augmented AUC": res["Augmented"]["AUC"],
            "Δ AUC":         round(res["Augmented"]["AUC"] - res["TRTR"]["AUC"], 4),
            "TRTR F1":       res["TRTR"]["F1"],
            "TSTR F1":       res["TSTR"]["F1"],
            "Augmented F1":  res["Augmented"]["F1"],
            "Δ F1":          round(res["Augmented"]["F1"] - res["TRTR"]["F1"], 4),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, "digital_twin_summary.csv")
    df.to_csv(path, index=False)
    print(f"\n[Summary] Saved: {path}")
    print("\n" + df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Digital Twin Experiment — CovidGAN (Fixed)")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  Data dir    : {DATA_DIR}")
    print(f"  Checkpoints : {COVIDGAN_DIR}")

    data_paths = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.xlsx")) +
        glob.glob(os.path.join(DATA_DIR, "*.csv"))
    )
    if not data_paths:
        print(f"\n[ERROR] No datasets found in {DATA_DIR}")
        return

    all_results = {}

    for data_path in data_paths:
        data_tag = _sanitize_name(data_path)
        print(f"\n\n{'='*60}")
        print(f"  Wave: {os.path.basename(data_path)}")
        print(f"{'='*60}")

            # Načítať surové (neškálované) dáta
        X_raw, y, feature_names, target_le = load_raw(data_path)
        n_classes = len(np.unique(y))
        data_dim  = X_raw.shape[1]

        # Načítať generátor
        ckpt_path = os.path.join(COVIDGAN_DIR,
                                  f"covidgan_checkpoint_{data_tag}.pth")
        try:
            G = load_generator(ckpt_path, data_dim, n_classes)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {e}")
            continue

        # Najprv Experiment B, aby sa získal správne natrénovaný scaler
        results = experiment_b_augmentation(
            G, X_raw, y, n_classes, target_le, data_tag
        )

        # Experiment A — použije scaler z Experimentu B (učený iba na trénovacej množine)
        scaler = results["scaler"]
        _, X_test_raw, _, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_scaled = scaler.transform(X_test_raw)

        experiment_a_qualitative(
            G, X_test_scaled, y_test, feature_names,
            scaler, target_le, n_classes, data_tag
        )

        all_results[data_tag] = results

    if all_results:
        save_summary(all_results)

    print("\n" + "=" * 60)
    print("  Digital Twin Experiment Complete")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
