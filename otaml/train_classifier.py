"""Train the fault classifier and run the feature-set ablation.

    python -m otaml.train_classifier
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from . import config as C
from .classifier import FEATURE_SETS, FaultDiagnoser, add_measurement_noise, build_features
from .surrogate import Surrogate


def main():
    sur = Surrogate.load()
    df = pd.read_csv(C.DATA_DIR / "faults.csv")
    df = df[df["ok"] == 1].reset_index(drop=True)
    y = df["fault"]
    idx_tr, idx_te = train_test_split(df.index, test_size=0.25, stratify=y, random_state=0)

    ablation, best = {}, None
    for kind in FEATURE_SETS:
        X = build_features(df, sur, kind)
        for name, mk in {
            "random_forest": lambda: RandomForestClassifier(400, n_jobs=-1, random_state=0),
            "hist_gb": lambda: HistGradientBoostingClassifier(max_iter=300, random_state=0),
        }.items():
            clf = mk().fit(X.loc[idx_tr], y.loc[idx_tr])
            pred = clf.predict(X.loc[idx_te])
            acc = accuracy_score(y.loc[idx_te], pred)
            f1 = f1_score(y.loc[idx_te], pred, average="macro")
            ablation[f"{kind} | {name}"] = dict(accuracy=acc, macro_f1=f1)
            print(f"{kind:20s} {name:14s} acc={acc:.4f} macroF1={f1:.4f}")
            if kind == "measured+residuals" and (best is None or f1 > best[0]):
                best = (f1, clf, pred)

    f1, clf, pred = best
    print("\nBest model report (measured+residuals):")
    print(classification_report(y.loc[idx_te], pred, digits=3))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y.loc[idx_te], pred, labels=C.FAULT_CLASSES,
                                            xticks_rotation=35, ax=ax, colorbar=False)
    ax.set_title("Fault classifier confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fault_confusion_matrix.png", dpi=140); plt.close(fig)

    abl = pd.DataFrame(ablation).T
    ax = abl["accuracy"].plot(kind="barh", figsize=(8, 4), xlim=(abl["accuracy"].min() - 0.05, 1.0))
    ax.set_title("Ablation: which features help fault diagnosis?")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "fault_ablation.png", dpi=140); plt.close()

    # ---- robustness to measurement noise ----------------------------------
    # (a) model trained on clean data, (b) model trained with noise augmentation (0 %, 1 %, 2 %)
    train_df, test_df = df.loc[idx_tr], df.loc[idx_te]
    aug = pd.concat([add_measurement_noise(train_df, lv, seed=i) for i, lv in enumerate((0.0, 0.01, 0.02))])
    y_aug = pd.concat([y.loc[idx_tr]] * 3)
    clf_aug = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(
        build_features(aug, sur, "measured+residuals"), y_aug)
    robust = {}
    print("\nNoise robustness (test accuracy):")
    for lv in (0.0, 0.01, 0.02, 0.05):
        Xn = build_features(add_measurement_noise(test_df, lv, seed=123), sur, "measured+residuals")
        a_clean = accuracy_score(y.loc[idx_te], clf.predict(Xn))
        a_aug = accuracy_score(y.loc[idx_te], clf_aug.predict(Xn))
        robust[f"{lv:.2f}"] = dict(trained_clean=a_clean, trained_noise_aug=a_aug)
        print(f"  noise {lv*100:>4.1f}%  clean-trained={a_clean:.3f}  noise-augmented={a_aug:.3f}")
    rb = pd.DataFrame(robust).T
    ax = rb.plot(marker="o", figsize=(7, 4), ylim=(0.5, 1.02))
    ax.set_xlabel("measurement noise (relative sigma)"); ax.set_ylabel("accuracy"); ax.grid(alpha=.3)
    ax.set_title("Fault classifier robustness to measurement noise")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "fault_noise_robustness.png", dpi=140); plt.close()

    FaultDiagnoser(sur, clf_aug).save()   # deployable model = noise-augmented
    (C.REPORT_DIR / "fault_metrics.json").write_text(json.dumps(dict(
        n_train=len(idx_tr), n_test=len(idx_te), ablation=ablation, noise_robustness=robust,
        best_report=classification_report(y.loc[idx_te], pred, output_dict=True)), indent=2))
    print("saved models/fault_clf.joblib and reports/")


if __name__ == "__main__":
    main()
