#!/usr/bin/env python3
"""Nested cross-validation confirmatória para predição de urocultura positiva.

Versão 2 (30/08/2026)
----------------------
* Exclui explicitamente 46 registros com Sexo == "I" (erro cadastral).
* Desenvolvimento: 2011--2023; teste temporal intocado: 2024.
* Nested CV 5 x 5 com StratifiedGroupKFold por paciente pseudonimizado.
* Braços: modelo compacto de quatro parâmetros (secundário), UROT e UROT+Gram.
* Elastic Net é o modelo de referência; boosting por histogramas é comparador.
* Imputação, codificação, escalonamento, pesos de classe, hiperparâmetros e
  limiares são aprendidos somente dentro do treino correspondente.
* Checkpoints são atômicos por fold externo e por ajuste temporal.

Exemplos
--------
Inicializar/auditar:
    python analysis/nested_cv_tese_v2.py --phase init

Executar um fold:
    python analysis/nested_cv_tese_v2.py --phase outer \
      --task UROT_40:elastic_net --fold 1

Executar o ajuste temporal de uma configuração:
    python analysis/nested_cv_tese_v2.py --phase temporal \
      --task UROT_40:elastic_net

Consolidar depois de todos os checkpoints:
    python analysis/nested_cv_tese_v2.py --phase aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import logit
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class SafeCatBoostClassifier(CatBoostClassifier):
    """Compatibiliza o valor nulo da grade sklearn com a API do CatBoost."""

    def set_params(self, **params: Any) -> "SafeCatBoostClassifier":
        if params.get("auto_class_weights", "__missing__") is None:
            params["auto_class_weights"] = "None"
        return super().set_params(**params)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "Matriz_Estatistica_v6_150859x52.csv"
EXCLUSION_LEDGER = ROOT / "data" / "Exclusoes_Sexo_Indeterminado_v6.csv"
OUT = ROOT / "outputs" / "nested_cv_arvores_v3_2026_09_02"
CHECKPOINTS = OUT / "checkpoints"
FIGURES = OUT / "figuras"
MODELS = OUT / "modelos"
DATA_OUT = OUT / "dados"
SEED = 20260829


CORE_UROT = [
    "PH", "GRAVID", "HEMAC", "LEUCOCITOS", "EPI_TOTAL", "ASP",
    "ESTERASE", "NITRIT", "HEMOGL", "CETONA", "GLIC", "PROT",
    "UROBIL", "BILIRR", "MUCO", "FLORA",
]

RARE_UROT = [
    "BROAD", "CIL_GRAXO", "GRAN", "HYAL", "DISMOR",
    "CIL_CEROCEROSOS", "CIL_LEUC", "CIL_HEMAT", "TRCHO", "BYST",
    "FAT", "SPRM", "CAOX", "FOSF_CA", "CISTINA",
    "CRISTAIS_LEUCINA", "FOSF_TRIP", "TIROS", "AC_URIC",
    "FSF_AMRF", "URT_AMRF",
]

CONTEXT_NUMERIC = ["Idade_anos"]
CONTEXT_CATEGORICAL = ["Sexo", "Origem"]
GRAM_NUMERIC = ["GR_EPIT", "GR_LEUC", "TGU"]
GRAM_CATEGORICAL = ["GRAM1", "GRAM2", "GRAM3"]

ARM_SPECS: dict[str, dict[str, Any]] = {
    "COMPACT_4": {
        "numeric": ["FLORA", "LEUCOCITOS", "ESTERASE", "NITRIT"],
        "categorical": [],
        "role": (
            "modelo secundário; hipótese formulada após exploração anterior "
            "da mesma base, portanto não constitui seleção inteiramente a priori"
        ),
    },
    "UROT_40": {
        "numeric": CORE_UROT + RARE_UROT + CONTEXT_NUMERIC,
        "categorical": CONTEXT_CATEGORICAL,
        "role": "braço confirmatório com 40 candidatos elegíveis por disponibilidade temporal",
    },
    "UROT_GRAM_46": {
        "numeric": CORE_UROT + RARE_UROT + CONTEXT_NUMERIC + GRAM_NUMERIC,
        "categorical": CONTEXT_CATEGORICAL + GRAM_CATEGORICAL,
        "role": "braço confirmatório UROT mais seis parâmetros de Gram",
    },
}

TREE_MODELS = ["random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"]
FULL_TASKS = [
    (arm, model)
    for arm in ("UROT_40", "UROT_GRAM_46")
    for model in TREE_MODELS
]

EXCLUDED_COLUMNS = {
    "DataEntregaMaterial": "usada apenas para separação temporal",
    "ANO": "usado apenas para separação temporal",
    "Paciente_grupo": "identificador pseudonimizado usado apenas no agrupamento",
    "Positivo": "desfecho",
    "COR": "excluída por descontinuidade histórica (disponível apenas desde 2016)",
    "Clinica": "metadado assistencial de baixa transportabilidade",
}

PALETTE = {
    "navy": "#234A63",
    "blue": "#4D88A8",
    "teal": "#62A89F",
    "orange": "#D58B4C",
    "red": "#B85C5C",
    "purple": "#77658C",
    "ink": "#20323D",
    "muted": "#6E808A",
    "grid": "#D9E2E7",
}


def select_font() -> str:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    for candidate in ("Aptos", "Carlito", "Liberation Sans", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


mpl.rcParams.update(
    {
        "font.family": select_font(),
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class RunConfig:
    seed: int = SEED
    outer_splits: int = 5
    inner_splits: int = 5
    n_jobs: int = 3
    bootstrap_reps: int = 300
    threshold_sensitivity_target: float = 0.95
    development_years: str = "2011-2023"
    temporal_test_year: int = 2024
    primary_metric: str = "roc_auc"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def ensure_dirs() -> None:
    for path in (OUT, CHECKPOINTS, FIGURES, MODELS, DATA_OUT):
        path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo não serializável: {type(value)!r}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_to_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Grava tabelas compartilhadas sem expor arquivos parciais a outros processos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_task(value: str) -> tuple[str, str]:
    try:
        arm, model = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use BRAÇO:MODELO") from exc
    task = (arm, model)
    if task not in FULL_TASKS:
        allowed = ", ".join(f"{a}:{m}" for a, m in FULL_TASKS)
        raise argparse.ArgumentTypeError(f"Tarefa inválida. Opções: {allowed}")
    return task


def load_data() -> pd.DataFrame:
    required = {
        "DataEntregaMaterial", "ANO", "Paciente_grupo", "Positivo", "Sexo",
        *ARM_SPECS["UROT_GRAM_46"]["numeric"],
        *ARM_SPECS["UROT_GRAM_46"]["categorical"],
        *EXCLUDED_COLUMNS,
    }
    data = pd.read_csv(SOURCE, sep=";", encoding="utf-8-sig", low_memory=False)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise AssertionError(f"Colunas obrigatórias ausentes: {missing}")
    if tuple(data.shape) != (150_859, 52):
        raise AssertionError(f"Dimensão inesperada: {data.shape}")
    if set(data["Sexo"].unique()) != {"F", "M"}:
        raise AssertionError("Sexo deve conter somente F/M após exclusão dos erros cadastrais")
    if data["Paciente_grupo"].isna().any():
        raise AssertionError("Paciente_grupo contém ausências")
    if data["Positivo"].isna().any() or set(data["Positivo"].unique()) != {0, 1}:
        raise AssertionError("Desfecho Positivo inválido")
    data["DataEntregaMaterial"] = pd.to_datetime(data["DataEntregaMaterial"], errors="raise")
    if not data["DataEntregaMaterial"].dt.year.eq(data["ANO"]).all():
        raise AssertionError("ANO diverge de DataEntregaMaterial")
    if data["ANO"].min() != 2011 or data["ANO"].max() != 2024:
        raise AssertionError("Janela temporal inesperada")
    data = data.reset_index(drop=True)
    data.insert(0, "row_id", np.arange(len(data), dtype=np.int64))
    return data


def predictor_inventory(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, spec in ARM_SPECS.items():
        for kind in ("numeric", "categorical"):
            for column in spec[kind]:
                series = data[column]
                rows.append(
                    {
                        "braco": arm,
                        "variavel": column,
                        "tipo_modelagem": "numérica" if kind == "numeric" else "categórica",
                        "papel": spec["role"],
                        "n_ausente": int(series.isna().sum()),
                        "percentual_ausente": float(series.isna().mean() * 100),
                        "n_niveis_ou_valores": int(series.nunique(dropna=True)),
                    }
                )
    for column, reason in EXCLUDED_COLUMNS.items():
        series = data[column]
        rows.append(
            {
                "braco": "NÃO_ENTRA_EM_X",
                "variavel": column,
                "tipo_modelagem": "metadado/desfecho",
                "papel": reason,
                "n_ausente": int(series.isna().sum()),
                "percentual_ausente": float(series.isna().mean() * 100),
                "n_niveis_ou_valores": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def make_outer_splits(
    dev: pd.DataFrame, config: RunConfig
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(
        n_splits=config.outer_splits, shuffle=True, random_state=config.seed
    )
    splits = list(
        splitter.split(
            np.zeros(len(dev)),
            dev["Positivo"].to_numpy(),
            dev["Paciente_grupo"].to_numpy(),
        )
    )
    assignment = pd.DataFrame(
        {
            "row_id": dev["row_id"].to_numpy(),
            "Paciente_grupo": dev["Paciente_grupo"].to_numpy(),
            "Positivo": dev["Positivo"].to_numpy(),
            "ANO": dev["ANO"].to_numpy(),
            "outer_fold": np.zeros(len(dev), dtype=np.int8),
        }
    )
    audit_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        assignment.loc[test_idx, "outer_fold"] = fold
        train_groups = set(dev.iloc[train_idx]["Paciente_grupo"])
        test_groups = set(dev.iloc[test_idx]["Paciente_grupo"])
        overlap = train_groups.intersection(test_groups)
        audit_rows.append(
            {
                "outer_fold": fold,
                "n_treino": len(train_idx),
                "n_teste": len(test_idx),
                "positivos_treino": int(dev.iloc[train_idx]["Positivo"].sum()),
                "positivos_teste": int(dev.iloc[test_idx]["Positivo"].sum()),
                "prevalencia_treino": float(dev.iloc[train_idx]["Positivo"].mean()),
                "prevalencia_teste": float(dev.iloc[test_idx]["Positivo"].mean()),
                "pacientes_treino": len(train_groups),
                "pacientes_teste": len(test_groups),
                "pacientes_sobrepostos": len(overlap),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not assignment["outer_fold"].between(1, config.outer_splits).all():
        raise AssertionError("Nem todas as linhas receberam fold externo")
    if audit["pacientes_sobrepostos"].sum() != 0:
        raise AssertionError("Há paciente compartilhado entre treino e teste externo")
    return splits, assignment, audit


def make_inner_splits(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train_idx, valid_idx in splits:
        if set(groups[train_idx]).intersection(groups[valid_idx]):
            raise AssertionError("Paciente compartilhado entre treino e validação internos")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[valid_idx])) != 2:
            raise AssertionError("Fold interno sem as duas classes")
    return splits


def build_preprocessor(
    numeric: list[str], categorical: list[str], scale: bool
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            )
        ]
        if scale:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def build_estimator(arm: str, model_name: str, seed: int) -> Pipeline:
    spec = ARM_SPECS[arm]
    if model_name == "elastic_net":
        model = LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            C=0.2,
            max_iter=800,
            tol=1e-3,
            random_state=seed,
        )
        return Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(spec["numeric"], spec["categorical"], scale=True),
                ),
                ("model", model),
            ]
        )
    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        )
        return Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(spec["numeric"], spec["categorical"], scale=False),
                ),
                ("model", model),
            ]
        )
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=5,
            n_jobs=1, random_state=seed,
        )
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=5,
            n_jobs=1, random_state=seed,
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
            n_jobs=1, random_state=seed,
        )
    elif model_name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=1, random_state=seed, verbosity=-1,
        )
    elif model_name == "catboost":
        model = SafeCatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            loss_function="Logloss", eval_metric="AUC", verbose=False,
            allow_writing_files=False, thread_count=1, random_seed=seed,
        )
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    return Pipeline([
        ("preprocess", build_preprocessor(spec["numeric"], spec["categorical"], scale=False)),
        ("model", model),
    ])


def explicit_grid(model_name: str) -> list[dict[str, list[Any]]]:
    if model_name == "elastic_net":
        candidates = [
            (0.03, 0.5, None),
            (0.10, 0.5, None),
            (0.30, 0.5, None),
            (1.00, 0.5, None),
            (0.10, 0.0, None),
            (0.30, 1.0, None),
            (0.10, 0.5, "balanced"),
            (0.30, 0.5, "balanced"),
        ]
        return [
            {
                "model__C": [c],
                "model__l1_ratio": [ratio],
                "model__class_weight": [weight],
            }
            for c, ratio, weight in candidates
        ]
    if model_name == "hist_gradient_boosting":
        candidates = [
            (15, 0.05, 1.0, 100, None),
            (31, 0.05, 1.0, 50, None),
            (31, 0.10, 1.0, 50, None),
            (63, 0.05, 1.0, 50, None),
            (31, 0.05, 5.0, 100, None),
            (31, 0.05, 1.0, 50, "balanced"),
        ]
        return [
            {
                "model__max_leaf_nodes": [leaves],
                "model__learning_rate": [learning_rate],
                "model__l2_regularization": [l2],
                "model__min_samples_leaf": [min_leaf],
                "model__class_weight": [weight],
            }
            for leaves, learning_rate, l2, min_leaf, weight in candidates
        ]
    if model_name in {"random_forest", "extra_trees"}:
        candidates = [
            (None, "sqrt", 5, None),
            (20, "sqrt", 10, None),
            (None, 0.5, 10, "balanced"),
        ]
        return [{
            "model__max_depth": [depth], "model__max_features": [features],
            "model__min_samples_leaf": [leaf], "model__class_weight": [weight],
        } for depth, features, leaf, weight in candidates]
    if model_name == "xgboost":
        candidates = [(4, 0.05, 10, 1.0), (6, 0.05, 10, 1.0), (6, 0.05, 25, 3.5)]
        return [{
            "model__max_depth": [depth], "model__learning_rate": [rate],
            "model__min_child_weight": [child], "model__scale_pos_weight": [weight],
        } for depth, rate, child, weight in candidates]
    if model_name == "lightgbm":
        candidates = [(15, 50, 0.0, None), (31, 50, 1.0, None), (31, 100, 1.0, "balanced")]
        return [{
            "model__num_leaves": [leaves], "model__min_child_samples": [child],
            "model__reg_lambda": [regularization], "model__class_weight": [weight],
        } for leaves, child, regularization, weight in candidates]
    if model_name == "catboost":
        candidates = [(5, 0.05, 3.0, None), (6, 0.05, 3.0, None), (6, 0.05, 10.0, "Balanced")]
        return [{
            "model__depth": [depth], "model__learning_rate": [rate],
            "model__l2_leaf_reg": [regularization], "model__auto_class_weights": [weight],
        } for depth, rate, regularization, weight in candidates]
    raise ValueError(model_name)


def scoring_dict() -> dict[str, str]:
    return {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "neg_brier": "neg_brier_score",
        "neg_log_loss": "neg_log_loss",
    }


def find_thresholds(y: np.ndarray, probability: np.ndarray, target: float) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    youden_idx = int(np.nanargmax(tpr - fpr))
    eligible = np.flatnonzero(tpr >= target)
    if len(eligible):
        specificity = 1.0 - fpr[eligible]
        best_specificity = np.max(specificity)
        tied = eligible[np.isclose(specificity, best_specificity)]
        sensitivity_idx = int(tied[np.argmax(thresholds[tied])])
        sens_threshold = float(thresholds[sensitivity_idx])
    else:
        sens_threshold = 0.0
    return {
        "fixo_0_5": 0.5,
        "youden_interno": float(np.clip(thresholds[youden_idx], 0.0, 1.0)),
        f"sensibilidade_{int(round(target * 100))}_interna": float(
            np.clip(sens_threshold, 0.0, 1.0)
        ),
    }


def calibration_parameters(
    y: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    x = logit(probability).reshape(-1, 1)
    try:
        calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        calibration.fit(x, y, sample_weight=sample_weight)
        return float(calibration.intercept_[0]), float(calibration.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def probability_metrics(
    y: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None
) -> dict[str, float]:
    intercept, slope = calibration_parameters(y, probability, sample_weight)
    observed = (
        float(np.average(y, weights=sample_weight))
        if sample_weight is not None
        else float(np.mean(y))
    )
    expected = (
        float(np.average(probability, weights=sample_weight))
        if sample_weight is not None
        else float(np.mean(probability))
    )
    return {
        "roc_auc": float(roc_auc_score(y, probability, sample_weight=sample_weight)),
        "pr_auc": float(average_precision_score(y, probability, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y, probability, sample_weight=sample_weight)),
        "log_loss": float(
            log_loss(y, probability, sample_weight=sample_weight, labels=[0, 1])
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "observed_expected_ratio": (
            float(observed / expected) if expected > 0 else float("nan")
        ),
        "prevalencia_observada": observed,
        "probabilidade_media": expected,
    }


def threshold_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float | np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    return binary_metrics(y, predicted, sample_weight)


def binary_metrics(
    y: np.ndarray,
    predicted: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1], sample_weight=sample_weight
    ).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "sensitivity": float(
            recall_score(y, predicted, sample_weight=sample_weight, zero_division=0)
        ),
        "specificity": float(specificity),
        "ppv": float(
            precision_score(y, predicted, sample_weight=sample_weight, zero_division=0)
        ),
        "npv": float(npv),
        "accuracy": float(accuracy_score(y, predicted, sample_weight=sample_weight)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted, sample_weight=sample_weight)
        ),
        "f1": float(
            f1_score(y, predicted, sample_weight=sample_weight, zero_division=0)
        ),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def coefficient_frame(
    estimator: Pipeline, arm: str, model_name: str, fold: str
) -> pd.DataFrame:
    if model_name != "elastic_net":
        return pd.DataFrame()
    feature_names = estimator.named_steps["preprocess"].get_feature_names_out()
    coefficients = estimator.named_steps["model"].coef_[0]
    return pd.DataFrame(
        {
            "braco": arm,
            "modelo": model_name,
            "fold": fold,
            "feature_transformada": feature_names,
            "coeficiente": coefficients,
            "odds_ratio_por_unidade_transformada": np.exp(
                np.clip(coefficients, -30, 30)
            ),
            "selecionada_nao_zero": np.abs(coefficients) > 1e-10,
        }
    )


def checkpoint_paths(arm: str, model_name: str, fold: int) -> dict[str, Path]:
    stem = CHECKPOINTS / f"{arm}__{model_name}__outer_{fold}"
    return {
        "meta": stem.with_suffix(".json"),
        "pred": stem.with_name(stem.name + "__predicoes.csv.gz"),
        "coef": stem.with_name(stem.name + "__coeficientes.csv.gz"),
        "cv": stem.with_name(stem.name + "__cv_results.csv.gz"),
    }


def final_checkpoint_paths(arm: str, model_name: str) -> dict[str, Path]:
    stem = CHECKPOINTS / f"{arm}__{model_name}__final_2011_2023"
    return {
        "meta": stem.with_suffix(".json"),
        "pred": stem.with_name(stem.name + "__predicoes_2024.csv.gz"),
        "coef": stem.with_name(stem.name + "__coeficientes.csv.gz"),
        "cv": stem.with_name(stem.name + "__cv_results.csv.gz"),
        "model": MODELS / f"Modelo_{arm}_{model_name}_treinado_2011_2023.joblib",
    }


def checkpoint_complete(paths: dict[str, Path], run_hash: str) -> bool:
    required = [paths["meta"], paths["pred"], paths["cv"]]
    if not all(path.exists() for path in required):
        return False
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    return meta.get("status") == "completed" and meta.get("run_hash") == run_hash


def grid_results_frame(grid: GridSearchCV, arm: str, model_name: str, fold: int | None) -> pd.DataFrame:
    results = pd.DataFrame(grid.cv_results_)
    keep = [
        column
        for column in results.columns
        if column.startswith("param_")
        or column.startswith("mean_test_")
        or column.startswith("std_test_")
        or column.startswith("rank_test_")
        or column in {"mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"}
    ]
    results = results[keep].copy()
    if fold is not None:
        results.insert(0, "outer_fold", fold)
    results.insert(0, "modelo", model_name)
    results.insert(0, "braco", arm)
    return results


def fit_outer_fold(
    dev: pd.DataFrame,
    arm: str,
    model_name: str,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: RunConfig,
    run_hash: str,
) -> dict[str, Any]:
    paths = checkpoint_paths(arm, model_name, fold)
    if checkpoint_complete(paths, run_hash):
        log(f"CHECKPOINT reutilizado: {arm} / {model_name} / fold {fold}")
        return json.loads(paths["meta"].read_text(encoding="utf-8"))

    started = time.perf_counter()
    spec = ARM_SPECS[arm]
    features = spec["numeric"] + spec["categorical"]
    train = dev.iloc[train_idx]
    test = dev.iloc[test_idx]
    x_train = train[features]
    y_train = train["Positivo"].to_numpy(dtype=int)
    groups_train = train["Paciente_grupo"].to_numpy()
    x_test = test[features]
    y_test = test["Positivo"].to_numpy(dtype=int)

    inner_splits = make_inner_splits(
        y_train,
        groups_train,
        n_splits=config.inner_splits,
        seed=config.seed + fold * 101,
    )
    estimator = build_estimator(arm, model_name, config.seed + fold)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=explicit_grid(model_name),
        scoring=scoring_dict(),
        refit=config.primary_metric,
        cv=inner_splits,
        n_jobs=config.n_jobs,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )
    log(
        f"AJUSTE: {arm} / {model_name} / fold {fold}: "
        f"{len(explicit_grid(model_name))} configurações x {config.inner_splits} folds internos"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        grid.fit(x_train, y_train)
        inner_oof = cross_val_predict(
            clone(grid.best_estimator_),
            x_train,
            y_train,
            cv=inner_splits,
            method="predict_proba",
            n_jobs=config.n_jobs,
        )[:, 1]

    best = grid.best_estimator_
    thresholds = find_thresholds(y_train, inner_oof, config.threshold_sensitivity_target)
    probability = best.predict_proba(x_test)[:, 1]
    prob_metrics = probability_metrics(y_test, probability)
    threshold_results = {
        name: {
            "threshold": float(value),
            **threshold_metrics(y_test, probability, value),
        }
        for name, value in thresholds.items()
    }
    predictions = pd.DataFrame(
        {
            "row_id": test["row_id"].to_numpy(),
            "Paciente_grupo": test["Paciente_grupo"].to_numpy(),
            "ANO": test["ANO"].to_numpy(),
            "Positivo": y_test,
            "probabilidade": probability,
            "outer_fold": fold,
            "braco": arm,
            "modelo": model_name,
        }
    )
    for name, value in thresholds.items():
        predictions[f"limiar__{name}"] = value
        predictions[f"predito__{name}"] = (probability >= value).astype(np.int8)

    cv_results = grid_results_frame(grid, arm, model_name, fold)
    coefficients = coefficient_frame(best, arm, model_name, str(fold))
    elapsed = time.perf_counter() - started
    meta: dict[str, Any] = {
        "status": "completed",
        "run_hash": run_hash,
        "completed_at": utc_now(),
        "braco": arm,
        "modelo": model_name,
        "outer_fold": fold,
        "n_train": len(train),
        "n_test": len(test),
        "n_groups_train": int(train["Paciente_grupo"].nunique()),
        "n_groups_test": int(test["Paciente_grupo"].nunique()),
        "best_inner_roc_auc": float(grid.best_score_),
        "best_params": grid.best_params_,
        "thresholds": thresholds,
        "probability_metrics": prob_metrics,
        "threshold_metrics": threshold_results,
        "elapsed_seconds": elapsed,
    }

    predictions.to_csv(paths["pred"], index=False, compression="gzip")
    cv_results.to_csv(paths["cv"], index=False, compression="gzip")
    if not coefficients.empty:
        coefficients.to_csv(paths["coef"], index=False, compression="gzip")
    write_json(paths["meta"], meta)
    log(
        f"CONCLUÍDO: {arm} / {model_name} / fold {fold} - "
        f"AUC={prob_metrics['roc_auc']:.4f}, PR-AUC={prob_metrics['pr_auc']:.4f}, "
        f"{elapsed / 60:.1f} min"
    )
    return meta


def fit_final_temporal(
    dev: pd.DataFrame,
    temporal: pd.DataFrame,
    arm: str,
    model_name: str,
    config: RunConfig,
    run_hash: str,
) -> dict[str, Any]:
    paths = final_checkpoint_paths(arm, model_name)
    if checkpoint_complete(paths, run_hash) and paths["model"].exists():
        log(f"CHECKPOINT temporal reutilizado: {arm} / {model_name}")
        return json.loads(paths["meta"].read_text(encoding="utf-8"))

    started = time.perf_counter()
    spec = ARM_SPECS[arm]
    features = spec["numeric"] + spec["categorical"]
    x_dev = dev[features]
    y_dev = dev["Positivo"].to_numpy(dtype=int)
    groups_dev = dev["Paciente_grupo"].to_numpy()
    inner_splits = make_inner_splits(
        y_dev, groups_dev, config.inner_splits, config.seed + 909
    )
    estimator = build_estimator(arm, model_name, config.seed + 909)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=explicit_grid(model_name),
        scoring=scoring_dict(),
        refit=config.primary_metric,
        cv=inner_splits,
        n_jobs=config.n_jobs,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )
    log(f"AJUSTE FINAL 2011-2023: {arm} / {model_name}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        grid.fit(x_dev, y_dev)
        dev_oof = cross_val_predict(
            clone(grid.best_estimator_),
            x_dev,
            y_dev,
            cv=inner_splits,
            method="predict_proba",
            n_jobs=config.n_jobs,
        )[:, 1]

    thresholds = find_thresholds(y_dev, dev_oof, config.threshold_sensitivity_target)
    best = grid.best_estimator_
    x_temporal = temporal[features]
    y_temporal = temporal["Positivo"].to_numpy(dtype=int)
    probability = best.predict_proba(x_temporal)[:, 1]
    seen_groups = set(dev["Paciente_grupo"])
    new_patient = ~temporal["Paciente_grupo"].isin(seen_groups)
    probability_all = probability_metrics(y_temporal, probability)
    threshold_all = {
        name: {
            "threshold": float(value),
            **threshold_metrics(y_temporal, probability, value),
        }
        for name, value in thresholds.items()
    }
    probability_new = (
        probability_metrics(y_temporal[new_patient], probability[new_patient])
        if new_patient.any()
        else {}
    )
    threshold_new = (
        {
            name: {
                "threshold": float(value),
                **threshold_metrics(
                    y_temporal[new_patient], probability[new_patient], value
                ),
            }
            for name, value in thresholds.items()
        }
        if new_patient.any()
        else {}
    )
    predictions = pd.DataFrame(
        {
            "row_id": temporal["row_id"].to_numpy(),
            "Paciente_grupo": temporal["Paciente_grupo"].to_numpy(),
            "ANO": temporal["ANO"].to_numpy(),
            "Positivo": y_temporal,
            "paciente_novo_em_2024": new_patient.to_numpy(),
            "probabilidade": probability,
            "braco": arm,
            "modelo": model_name,
        }
    )
    for name, value in thresholds.items():
        predictions[f"limiar__{name}"] = value
        predictions[f"predito__{name}"] = (probability >= value).astype(np.int8)

    cv_results = grid_results_frame(grid, arm, model_name, None)
    coefficients = coefficient_frame(best, arm, model_name, "final_2011_2023")
    joblib.dump(best, paths["model"], compress=3)
    elapsed = time.perf_counter() - started
    meta: dict[str, Any] = {
        "status": "completed",
        "run_hash": run_hash,
        "completed_at": utc_now(),
        "braco": arm,
        "modelo": model_name,
        "n_development": len(dev),
        "n_temporal_2024": len(temporal),
        "n_temporal_new_patient": int(new_patient.sum()),
        "n_temporal_seen_patient": int((~new_patient).sum()),
        "best_inner_roc_auc": float(grid.best_score_),
        "best_params": grid.best_params_,
        "thresholds": thresholds,
        "probability_metrics_all_2024": probability_all,
        "threshold_metrics_all_2024": threshold_all,
        "probability_metrics_new_patients_2024": probability_new,
        "threshold_metrics_new_patients_2024": threshold_new,
        "elapsed_seconds": elapsed,
        "model_path": str(paths["model"]),
    }
    predictions.to_csv(paths["pred"], index=False, compression="gzip")
    cv_results.to_csv(paths["cv"], index=False, compression="gzip")
    if not coefficients.empty:
        coefficients.to_csv(paths["coef"], index=False, compression="gzip")
    write_json(paths["meta"], meta)
    log(
        f"TESTE 2024 CONCLUÍDO: {arm} / {model_name} - "
        f"AUC={probability_all['roc_auc']:.4f}, PR-AUC={probability_all['pr_auc']:.4f}, "
        f"{elapsed / 60:.1f} min"
    )
    return meta


def prepare_run(config: RunConfig) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[np.ndarray, np.ndarray]],
    str,
]:
    ensure_dirs()
    source_hash = sha256_file(SOURCE)
    protocol: dict[str, Any] = {
        **asdict(config),
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "source_expected_shape": [150_859, 52],
        "sex_registry_errors_excluded": 46,
        "arms": ARM_SPECS,
        "tasks": [f"{arm}:{model}" for arm, model in FULL_TASKS],
        "hyperparameter_candidates": {
            model: explicit_grid(model)
            for model in sorted({model for _, model in FULL_TASKS})
        },
        "excluded_columns": EXCLUDED_COLUMNS,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
            "joblib": joblib.__version__,
            "xgboost": __import__("xgboost").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "catboost": __import__("catboost").__version__,
        },
        "leakage_controls": [
            "2024 nunca participa de seleção, ajuste ou limiar",
            "pacientes não atravessam folds internos ou externos",
            "pré-processamento dentro do Pipeline",
            "hiperparâmetros escolhidos somente no loop interno",
            "limiares derivados de predições internas out-of-fold",
            "sem seleção global pelo desfecho nos braços confirmatórios",
            "sem SMOTE",
        ],
    }
    run_hash = config_hash(protocol)
    config_path = OUT / "Configuracao_Congelada.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("run_hash") != run_hash:
            raise AssertionError(
                "A configuração existente tem outro hash; use outro diretório de saída"
            )
    else:
        protocol["run_hash"] = run_hash
        protocol["created_at"] = utc_now()
        write_json(config_path, protocol)
    log(f"Configuração congelada: run_hash={run_hash}")

    data = load_data()
    dev = data[data["ANO"].between(2011, 2023)].reset_index(drop=True)
    temporal = data[data["ANO"].eq(2024)].reset_index(drop=True)
    log(
        f"Dados: {len(dev):,} desenvolvimento ({dev['Positivo'].mean():.2%} positivos); "
        f"{len(temporal):,} teste temporal ({temporal['Positivo'].mean():.2%} positivos)"
    )
    outer_splits, assignment, outer_audit = make_outer_splits(dev, config)
    atomic_to_csv(
        assignment, OUT / "Atribuicao_Folds_Externos.csv.gz",
        index=False, compression="gzip",
    )
    atomic_to_csv(outer_audit, OUT / "Auditoria_Folds_Externos.csv", index=False)
    atomic_to_csv(predictor_inventory(data), OUT / "Inventario_Preditores.csv", index=False)
    audit = build_methodology_audit(data, dev, temporal, outer_audit)
    atomic_to_csv(audit, OUT / "Auditoria_Metodologica.csv", index=False)
    log("Auditoria pré-modelagem: PASSOU; zero pacientes cruzando folds externos")
    return data, dev, temporal, outer_splits, run_hash


def build_methodology_audit(
    data: pd.DataFrame,
    dev: pd.DataFrame,
    temporal: pd.DataFrame,
    outer_audit: pd.DataFrame,
) -> pd.DataFrame:
    dev_groups = set(dev["Paciente_grupo"])
    temporal_groups = set(temporal["Paciente_grupo"])
    checks = [
        ("dimensão da matriz corrigida", len(data) == 150_859, str(data.shape)),
        ("sexo restrito a F/M", set(data["Sexo"].unique()) == {"F", "M"}, str(sorted(data["Sexo"].unique()))),
        ("desfecho estritamente binário", set(data["Positivo"].unique()) == {0, 1}, str(sorted(data["Positivo"].unique()))),
        ("desenvolvimento restrito a 2011-2023", dev["ANO"].between(2011, 2023).all(), f"{dev['ANO'].min()}-{dev['ANO'].max()}"),
        ("teste temporal restrito a 2024", temporal["ANO"].eq(2024).all(), f"n={len(temporal)}"),
        ("zero sobreposição paciente treino/teste externo", outer_audit["pacientes_sobrepostos"].sum() == 0, f"soma={outer_audit['pacientes_sobrepostos'].sum()}"),
        ("40 candidatos no braço UROT", len(ARM_SPECS["UROT_40"]["numeric"] + ARM_SPECS["UROT_40"]["categorical"]) == 40, "40"),
        ("46 candidatos no braço UROT+Gram", len(ARM_SPECS["UROT_GRAM_46"]["numeric"] + ARM_SPECS["UROT_GRAM_46"]["categorical"]) == 46, "46"),
        ("4 candidatos no modelo compacto", len(ARM_SPECS["COMPACT_4"]["numeric"]) == 4, "4"),
        ("COR excluída", all("COR" not in spec["numeric"] + spec["categorical"] for spec in ARM_SPECS.values()), "descontinuidade 2011-2015"),
        ("Clínica excluída", all("Clinica" not in spec["numeric"] + spec["categorical"] for spec in ARM_SPECS.values()), "metadado assistencial"),
    ]
    rows = [
        {
            "checagem": name,
            "resultado": "PASSOU" if passed else "FALHOU",
            "evidencia": evidence,
        }
        for name, passed, evidence in checks
    ]
    rows.extend(
        [
            {"checagem": "pacientes em 2011-2023", "resultado": "INFORMATIVO", "evidencia": str(len(dev_groups))},
            {"checagem": "pacientes em 2024", "resultado": "INFORMATIVO", "evidencia": str(len(temporal_groups))},
            {"checagem": "pacientes de 2024 já observados antes", "resultado": "INFORMATIVO", "evidencia": str(len(dev_groups.intersection(temporal_groups)))},
            {"checagem": "pacientes novos em 2024", "resultado": "INFORMATIVO", "evidencia": str(len(temporal_groups.difference(dev_groups)))},
        ]
    )
    audit = pd.DataFrame(rows)
    if (audit["resultado"] == "FALHOU").any():
        raise AssertionError("Auditoria metodológica falhou")
    return audit


def validate_outer_checkpoint(
    dev: pd.DataFrame,
    assignment: pd.DataFrame,
    arm: str,
    model: str,
    fold: int,
    run_hash: str,
) -> dict[str, Any]:
    paths = checkpoint_paths(arm, model, fold)
    if not checkpoint_complete(paths, run_hash):
        raise AssertionError(f"Checkpoint incompleto: {arm}/{model}/fold {fold}")
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    pred = pd.read_csv(paths["pred"])
    expected = set(assignment.loc[assignment["outer_fold"].eq(fold), "row_id"])
    auc = roc_auc_score(pred["Positivo"], pred["probabilidade"])
    pr_auc = average_precision_score(pred["Positivo"], pred["probabilidade"])
    checks = [
        len(pred) == len(expected),
        pred["row_id"].is_unique,
        set(pred["row_id"]) == expected,
        np.isfinite(pred["probabilidade"]).all(),
        abs(auc - meta["probability_metrics"]["roc_auc"]) < 1e-12,
        abs(pr_auc - meta["probability_metrics"]["pr_auc"]) < 1e-12,
    ]
    if not all(checks):
        raise AssertionError(f"Validação falhou: {arm}/{model}/fold {fold}")
    return meta


def load_all_checkpoints(
    dev: pd.DataFrame,
    temporal: pd.DataFrame,
    assignment: pd.DataFrame,
    run_hash: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    outer_meta: list[dict[str, Any]] = []
    temporal_meta: list[dict[str, Any]] = []
    outer_predictions: list[pd.DataFrame] = []
    temporal_predictions: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    cv_results: list[pd.DataFrame] = []
    missing: list[str] = []

    for arm, model in FULL_TASKS:
        for fold in range(1, 6):
            paths = checkpoint_paths(arm, model, fold)
            if not checkpoint_complete(paths, run_hash):
                missing.append(f"{arm}:{model}:outer_{fold}")
                continue
            outer_meta.append(
                validate_outer_checkpoint(
                    dev, assignment, arm, model, fold, run_hash
                )
            )
            outer_predictions.append(pd.read_csv(paths["pred"]))
            cv_results.append(pd.read_csv(paths["cv"]))
            if paths["coef"].exists():
                coefficients.append(pd.read_csv(paths["coef"]))

        paths = final_checkpoint_paths(arm, model)
        if not checkpoint_complete(paths, run_hash) or not paths["model"].exists():
            missing.append(f"{arm}:{model}:temporal")
            continue
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        pred = pd.read_csv(paths["pred"])
        auc = roc_auc_score(pred["Positivo"], pred["probabilidade"])
        if len(pred) != len(temporal) or not pred["row_id"].is_unique:
            raise AssertionError(f"Predições temporais inválidas: {arm}/{model}")
        if abs(auc - meta["probability_metrics_all_2024"]["roc_auc"]) >= 1e-12:
            raise AssertionError(f"AUC temporal divergente: {arm}/{model}")
        temporal_meta.append(meta)
        temporal_predictions.append(pred)
        cv_results.append(pd.read_csv(paths["cv"]))
        if paths["coef"].exists():
            coefficients.append(pd.read_csv(paths["coef"]))

    if missing:
        raise RuntimeError("Checkpoints pendentes: " + ", ".join(missing))

    outer_pred = pd.concat(outer_predictions, ignore_index=True)
    temporal_pred = pd.concat(temporal_predictions, ignore_index=True)
    coefficient_frame_all = (
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    )
    cv_frame_all = pd.concat(cv_results, ignore_index=True)
    expected_outer = len(dev) * len(FULL_TASKS)
    if len(outer_pred) != expected_outer:
        raise AssertionError(
            f"Número inesperado de predições externas: {len(outer_pred)} != {expected_outer}"
        )
    if outer_pred.groupby(["braco", "modelo", "row_id"]).size().ne(1).any():
        raise AssertionError("Linha com zero ou mais de uma predição externa")
    return (
        outer_meta,
        temporal_meta,
        outer_pred,
        temporal_pred,
        coefficient_frame_all,
        cv_frame_all,
    )


def flatten_outer_meta(
    meta_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    for meta in meta_rows:
        base = {
            "braco": meta["braco"],
            "modelo": meta["modelo"],
            "outer_fold": meta["outer_fold"],
        }
        probability_rows.append({**base, **meta["probability_metrics"]})
        for rule, metrics in meta["threshold_metrics"].items():
            threshold_rows.append({**base, "regra_limiar": rule, **metrics})
        hyper_rows.append(
            {
                **base,
                "best_inner_roc_auc": meta["best_inner_roc_auc"],
                "best_params_json": json.dumps(
                    meta["best_params"], ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return (
        pd.DataFrame(probability_rows),
        pd.DataFrame(threshold_rows),
        pd.DataFrame(hyper_rows),
    )


def flatten_temporal_meta(
    meta_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    for meta in meta_rows:
        base = {"braco": meta["braco"], "modelo": meta["modelo"]}
        for population, metrics_key, threshold_key in [
            ("todos_2024", "probability_metrics_all_2024", "threshold_metrics_all_2024"),
            (
                "pacientes_novos_2024",
                "probability_metrics_new_patients_2024",
                "threshold_metrics_new_patients_2024",
            ),
        ]:
            metrics = meta.get(metrics_key, {})
            if not metrics:
                continue
            probability_rows.append(
                {**base, "populacao": population, **metrics}
            )
            for rule, values in meta.get(threshold_key, {}).items():
                threshold_rows.append(
                    {
                        **base,
                        "populacao": population,
                        "regra_limiar": rule,
                        **values,
                    }
                )
        hyper_rows.append(
            {
                **base,
                "best_inner_roc_auc": meta["best_inner_roc_auc"],
                "best_params_json": json.dumps(
                    meta["best_params"], ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return (
        pd.DataFrame(probability_rows),
        pd.DataFrame(threshold_rows),
        pd.DataFrame(hyper_rows),
    )


def metric_summary(
    fold_probability: pd.DataFrame, fold_threshold: pd.DataFrame
) -> pd.DataFrame:
    probability_long = fold_probability.melt(
        id_vars=["braco", "modelo", "outer_fold"],
        value_vars=[
            "roc_auc", "pr_auc", "brier", "log_loss",
            "calibration_intercept", "calibration_slope",
            "prevalencia_observada", "probabilidade_media",
        ],
        var_name="metrica",
        value_name="valor",
    )
    probability_long["regra_limiar"] = "probabilidade"
    threshold_long = fold_threshold.melt(
        id_vars=["braco", "modelo", "outer_fold", "regra_limiar"],
        value_vars=[
            "sensitivity", "specificity", "ppv", "npv", "accuracy",
            "balanced_accuracy", "f1",
        ],
        var_name="metrica",
        value_name="valor",
    )
    combined = pd.concat([probability_long, threshold_long], ignore_index=True)
    return (
        combined.groupby(["braco", "modelo", "regra_limiar", "metrica"], as_index=False)
        .agg(media_folds=("valor", "mean"), dp_folds=("valor", "std"), minimo=("valor", "min"), maximo=("valor", "max"), n_folds=("valor", "count"))
    )


def bootstrap_cluster_metrics(
    predictions: pd.DataFrame,
    reps: int,
    seed: int,
    population: str,
) -> pd.DataFrame:
    y = predictions["Positivo"].to_numpy(dtype=int)
    probability = predictions["probabilidade"].to_numpy(dtype=float)
    groups, inverse = np.unique(
        predictions["Paciente_grupo"].to_numpy(), return_inverse=True
    )
    n_groups = len(groups)
    rng = np.random.default_rng(seed)
    rules = {
        column.replace("predito__", "", 1): predictions[column].to_numpy(dtype=int)
        for column in predictions.columns
        if column.startswith("predito__")
    }
    point_probability = probability_metrics(y, probability)
    point_threshold = {
        rule: binary_metrics(y, predicted) for rule, predicted in rules.items()
    }
    probability_draws: dict[str, list[float]] = {
        metric: [] for metric in ("roc_auc", "pr_auc", "brier", "log_loss")
    }
    threshold_draws: dict[tuple[str, str], list[float]] = {}
    selected_threshold_metrics = [
        "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "f1"
    ]
    for _ in range(reps):
        sampled = rng.integers(0, n_groups, size=n_groups)
        group_weight = np.bincount(sampled, minlength=n_groups).astype(float)
        row_weight = group_weight[inverse]
        if np.sum(row_weight[y == 1]) == 0 or np.sum(row_weight[y == 0]) == 0:
            continue
        probability_draws["roc_auc"].append(
            roc_auc_score(y, probability, sample_weight=row_weight)
        )
        probability_draws["pr_auc"].append(
            average_precision_score(y, probability, sample_weight=row_weight)
        )
        probability_draws["brier"].append(
            brier_score_loss(y, probability, sample_weight=row_weight)
        )
        probability_draws["log_loss"].append(
            log_loss(y, probability, sample_weight=row_weight, labels=[0, 1])
        )
        for rule, predicted in rules.items():
            values = binary_metrics(y, predicted, row_weight)
            for metric in selected_threshold_metrics:
                threshold_draws.setdefault((rule, metric), []).append(values[metric])

    rows: list[dict[str, Any]] = []
    for metric, draws in probability_draws.items():
        array = np.asarray(draws, dtype=float)
        rows.append(
            {
                "populacao": population,
                "regra_limiar": "probabilidade",
                "metrica": metric,
                "estimativa": point_probability[metric],
                "ic95_inferior": float(np.nanpercentile(array, 2.5)),
                "ic95_superior": float(np.nanpercentile(array, 97.5)),
                "bootstrap_reps_validas": int(np.isfinite(array).sum()),
                "unidade_reamostragem": "Paciente_grupo",
            }
        )
    for (rule, metric), draws in threshold_draws.items():
        array = np.asarray(draws, dtype=float)
        rows.append(
            {
                "populacao": population,
                "regra_limiar": rule,
                "metrica": metric,
                "estimativa": point_threshold[rule][metric],
                "ic95_inferior": float(np.nanpercentile(array, 2.5)),
                "ic95_superior": float(np.nanpercentile(array, 97.5)),
                "bootstrap_reps_validas": int(np.isfinite(array).sum()),
                "unidade_reamostragem": "Paciente_grupo",
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_auc_differences(
    predictions: pd.DataFrame, reps: int, seed: int, population: str
) -> pd.DataFrame:
    wide = predictions.pivot_table(
        index=["row_id", "Paciente_grupo", "Positivo"],
        columns=["braco", "modelo"],
        values="probabilidade",
        aggfunc="first",
    )
    configs = set(wide.columns)
    comparisons: list[tuple[tuple[str, str], tuple[str, str], str]] = []
    candidate_pairs = [
        (("UROT_GRAM_46", "elastic_net"), ("UROT_40", "elastic_net"), "UROT+Gram menos UROT - Elastic Net"),
        (("UROT_GRAM_46", "hist_gradient_boosting"), ("UROT_40", "hist_gradient_boosting"), "UROT+Gram menos UROT - boosting"),
        (("UROT_40", "elastic_net"), ("COMPACT_4", "elastic_net"), "UROT menos compacto - Elastic Net"),
        (("UROT_GRAM_46", "elastic_net"), ("COMPACT_4", "elastic_net"), "UROT+Gram menos compacto - Elastic Net"),
        (("UROT_40", "hist_gradient_boosting"), ("UROT_40", "elastic_net"), "boosting menos Elastic Net - UROT"),
        (("UROT_GRAM_46", "hist_gradient_boosting"), ("UROT_GRAM_46", "elastic_net"), "boosting menos Elastic Net - UROT+Gram"),
    ]
    for left, right, label in candidate_pairs:
        if left in configs and right in configs:
            comparisons.append((left, right, label))

    rows: list[dict[str, Any]] = []
    for comparison_index, (left, right, label) in enumerate(comparisons):
        subset = wide[[left, right]].dropna()
        index_frame = subset.index.to_frame(index=False)
        y = index_frame["Positivo"].to_numpy(dtype=int)
        groups, inverse = np.unique(
            index_frame["Paciente_grupo"].to_numpy(), return_inverse=True
        )
        p_left = subset[left].to_numpy(dtype=float)
        p_right = subset[right].to_numpy(dtype=float)
        point = roc_auc_score(y, p_left) - roc_auc_score(y, p_right)
        rng = np.random.default_rng(seed + comparison_index * 101)
        draws: list[float] = []
        for _ in range(reps):
            sampled = rng.integers(0, len(groups), size=len(groups))
            group_weight = np.bincount(sampled, minlength=len(groups)).astype(float)
            row_weight = group_weight[inverse]
            if np.sum(row_weight[y == 1]) == 0 or np.sum(row_weight[y == 0]) == 0:
                continue
            draws.append(
                roc_auc_score(y, p_left, sample_weight=row_weight)
                - roc_auc_score(y, p_right, sample_weight=row_weight)
            )
        array = np.asarray(draws, dtype=float)
        p_two_sided = min(
            1.0, 2 * min(float(np.mean(array <= 0)), float(np.mean(array >= 0)))
        )
        rows.append(
            {
                "populacao": population,
                "comparacao": label,
                "configuracao_esquerda": f"{left[0]} | {left[1]}",
                "configuracao_direita": f"{right[0]} | {right[1]}",
                "delta_auc": float(point),
                "ic95_inferior": float(np.percentile(array, 2.5)),
                "ic95_superior": float(np.percentile(array, 97.5)),
                "p_bootstrap_bilateral": float(p_two_sided),
                "bootstrap_reps_validas": len(array),
                "unidade_reamostragem": "Paciente_grupo",
            }
        )
    return pd.DataFrame(rows)


def decision_curve_table(
    predictions: pd.DataFrame, population: str
) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.01, 0.81, 0.01), 2)
    rows: list[dict[str, Any]] = []
    for (arm, model), group in predictions.groupby(["braco", "modelo"], sort=False):
        y = group["Positivo"].to_numpy(dtype=int)
        probability = group["probabilidade"].to_numpy(dtype=float)
        n = len(y)
        prevalence = y.mean()
        for threshold in thresholds:
            predicted = probability >= threshold
            tp = np.sum(predicted & (y == 1))
            fp = np.sum(predicted & (y == 0))
            odds = threshold / (1.0 - threshold)
            net_benefit = tp / n - fp / n * odds
            treat_all = prevalence - (1.0 - prevalence) * odds
            rows.extend(
                [
                    {"populacao": population, "braco": arm, "modelo": model, "estrategia": "modelo", "limiar": threshold, "beneficio_liquido": net_benefit},
                    {"populacao": population, "braco": arm, "modelo": model, "estrategia": "tratar_todos", "limiar": threshold, "beneficio_liquido": treat_all},
                    {"populacao": population, "braco": arm, "modelo": model, "estrategia": "tratar_nenhum", "limiar": threshold, "beneficio_liquido": 0.0},
                ]
            )
    return pd.DataFrame(rows)


def style_map() -> dict[tuple[str, str], tuple[str, str, str]]:
    return {
        ("COMPACT_4", "elastic_net"): (PALETTE["purple"], ":", "Compacto 4 - Elastic Net"),
        ("UROT_40", "elastic_net"): (PALETTE["navy"], "-", "UROT - Elastic Net"),
        ("UROT_40", "hist_gradient_boosting"): (PALETTE["orange"], "--", "UROT - boosting"),
        ("UROT_GRAM_46", "elastic_net"): (PALETTE["teal"], "-", "UROT+Gram - Elastic Net"),
        ("UROT_GRAM_46", "hist_gradient_boosting"): (PALETTE["red"], "--", "UROT+Gram - boosting"),
    }


def save_figure(figure: plt.Figure, stem: Path, dpi: int = 320) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=dpi)
    figure.savefig(stem.with_suffix(".svg"))
    plt.close(figure)


def plot_discrimination(predictions: pd.DataFrame, temporal: bool, stem: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for (arm, model), group in predictions.groupby(["braco", "modelo"], sort=False):
        color, line_style, label = style_map()[(arm, model)]
        y = group["Positivo"].to_numpy(dtype=int)
        probability = group["probabilidade"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        auc = roc_auc_score(y, probability)
        average_precision = average_precision_score(y, probability)
        axes[0].plot(fpr, tpr, color=color, linestyle=line_style, linewidth=2, label=f"{label} (AUC {auc:.3f})")
        axes[1].plot(recall, precision, color=color, linestyle=line_style, linewidth=2, label=f"{label} (AP {average_precision:.3f})")
    axes[0].plot([0, 1], [0, 1], color=PALETTE["muted"], linewidth=1, linestyle=":")
    axes[0].set(xlabel="1 - especificidade", ylabel="Sensibilidade", title="Curvas ROC")
    baseline = predictions.drop_duplicates("row_id")["Positivo"].mean()
    axes[1].axhline(baseline, color=PALETTE["muted"], linewidth=1, linestyle=":")
    axes[1].set(xlabel="Sensibilidade (recall)", ylabel="Valor preditivo positivo (precision)", title="Curvas precisão-recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(frameon=False, fontsize=7.5, loc="lower right" if axis is axes[0] else "upper right")
    figure.suptitle(
        "Validação temporal em 2024" if temporal else "Predições out-of-fold da nested CV (2011-2023)",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, stem)


def plot_calibration(predictions: pd.DataFrame, temporal: bool, stem: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6.4))
    axis.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle=":", label="Calibração perfeita")
    for (arm, model), group in predictions.groupby(["braco", "modelo"], sort=False):
        color, line_style, label = style_map()[(arm, model)]
        observed, predicted = calibration_curve(
            group["Positivo"], group["probabilidade"], n_bins=10, strategy="quantile"
        )
        axis.plot(predicted, observed, color=color, linestyle=line_style, marker="o", linewidth=1.8, markersize=4, label=label)
    axis.set(xlabel="Probabilidade predita média", ylabel="Frequência observada", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Calibração em 2024" if temporal else "Calibração out-of-fold (2011-2023)", fontweight="bold")
    axis.legend(frameon=False, fontsize=7.5, loc="upper left")
    figure.tight_layout()
    save_figure(figure, stem)


def plot_fold_auc(fold_probability: pd.DataFrame, stem: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 5.6))
    labels: list[str] = []
    groups: list[np.ndarray] = []
    colors: list[str] = []
    for arm, model in FULL_TASKS:
        subset = fold_probability.query("braco == @arm and modelo == @model")["roc_auc"].to_numpy()
        color, _, label = style_map()[(arm, model)]
        labels.append(label.replace(" - ", "\n"))
        groups.append(subset)
        colors.append(color)
    positions = np.arange(1, len(groups) + 1)
    for position, values, color in zip(positions, groups, colors):
        axis.scatter(np.repeat(position, len(values)), values, color=color, s=45, alpha=0.9, zorder=3)
        axis.errorbar(position, np.mean(values), yerr=np.std(values, ddof=1), color=PALETTE["ink"], fmt="D", markersize=5, capsize=5, zorder=4)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("AUC ROC no fold externo")
    axis.set_title("Estabilidade da discriminação entre folds externos", fontweight="bold")
    axis.set_ylim(max(0.5, min(map(np.min, groups)) - 0.02), min(1.0, max(map(np.max, groups)) + 0.02))
    figure.tight_layout()
    save_figure(figure, stem)


def plot_decision_curves(table: pd.DataFrame, population: str, stem: Path) -> None:
    subset = table[(table["populacao"] == population) & (table["limiar"] <= 0.50)]
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    baseline = subset[(subset["braco"] == "UROT_40") & (subset["modelo"] == "elastic_net")]
    for strategy, color, line_style, label in [
        ("tratar_todos", PALETTE["muted"], "--", "Cultivar todos"),
        ("tratar_nenhum", "#AAB6BC", ":", "Cultivar nenhum"),
    ]:
        part = baseline[baseline["estrategia"] == strategy]
        axis.plot(part["limiar"], part["beneficio_liquido"], color=color, linestyle=line_style, linewidth=1.5, label=label)
    for (arm, model), group in subset[subset["estrategia"] == "modelo"].groupby(["braco", "modelo"], sort=False):
        color, line_style, label = style_map()[(arm, model)]
        axis.plot(group["limiar"], group["beneficio_liquido"], color=color, linestyle=line_style, linewidth=2, label=label)
    axis.set(xlabel="Limiar de probabilidade", ylabel="Benefício líquido", xlim=(0.01, 0.50))
    axis.set_title("Curvas de decisão - " + ("2024" if population == "todos_2024" else "nested CV 2011-2023"), fontweight="bold")
    axis.legend(frameon=False, fontsize=7.5, ncol=2)
    figure.tight_layout()
    save_figure(figure, stem)


def build_report(
    data: pd.DataFrame,
    dev: pd.DataFrame,
    temporal: pd.DataFrame,
    fold_probability: pd.DataFrame,
    temporal_probability: pd.DataFrame,
    fold_summary: pd.DataFrame,
    delta_nested: pd.DataFrame,
    delta_temporal: pd.DataFrame,
    ci_table: pd.DataFrame,
) -> str:
    summary = (
        fold_summary.query("regra_limiar == 'probabilidade' and metrica in ['roc_auc','pr_auc','brier','calibration_intercept','calibration_slope']")
        .pivot_table(index=["braco", "modelo"], columns="metrica", values=["media_folds", "dp_folds"])
    )
    lines = [
        "# Nested cross-validation confirmatória - UROC/UROT/Gram",
        "",
        f"Gerado em {utc_now()}.",
        "",
        "## Síntese técnica",
        "",
        f"A coorte corrigida contém {len(data):,} pedidos após a exclusão de 46 registros com sexo indeterminado, considerados erros cadastrais. O desenvolvimento utilizou {len(dev):,} pedidos de 2011-2023; {len(temporal):,} pedidos de 2024 permaneceram fora de seleção, ajuste e definição de limiar.".replace(",", "."),
        "",
        "## Desempenho médio nos cinco folds externos",
        "",
        "| Braço | Modelo | AUC ROC, média ± DP | PR-AUC, média ± DP | Brier, média ± DP | Intercepto de calibração | Inclinação de calibração |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, model in FULL_TASKS:
        row = summary.loc[(arm, model)]
        lines.append(
            "| " + " | ".join(
                [
                    arm,
                    model,
                    f"{row[('media_folds','roc_auc')]:.4f} ± {row[('dp_folds','roc_auc')]:.4f}",
                    f"{row[('media_folds','pr_auc')]:.4f} ± {row[('dp_folds','pr_auc')]:.4f}",
                    f"{row[('media_folds','brier')]:.4f} ± {row[('dp_folds','brier')]:.4f}",
                    f"{row[('media_folds','calibration_intercept')]:.3f}",
                    f"{row[('media_folds','calibration_slope')]:.3f}",
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Validação temporal em 2024",
            "",
            "| Braço | Modelo | População | AUC ROC | PR-AUC | Brier | Intercepto | Inclinação |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in temporal_probability.iterrows():
        lines.append(
            f"| {row['braco']} | {row['modelo']} | {row['populacao']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['brier']:.4f} | {row['calibration_intercept']:.3f} | {row['calibration_slope']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Comparações pareadas de AUC",
            "",
            "As diferenças foram reamostradas por paciente, preservando a dependência entre pedidos do mesmo indivíduo. Valores positivos favorecem a configuração à esquerda.",
            "",
            "### Nested CV",
            "",
            delta_nested.to_markdown(index=False, floatfmt=".4f"),
            "",
            "### Teste temporal de 2024",
            "",
            delta_temporal.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Controles contra vazamento",
            "",
            "- divisão temporal definida antes da modelagem;",
            "- agrupamento por paciente nos ciclos externo e interno;",
            "- pré-processamento dentro do pipeline;",
            "- hiperparâmetros e limiares derivados somente no treino interno;",
            "- mesma atribuição de folds para todas as configurações;",
            "- modelo compacto relatado como secundário, pois suas quatro variáveis foram formuladas após exploração anterior da mesma base.",
            "",
            "## Limitações e interpretação",
            "",
            "A validação de 2024 é temporal no mesmo serviço, não validação geográfica externa. Pacientes já observados antes de 2024 e pacientes novos são apresentados separadamente. A escolha por AUC pode selecionar pesos de classe que preservam a discriminação, mas deslocam o nível absoluto das probabilidades; por isso calibração, Brier e curvas de decisão são interpretados separadamente. Os intervalos de confiança foram obtidos por bootstrap agrupado por paciente.",
            "",
            f"A tabela de intervalos contém {len(ci_table):,} estimativas/intervalos.".replace(",", "."),
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_results(
    config: RunConfig,
    data: pd.DataFrame,
    dev: pd.DataFrame,
    temporal: pd.DataFrame,
    run_hash: str,
) -> Path:
    assignment = pd.read_csv(OUT / "Atribuicao_Folds_Externos.csv.gz")
    (
        outer_meta,
        temporal_meta,
        outer_predictions,
        temporal_predictions,
        coefficients,
        cv_results,
    ) = load_all_checkpoints(dev, temporal, assignment, run_hash)

    outer_predictions.to_csv(
        OUT / "Predicoes_Outer_OOF.csv.gz", index=False, compression="gzip"
    )
    temporal_predictions.to_csv(
        OUT / "Predicoes_Temporais_2024.csv.gz", index=False, compression="gzip"
    )
    fold_probability, fold_threshold, fold_hyper = flatten_outer_meta(outer_meta)
    temporal_probability, temporal_threshold, temporal_hyper = flatten_temporal_meta(temporal_meta)
    fold_probability.to_csv(OUT / "Resultados_Probabilidade_Folds_Externos.csv", index=False)
    fold_threshold.to_csv(OUT / "Resultados_Limiares_Folds_Externos.csv", index=False)
    fold_hyper.to_csv(OUT / "Hiperparametros_Selecionados_Folds.csv", index=False)
    temporal_probability.to_csv(OUT / "Resultados_Temporais_2024_Probabilidade.csv", index=False)
    temporal_threshold.to_csv(OUT / "Resultados_Temporais_2024_Limiares.csv", index=False)
    temporal_hyper.to_csv(OUT / "Hiperparametros_Modelos_Finais.csv", index=False)
    cv_results.to_csv(OUT / "Resultados_Completos_Grid.csv.gz", index=False, compression="gzip")
    if not coefficients.empty:
        coefficients.to_csv(OUT / "Coeficientes_Elastic_Net.csv.gz", index=False, compression="gzip")
    fold_summary = metric_summary(fold_probability, fold_threshold)
    fold_summary.to_csv(OUT / "Resumo_Metricas_Nested_CV.csv", index=False)

    log(f"Bootstrap agrupado por paciente: {config.bootstrap_reps} réplicas")
    ci_frames: list[pd.DataFrame] = []
    for config_index, ((arm, model), group) in enumerate(
        outer_predictions.groupby(["braco", "modelo"], sort=False)
    ):
        frame = bootstrap_cluster_metrics(
            group,
            reps=config.bootstrap_reps,
            seed=config.seed + config_index * 17,
            population="nested_cv_oof_2011_2023",
        )
        frame.insert(0, "modelo", model)
        frame.insert(0, "braco", arm)
        ci_frames.append(frame)
    for config_index, ((arm, model), group) in enumerate(
        temporal_predictions.groupby(["braco", "modelo"], sort=False), start=100
    ):
        for population, subset in [
            ("todos_2024", group),
            ("pacientes_novos_2024", group[group["paciente_novo_em_2024"]]),
        ]:
            frame = bootstrap_cluster_metrics(
                subset,
                reps=config.bootstrap_reps,
                seed=config.seed + config_index * 17 + len(ci_frames),
                population=population,
            )
            frame.insert(0, "modelo", model)
            frame.insert(0, "braco", arm)
            ci_frames.append(frame)
    ci_table = pd.concat(ci_frames, ignore_index=True)
    ci_table.to_csv(OUT / "Intervalos_Confianca_Bootstrap_Paciente.csv", index=False)

    delta_nested = paired_bootstrap_auc_differences(
        outer_predictions,
        reps=config.bootstrap_reps,
        seed=config.seed + 7000,
        population="nested_cv_oof_2011_2023",
    )
    delta_temporal = paired_bootstrap_auc_differences(
        temporal_predictions,
        reps=config.bootstrap_reps,
        seed=config.seed + 8000,
        population="todos_2024",
    )
    delta_nested.to_csv(OUT / "Comparacoes_Pareadas_AUC_Nested_CV.csv", index=False)
    delta_temporal.to_csv(OUT / "Comparacoes_Pareadas_AUC_2024.csv", index=False)

    decision_nested = decision_curve_table(outer_predictions, "nested_cv_oof_2011_2023")
    decision_temporal = decision_curve_table(temporal_predictions, "todos_2024")
    decision = pd.concat([decision_nested, decision_temporal], ignore_index=True)
    decision.to_csv(OUT / "Analise_Curva_Decisao.csv.gz", index=False, compression="gzip")

    plot_discrimination(outer_predictions, False, FIGURES / "01_ROC_PR_Nested_CV")
    plot_discrimination(temporal_predictions, True, FIGURES / "02_ROC_PR_Temporal_2024")
    plot_calibration(outer_predictions, False, FIGURES / "03_Calibracao_Nested_CV")
    plot_calibration(temporal_predictions, True, FIGURES / "04_Calibracao_Temporal_2024")
    plot_fold_auc(fold_probability, FIGURES / "05_AUC_por_Fold_Externo")
    plot_decision_curves(decision, "nested_cv_oof_2011_2023", FIGURES / "06_Curva_Decisao_Nested_CV")
    plot_decision_curves(decision, "todos_2024", FIGURES / "07_Curva_Decisao_Temporal_2024")

    report = build_report(
        data,
        dev,
        temporal,
        fold_probability,
        temporal_probability,
        fold_summary,
        delta_nested,
        delta_temporal,
        ci_table,
    )
    (OUT / "Relatorio_Tecnico_Nested_CV.md").write_text(report, encoding="utf-8")
    shutil.copy2(__file__, OUT / "nested_cv_tese_v2.py")
    shutil.copy2(SOURCE, DATA_OUT / SOURCE.name)
    shutil.copy2(EXCLUSION_LEDGER, DATA_OUT / EXCLUSION_LEDGER.name)

    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "MANIFESTO_SHA256.csv":
            manifest_rows.append(
                {
                    "arquivo": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(OUT / "MANIFESTO_SHA256.csv", index=False)
    archive_base = ROOT / "outputs" / "Pacote_Nested_CV_Arvores_v3_2026-09-02"
    archive = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=OUT.parent,
            base_dir=OUT.name,
        )
    )
    log(f"ANÁLISE CONSOLIDADA: {archive}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["init", "outer", "temporal", "aggregate", "full"],
        required=True,
    )
    parser.add_argument("--task", type=parse_task)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()

    if args.phase in {"outer", "temporal"} and args.task is None:
        parser.error("--task é obrigatório para esta fase")
    if args.phase == "outer" and args.fold is None:
        parser.error("--fold é obrigatório para a fase outer")

    config = RunConfig(n_jobs=args.jobs, bootstrap_reps=args.bootstrap)
    data, dev, temporal, outer_splits, run_hash = prepare_run(config)
    if args.phase == "init":
        log("INICIALIZAÇÃO CONCLUÍDA")
        return
    if args.phase == "outer":
        arm, model = args.task
        train_idx, test_idx = outer_splits[args.fold - 1]
        fit_outer_fold(
            dev, arm, model, args.fold, train_idx, test_idx, config, run_hash
        )
        assignment = pd.read_csv(OUT / "Atribuicao_Folds_Externos.csv.gz")
        validate_outer_checkpoint(
            dev, assignment, arm, model, args.fold, run_hash
        )
        log("CHECKPOINT VALIDADO")
        return
    if args.phase == "temporal":
        arm, model = args.task
        fit_final_temporal(dev, temporal, arm, model, config, run_hash)
        log("CHECKPOINT TEMPORAL VALIDADO")
        return
    if args.phase == "aggregate":
        aggregate_results(config, data, dev, temporal, run_hash)
        return
    if args.phase == "full":
        for arm, model in FULL_TASKS:
            for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
                fit_outer_fold(
                    dev, arm, model, fold, train_idx, test_idx, config, run_hash
                )
            fit_final_temporal(dev, temporal, arm, model, config, run_hash)
        aggregate_results(config, data, dev, temporal, run_hash)


if __name__ == "__main__":
    main()
