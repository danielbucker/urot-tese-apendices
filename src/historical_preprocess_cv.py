#!/usr/bin/env python3
"""PIPELINE HISTÓRICO — PRESERVADO PARA RASTREABILIDADE, NÃO USAR NA ANÁLISE FINAL.

Esta transcrição consolidada documenta a primeira estratégia exploratória usada
no projeto. Ela contém escolhas posteriormente consideradas inadequadas para a
inferência final: imputação e seleção de variáveis antes da validação, inclusão
de identificadores, KFold sem estratificação/agrupamento e particionamento sem
teste temporal. Para manter a regra cadastral da coorte, qualquer reexecução
deste artefato também exclui sexo ausente ou indeterminado. O pipeline
definitivo está em nested_temporal_validation.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Caminhos originalmente usados em ambiente Google Colab.
ROOT = Path("/content/drive/MyDrive/UFMG/TESE_DANIEL")
EUR = ROOT / "EUR_CONSOLIDADO.csv"
GRAM = ROOT / "GRAM_CONSOLIDADO.csv"
CULTURA = ROOT / "UROCULTURA_CONSOLIDADA.csv"
OUT = ROOT / "RESULTADOS_EXPLORATORIOS"


def normalize_columns(frame):
    frame = frame.copy()
    frame.columns = (frame.columns.str.strip().str.upper()
                     .str.normalize("NFKD")
                     .str.encode("ascii", errors="ignore").str.decode("ascii")
                     .str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_"))
    return frame


def read_source(path, sep=";"):
    return normalize_columns(pd.read_csv(path, sep=sep, low_memory=False,
                                         encoding="latin1", decimal=","))


def parse_date(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def normalize_sex(series):
    x = series.astype("string").str.upper().str.strip()
    return x.replace({"FEMININO": "F", "MASCULINO": "M", "FEM": "F", "MASC": "M"})


def normalize_result(series):
    x = (series.astype("string").str.upper().str.normalize("NFKD")
         .str.encode("ascii", errors="ignore").str.decode("ascii"))
    out = pd.Series(np.nan, index=x.index)
    out[x.str.contains("NEGATIV|SEM CRESCIMENTO", na=False)] = 0
    out[x.str.contains("POSITIV|CRESCIMENTO|ISOLADO", na=False)] = 1
    return out


def merge_exports():
    eur = read_source(EUR)
    gram = read_source(GRAM)
    culture = read_source(CULTURA)
    for frame in (eur, gram, culture):
        if "DATA_PEDIDO" in frame:
            frame["DATA_PEDIDO"] = parse_date(frame["DATA_PEDIDO"])
    data = culture.merge(eur, on="NUMEROPEDIDO", how="left", suffixes=("_CULT", "_EUR"))
    data = data.merge(gram, on="NUMEROPEDIDO", how="left", suffixes=("", "_GRAM"))
    data["TARGET"] = normalize_result(data["RESULTADO_CULTURA"])
    if "SEXO" not in data:
        raise ValueError("Campo SEXO ausente; não é possível aplicar a elegibilidade cadastral.")
    data["SEXO"] = normalize_sex(data["SEXO"])
    return data[data["TARGET"].notna() & data["SEXO"].isin(["F", "M"])].copy()


def historic_cleaning(data):
    data = data.copy()
    # Na etapa histórica, ausências foram preenchidas globalmente.
    numeric = data.select_dtypes(include=np.number).columns
    categorical = data.columns.difference(numeric)
    for col in numeric:
        data[col] = data[col].fillna(data[col].median())
    for col in categorical:
        mode = data[col].mode(dropna=True)
        data[col] = data[col].fillna(mode.iloc[0] if len(mode) else "AUSENTE")
    return data


def initial_predictors(data):
    # Identificadores foram mantidos nesta etapa histórica.
    candidates = [
        "REGISTRO", "NUMEROPEDIDO", "IDADE", "SEXO", "ORIGEM",
        "LEUCOCITOS", "HEMACIAS", "CELULAS_EPITELIAIS", "BACTERIAS",
        "ESTERASE", "NITRITO", "PH", "DENSIDADE", "ASPECTO",
        "GRAM_RESULTADO", "GRAM_QUANTIFICACAO", "ANO", "MES",
    ]
    return [c for c in candidates if c in data]


def transform_matrix(x):
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = x.columns.difference(numeric).tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return prep, numeric, categorical


def global_feature_selection(x, y, top_n=30):
    # ALERTA: ajuste em todos os dados antes da CV; preservado apenas para auditoria.
    prep, _, _ = transform_matrix(x)
    xt = prep.fit_transform(x)
    selector = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    selector.fit(xt, y)
    order = np.argsort(selector.feature_importances_)[::-1][:top_n]
    return xt[:, order], prep, order, selector.feature_importances_[order]


def evaluate_cv(model, x, y, name):
    # ALERTA: KFold não preserva paciente nem prevalência por dobra.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    p = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, p)
    print(f"{name}: AUC={auc:.4f}")
    return {"modelo": name, "auc": auc, "pred": p}


def logistic_no_interactions():
    return LogisticRegression(max_iter=3000, C=1.0)


def interaction_terms(x):
    # Interações exploratórias simples após transformação.
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    return poly.fit_transform(x), poly


def xgb_model():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )


def lgbm_model():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )


def run_model_comparison(x_selected, y):
    results = []
    results.append(evaluate_cv(logistic_no_interactions(), x_selected, y,
                               "Regressão logística sem interações"))
    x_inter, _ = interaction_terms(x_selected)
    results.append(evaluate_cv(LogisticRegression(max_iter=3000), x_inter, y,
                               "Regressão logística com interações"))
    results.append(evaluate_cv(RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1), x_selected, y,
        "Random Forest"))
    try:
        results.append(evaluate_cv(xgb_model(), x_selected, y, "XGBoost"))
    except ImportError:
        pass
    try:
        results.append(evaluate_cv(lgbm_model(), x_selected, y, "LightGBM"))
    except ImportError:
        pass
    return results


def old_holdout_artifact(x, y):
    # Holdout exploratório antigo; não constitui validação final.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y)
    model = xgb_model()
    model.fit(x_train, y_train)
    p = model.predict_proba(x_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "auc": roc_auc_score(y_test, p),
        "acuracia": accuracy_score(y_test, pred),
        "sensibilidade": tp / (tp + fn),
        "especificidade": tn / (tn + fp),
        "vpn": tn / (tn + fn),
        "relatorio": classification_report(y_test, pred, output_dict=True),
    }


def export_results(results, y, out):
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in results])
    summary.to_csv(out / "comparacao_modelos_historica.csv", index=False)
    for result in results:
        safe = result["modelo"].lower().replace(" ", "_")
        pd.DataFrame({"y": y, "probabilidade": result["pred"]}).to_csv(
            out / f"predicoes_{safe}.csv", index=False)


def main():
    data = merge_exports()
    data = historic_cleaning(data)
    predictors = initial_predictors(data)
    x, y = data[predictors], data["TARGET"].astype(int)
    x_selected, prep, order, importance = global_feature_selection(x, y, top_n=30)
    results = run_model_comparison(x_selected, y)
    export_results(results, y, OUT)
    try:
        holdout = old_holdout_artifact(x_selected, y)
        pd.Series({k: v for k, v in holdout.items() if k != "relatorio"}).to_csv(
            OUT / "holdout_xgb_historico.csv")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
