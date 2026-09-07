#!/usr/bin/env python3
"""Construção da coorte final e análises de prevalência.

Unidade analítica: pedido/amostra (NumeroPedido).
Desfecho: positividade microbiológica da urocultura, não diagnóstico clínico de ITU.

O arquivo de entrada deve ser uma exportação já harmonizada com, no mínimo:
NumeroPedido, Registro, data_pedido, sexo, idade_anos, resultado_cultura,
elegivel_protocolo_final e motivo_exclusao_final. Dados estruturais ausentes não
são convertidos em zero. Sexo ausente ou indeterminado é tratado como erro
cadastral e excluído da coorte antes de qualquer análise.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


AGE_BINS = [-np.inf, 0.0767, 2, 6, 10, 20, 40, 60, 75, 90, np.inf]
AGE_LABELS = [
    "Recém-nascido", "Lactente", "Primeira infância", "Segunda infância",
    "Adolescente", "Adulto jovem", "Adulto maduro", "Idoso jovem",
    "Idoso médio/longevo", "Muito longevo",
]


def normalize_text(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.normalize("NFKD")
            .str.encode("ascii", errors="ignore").str.decode("ascii")
            .str.upper().str.strip())


def normalize_sex(s: pd.Series) -> pd.Series:
    """Harmoniza sexo cadastral; somente F e M são categorias elegíveis."""
    x = normalize_text(s)
    return x.replace({
        "FEMININO": "F", "FEM": "F", "FEMALE": "F",
        "MASCULINO": "M", "MASC": "M", "MALE": "M",
    })


def classify_culture(s: pd.Series) -> pd.Series:
    """Mapeia o laudo harmonizado para positivo/negativo; ambíguos ficam NA."""
    x = normalize_text(s)
    positive = x.str.contains(
        r"POSITIV|CRESCIMENTO|ISOLAD|UFC|CFU", regex=True, na=False
    ) & ~x.str.contains(r"SEM CRESCIMENTO|NAO HOUVE CRESCIMENTO", regex=True, na=False)
    negative = x.str.contains(
        r"NEGATIV|SEM CRESCIMENTO|NAO HOUVE CRESCIMENTO", regex=True, na=False
    )
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out.loc[negative] = 0
    out.loc[positive] = 1
    return out


def coalesce_order(group: pd.DataFrame) -> pd.Series:
    """Colapsa linhas redundantes de um mesmo pedido sem inventar valores."""
    row = group.sort_values("data_pedido").iloc[0].copy()
    for col in group.columns:
        vals = group[col].dropna().unique()
        if len(vals) == 1:
            row[col] = vals[0]
        elif col == "cultura_positiva" and len(vals):
            row[col] = int(np.max(vals))
        elif len(vals) > 1:
            row[col] = vals[-1]
    row["n_linhas_origem"] = len(group)
    return row


def build_final_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = raw.copy()
    d["data_pedido"] = pd.to_datetime(d["data_pedido"], errors="coerce")
    if "cultura_positiva" not in d:
        d["cultura_positiva"] = classify_culture(d["resultado_cultura"])

    n_raw = len(d)
    orders = (d.groupby("NumeroPedido", dropna=False, sort=False, group_keys=False)
              .apply(coalesce_order, include_groups=False).reset_index())
    n_orders = len(orders)

    eligible = orders["elegivel_protocolo_final"].fillna(False).astype(bool)
    interpretable = orders["cultura_positiva"].isin([0, 1])
    orders["sexo"] = normalize_sex(orders["sexo"])
    valid_sex = orders["sexo"].isin(["F", "M"])
    protocol_eligible = eligible & interpretable
    final_mask = protocol_eligible & valid_sex
    final = orders.loc[final_mask].copy()

    n_other_exclusions = int((~protocol_eligible).sum())
    n_invalid_sex = int((protocol_eligible & ~valid_sex).sum())
    n_total_exclusions = n_other_exclusions + n_invalid_sex

    audit = pd.DataFrame({
        "etapa": [
            "Linhas brutas de resultado",
            "Pedidos únicos após colapso de redundâncias",
            "Exclusões pelos demais critérios de elegibilidade",
            "Exclusões por sexo ausente/indeterminado (erro cadastral)",
            "Total de exclusões pelo protocolo final",
            "Coorte analítica final",
        ],
        "n": [n_raw, n_orders, n_other_exclusions, n_invalid_sex,
              n_total_exclusions, len(final)],
    })
    reasons = orders["motivo_exclusao_final"].astype("string").fillna(
        "Não especificado no campo de auditoria")
    reasons.loc[protocol_eligible & ~valid_sex] = (
        "Sexo ausente/indeterminado — erro cadastral")
    exclusions = (reasons.loc[~final_mask].value_counts(dropna=False)
                  .rename_axis("motivo").reset_index(name="n"))
    exclusions.to_csv("exclusoes_por_motivo.csv", index=False)
    return final, audit


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def prevalence_table(d: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in d.groupby(groups, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(g)
        k = int(g["cultura_positiva"].sum())
        lo, hi = wilson(k, n)
        rows.append(dict(zip(groups, keys)) | {
            "positivas": k, "total": n, "prevalencia": k / n,
            "ic95_inferior": lo, "ic95_superior": hi,
        })
    return pd.DataFrame(rows)


def odds_ratio_2x2(f_pos: int, f_neg: int, m_pos: int, m_neg: int):
    # Correção de Haldane-Anscombe apenas se alguma célula for zero.
    cells = np.array([f_pos, f_neg, m_pos, m_neg], dtype=float)
    if np.any(cells == 0):
        cells += 0.5
    a, b, c, d = cells
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(np.sum(1 / cells))
    z = log_or / se
    p = chi2.sf(z * z, 1)
    return np.exp(log_or), np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se), p


def sex_by_age_or(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for age, g in d[d["sexo"].isin(["F", "M"])].groupby("faixa_etaria", observed=True):
        f = g[g["sexo"] == "F"]["cultura_positiva"]
        m = g[g["sexo"] == "M"]["cultura_positiva"]
        est, lo, hi, p = odds_ratio_2x2(int(f.sum()), int((1-f).sum()),
                                        int(m.sum()), int((1-m).sum()))
        rows.append({"faixa_etaria": age, "OR_F_vs_M": est,
                     "IC95_inf": lo, "IC95_sup": hi, "p": p})
    out = pd.DataFrame(rows)
    out["p_BH"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def wald_interaction_from_model(d: pd.DataFrame) -> pd.DataFrame:
    """Teste de Wald conjunto da interação sexo×faixa etária."""
    import statsmodels.formula.api as smf
    x = d[d["sexo"].isin(["F", "M"])].copy()
    model = smf.logit("cultura_positiva ~ C(sexo) * C(faixa_etaria)", data=x).fit(disp=False)
    names = list(model.params.index)
    idx = [i for i, name in enumerate(names) if ":" in name]
    R = np.zeros((len(idx), len(names)))
    for r, c in enumerate(idx):
        R[r, c] = 1
    wt = model.wald_test(R, scalar=True)
    return pd.DataFrame({"chi2": [float(wt.statistic)], "gl": [len(idx)], "p": [float(wt.pvalue)]})


def first_order_sensitivity(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    d = d[d["sexo"].isin(["F", "M"])].copy()
    first = (d.sort_values(["Registro", "data_pedido", "NumeroPedido"])
             .drop_duplicates("Registro", keep="first"))
    full = prevalence_table(d, ["sexo", "faixa_etaria"]).rename(columns={"prevalencia":"prev_todos"})
    one = prevalence_table(first, ["sexo", "faixa_etaria"]).rename(columns={"prevalencia":"prev_primeiro"})
    out = full.merge(one[["sexo", "faixa_etaria", "prev_primeiro"]],
                     on=["sexo", "faixa_etaria"], how="left")
    out["diferenca_pp"] = 100 * (out["prev_todos"] - out["prev_primeiro"]).abs()
    return out, float(out["diferenca_pp"].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=Path("resultados_prevalencia"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    final, audit = build_final_cohort(raw)
    if not final["sexo"].isin(["F", "M"]).all():
        raise RuntimeError("A coorte final contém sexo ausente ou indeterminado.")
    final["idade_anos"] = pd.to_numeric(final["idade_anos"], errors="coerce")
    final["faixa_etaria"] = pd.cut(final["idade_anos"], AGE_BINS,
                                    labels=AGE_LABELS, right=False, ordered=True)

    audit.to_csv(args.out / "fluxo_coorte.csv", index=False)
    prevalence_table(final, ["sexo"]).to_csv(args.out / "prevalencia_sexo.csv", index=False)
    age_valid = final[final["idade_anos"].ge(0) & final["faixa_etaria"].notna()].copy()
    prevalence_table(age_valid, ["sexo", "faixa_etaria"]).to_csv(
        args.out / "prevalencia_sexo_idade.csv", index=False)
    sex_by_age_or(age_valid).to_csv(args.out / "or_sexo_por_idade.csv", index=False)
    wald_interaction_from_model(age_valid).to_csv(args.out / "wald_interacao.csv", index=False)
    sens, max_pp = first_order_sensitivity(age_valid)
    sens.to_csv(args.out / "sensibilidade_primeiro_pedido.csv", index=False)
    (args.out / "sensibilidade_resumo.txt").write_text(
        f"Maior diferença absoluta entre estratos: {max_pp:.2f} pontos percentuais\n",
        encoding="utf-8")
    final.to_parquet(args.out / "coorte_final.parquet", index=False)


if __name__ == "__main__":
    main()
