#!/usr/bin/env python3
"""Cálculo dos índices inflamatórios urinários e preparação do escore local.

Implementa a raiz da média dos quadrados proposta por Gu et al. (2020).
Campos estruturalmente ausentes permanecem NA; não são interpretados como zero.
Quando o campo sexo está presente, registros ausentes ou indeterminados são
excluídos como erros cadastrais antes do cálculo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


ORDINAL_COMMON = {
    "NEGATIVO": 0.0, "NEG": 0.0, "AUSENTE": 0.0,
    "+/-": 0.5, "±": 0.5, "TRACOS": 0.5, "TRACO": 0.5,
    "1+": 1.0, "+": 1.0, "2+": 2.0, "++": 2.0,
    "3+": 3.0, "+++": 3.0, "4+": 4.0, "++++": 4.0,
}

WBCC_MAP = {
    "NEGATIVO": 0.0, "AUSENTE": 0.0, "OCASIONAL": 0.5,
    "RARO": 1.0, "RAROS": 1.0, "POUCO": 2.0, "POUCOS": 2.0,
    "MEDIO": 3.0, "MODERADO": 3.0, "MASSA": 4.0, "MUITOS": 4.0,
}


def norm(x: pd.Series) -> pd.Series:
    return (x.astype("string").str.normalize("NFKD")
            .str.encode("ascii", errors="ignore").str.decode("ascii")
            .str.upper().str.strip())


def valid_registered_sex(x: pd.Series) -> pd.Series:
    sex = norm(x).replace({
        "FEMININO": "F", "FEM": "F", "FEMALE": "F",
        "MASCULINO": "M", "MASC": "M", "MALE": "M",
    })
    return sex.isin(["F", "M"])


def ordinal(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    x = norm(series)
    out = x.map(mapping | ORDINAL_COMMON).astype("Float64")
    numeric = pd.to_numeric(x.str.replace(",", ".", regex=False), errors="coerce")
    return out.fillna(numeric)


def wbc_hpf_to_ul(hpf: pd.Series) -> pd.Series:
    """Conversão local documentada: leucócitos/HPF × 3,15 = células/µL."""
    return pd.to_numeric(hpf, errors="coerce") * 3.15


def upper_limit_ratio_class(value: pd.Series, upper_reference: float) -> pd.Series:
    """Classes locais: 0; ≤5; ≤10; ≤20; ≤30; ≤40; ≤50; >50 vezes o VR."""
    x = pd.to_numeric(value, errors="coerce")
    ratio = x / upper_reference
    result = pd.Series(pd.NA, index=x.index, dtype="Float64")
    result.loc[x.eq(0)] = 0
    result.loc[ratio.gt(0) & ratio.le(5)] = 1
    result.loc[ratio.gt(5) & ratio.le(10)] = 2
    result.loc[ratio.gt(10) & ratio.le(20)] = 3
    result.loc[ratio.gt(20) & ratio.le(30)] = 4
    result.loc[ratio.gt(30) & ratio.le(40)] = 5
    result.loc[ratio.gt(40) & ratio.le(50)] = 6
    result.loc[ratio.gt(50)] = 7
    return result


def rms(*values: pd.Series) -> pd.Series:
    frame = pd.concat(values, axis=1)
    # A fórmula só é calculada quando todos os itens requeridos estão presentes.
    return np.sqrt(frame.pow(2).mean(axis=1, skipna=False)).astype("Float64")


def calculate_uii(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    a = ordinal(d["esterase"], ORDINAL_COMMON)
    b = ordinal(d["nitrito"], ORDINAL_COMMON)
    if "leucocitos_ul" in d:
        wbc_ul = pd.to_numeric(d["leucocitos_ul"], errors="coerce")
    else:
        wbc_ul = wbc_hpf_to_ul(d["leucocitos_hpf"])
    c = upper_limit_ratio_class(wbc_ul, 15.75)
    d_bacteria = ordinal(d["bacterias"], ORDINAL_COMMON)
    e = ordinal(d["aglomerados_leucocitarios"], WBCC_MAP)

    # O limite superior para células epiteliais deve ser informado no dado
    # harmonizado segundo sexo/método; 5/µL é o padrão quando aplicável.
    ec = pd.to_numeric(d["celulas_epiteliais_ul"], errors="coerce")
    ec_uln = pd.to_numeric(d.get("vr_sup_celulas_epiteliais", 5.0), errors="coerce")
    f = pd.Series(pd.NA, index=d.index, dtype="Float64")
    for upper in pd.Series(ec_uln, index=d.index).dropna().unique():
        mask = pd.Series(ec_uln, index=d.index).eq(upper)
        f.loc[mask] = upper_limit_ratio_class(ec.loc[mask], float(upper))

    d["leucocitos_ul_calculado"] = wbc_ul
    d["a_esterase"] = a
    d["b_nitrito"] = b
    d["c_leucocitos"] = c
    d["d_bacterias"] = d_bacteria
    d["e_aglomerados"] = e
    d["f_epiteliais"] = f
    d["UII1"] = rms(a, b)
    d["UII2"] = rms(a, b, c)
    d["UII3"] = rms(a, b, c, d_bacteria)
    d["UII4"] = rms(c, d_bacteria)
    d["UII5"] = rms(c, d_bacteria, e, f)
    d["UII6"] = rms(a, b, c, d_bacteria, e, f)
    return d


def local_score_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Preditores candidatos do escore logístico simplificado local."""
    out = pd.DataFrame(index=df.index)
    out["FLORA"] = ordinal(df["bacterias"], ORDINAL_COMMON)
    out["LEUCOCITOS"] = pd.to_numeric(
        df.get("leucocitos_ul_calculado", df.get("leucocitos_ul")), errors="coerce")
    out["ESTERASE"] = ordinal(df["esterase"], ORDINAL_COMMON)
    out["NITRITO"] = ordinal(df["nitrito"], ORDINAL_COMMON)
    return out


def points_from_logistic(coef: pd.Series, reference: float | None = None) -> pd.Series:
    """Transforma coeficientes em pontos; a escala deve ser validada internamente."""
    nonzero = coef.drop(labels=["const"], errors="ignore").abs()
    scale = reference or float(nonzero[nonzero.gt(0)].min())
    return (coef / scale).round().astype(int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    data = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    if "sexo" in data:
        data = data.loc[valid_registered_sex(data["sexo"])].copy()
    out = calculate_uii(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        out.to_parquet(args.output, index=False)
    else:
        out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
