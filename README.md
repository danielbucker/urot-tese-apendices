# Códigos-fonte da tese — Apêndices A–D

Pacote analítico completo referenciado na Seção 6.16 ("Reprodutibilidade e
códigos") e na Tabela 15 da tese de doutorado de Daniel Henrique Bücker
(Faculdade de Medicina, UFMG), sobre predição de positividade de urocultura
a partir de exame de urina rotina (EUR) e bacterioscopia pelo Gram.

Para a reprodução mínima e comentada dos braços finais UROT_40 e COMPACT_4,
veja o repositório separado
[`urot-compact-reproducao`](https://github.com/danielbucker/urot-compact-reproducao).
Este repositório aqui preserva o rastro metodológico integral, incluindo um
pipeline histórico descontinuado.

## Mapa dos arquivos

| Arquivo (`src/`) | Apêndice na tese | O que faz |
|---|---|---|
| `historical_preprocess_cv.py` | A | Pipeline exploratório inicial. **Preservado apenas para rastreabilidade — não usar na análise final.** Contém escolhas metodológicas posteriormente descartadas (imputação/seleção antes da validação, KFold sem agrupamento por paciente, sem teste temporal). |
| `final_cohort_prevalence.py` | B | Construção da coorte final e análises de prevalência. |
| `uii_and_score.py` | C | Cálculo dos índices inflamatórios urinários (UII1–UII6) e preparação do escore compacto local. |
| `nested_cv_tese_v2.py` | D | Núcleo do pipeline de validação agrupada, aninhada e temporal (Elastic Net e HistGradientBoosting). |
| `nested_cv_tree_extension_v3.py` | D (extensão) | Extensão do pipeline de validação para Random Forest, Extra Trees, XGBoost, LightGBM e CatBoost, com as mesmas atribuições de dobras e saídas auditáveis. |

`historical_preprocess_cv.py` e `nested_cv_tese_v2.py`/`nested_cv_tree_extension_v3.py`
foram extraídos/copiados a partir do texto integral em fonte monoespaçada
reproduzido nos apêndices da tese e do pacote analítico correspondente, para
manter os arquivos `.py` executáveis disponíveis separadamente do documento.

## Disponibilidade dos dados

Os dados de urocultura, EUR e Gram em nível de paciente **não estão incluídos
neste repositório**. São dado de saúde pseudonimizado, sujeito à LGPD e à
aprovação do Comitê de Ética em Pesquisa (CEP) que autorizou o estudo. Podem
ser disponibilizados mediante solicitação ao pesquisador responsável e
aprovação do CEP correspondente.

As saídas geradas pelos scripts (checkpoints, modelos treinados, predições,
figuras) também não estão incluídas por padrão — parte delas é derivada
diretamente dos dados de paciente.

## Ambiente

Testado com Python 3.12.14. Para reproduzir o ambiente exato:

```bash
conda create -n urot-tese -c conda-forge python=3.12 \
    numpy==2.5.3 pandas==3.0.5 scipy==1.18.0 scikit-learn==1.9.0 \
    statsmodels==0.15.0 matplotlib==3.11.1 joblib==1.6.0 \
    lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10
```

ou, com pip, em um ambiente Python 3.12 já ativo:

```bash
pip install -r requirements.txt
```

## Como executar

Os scripts em `src/` têm convenções de entrada diferentes:

- **`historical_preprocess_cv.py`** (Apêndice A): usa caminhos absolutos do
  Google Colab original (`/content/drive/...`). É preservado como está, por
  rastreabilidade — não roda sem editar os caminhos, e não deve ser usado
  para gerar resultados novos.
- **`final_cohort_prevalence.py`** (Apêndice B) e **`uii_and_score.py`**
  (Apêndice C): recebem caminho de entrada e saída por linha de comando:

  ```bash
  python src/final_cohort_prevalence.py caminho/para/entrada.csv --out resultados_prevalencia
  python src/uii_and_score.py caminho/para/entrada.csv caminho/para/saida.csv
  ```

- **`nested_cv_tese_v2.py`** e **`nested_cv_tree_extension_v3.py`**
  (Apêndice D): esperam a estrutura de pastas
  `<raiz>/data/Matriz_Estatistica_v6_150859x52.csv` e
  `<raiz>/data/Exclusoes_Sexo_Indeterminado_v6.csv`, com o script executado
  de dentro de `src/` (a raiz é resolvida como um nível acima do script).
  Coloque os arquivos de dados em `data/` na raiz deste repositório (ver
  seção acima) e rode:

  ```bash
  python src/nested_cv_tese_v2.py
  python src/nested_cv_tree_extension_v3.py
  ```

  As saídas são gravadas em `outputs/nested_cv_arvores_v3_2026_09_02/`
  (checkpoints, modelos, figuras, predições e relatório técnico).
