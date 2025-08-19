#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import logging
import warnings
from datetime import datetime, timedelta
from datetime import datetime as _dt
import unicodedata
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import dcc, html
from dash.dependencies import Input, Output
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
from plotly.figure_factory import create_dendrogram
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance as ssd
# opcional: plotly.express se você usa px em callbacks
import plotly.express as px  # noqa

load_dotenv()
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------------------
# LOGGING (Cloud Run envia stdout/stderr para Cloud Logging)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("app")

# ------------------------------------------------------------------------------
# PARÂMETROS GERAIS
# ------------------------------------------------------------------------------
START_FILTER = pd.Timestamp("2021-06-30", tz=None)
WINDOW       = 252
MINP         = 126
ROLL_MM      = 5
CLIP_RET     = 0.20
TRIM_Q       = 0.10
SCALE_FIX    = 10

# Outliers Sharpe
SH_LOW_Q     = 0.01
SH_HIGH_Q    = 0.01
MAD_K        = 6

# ------------------------------------------------------------------------------
# REQUESTS SESSION COM RETRIES (robusto em Cloud Run)
# ------------------------------------------------------------------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _build_session():
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s = requests.Session()
    s.headers.update({"User-Agent": "fi-mm-dash/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

HTTP = _build_session()
OFFLINE = os.getenv("OFFLINE", "0") == "1"  # força modo offline se quiser

# ------------------------------------------------------------------------------
# I/O HELPERS (seguros para Cloud Run; suportam local e GCS)
# ------------------------------------------------------------------------------
def _is_gcs_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("gs://")

def _read_parquet_safe(path: str, empty_cols=None):
    """
    Tenta ler parquet de:
      - GCS (se gcsfs disponível e path = gs://...)
      - local (image/container)
    Retorna DataFrame vazio em caso de erro, sem quebrar o start.
    """
    try:
        if not path:
            raise FileNotFoundError("caminho vazio")

        if _is_gcs_path(path):
            try:
                import gcsfs  # noqa: F401
            except Exception as e:
                log.warning("gcsfs não disponível para %s: %s", path, e)
                return pd.DataFrame(columns=empty_cols or [])

        if _is_gcs_path(path) or Path(path).exists():
            return pd.read_parquet(path)
        else:
            log.warning("Parquet não encontrado: %s", path)
    except Exception as e:
        log.warning("Falha lendo parquet %s: %s", path, e)
    return pd.DataFrame(columns=empty_cols or [])

def _read_excel_safe(path: str, parse_dates=None, index_col=None, sheet_name=0):
    """
    Lê Excel (local/GCS) de forma segura.
    - Usa gcsfs se for gs://
    - Tenta engine default; se falhar, retorna DF vazio
    """
    try:
        if not path:
            raise FileNotFoundError("caminho vazio")
        if _is_gcs_path(path):
            try:
                import gcsfs  # noqa: F401
            except Exception as e:
                log.warning("gcsfs não disponível para %s: %s", path, e)
                return pd.DataFrame()
        if _is_gcs_path(path) or Path(path).exists():
            return pd.read_excel(path, parse_dates=parse_dates, sheet_name=sheet_name)
        else:
            log.warning("Excel não encontrado: %s", path)
    except Exception as e:
        log.warning("Falha lendo excel %s: %s", path, e)
    return pd.DataFrame()

def _load_index_series(url: str, name: str, timeout=8) -> pd.Series:
    """Carrega série de índice (CDI/IHFA) com retries e fallback a série vazia."""
    if OFFLINE:
        log.info("OFFLINE=1 → pulando fetch de %s", name)
        return pd.Series(dtype="float64")
    try:
        r = HTTP.get(url, timeout=timeout)
        if r.status_code != 200:
            log.warning("%s HTTP %s em %s", name, r.status_code, url)
            return pd.Series(dtype="float64")
        dados = r.json()
        if isinstance(dados, dict) and "quotes" in dados:
            df_q = pd.DataFrame(dados["quotes"])
        else:
            df_q = pd.DataFrame(pd.DataFrame(dados)["quotes"].tolist())
        # datas em UTC → tira tz pra ficar alinhado com pandas naive
        df_q["date"]  = pd.to_datetime(df_q["d"], unit="ms", utc=True).dt.tz_localize(None)
        df_q["value"] = pd.to_numeric(df_q["c"], errors="coerce")
        s = df_q.set_index("date")["value"].sort_index().dropna()
        return s
    except Exception as e:
        log.warning("%s indisponível: %s", name, e)
        return pd.Series(dtype="float64")

# ------------------------------------------------------------------------------
# SÉRIES DE PREÇO E MAPPING (paths via env; default para diretório 'data/')
# ------------------------------------------------------------------------------
DFP_PATH      = os.getenv("DFP_PARQUET",      "data/df_p.parquet")
MAPPING_PATH  = os.getenv("MAPPING_PARQUET",  "data/mapping.parquet")
ATU_PATH      = os.getenv("ATUARIAL_XLSX",    "data/serie_historica_atuarial.xlsx")

df_p = _read_parquet_safe(DFP_PATH)
if not isinstance(df_p.index, pd.DatetimeIndex):
    # converte índice pra datetime; se falhar, substitui por índice vazio
    try:
        df_p.index = pd.to_datetime(df_p.index, utc=True).tz_localize(None)
    except Exception:
        df_p = pd.DataFrame(index=pd.to_datetime([]))

mapping_df = _read_parquet_safe(MAPPING_PATH, empty_cols=["Ativo", "Nome"])
mapping = (
    dict(zip(mapping_df.get("Ativo", []), mapping_df.get("Nome", [])))
    if not mapping_df.empty else {}
)

# lista base de fundos (se não houver mapping, usa todas as colunas disponíveis)
funds = [c for c in df_p.columns if c in mapping] if mapping else list(df_p.columns)

# ------------------------------------------------------------------------------
# BENCHMARKS (CDI/IHFA com fallback) + Meta Atuarial (planilha)
# ------------------------------------------------------------------------------
cdi_idx  = _load_index_series("https://api.maisretorno.com/v3/indexes/quotes/cdi",  "CDI")
ihfa_idx = _load_index_series("https://api.maisretorno.com/v3/indexes/quotes/ihfa", "IHFA")

df_atu = _read_excel_safe(ATU_PATH, parse_dates=["Data"])
if not df_atu.empty and "Data" in df_atu.columns:
    try:
        df_atu = df_atu.set_index(pd.to_datetime(df_atu["Data"], utc=True).dt.tz_localize(None)).sort_index()
        atu_series = pd.to_numeric(df_atu.get("Cota", pd.Series(dtype="float64")), errors="coerce").dropna()
        atu_series.index.name = "Data"
    except Exception as e:
        log.warning("Falha ao processar Meta Atuarial: %s", e)
        atu_series = pd.Series(dtype="float64")
else:
    log.warning("Meta Atuarial indisponível ou sem coluna 'Data'")
    atu_series = pd.Series(dtype="float64")

# ------------------------------------------------------------------------------
# CÁLCULOS GLOBAIS (sem filtros)
# ------------------------------------------------------------------------------
# Retornos diários dos fundos
df_ret_full = df_p.pct_change(fill_method=None) if not df_p.empty else pd.DataFrame(index=pd.to_datetime([]))

# CDI diário reindexado ao calendário dos fundos
if not cdi_idx.empty and not df_ret_full.empty:
    cdi_daily_idx = cdi_idx.reindex(df_ret_full.index, method="ffill")
    cdi_ret_daily = cdi_daily_idx.pct_change().fillna(0)
else:
    cdi_ret_daily = pd.Series(0.0, index=df_ret_full.index)

# Vol anualizada em %
if not df_ret_full.empty:
    df_vol_full = (
        df_ret_full
        .rolling(WINDOW, min_periods=MINP)
        .std()
        .replace(0, pd.NA)
        * (WINDOW ** 0.5) * 100
    )
else:
    df_vol_full = pd.DataFrame(index=pd.to_datetime([]))

# ------------------------------------------------------------------------------
# MORTALIDADE — caminho via env; funciona com gs:// ou local (Cloud Run-friendly)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# MORTALIDADE — caminho via env; funciona com gs:// ou local (Cloud Run-friendly)
# ------------------------------------------------------------------------------
MORT_PATH = os.getenv("MORTALIDADE_XLSX", "data/inicio_fim.xlsx")

def _strip_accents(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch))

def _norm(s):
    return _strip_accents(s).strip().lower()

def _pick_col(df, *aliases):
    norm = {_norm(c): c for c in df.columns}
    for a in aliases:
        k = _norm(a)
        if k in norm:
            return norm[k]
    for a in aliases:
        k = _norm(a)
        for nk, orig in norm.items():
            if nk.startswith(k):
                return orig
    raise KeyError(f"Não achei {aliases} em {MORT_PATH}")

def _digits(x):
    return re.sub(r"\D", "", "" if x is None else str(x))

def _parse_excel_date(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() in {"", "-", "nan", "NaT", "None"}):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, _dt, np.datetime64)):
        return pd.to_datetime(x, errors="coerce")
    dt = pd.to_datetime(str(x).strip(), dayfirst=True, errors="coerce")
    if pd.notna(dt):
        return dt
    num = pd.to_numeric(str(x).strip(), errors="coerce")
    if pd.notna(num):
        base = pd.Timestamp("1899-12-30")
        return base + pd.to_timedelta(int(num), unit="D")
    return pd.NaT

# --- leitura segura da planilha de mortalidade (local/gs://) ---
_raw = _read_excel_safe(MORT_PATH)

if _raw.empty or _raw.columns.empty:
    log.warning("Planilha de mortalidade vazia/ausente: %s", MORT_PATH)
    df_life_all = pd.DataFrame(columns=["Codigo", "CodigoFinal", "Nome", "DataInicio", "DataFim"])
    MORT_MIN = pd.Timestamp("2000-01-01")
    MORT_MAX = pd.Timestamp.today().normalize()
else:
    def _safe_pick(df, *aliases, required=False):
        try:
            return _pick_col(df, *aliases)
        except KeyError:
            if required:
                raise
            return None

    col_codigo = _safe_pick(_raw, "Código", "Codigo", required=True)
    col_fim    = _safe_pick(_raw, "Data Fim", "Fim", required=True)
    col_ini    = _safe_pick(_raw, "Data Início", "Data Inicio", "Inicio", required=False)

    col_ativo = None
    for c in _raw.columns:
        if _norm(c).startswith("ativo"):
            col_ativo = c
            break

    rename_map = {col_codigo: "Codigo", col_fim: "DataFim"}
    if col_ini is not None:
        rename_map[col_ini] = "DataInicio"
    if col_ativo is not None:
        rename_map[col_ativo] = "Ativo"

    df_life_all = _raw.rename(columns=rename_map).copy()

    for needed in ["Codigo", "DataFim", "DataInicio", "Ativo", "Nome"]:
        if needed not in df_life_all.columns:
            df_life_all[needed] = pd.NA

    df_life_all["Codigo"] = df_life_all["Codigo"].apply(_digits)
    df_life_all["Ativo"]  = df_life_all["Ativo"].apply(_digits)

    df_life_all["CodigoFinal"] = np.where(
        df_life_all["Codigo"].astype(str) != "", df_life_all["Codigo"], df_life_all["Ativo"]
    )

    df_life_all["DataInicio"] = df_life_all["DataInicio"].apply(_parse_excel_date)
    df_life_all["DataFim"]    = df_life_all["DataFim"].apply(_parse_excel_date)

    _dates_for_minmax = []
    if df_life_all["DataInicio"].notna().any():
        _dates_for_minmax.append(df_life_all["DataInicio"].dropna())
    if df_life_all["DataFim"].notna().any():
        _dates_for_minmax.append(df_life_all["DataFim"].dropna())

    if _dates_for_minmax:
        _all_dates = pd.concat(_dates_for_minmax, ignore_index=True)
        MORT_MIN = _all_dates.min().normalize()
        MORT_MAX = _all_dates.max().normalize()
    else:
        MORT_MIN = pd.Timestamp("2000-01-01")
        MORT_MAX = pd.Timestamp.today().normalize()

# (opcional) df_deaths se você usa em algum lugar
df_deaths = df_life_all[
    (df_life_all.get("CodigoFinal", "").astype(str) != "") & df_life_all["DataFim"].notna()
].copy()
if not df_deaths.empty:
    df_deaths["Ano"] = df_deaths["DataFim"].dt.year.astype("Int64")

# ======================
# Conjuntos de códigos por estilo (Macro / Long&Short)
# ======================
macro_codes = [
    "532673","465501","541427","581194","348341","285560",
    "396052","227552","323683","469025","419011","320153",
    "417890","413259","443808"
]
ls_codes = ["221260","573019","456748","541133","342092"]
codes    = macro_codes + ls_codes

macro_cols   = [c for c in df_ret_full.columns if any(c.startswith(code) for code in macro_codes)]
ls_cols      = [c for c in df_ret_full.columns if any(c.startswith(code) for code in ls_codes)]
macro_labels = [mapping.get(c, c) for c in macro_cols]
ls_labels    = [mapping.get(c, c) for c in ls_cols]

fund_codes   = [c for c in df_ret_full.columns if any(c.startswith(code) for code in codes)]
fund_options = [{"label": mapping.get(c, c), "value": c} for c in fund_codes]

# ======================
# mapping_all.parquet — leitura segura (local/GCS) + normalizações resilientes
# ======================
DF_ALL_PATH = os.getenv("MAPPING_ALL_PARQUET", "data/mapping_all.parquet")
_EXPECTED_COLS = [
    "CNPJ","Gestora|","Patrimônio|||em milhares","Média de|Cotistas|1 mês|em unidades",
    "Data do|Início da Série","Comp carteira|3m Antes Ult|em %|Ihfa",
    "Multigestor","Fundo|exclusivo","Forma de|condomínio","Investimento|no Exterior",
    "Restrito","Ativo"
]
df_all = _read_parquet_safe(DF_ALL_PATH, empty_cols=_EXPECTED_COLS)
if df_all.empty:
    log.warning("mapping_all.parquet vazio/ausente em %s — filtros MEG terão universo reduzido.", DF_ALL_PATH)

for col in _EXPECTED_COLS:
    if col not in df_all.columns:
        df_all[col] = pd.NA

# Normalizações (sem KeyError mesmo se col faltar)
df_all["CNPJ"] = (
    df_all.get("CNPJ", pd.Series(dtype=str))
          .astype(str)
          .str.replace(r"\D", "", regex=True)
          .str.zfill(14)
)
df_all["Gestora|"] = df_all.get("Gestora|", "").astype(str).str.strip()
df_all["Patrimônio|||em milhares"] = pd.to_numeric(
    df_all.get("Patrimônio|||em milhares", pd.Series(dtype="float64")), errors="coerce"
)
df_all["Média de|Cotistas|1 mês|em unidades"] = pd.to_numeric(
    df_all.get("Média de|Cotistas|1 mês|em unidades", pd.Series(dtype="float64")), errors="coerce"
)
df_all["Data do|Início da Série"] = pd.to_datetime(
    df_all.get("Data do|Início da Série", pd.Series(dtype="datetime64[ns]")), errors="coerce"
)
df_all["Comp carteira|3m Antes Ult|em %|Ihfa"] = pd.to_numeric(
    df_all.get("Comp carteira|3m Antes Ult|em %|Ihfa", pd.Series(dtype="float64")), errors="coerce"
)
df_all["Multigestor"]            = df_all.get("Multigestor", "").astype(str)
df_all["Fundo|exclusivo"]        = df_all.get("Fundo|exclusivo", "").astype(str)
df_all["Forma de|condomínio"]    = df_all.get("Forma de|condomínio", "").astype(str)
df_all["Investimento|no Exterior"]= df_all.get("Investimento|no Exterior", "").astype(str)
df_all["Restrito"]               = df_all.get("Restrito", "").astype(str)
df_all["Ativo"]                  = df_all.get("Ativo", "").astype(str).str.replace(r"\D", "", regex=True)

# IHFA flag
has_ihfa_mask = df_all["Comp carteira|3m Antes Ult|em %|Ihfa"].notna()
gestoras_com_ihfa = df_all.loc[has_ihfa_mask, "Gestora|"].dropna().unique()
df_all["Gestora com histórico IHFA"] = df_all["Gestora|"].apply(
    lambda g: "SIM" if g in set(gestoras_com_ihfa) else "NÃO"
)

# ======================
# Parâmetros de atuação e janela fixa documentais (mantidos)
# ======================
BASE_ATUACAO = pd.Timestamp("2025-06-30")
ANOS_ATUACAO_MIN = 4
data_corte = BASE_ATUACAO - pd.DateOffset(years=ANOS_ATUACAO_MIN)  # 2021-06-30

WINDOW48_START = pd.Timestamp("2021-06-30")
WINDOW48_END   = pd.Timestamp("2025-06-30")
REBAL_DATES    = pd.to_datetime([
    "2021-06-30", "2022-06-30", "2023-06-30", "2024-06-30", "2025-06-30"
])

# ======================
# Filtros MEG (sem KeyError) + df_pl robusto
# ======================
filter_funcs = {
    'patrimonio': lambda d: pd.to_numeric(d.get('Patrimônio|||em milhares'), errors='coerce') > 300_000,
    'atuacao':     lambda d: pd.to_datetime(d.get('Data do|Início da Série'), errors='coerce') <= data_corte,
    'cotistas':    lambda d: pd.to_numeric(d.get('Média de|Cotistas|1 mês|em unidades'), errors='coerce') > 100,
    'multigestor': lambda d: ~d.get('Multigestor', pd.Series("", index=d.index)).isin(['Multigestor', 'Espelho']),
    'exclusivo':   lambda d: d.get('Fundo|exclusivo', pd.Series("", index=d.index)) != 'Sim',
    'condominio':  lambda d: d.get('Forma de|condomínio', pd.Series("", index=d.index)) != 'Fechado',
    'exterior':    lambda d: ~d.get('Investimento|no Exterior', pd.Series("", index=d.index)).isin(['', 'Até 100%']),
    'restrito':    lambda d: d.get('Restrito', pd.Series("", index=d.index)) != 'Sim',
    'ihfa':        lambda d: d.get('Gestora com histórico IHFA', pd.Series("", index=d.index)) == 'SIM',
}

if "Ativo" in df_all.columns:
    df_pl = pd.to_numeric(df_all.set_index("Ativo").get("Patrimônio|||em milhares"), errors="coerce").fillna(0.0) * 1000.0
else:
    df_pl = pd.Series(dtype="float64")


# ======================
# FUNÇÕES AUXILIARES
# ======================
def trim_mean_series(row, q=TRIM_Q):
    vals = row.dropna().sort_values()
    n = len(vals)
    if n == 0:
        return np.nan
    k = int(n * q)
    if 2 * k >= n:
        return vals.mean()
    return vals.iloc[k:n-k].mean()

def robust_cross_section_mean(df_values):
    clipped = df_values.clip(lower=-CLIP_RET, upper=CLIP_RET)
    return clipped.apply(trim_mean_series, axis=1)


def get_allowed_activos_at(ref_date: pd.Timestamp, selected_filters):
    """
    Aplica os filtros do MEG NA DATA DE REBALANCE (ref_date).
    Somente a condição de 'atuacao' muda com a data (4 anos antes).
    Retorna apenas códigos existentes em df_p (com séries).
    """
    d = df_all.copy()
    selected_filters = selected_filters or []

    for f in selected_filters:
        if f == 'patrimonio':
            d = d[d['Patrimônio|||em milhares'] > 300_000]
        elif f == 'atuacao':
            cutoff = ref_date - pd.DateOffset(years=ANOS_ATUACAO_MIN)
            d = d[d['Data do|Início da Série'] <= cutoff]
        elif f == 'cotistas':
            d = d[d['Média de|Cotistas|1 mês|em unidades'] > 100]
        elif f == 'multigestor':
            d = d[~d['Multigestor'].isin(['Multigestor', 'Espelho'])]
        elif f == 'exclusivo':
            d = d[d['Fundo|exclusivo'] != 'Sim']
        elif f == 'condominio':
            d = d[d['Forma de|condomínio'] != 'Fechado']
        elif f == 'exterior':
            d = d[~d['Investimento|no Exterior'].isin(['', 'Até 100%'])]
        elif f == 'restrito':
            d = d[d['Restrito'] != 'Sim']
        elif f == 'ihfa':
            d = d[d['Gestora com histórico IHFA'] == 'SIM']

    ativos = d['Ativo'].unique().tolist()
    return [a for a in ativos if a in df_p.columns]


def _ind_xs_series(metric: str, cols: list[str]) -> pd.Series:
    """
    Série diária cross-section para a indústria (sem encadear períodos).
    - 'ret': robust_cross_section_mean dos retornos diários
    - 'vol': média aparada da vol diária (%)
    - 'sharpe': média aparada do Sharpe diário
    Em todos os casos, aplica suavização ROLL_MM.
    """
    if not cols:
        return pd.Series(dtype='float64')

    if metric == 'ret':
        arr = df_ret_full[cols]
        xs = robust_cross_section_mean(arr)
    elif metric == 'vol':
        arr = df_vol_full[cols]
        xs = arr.apply(trim_mean_series, axis=1)
    else:  # 'sharpe'
        arr = sharpe_full[cols]
        xs = arr.apply(trim_mean_series, axis=1)

    xs = xs.rolling(ROLL_MM, min_periods=1).mean()
    return xs.dropna()


def industry_rebalanced_series(metric: str, selected_filters, sd, ed):
    """
    Constrói a série de MÉDIA DA INDÚSTRIA com rebalanceamento ANUAL:
      - Janela fixa: WINDOW48_START → WINDOW48_END (48 meses)
      - Cohort fixo entre datas de rebalance (30/06 de cada ano)
      - Filtros MEG aplicados NA DATA DO REBALANCE
    Retorna: (serie, dict_contagem_por_ano, sd_efetivo, ed_efetivo)
    """
    sd = pd.to_datetime(sd); ed = pd.to_datetime(ed)
    sd = max(sd, WINDOW48_START); ed = min(ed, WINDOW48_END)
    if sd >= ed:
        return pd.Series(dtype='float64'), {}, sd, ed

    seg_series = []
    counts_by_year = {}

    # percorre os blocos entre rebalances
    for i in range(len(REBAL_DATES) - 1):
        r0, r1 = REBAL_DATES[i], REBAL_DATES[i + 1]  # [r0, r1]
        seg_start = max(sd, r0)
        seg_end   = min(ed, r1)
        if seg_start >= seg_end:
            continue

        cohort_all = get_allowed_activos_at(r0, selected_filters)

        # mantém apenas fundos com dados no período do segmento
        if metric == 'ret':
            cols = [c for c in cohort_all if not df_ret_full[c].loc[seg_start:seg_end].dropna().empty]
        elif metric == 'vol':
            cols = [c for c in cohort_all if not df_vol_full[c].loc[seg_start:seg_end].dropna().empty]
        else:
            cols = [c for c in cohort_all if not sharpe_full[c].loc[seg_start:seg_end].dropna().empty]

        if not cols:
            continue

        counts_by_year[r0.year] = len(cols)

        s = _ind_xs_series(metric, cols).loc[seg_start:seg_end]
        if not s.empty:
            seg_series.append(s)

    if not seg_series:
        return pd.Series(dtype='float64'), counts_by_year, sd, ed

    s = pd.concat(seg_series).sort_index()

    if metric == 'ret':
        s = (1 + s).cumprod()
        s = (s / s.iloc[0]) * 100.0

    return s, counts_by_year, sd, ed



def winsorize_series(s, q_low=SH_LOW_Q, q_high=SH_HIGH_Q):
    if s.dropna().empty:
        return s
    ql, qh = s.quantile([q_low, 1 - q_high])
    return s.clip(lower=ql, upper=qh)

def mad_clip_series(s, k=MAD_K):
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return s
    return s.clip(lower=med - k * mad, upper=med + k * mad)

def safe_winsor_mad(col):
    s = col.dropna()
    if len(s) < 30:
        return col
    s = winsorize_series(s)
    s = mad_clip_series(s)
    return col.combine_first(s)

def rebase100(series):
    if series is None or len(series) == 0:
        return series
    first = series.iloc[0]
    if pd.isna(first) or first == 0:
        return series
    return series / first * 100

# lista de fundos base (sem filtros)
funds = [f for f in df_p.columns if f in mapping]

def get_allowed_activos(selected_filters):
    # se não há filtros selecionados, retorna todos os fundos
    if not selected_filters:
        return funds
    df_f = df_all.copy()
    for f in selected_filters:
        df_f = df_f[filter_funcs[f](df_f)]
    ativos = df_f['Ativo'].unique().tolist()
    # garante só códigos que existam em df_p
    return [a for a in ativos if a in df_p.columns]

def start_at_first_valid(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    idx = s.first_valid_index()
    return s.loc[idx:] if idx is not None else s



def base100_from(s: pd.Series) -> pd.Series:
    # rebase em 100 a partir da 1ª observação válida
    s = s.dropna()
    if s.empty:
        return s
    return (s / s.iloc[0]) * 100

def _annot_fig(title, msg, height=350):
    fig = go.Figure()
    fig.update_layout(template='plotly_white', title=title, height=height)
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=14))
    return fig
# --- helper: pega o 1º valor válido e rebasa em 100 ---
def base100_series(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return (s / s.iloc[0]) * 100

def _empty_fig(title, msg, height=350):
    return _annot_fig(title, msg, height)



# ---------- Fonte única para mortalidade: df_life_all ----------
_life_df = globals().get("df_life_all", None)
if _life_df is None or not isinstance(_life_df, pd.DataFrame):
    _life_df = pd.DataFrame(columns=["Codigo", "CodigoFinal", "Nome"])

# coluna de código final (numérica)
_code_col = "CodigoFinal" if "CodigoFinal" in _life_df.columns else ("Codigo" if "Codigo" in _life_df.columns else None)

# mapping pode não existir
_mapping = globals().get("mapping", {}) or {}

def label_for_code(code: str) -> str:
    code = str(code)
    # 1) nome pelo mapping (prefixo do Ativo começa com o código)
    nm = next((v for k, v in _mapping.items() if str(k).startswith(code)), None)
    if nm:
        return nm
    # 2) nome vindo da planilha
    if _code_col and "Nome" in _life_df.columns:
        row = _life_df.loc[_life_df[_code_col].astype(str) == code, "Nome"]
        if not row.empty:
            val = str(row.iloc[0]).strip()
            if val:
                return val
    # 3) fallback
    return code

# códigos únicos (ordenados por label)
if _code_col:
    codes = (
        _life_df[_code_col]
        .astype(str).str.strip()
        .replace("", np.nan).dropna()
        .drop_duplicates()
        .tolist()
    )
    codes.sort(key=lambda c: (label_for_code(c) or "").lower())
else:
    codes = []

fund_opts = [{"label": label_for_code(c), "value": c} for c in codes]
default_values = [codes[0]] if codes else []
default_code   = default_values[0] if default_values else None  # se algum ponto do código ainda espera escalar





excess_ret   = df_ret_full.sub(cdi_ret_daily, axis=0)
roll_mean_e  = excess_ret.rolling(WINDOW, min_periods=MINP).mean()
roll_std     = df_ret_full.rolling(WINDOW, min_periods=MINP).std().replace(0, np.nan)
sharpe_full  = (roll_mean_e / roll_std).apply(safe_winsor_mad).replace([np.inf, -np.inf], np.nan)

df_ret   = df_ret_full[df_ret_full.index >= START_FILTER]
df_vol   = df_vol_full[df_vol_full.index >= START_FILTER]
df_sh    = sharpe_full[sharpe_full.index >= START_FILTER]

ret_cs_mean = robust_cross_section_mean(df_ret)
ret_cs_mm   = ret_cs_mean.rolling(ROLL_MM, min_periods=1).mean()
ret_ind     = rebase100((1 + ret_cs_mm).cumprod())

vol_tmp     = df_vol / SCALE_FIX
vol_cs_mean = vol_tmp.apply(trim_mean_series, axis=1)
vol_ind     = vol_cs_mean.rolling(ROLL_MM, min_periods=1).mean()

sh_cs_mean  = df_sh.apply(trim_mean_series, axis=1)
sh_ind      = sh_cs_mean.rolling(ROLL_MM, min_periods=1).mean()

ativos_por_ano = (
    df_p.stack(dropna=True)
        .reset_index(level=1)
        .rename(columns={"level_1": "Ativo", 0: "Preco"})
        .assign(Ano=lambda d: d.index.year)
        .groupby("Ano")["Ativo"].nunique()
)

# ======================
# UI: dropdowns e filtros
# ======================
metricas = [
    {'label': 'Retorno Acumulado', 'value': 'retorno'},
    {'label': 'Volatilidade',      'value': 'volatilidade'},
    {'label': 'Sharpe',            'value': 'sharpe'}
]

filter_options = [
    {'label': 'Patrimônio > 300 mi', 'value': 'patrimonio'},
    {'label': 'Atuação > 4 anos',    'value': 'atuacao'},
    {'label': 'Cotistas > 100',      'value': 'cotistas'},
    {'label': 'Sem multigestor/espelho', 'value': 'multigestor'},
    {'label': 'Fundo exclusivo ≠ Sim',   'value': 'exclusivo'},
    {'label': 'Condomínio aberto',       'value': 'condominio'},
    {'label': "Investimento no Exterior ‘Até 40%%’ excluído", 'value': 'exterior'},
    {'label': 'Restritos ≠ Sim',         'value': 'restrito'},
    {'label': 'Gestora com histórico IHFA', 'value': 'ihfa'},
]

app = dash.Dash(__name__, suppress_callback_exceptions=True)


from datetime import datetime, timedelta
from dash import html, dcc

def serve_layout():
    """
    Layout seguro: lida com ausências de dados/arquivos, normaliza limites de datas
    e evita NameError/KeyError vindo de variáveis globais.
    """
    # Datas-base para seletores
    today = datetime.today().date()
    ten_years_ago = today - timedelta(days=3650)

    # ─────────────────────────────────────────────────────────────────────────────
    # Bases globais com fallback seguro
    # ─────────────────────────────────────────────────────────────────────────────
    _life_df   = globals().get("df_life_all")
    _mapping   = globals().get("mapping") or {}
    _df_ret    = globals().get("df_ret_full", pd.DataFrame(index=pd.to_datetime([])))
    _ret_ind   = globals().get("ret_ind", pd.Series(dtype="float64"))
    _cdi_idx   = globals().get("cdi_idx", pd.Series(dtype="float64"))
    funds_local = list(globals().get("funds") or [])
    filter_opts = list(globals().get("filter_options") or [])

    # Mortalidade: dataframe mínimo se não houver
    if not isinstance(_life_df, pd.DataFrame):
        _life_df = pd.DataFrame(columns=["Codigo", "CodigoFinal", "Nome", "DataInicio", "DataFim"])

    # Coluna de código (preferir CodigoFinal)
    _code_col = "CodigoFinal" if "CodigoFinal" in _life_df.columns else ("Codigo" if "Codigo" in _life_df.columns else None)

    # Helper p/ rótulo: tenta mapping; se não, usa Nome da planilha; senão, o próprio código
    def _label_for_code(code: str) -> str:
        code = str(code)
        nm = next((v for k, v in _mapping.items() if str(k).startswith(code)), None)
        if nm:
            return nm
        if _code_col and "Nome" in _life_df.columns:
            row = _life_df.loc[_life_df[_code_col].astype(str) == code, "Nome"]
            if not row.empty:
                val = str(row.iloc[0]).strip()
                if val:
                    return val
        return code

    # Opções do dropdown de mortalidade
    if _code_col and not _life_df.empty:
        all_codes = (
            _life_df[_code_col]
            .astype(str).str.strip()
            .replace("", pd.NA).dropna().unique().tolist()
        )
        all_codes.sort(key=lambda c: (_label_for_code(c) or "").lower())
        fund_opts = [{"label": _label_for_code(c), "value": c} for c in all_codes]
        default_values = [all_codes[0]] if all_codes else []
    else:
        fund_opts, default_values = [], []

    # Limites do seletor de mortalidade (recalcula se não existirem/invalidos)
    _MORT_MIN = globals().get("MORT_MIN")
    _MORT_MAX = globals().get("MORT_MAX")

    def _recompute_mort_bounds():
        _dates = []
        if "DataInicio" in _life_df.columns and _life_df["DataInicio"].notna().any():
            _dates.append(_life_df["DataInicio"].dropna())
        if "DataFim" in _life_df.columns and _life_df["DataFim"].notna().any():
            _dates.append(_life_df["DataFim"].dropna())
        if _dates:
            _all = pd.concat(_dates, ignore_index=True)
            return _all.min().normalize(), _all.max().normalize()
        return pd.Timestamp("2000-01-01"), pd.Timestamp.today().normalize()

    if not isinstance(_MORT_MIN, pd.Timestamp) or not isinstance(_MORT_MAX, pd.Timestamp) or pd.isna(_MORT_MIN) or pd.isna(_MORT_MAX) or (_MORT_MIN > _MORT_MAX):
        _MORT_MIN, _MORT_MAX = _recompute_mort_bounds()

    # --------- helpers de datas seguras ----------
    def _idx_bounds_from(series_like, fallback_min, fallback_max):
        """
        Retorna (start_date, end_date) como objetos date, usando fallbacks se necessário.
        Aceita Series/DataFrame indexados por DatetimeIndex.
        """
        try:
            if isinstance(series_like, (pd.Series, pd.DataFrame)) and not series_like.empty:
                smin = series_like.index.min()
                smax = series_like.index.max()
                if pd.notna(smin) and pd.notna(smax):
                    return smin.date(), smax.date()
            # fallback
            return pd.to_datetime(fallback_min).date(), pd.to_datetime(fallback_max).date()
        except Exception:
            return pd.to_datetime(fallback_min).date(), pd.to_datetime(fallback_max).date()

    # bounds baseados no df_ret_full (fallback geral)
    base_min = _df_ret.index.min() if not _df_ret.empty else pd.Timestamp("2010-01-01")
    base_max = _df_ret.index.max() if not _df_ret.empty else pd.Timestamp.today()

    # DatePicker: CDI/IHFA/Meta (limitar a 10 anos)
    cdi_start_all, cdi_end_all = _idx_bounds_from(_cdi_idx, base_min, base_max)
    cdi_start_default = max(pd.to_datetime(cdi_start_all).date(), ten_years_ago)
    cdi_end_default   = min(pd.to_datetime(cdi_end_all).date(), today)
    if cdi_start_default > cdi_end_default:
        # fallback coerente
        cdi_start_default, cdi_end_default = cdi_start_all, cdi_end_all

    # Indústria (usar ret_ind com fallback para df_ret_full)
    ind_start_all, ind_end_all = _idx_bounds_from(
        _ret_ind if isinstance(_ret_ind, (pd.Series, pd.DataFrame)) and not getattr(_ret_ind, "empty", True) else _df_ret,
        base_min, base_max
    )

    # DatePickers que usam df_ret_full direto (com fallback)
    df_start_all, df_end_all = _idx_bounds_from(_df_ret, base_min, base_max)

    # Códigos de classes (podem não existir, então fallback)
    macro_codes = list(globals().get("macro_codes") or [])
    ls_codes    = list(globals().get("ls_codes") or [])
    codes       = list(globals().get("codes") or (macro_codes + ls_codes))

    # Fundos/códigos do PCA (podem não existir)
    fund_options = list(globals().get("fund_options") or [])
    fund_codes   = list(globals().get("fund_codes") or [])

    # ─────────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────────
    return html.Div(
        [
            html.H1("Painel FI Multimercado", style={"textAlign": "center"}),

            # Filtros MEG 143
            html.H2("Filtros MEG 143"),
            dcc.Checklist(
                id="meg-filters",
                options=filter_opts,
                value=[],
                inputStyle={"marginRight": "5px"}
            ),

            # Benchmarks: CDI, IHFA & Meta Atuarial
            html.H2("Benchmarks: CDI, IHFA & Meta Atuarial"),
            dcc.Checklist(
                id="benchmarks-selector",
                options=[
                    {"label": "CDI", "value": "cdi"},
                    {"label": "IHFA", "value": "ihfa"},
                    {"label": "Meta Atuarial", "value": "atu"},
                ],
                value=["cdi", "ihfa", "atu"],
                inputStyle={"marginRight": "5px"}
            ),

            # Índice Acumulado (Base 100 no range)
            html.H2("Índice Acumulado (Base 100 no range)"),
            dcc.DatePickerRange(
                id="cdi-date-picker",
                start_date=cdi_start_default,
                end_date=cdi_end_default,
                min_date_allowed=cdi_start_all,
                max_date_allowed=min(today, cdi_end_all),
                display_format="YYYY-MM-DD"
            ),
            dcc.Graph(id="cdi-chart", style={"marginTop": "20px"}),

            # ─────────────────────────────────────────────
            # Mortalidade (ANUAL): Mortes/Ativos + bolhas Início/Fim
            # ─────────────────────────────────────────────
            html.H2("Mortalidade (Anual) — Mortes / Total de Fundos"),
            dcc.DatePickerRange(
                id="mort-date-picker",
                start_date=_MORT_MIN.date(),
                end_date=_MORT_MAX.date(),
                min_date_allowed=_MORT_MIN.date(),
                max_date_allowed=_MORT_MAX.date(),
                display_format="YYYY-MM-DD",
                style={"marginTop": "4px"}
            ),
            dcc.Dropdown(
                id="mort-fund-select",
                options=fund_opts,
                value=default_values,
                multi=True,
                placeholder="Selecione um ou mais fundos…",
                clearable=True,
                style={"width": "50%", "marginTop": "10px"}
            ),
            dcc.Graph(id="mortality-chart", style={"marginTop": "14px"}),

            # Médias da Indústria (Robustas, Base 100 no range)
            html.H2("Médias da Indústria (Robustas, Base 100 no range)"),
            dcc.DatePickerRange(
                id="ind-date-picker",
                start_date=ind_start_all,
                end_date=ind_end_all,
                min_date_allowed=ind_start_all,
                max_date_allowed=ind_end_all,
                display_format="YYYY-MM-DD"
            ),
            dcc.RadioItems(
                id="ind-metrics-selector",
                options=[
                    {"label": "Retorno Acumulado (Base 100)", "value": "ret"},
                    {"label": "Volatilidade", "value": "vol"},
                    {"label": "Sharpe", "value": "sharpe"},
                ],
                value="ret",
                inline=True,
                style={"marginTop": "10px"},
                inputStyle={"marginRight": "6px"}
            ),
            dcc.Dropdown(
                id="ind-funds-compare",
                options=[{"label": _mapping.get(a, a), "value": a} for a in funds_local],
                value=[],
                multi=True,
                placeholder="Selecione fundos para comparar com a média da indústria",
                style={"width": "60%", "marginTop": "10px"}
            ),
            dcc.Graph(id="ind-comp-chart", style={"marginTop": "20px"}),

            html.Div(
                [
                    dcc.Graph(id="ret-ind-chart"),
                    dcc.Graph(id="vol-ind-chart"),
                    dcc.Graph(id="sh-ind-chart"),
                    dcc.Graph(id="ativos-chart"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                    "marginTop": "40px"
                }
            ),

            # Correlação de Retornos Diários
            html.H2("Correlação de Retornos Diários"),
            dcc.DatePickerRange(
                id="corr-date-picker",
                start_date=df_start_all,
                end_date=df_end_all,
                min_date_allowed=df_start_all,
                max_date_allowed=df_end_all,
                display_format="YYYY-MM-DD",
                style={"marginTop": "20px"}
            ),
            html.Div(
                [
                    dcc.Graph(id="macro-corr-heatmap", style={"width": "100%", "marginTop": "40px"}),
                    dcc.Graph(id="ls-corr-heatmap", style={"width": "100%", "marginTop": "40px"}),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "40px"}
            ),

            # Correlação Rolling (ρ)
            html.H2("Correlação Rolling (ρ)"),
            dcc.DatePickerRange(
                id="rolling-corr-date-picker",
                start_date=df_start_all,
                end_date=df_end_all,
                min_date_allowed=df_start_all,
                max_date_allowed=df_end_all,
                display_format="YYYY-MM-DD",
                style={"marginBottom": "10px"}
            ),
            html.Label("Selecione até dois fundos:"),
            dcc.Dropdown(
                id="fund-selector",
                options=[{"label": _mapping.get(a, a), "value": a} for a in funds_local],
                value=funds_local[:2] if len(funds_local) >= 2 else funds_local,
                multi=True,
                searchable=True,
                placeholder="Digite e selecione fundos...",
                style={"width": "60%", "marginBottom": "20px"}
            ),
            dcc.Graph(id="rolling-corr-graph", style={"marginBottom": "40px"}),

            # Correlação Média x PL x Volatilidade
            html.H2("Correlação Média x PL x Volatilidade"),
            html.Div(
                [
                    dcc.DatePickerRange(
                        id="corrpl-date-picker",
                        start_date=df_start_all,
                        end_date=df_end_all,
                        min_date_allowed=df_start_all,
                        max_date_allowed=df_end_all,
                        display_format="YYYY-MM-DD",
                    ),
                    dcc.RadioItems(
                        id="corrpl-group",
                        options=[
                            {"label": "Todos",              "value": "all"},
                            {"label": "Todos - Macro",      "value": "macro"},
                            {"label": "Todos - Long/Short", "value": "ls"},
                        ],
                        value="all",
                        inline=True,
                        style={"marginLeft": "16px"}
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "16px",
                       "flexWrap": "wrap", "marginBottom": "8px"}
            ),
            dcc.Dropdown(
                id="corrpl-highlight",
                options=[{"label": f"{_mapping.get(a,a)} ({a})", "value": a}
                         for a in funds_local if any(a.startswith(code) for code in (macro_codes + ls_codes))],
                value=None,
                placeholder="Destaque um fundo (opcional)…",
                style={"width": "40%", "marginBottom": "8px"}
            ),
            dcc.Graph(id="corrpl-scatter", style={"marginBottom": "40px"}),

            # Bubble Chart 3D
            html.H2("Bubble Chart 3D: Correlação × Volatilidade × Retorno"),
            html.Div(
                [
                    dcc.DatePickerRange(
                        id="bubble-date-picker",
                        start_date=START_FILTER.date(),
                        end_date=today,
                        min_date_allowed=START_FILTER.date(),
                        max_date_allowed=today,
                        display_format="YYYY-MM-DD",
                        style={"marginRight": "20px"}
                    ),
                    dcc.Dropdown(
                        id="bubble-funds-select",
                        options=[
                            {"label": _mapping.get(a, a), "value": a}
                            for a in funds_local
                            if any(a.startswith(code) for code in macro_codes + ls_codes)
                        ],
                        multi=True,
                        placeholder="Selecione fundos...",
                        style={"width": "400px"}
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}
            ),
            dcc.Graph(id="bubble3d", style={"marginBottom": "40px"}),

            # PCA 2D
            html.H2("PCA 2D de Fundos Multimercado"),
            html.Div(
                [
                    dcc.Dropdown(
                        id="pca-fundos",
                        options=fund_options,
                        value=fund_codes[:5] if fund_codes else [],
                        multi=True,
                        placeholder="Selecione fundos...",
                        style={"width": "300px", "marginRight": "20px"}
                    ),
                    dcc.DatePickerRange(
                        id="pca-date-range",
                        start_date=START_FILTER.date(),
                        end_date=today,
                        display_format="YYYY-MM-DD"
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}
            ),
            dcc.Graph(id="pca-graph", style={"height": "600px"}),

            # Dendrograma por Faixa de Volatilidade
            html.H2("Dendrograma por Faixa de Volatilidade"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Faixa de Volatilidade (anualizada, %)"),
                            dcc.RangeSlider(
                                id="dendo-vol-range",
                                min=0, max=60, step=0.5,
                                value=[5, 25],
                                allowCross=False,
                                marks={i: f"{i}%" for i in range(0, 61, 5)},
                                tooltip={"always_visible": False, "placement": "bottom"}
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block",
                               "verticalAlign": "top", "paddingRight": "10px"}
                    ),
                    html.Div(
                        [
                            html.Label("Selecione os Fundos"),
                            dcc.Dropdown(
                                id="dendo-funds-select",
                                options=[
                                    {"label": _mapping.get(c, c), "value": c}
                                    for c in funds_local if any(c.startswith(code) for code in codes)
                                ],
                                value=[c for c in funds_local if any(c.startswith(code) for code in codes)][:10],
                                multi=True,
                                placeholder="Escolha os fundos para o dendrograma..."
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}
                    ),
                ],
                style={"marginBottom": "10px"}
            ),
            dcc.Graph(id="dendo-graph", style={"height": "700px"}),

            # Análise por Fundo (Base 100 no range)
            html.H2("Análise por Fundo (Base 100 no range)"),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id=f"fund{i}-dropdown",
                                options=[{"label": _mapping.get(a, a), "value": a} for a in funds_local],
                                placeholder=f"Fundo {i}"
                            ) for i in range(1, 6)
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "10px"}
                    ),
                    dcc.Dropdown(
                        id="metric-dropdown",
                        options=[
                            {"label": "Retorno Acumulado", "value": "retorno"},
                            {"label": "Volatilidade", "value": "volatilidade"},
                            {"label": "Sharpe", "value": "sharpe"}
                        ],
                        value="retorno",
                        style={"width": "30%", "marginTop": "20px"}
                    ),
                    dcc.Checklist(
                        id="fund-benchmarks-selector",
                        options=[
                            {"label": "CDI",  "value": "cdi"},
                            {"label": "IHFA", "value": "ihfa"},
                            {"label": "Meta Atuarial", "value": "atu"}
                        ],
                        value=["cdi", "ihfa", "atu"],
                        inputStyle={"marginRight": "5px"},
                        style={"marginTop": "20px"}
                    ),
                    dcc.DatePickerRange(
                        id="fund-date-picker",
                        start_date=df_start_all,
                        end_date=df_end_all,
                        min_date_allowed=df_start_all,
                        max_date_allowed=df_end_all,
                        display_format="YYYY-MM-DD",
                        style={"marginTop": "20px"}
                    ),
                    dcc.Graph(id="fund-metric-chart", style={"marginTop": "20px"}),
                ],
                style={"marginTop": "20px"}
            ),
        ],
        style={"maxWidth": "1600px", "width": "95%", "margin": "0 auto", "padding": "20px"}
    )

# Vincule o layout/servidor APENAS uma vez, fora da função:
app.layout = serve_layout
server = app.server


from datetime import timedelta

from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output

@app.callback(
    Output('cdi-chart', 'figure'),
    Input('benchmarks-selector', 'value'),
    Input('cdi-date-picker',      'start_date'),
    Input('cdi-date-picker',      'end_date'),
)
def update_cdi_chart(benchmarks, start_date, end_date):
    # 1) converte datas e limita a 10 anos
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    if (ed - sd).days > 3650:
        sd = ed - timedelta(days=3650)

    fig = go.Figure()

    # 2) CDI (Base 100 diário)
    if 'cdi' in benchmarks:
        cdi_daily = cdi_idx.loc[sd:ed].copy()
        if not cdi_daily.empty:
            ret_d = cdi_daily.pct_change().fillna(0)
            cum_d = (1 + ret_d).cumprod()
            base = cum_d.iloc[0]
            if base != 0 and not pd.isna(base):
                s_cdi = cum_d.div(base).mul(100)
                fig.add_trace(go.Scatter(
                    x=s_cdi.index,
                    y=s_cdi.values,
                    mode='lines',
                    name='CDI (Base 100 Diário)'
                ))

    # 3) IHFA (Base 100 diário)
    if 'ihfa' in benchmarks:
        ihfa_daily = ihfa_idx.loc[sd:ed].copy()
        if not ihfa_daily.empty:
            ret_i = ihfa_daily.pct_change().fillna(0)
            cum_i = (1 + ret_i).cumprod()
            base_i = cum_i.iloc[0]
            if base_i != 0 and not pd.isna(base_i):
                s_ihfa = cum_i.div(base_i).mul(100)
                fig.add_trace(go.Scatter(
                    x=s_ihfa.index,
                    y=s_ihfa.values.flatten(),
                    mode='lines',
                    name='IHFA (Base 100 Diário)'
                ))

    # 4) Meta Atuarial (Base 100)
    if 'atu' in benchmarks:
        atu_slice = atu_series.loc[sd:ed].dropna()
        if not atu_slice.empty:
            base_a = atu_slice.iloc[0]
            if base_a != 0 and not pd.isna(base_a):
                s_atu = atu_slice.div(base_a).mul(100)
                fig.add_trace(go.Scatter(
                    x=s_atu.index,
                    y=s_atu.values,
                    mode='lines',
                    name='Meta Atuarial (Base 100)'
                ))

    # 5) layout final
    fig.update_layout(
        title='Índices Acumulados (Base 100)',
        xaxis_title='Data',
        yaxis_title='Índice (Base 100)',
        template='plotly_white',
        height=400
    )
    fig.update_yaxes(tickformat='.2f')

    return fig


# ======================
# CALLBACK: INDÚSTRIA (com linha estática sem MEG)
# ======================
@app.callback(
    Output('ret-ind-chart', 'figure'),
    Output('vol-ind-chart', 'figure'),
    Output('sh-ind-chart', 'figure'),
    Output('ativos-chart', 'figure'),
    Input('ind-date-picker', 'start_date'),
    Input('ind-date-picker', 'end_date'),
    Input('meg-filters', 'value')
)
def update_industry_charts(start_date, end_date, selected_filters):
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    # === TRAVA DA JANELA (fixa 30/06/2021 → 30/06/2025) ===
    LOCK_START = pd.Timestamp("2021-06-30")
    LOCK_END   = pd.Timestamp("2025-06-30")
    sd, ed     = LOCK_START, LOCK_END

    # MEG ativo? (há ao menos 1 filtro)
    meg_active = bool(selected_filters)
    # universo MEG: se não houver filtros, usa TODOS os fundos
    ativos = get_allowed_activos(selected_filters or [])
    if not meg_active:
        ativos = funds[:]  # todos

    # ======================
    # Helpers do REBALANCEAMENTO (apenas para a linha MEG) + contagem anual
    # ======================
    WINDOW_START = pd.Timestamp("2021-06-30")
    DATA_MIN = df_ret_full.index.min()
    DATA_MAX = df_ret_full.index.max()
    RANGE_START = max(sd, WINDOW_START, DATA_MIN)
    RANGE_END   = min(ed, DATA_MAX)

    def clamp_dates(s0, s1):
        s0 = max(pd.to_datetime(s0), DATA_MIN)
        s1 = min(pd.to_datetime(s1), DATA_MAX)
        return s0, s1

    def gen_rebal_dates(start, end):
        """Quebra em janelas com âncora 30/06 de cada ano."""
        if start >= end:
            return pd.to_datetime([start, end])
        dates = [start]
        y = max(2021, start.year)
        anchor = pd.Timestamp(year=y, month=6, day=30)
        if anchor <= start:
            y += 1
            anchor = pd.Timestamp(year=y, month=6, day=30)
        while anchor < end:
            dates.append(anchor)
            y += 1
            anchor = pd.Timestamp(year=y, month=6, day=30)
        dates.append(end)
        return pd.to_datetime(dates)

    def get_allowed_activos_at(ref_date, filters):
        """Aplica filtros MEG NA DATA DO REBAL (atuacao dinâmica)."""
        d = df_all.copy()
        filters = filters or []
        for f in filters:
            if f == 'patrimonio':
                d = d[d['Patrimônio|||em milhares'] > 300_000]
            elif f == 'atuacao':
                cutoff = ref_date - pd.DateOffset(years=ANOS_ATUACAO_MIN)
                d = d[d['Data do|Início da Série'] <= cutoff]
            elif f == 'cotistas':
                d = d[d['Média de|Cotistas|1 mês|em unidades'] > 100]
            elif f == 'multigestor':
                d = d[~d['Multigestor'].isin(['Multigestor', 'Espelho'])]
            elif f == 'exclusivo':
                d = d[d['Fundo|exclusivo'] != 'Sim']
            elif f == 'condominio':
                d = d[d['Forma de|condomínio'] != 'Fechado']
            elif f == 'exterior':
                d = d[~d['Investimento|no Exterior'].isin(['', 'Até 100%'])]
            elif f == 'restrito':
                d = d[d['Restrito'] != 'Sim']
            elif f == 'ihfa':
                d = d[d['Gestora com histórico IHFA'] == 'SIM']
        ativos_ok = d['Ativo'].unique().tolist()
        return [a for a in ativos_ok if a in df_p.columns]

    def _ind_xs_series(metric, cols):
        """Série diária cross-section (média robusta) com suavização ROLL_MM."""
        if not cols:
            return pd.Series(dtype='float64')
        if metric == 'ret':
            arr = df_ret_full[cols]
            xs = robust_cross_section_mean(arr)
        elif metric == 'vol':
            arr = df_vol_full[cols]
            xs = arr.apply(trim_mean_series, axis=1)
        else:  # 'sharpe'
            arr = sharpe_full[cols]
            xs = arr.apply(trim_mean_series, axis=1)
        return xs.rolling(ROLL_MM, min_periods=1).mean().dropna()

    def industry_meg_rebalanced(metric, s0, s1):
        """Constrói a linha MEG rebalanced: coorte por janela 30/06→30/06 e contagem anual."""
        s0, s1 = clamp_dates(s0, s1)
        s0 = max(s0, WINDOW_START)
        if s0 >= s1:
            return pd.Series(dtype='float64'), {}
        segments = []
        counts_by_year = {}
        rebal_dates = gen_rebal_dates(s0, s1)
        for i in range(len(rebal_dates) - 1):
            r0, r1 = rebal_dates[i], rebal_dates[i+1]
            cohort = get_allowed_activos_at(r0, selected_filters or [])
            if not cohort:
                continue
            # mantém apenas fundos com dados no segmento
            if metric == 'ret':
                cols = [c for c in cohort if not df_ret_full[c].loc[r0:r1].dropna().empty]
            elif metric == 'vol':
                cols = [c for c in cohort if not df_vol_full[c].loc[r0:r1].dropna().empty]
            else:
                cols = [c for c in cohort if not sharpe_full[c].loc[r0:r1].dropna().empty]
            if not cols:
                continue
            counts_by_year[r0.year] = len(cols)
            s = _ind_xs_series(metric, cols).loc[r0:r1]
            if not s.empty:
                segments.append(s)
        if not segments:
            return pd.Series(dtype='float64'), counts_by_year
        s = pd.concat(segments).sort_index()
        if metric == 'ret':
            s = (1 + s).cumprod()
            s = (s / s.iloc[0]) * 100.0
        return s, counts_by_year

    def static_cohort_counts(s0, s1):
        """Contagem anual (30/06→30/06) para TODOS os fundos, sem filtros."""
        s0, s1 = clamp_dates(s0, s1)
        s0 = max(s0, WINDOW_START)
        rebal_dates = gen_rebal_dates(s0, s1)
        counts = {}
        for i in range(len(rebal_dates) - 1):
            r0, r1 = rebal_dates[i], rebal_dates[i+1]
            # fundos com QUALQUER dado de preço/retorno no segmento
            cols = [c for c in funds if not df_ret_full[c].loc[r0:r1].dropna().empty]
            if cols:
                counts[r0.year] = len(cols)
        return counts

    # ======================
    # RETORNO — Indústria (MEG rebalanced) + Estática (original com clamp)
    # ======================
    # MEG
    if meg_active:
        ret_slice, counts_meg = industry_meg_rebalanced('ret', RANGE_START, RANGE_END)
    else:
        df_ret_meg = df_ret_full[ativos].loc[sd:ed].dropna(how='all')
        if not df_ret_meg.empty:
            ret_cs_meg     = robust_cross_section_mean(df_ret_meg)
            ret_smooth_meg = ret_cs_meg.rolling(ROLL_MM, min_periods=1).mean().dropna()
            ret_slice      = base100_from(((1 + ret_smooth_meg).cumprod()).dropna())
        else:
            ret_slice = pd.Series(dtype='float64')

    # Estática (todos os fundos), só quando MEG está ativo — mantém método, mas com datas “clampadas”
    if meg_active:
        s0, s1 = clamp_dates(sd, ed)
        df_ret_all = df_ret_full[funds].loc[s0:s1].dropna(how='all')
        if not df_ret_all.empty:
            ret_cs_all     = robust_cross_section_mean(df_ret_all)
            ret_smooth_all = ret_cs_all.rolling(ROLL_MM, min_periods=1).mean().dropna()
            ret_slice_all  = base100_from(((1 + ret_smooth_all).cumprod()).dropna())
        else:
            ret_slice_all = pd.Series(dtype='float64')
    else:
        ret_slice_all = pd.Series(dtype='float64')  # evita duplicar

    fig_ret = go.Figure()
    added = False
    if not ret_slice.empty:
        fig_ret.add_trace(go.Scatter(
            x=ret_slice.index, y=ret_slice.values, mode='lines',
            name='Indústria (MEG)' if meg_active else 'Indústria (todos)',
            line=dict(width=3)
        ))
        added = True
    if not ret_slice_all.empty:
        fig_ret.add_trace(go.Scatter(
            x=ret_slice_all.index, y=ret_slice_all.values, mode='lines',
            name='Indústria (estática — sem MEG)',
            line=dict(dash='dash', width=2)
        ))
        added = True
    if not added:
        fig_ret = _annot_fig('Retorno Médio da Indústria (Base 100)',
                             'Sem dados válidos no período/filtros. Amplie o intervalo ou remova filtros MEG.')
    else:
        fig_ret.update_layout(title='Retorno Médio da Indústria (Base 100)',
                              xaxis_title='Data', yaxis_title='Índice (Base 100)', template='plotly_white')
        fig_ret.update_yaxes(tickformat='.2f')

    # ======================
    # VOL — Indústria (MEG rebalanced) + Estática (original com clamp)
    # ======================
    if meg_active:
        vol_cs_meg, _ = industry_meg_rebalanced('vol', RANGE_START, RANGE_END)
    else:
        df_vol_meg = df_vol_full[ativos].loc[sd:ed].dropna(how='all')
        if not df_vol_meg.empty:
            vol_cs_meg = df_vol_meg.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
        else:
            vol_cs_meg = pd.Series(dtype='float64')

    if meg_active:
        s0, s1 = clamp_dates(sd, ed)
        df_vol_all = df_vol_full[funds].loc[s0:s1].dropna(how='all')
        if not df_vol_all.empty:
            vol_cs_all = df_vol_all.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
        else:
            vol_cs_all = pd.Series(dtype='float64')
    else:
        vol_cs_all = pd.Series(dtype='float64')

    fig_vol = go.Figure()
    added = False
    if not vol_cs_meg.empty:
        fig_vol.add_trace(go.Scatter(
            x=vol_cs_meg.index, y=vol_cs_meg.values, mode='lines',
            name='Indústria (MEG)' if meg_active else 'Indústria (todos)',
            line=dict(width=3)
        ))
        added = True
    if not vol_cs_all.empty:
        fig_vol.add_trace(go.Scatter(
            x=vol_cs_all.index, y=vol_cs_all.values, mode='lines',
            name='Indústria (estática — sem MEG)', line=dict(dash='dash', width=2)
        ))
        added = True
    if not added:
        fig_vol = _annot_fig('Volatilidade Média da Indústria',
                             'Sem dados válidos (intervalo muito curto?). Tente ≥ 6 meses por causa do MINP=126.')
    else:
        fig_vol.update_layout(title='Volatilidade Média da Indústria',
                              xaxis_title='Data', yaxis_title='Volatilidade (%)', template='plotly_white')
        fig_vol.update_yaxes(tickformat='.2f')

    # ======================
    # SHARPE — Indústria (MEG rebalanced) + Estática (original com clamp)
    # ======================
    if meg_active:
        sh_cs_meg, _ = industry_meg_rebalanced('sharpe', RANGE_START, RANGE_END)
    else:
        df_sh_meg = sharpe_full[ativos].loc[sd:ed].dropna(how='all')
        if not df_sh_meg.empty:
            sh_cs_meg = df_sh_meg.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
        else:
            sh_cs_meg = pd.Series(dtype='float64')

    if meg_active:
        s0, s1 = clamp_dates(sd, ed)
        df_sh_all = sharpe_full[funds].loc[s0:s1].dropna(how='all')
        if not df_sh_all.empty:
            sh_cs_all = df_sh_all.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
        else:
            sh_cs_all = pd.Series(dtype='float64')
    else:
        sh_cs_all = pd.Series(dtype='float64')

    fig_sh = go.Figure()
    added = False
    if not sh_cs_meg.empty:
        fig_sh.add_trace(go.Scatter(
            x=sh_cs_meg.index, y=sh_cs_meg.values, mode='lines',
            name='Indústria (MEG)' if meg_active else 'Indústria (todos)',
            line=dict(width=3)
        ))
        added = True
    if not sh_cs_all.empty:
        fig_sh.add_trace(go.Scatter(
            x=sh_cs_all.index, y=sh_cs_all.values, mode='lines',
            name='Indústria (estática — sem MEG)', line=dict(dash='dash', width=2)
        ))
        added = True
    if not added:
        fig_sh = _annot_fig('Sharpe Médio da Indústria',
                            'Sem dados válidos no período. Ajuste datas/filtros.')
    else:
        fig_sh.update_layout(title='Sharpe Médio da Indústria',
                             xaxis_title='Data', yaxis_title='Sharpe', template='plotly_white')
        fig_sh.update_yaxes(tickformat='.2f')

    # ======================
    # Nº de fundos por ano — agora por COHORT anual 30/06→30/06
    # ======================
    if meg_active:
        counts = counts_meg if 'counts_meg' in locals() else {}
        if not counts:
            # fallback: mostra todos (cohort anual sem filtros) para não ficar vazio
            counts = static_cohort_counts(RANGE_START, RANGE_END)
        anos = list(range(max(2021, RANGE_START.year), RANGE_END.year + 1))
        serie_counts = pd.Series(counts).reindex(anos).fillna(method='ffill').fillna(0).astype(int)
        fig_ativ = go.Figure(go.Scatter(
            x=serie_counts.index, y=serie_counts.values,
            mode='markers+lines', name='Nº de Fundos (cohort anual)'
        ))
        fig_ativ.update_layout(
            title='Evolução Anual do Nº de FI Multimercado (cohort 30/06→30/06)',
            xaxis_title='Ano', yaxis_title='Número de Fundos',
            template='plotly_white'
        )
    else:
        # Sem MEG: cohort anual para TODOS os fundos
        counts_all = static_cohort_counts(RANGE_START, RANGE_END)
        anos = list(range(max(2021, RANGE_START.year), RANGE_END.year + 1))
        serie_counts = pd.Series(counts_all).reindex(anos).fillna(method='ffill').fillna(0).astype(int)
        fig_ativ = go.Figure(go.Scatter(
            x=serie_counts.index, y=serie_counts.values,
            mode='markers+lines', name='Nº de Fundos (todos — cohort anual)'
        ))
        fig_ativ.update_layout(
            title='Evolução Anual do Nº de FI Multimercado (todos — cohort 30/06→30/06)',
            xaxis_title='Ano', yaxis_title='Número de Fundos',
            template='plotly_white'
        )

    return fig_ret, fig_vol, fig_sh, fig_ativ






# --- opções do dropdown "ind-funds-compare" sensíveis ao MEG ---
from dash.dependencies import Input, Output, State

# ======================
# CALLBACK: COMPARAÇÃO (MEG = rebalanced 48m | estática dos fundos não muda)
# ======================
@app.callback(
    Output('ind-comp-chart', 'figure'),
    Input('ind-date-picker', 'start_date'),
    Input('ind-date-picker', 'end_date'),
    Input('meg-filters', 'value'),
    Input('ind-metrics-selector', 'value'),   # 'ret' | 'vol' | 'sharpe'
    Input('ind-funds-compare', 'value')       # lista de códigos
)
def update_industry_comparison(start_date, end_date, selected_filters, selected_metric, selected_funds):
    sd = pd.to_datetime(start_date); ed = pd.to_datetime(end_date)

    WINDOW48_START = pd.Timestamp("2021-06-30")
    WINDOW48_END   = pd.Timestamp("2025-06-30")
    REBAL_DATES    = pd.to_datetime(["2021-06-30","2022-06-30","2023-06-30","2024-06-30","2025-06-30"])

    def get_allowed_activos_at(ref_date, filters):
        d = df_all.copy()
        filters = filters or []
        for f in filters:
            if f == 'patrimonio':
                d = d[d['Patrimônio|||em milhares'] > 300_000]
            elif f == 'atuacao':
                cutoff = ref_date - pd.DateOffset(years=ANOS_ATUACAO_MIN)
                d = d[d['Data do|Início da Série'] <= cutoff]
            elif f == 'cotistas':
                d = d[d['Média de|Cotistas|1 mês|em unidades'] > 100]
            elif f == 'multigestor':
                d = d[~d['Multigestor'].isin(['Multigestor', 'Espelho'])]
            elif f == 'exclusivo':
                d = d[d['Fundo|exclusivo'] != 'Sim']
            elif f == 'condominio':
                d = d[d['Forma de|condomínio'] != 'Fechado']
            elif f == 'exterior':
                d = d[~d['Investimento|no Exterior'].isin(['', 'Até 100%'])]
            elif f == 'restrito':
                d = d[d['Restrito'] != 'Sim']
            elif f == 'ihfa':
                d = d[d['Gestora com histórico IHFA'] == 'SIM']
        ativos = d['Ativo'].unique().tolist()
        return [a for a in ativos if a in df_p.columns]

    def _ind_xs_series(metric, cols):
        if not cols:
            return pd.Series(dtype='float64')
        if metric == 'ret':
            arr = df_ret_full[cols]; xs = robust_cross_section_mean(arr)
        elif metric == 'vol':
            arr = df_vol_full[cols]; xs = arr.apply(trim_mean_series, axis=1)
        else:
            arr = sharpe_full[cols]; xs = arr.apply(trim_mean_series, axis=1)
        return xs.rolling(ROLL_MM, min_periods=1).mean().dropna()

    def industry_rebalanced_series(metric, filters, s0, s1):
        s0 = max(pd.to_datetime(s0), WINDOW48_START)
        s1 = min(pd.to_datetime(s1), WINDOW48_END)
        if s0 >= s1:
            return pd.Series(dtype='float64'), {}, s0, s1

        segs = []
        for i in range(len(REBAL_DATES) - 1):
            r0, r1 = REBAL_DATES[i], REBAL_DATES[i+1]
            seg_start, seg_end = max(s0, r0), min(s1, r1)
            if seg_start >= seg_end:
                continue
            cohort = get_allowed_activos_at(r0, filters)

            if metric == 'ret':
                cols = [c for c in cohort if not df_ret_full[c].loc[seg_start:seg_end].dropna().empty]
            elif metric == 'vol':
                cols = [c for c in cohort if not df_vol_full[c].loc[seg_start:seg_end].dropna().empty]
            else:
                cols = [c for c in cohort if not sharpe_full[c].loc[seg_start:seg_end].dropna().empty]
            if not cols:
                continue

            s = _ind_xs_series(metric, cols).loc[seg_start:seg_end]
            if not s.empty:
                segs.append(s)

        if not segs:
            return pd.Series(dtype='float64'), {}, s0, s1

        out = pd.concat(segs).sort_index()
        if metric == 'ret':
            out = (1 + out).cumprod()
            out = (out / out.iloc[0]) * 100.0
        return out, {}, s0, s1

    metric_key = 'ret' if selected_metric == 'ret' else ('vol' if selected_metric == 'vol' else 'sharpe')

    if selected_filters:
        ind_series, _, rsd, red = industry_rebalanced_series(metric_key, selected_filters, sd, ed)
        if ind_series.empty:
            title = {"ret":"Média da Indústria — Retorno (Base 100)",
                     "vol":"Média da Indústria — Volatilidade",
                     "sharpe":"Média da Indústria — Sharpe"}[selected_metric]
            return _empty_fig(title, "Sem dados disponíveis no período/filtros.")
    else:
        # sem filtros → média original de TODOS (como antes)
        rsd, red = sd, ed
        if metric_key == 'ret':
            df_ret = df_ret_full[funds].loc[rsd:red]
            if df_ret.empty:
                return _empty_fig("Média da Indústria — Retorno (Base 100)", "Sem dados disponíveis no período.")
            ret_cs = robust_cross_section_mean(df_ret)
            ind_series = (1 + ret_cs.rolling(ROLL_MM, min_periods=1).mean()).cumprod().dropna()
            ind_series = start_at_first_valid(ind_series)
            ind_series = (ind_series / ind_series.iloc[0]) * 100
        elif metric_key == 'vol':
            df_vol = df_vol_full[funds].loc[rsd:red]
            if df_vol.empty:
                return _empty_fig("Média da Indústria — Volatilidade", "Sem dados disponíveis no período.")
            ind_series = df_vol.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
        else:
            df_sh = sharpe_full[funds].loc[rsd:red]
            if df_sh.empty:
                return _empty_fig("Média da Indústria — Sharpe", "Sem dados disponíveis no período.")
            ind_series = df_sh.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()

    y_title = {'ret':'Retorno (Base 100)', 'vol':'Volatilidade (%)', 'sharpe':'Sharpe'}[selected_metric]
    metric_label = y_title

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ind_series.index, y=ind_series.values, mode='lines',
        name='Média da Indústria' if not selected_filters else 'Média da Indústria (rebal anual)',
        line=dict(width=3)
    ))

    # Sobreposição dos fundos escolhidos (sem alterar o cálculo dos fundos)
    ativos_validos = get_allowed_activos(selected_filters or [])
    sel = [f for f in (selected_funds or [])
           if (selected_filters and f in ativos_validos) or (not selected_filters and f in funds)]

    for f in sel:
        if selected_metric == 'ret':
            s = (1 + df_ret_full[f].loc[rsd:red].fillna(0.0)).cumprod().dropna()
            if s.empty:
                continue
            s = start_at_first_valid(s)
            s = (s / s.iloc[0]) * 100
        elif selected_metric == 'vol':
            s = df_vol_full[f].loc[rsd:red].rolling(ROLL_MM, min_periods=1).mean().dropna()
        else:
            s = sharpe_full[f].loc[rsd:red].rolling(ROLL_MM, min_periods=1).mean().dropna()
        if not s.empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=mapping.get(f, f), opacity=0.9))

    ttl_suffix = (f" — Rebalanceamento anual • 48m: {WINDOW48_START.date()} → {WINDOW48_END.date()}"
                  if selected_filters else "")
    fig.update_layout(
        title=f"Média da Indústria — {metric_label}{ttl_suffix}",
        template='plotly_white', height=450, xaxis_title='Data'
    )
    fig.update_yaxes(title_text=y_title, tickformat='.2f')
    return fig




@app.callback(
    Output('macro-corr-heatmap', 'figure'),
    Output('ls-corr-heatmap',    'figure'),
    Input('corr-date-picker',    'start_date'),
    Input('corr-date-picker',    'end_date')
)
def update_corr_heatmaps(start_date, end_date):
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    df = df_ret_full.loc[sd:ed]

    # 1) Matriz de correlação
    macro_corr = df[macro_cols].corr()
    ls_corr    = df[ls_cols].corr()

    # 2) Heatmap Macro
    fig_macro = go.Figure(go.Heatmap(
        z=macro_corr.values,
        x=macro_labels,
        y=macro_labels,
        colorscale='Viridis', zmin=-1, zmax=1,
        colorbar=dict(title='ρ')
    ))
    # Anotações
    for i, row in enumerate(macro_corr.values):
        for j, val in enumerate(row):
            fig_macro.add_annotation(
                x=macro_labels[j], y=macro_labels[i],
                text=f"{val:.2f}", showarrow=False,
                font=dict(color='white' if abs(val) > 0.5 else 'black')
            )
    # Ajuste de largura baseado no número de fundos
    n = len(macro_cols)
    fig_macro.update_layout(
        title='Correlação — Macro',
        xaxis_tickangle=-45,
        margin=dict(t=60, b=100),
        width=70 * n,     # 70px por célula
        height=70 * n     # idem para altura, deixe quadrado
    )

    # 3) Heatmap Long/Short
    fig_ls = go.Figure(go.Heatmap(
        z=ls_corr.values,
        x=ls_labels,
        y=ls_labels,
        colorscale='Viridis', zmin=-1, zmax=1,
        colorbar=dict(title='ρ')
    ))
    for i, row in enumerate(ls_corr.values):
        for j, val in enumerate(row):
            fig_ls.add_annotation(
                x=ls_labels[j], y=ls_labels[i],
                text=f"{val:.2f}", showarrow=False,
                font=dict(color='white' if abs(val) > 0.5 else 'black')
            )
    fig_ls.update_layout(
        title='Correlação — Long/Short',
        xaxis_tickangle=-45,
        margin=dict(t=60, b=100),
        height=500
    )

    return fig_macro, fig_ls



# Sincroniza o dropdown de comparação com os filtros do MEG
@app.callback(
    Output('ind-funds-compare', 'options'),
    Output('ind-funds-compare', 'value'),
    Input('meg-filters', 'value'),
    State('ind-funds-compare', 'value')
)
def sync_ind_funds_compare(selected_filters, current_value):
    # universo permitido pelo MEG (se vazio, volta a todos)
    ativos = get_allowed_activos(selected_filters or [])

    # opções visíveis no dropdown
    options = [{'label': mapping.get(a, a), 'value': a} for a in ativos]

    # normaliza valor atual → lista
    if isinstance(current_value, list):
        cur = current_value
    elif current_value:
        cur = [current_value]
    else:
        cur = []

    # mantém apenas fundos válidos pelo MEG
    cur = [c for c in cur if c in ativos]

    # se esvaziou, não pré-seleciona nada (deixa o usuário escolher)
    return options, cur







@app.callback(
    Output('fund-metric-chart', 'figure'),
    Input('fund1-dropdown',           'value'),
    Input('fund2-dropdown',           'value'),
    Input('fund3-dropdown',           'value'),
    Input('fund4-dropdown',           'value'),
    Input('fund5-dropdown',           'value'),
    Input('metric-dropdown',          'value'),
    Input('fund-benchmarks-selector', 'value'),
    Input('fund-date-picker',         'start_date'),
    Input('fund-date-picker',         'end_date'),
    Input('meg-filters',              'value'),
)
def update_fund_chart(f1, f2, f3, f4, f5,
                      metrica, benchmarks,
                      start_date, end_date, meg_filters):

    # 1) converter datas e ativos permitidos
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    ativos = get_allowed_activos(meg_filters)

    # 2) seleciona até 5 fundos válidos
    candidatos = [f1, f2, f3, f4, f5]
    selected   = [f for f in candidatos if f in ativos][:5]
    nomes      = [mapping[f] for f in selected]
    if not selected:
        return go.Figure()

    # 3) escolhe fonte e função
    if metrica == 'retorno':
        df_src = df_ret_full

        def fn(s):
            # acumulação composta diária exata: produto de (1 + retorno)
            return (1 + s).cumprod()

        y_label, fmt = 'Retorno (Base 100)', 'Retorno Acumulado'
    elif metrica == 'volatilidade':
        df_src, fn = df_vol_full, lambda s: s.rolling(ROLL_MM, min_periods=1).mean()
        y_label, fmt = 'Volatilidade', 'Volatilidade'
    else:
        df_src, fn = sharpe_full, lambda s: s.rolling(ROLL_MM, min_periods=1).mean()
        y_label, fmt = 'Sharpe', 'Sharpe'

    # 4) coleta séries de fundos
    series = {}
    for f, nome in zip(selected, nomes):
        s = fn(df_src[f]).dropna().loc[sd:ed]
        if not s.empty:
            series[nome] = s

    # 5) adiciona benchmarks **só se for Retorno**, usando a mesma fn
    if metrica == 'retorno':
        if 'cdi' in benchmarks:
            s_cdi = fn(cdi_idx.pct_change().fillna(0)).loc[sd:ed]
            if not s_cdi.empty:
                series['CDI (BM)'] = s_cdi
        if 'ihfa' in benchmarks:
            s_ihfa = fn(ihfa_idx.pct_change().fillna(0)).loc[sd:ed]
            if not s_ihfa.empty:
                series['IHFA (BM)'] = s_ihfa
        if 'atu' in benchmarks:
            s_atu = atu_series.loc[sd:ed]
            if not s_atu.empty:
                series['Meta Atuarial'] = s_atu

    # 6) define data comum de início (mais recente entre todas)
    datas = [s.index.min() for s in series.values()]
    start_common = max(datas)

    # 7) trunca e, se for Retorno, rebase em 100
    for name, s in list(series.items()):
        s2 = s.loc[start_common:ed]
        if s2.empty:
            series.pop(name)
        else:
            series[name] = s2.div(s2.iloc[0]).mul(100) if metrica == 'retorno' else s2

    # 8) se não sobrou nada, retorna vazio
    if not series:
        return go.Figure()

    # 9) monta figura
    fig = go.Figure()
    for name, s in series.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=name))

    # 10) título dinâmico
    title = f"{fmt} — {' vs '.join(nomes)}"
    if metrica == 'retorno':
        if 'cdi'  in benchmarks: title += ' vs CDI'
        if 'ihfa' in benchmarks: title += ' vs IHFA'
        if 'atu'  in benchmarks: title += ' vs Meta Atuarial'

    # 11) layout final
    fig.update_layout(
        title=title,
        xaxis_title='Data',
        yaxis_title=y_label,
        xaxis=dict(range=[start_common, ed]),
        template='plotly_white'
    )
    fig.update_yaxes(tickformat='.2f')

    return fig



# ======================
# CALLBACK: ATUALIZA FUNDOS DISPONÍVEIS
# ======================
@app.callback(
    [
        # opções para cada um dos 5 dropdowns
        Output('fund1-dropdown', 'options'),
        Output('fund2-dropdown', 'options'),
        Output('fund3-dropdown', 'options'),
        Output('fund4-dropdown', 'options'),
        Output('fund5-dropdown', 'options'),
        # valor inicial para cada dropdown
        Output('fund1-dropdown', 'value'),
        Output('fund2-dropdown', 'value'),
        Output('fund3-dropdown', 'value'),
        Output('fund4-dropdown', 'value'),
        Output('fund5-dropdown', 'value'),
    ],
    Input('meg-filters', 'value')
)
def update_fund_dropdowns(selected_filters):
    # Determina os ativos permitidos pelos filtros
    ativos = get_allowed_activos(selected_filters)

    # Constrói a lista de opções
    options = [{'label': mapping[a], 'value': a} for a in ativos]

    # Valor padrão: primeiro ativo em fund1; os demais começam vazios
    default1 = ativos[0] if ativos else None

    return (
        options,  # fund1 opções
        options,  # fund2 opções
        options,  # fund3 opções
        options,  # fund4 opções
        options,  # fund5 opções
        default1, # fund1 valor
        None,     # fund2 valor
        None,     # fund3 valor
        None,     # fund4 valor
        None      # fund5 valor
    )




from dash.dependencies import Input, Output, State

@app.callback(
    Output("fund-selector", "options"),
    Output("fund-selector", "value"),
    Input("meg-filters", "value"),
    State("fund-selector", "value"),  # usa State pra evitar dependência circular
)
def sync_fund_selector_with_meg(selected_filters, current_value):
    ativos = get_allowed_activos(selected_filters or [])
    if not ativos:
        ativos = funds

    options = [{'label': mapping[a], 'value': a} for a in ativos]

    # normaliza o valor atual
    if isinstance(current_value, list):
        current = current_value
    elif current_value:
        current = [current_value]
    else:
        current = []

    # mantém só os válidos
    filtered = [f for f in current if f in ativos]

    # garante dois fundos válidos (mesma lógica de fallback do rolling)
    pair = filtered[:2]
    if len(pair) < 2:
        for a in ativos:
            if a not in pair:
                pair.append(a)
            if len(pair) == 2:
                break

    if len(pair) == 0:
        pair = funds[:2]
    elif len(pair) == 1:
        other = [f for f in funds if f != pair[0]]
        pair = [pair[0], other[0] if other else pair[0]]

    return options, pair



@app.callback(
    Output("rolling-corr-graph", "figure"),
    [
        Input("rolling-corr-date-picker", "start_date"),
        Input("rolling-corr-date-picker", "end_date"),
        Input("fund-selector", "value"),
        Input("meg-filters", "value"),  # adicionado
    ]
)
def update_rolling_correlation(start_date, end_date, selected_cols, selected_filters):
    # determina ativos válidos pelos filtros MEG (fallback para todos)
    ativos = get_allowed_activos(selected_filters or [])
    if not ativos:
        ativos = funds

    # constrói par de fundos válidos
    if not selected_cols:
        pair = ativos[:2]
    else:
        sel = selected_cols if isinstance(selected_cols, list) else [selected_cols]
        filtered = [f for f in sel if f in ativos]
        if len(filtered) >= 2:
            pair = filtered[:2]
        else:
            needed = 2 - len(filtered)
            addition = [f for f in ativos if f not in filtered][:needed]
            pair = filtered + addition
            if len(pair) < 2:
                extra = [f for f in funds if f not in pair][: (2 - len(pair))]
                pair = pair + extra

    # garantir exatamente dois fundos
    if len(pair) == 0:
        pair = funds[:2]
    elif len(pair) == 1:
        other = [f for f in funds if f != pair[0]]
        pair = [pair[0], other[0] if other else pair[0]]

    f1, f2 = pair[0], pair[1]

    # slice por data
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    df_pair = df_ret_full.loc[sd:ed, [f1, f2]].dropna()

    if df_pair.empty:
        return go.Figure()  # sem dados válidos

    fig = go.Figure()
    for w in [252, 504, 756]:
        corr = df_pair[f1].rolling(window=w).corr(df_pair[f2])
        fig.add_trace(go.Scatter(
            x=corr.index,
            y=corr,
            mode="lines",
            name=f"Janela {w} dias"
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    labels = {c: mapping.get(c, c) for c in (f1, f2)}
    fig.update_layout(
        title=(
            f"Correlação Rolling entre<br>"
            f"{labels[f1]} e {labels[f2]}"
        ),
        xaxis_title="Data",
        yaxis_title="Correlação (ρ)",
        template="plotly_white",
        height=500,
        margin={"l": 40, "r": 20, "t": 80, "b": 40}
    )
    fig.update_yaxes(tickformat='.2f')

    return fig



# Sincroniza as opções do "Destaque" com os filtros do MEG
@app.callback(
    Output('corrpl-highlight', 'options'),
    Output('corrpl-highlight', 'value'),
    Input('meg-filters', 'value'),
    State('corrpl-highlight', 'value')
)
def sync_corrpl_controls(meg_filters, current_focus):
    ativos = get_allowed_activos(meg_filters or [])
    codes = macro_codes + ls_codes
    allowed = [c for c in ativos if any(c.startswith(code) for code in codes)]
    opts = [{'label': f"{mapping.get(c,c)} ({c})", 'value': c} for c in allowed]
    focus = current_focus if current_focus in allowed else None
    return opts, focus


# Figura PL × Volatilidade (cor = média de correlação)
@app.callback(
    Output('corrpl-scatter', 'figure'),
    Input('corrpl-date-picker', 'start_date'),
    Input('corrpl-date-picker', 'end_date'),
    Input('meg-filters',        'value'),
    Input('corrpl-group',       'value'),   # 'all' | 'macro' | 'ls'
    Input('corrpl-highlight',   'value')
)
def update_corrpl_scatter(start_date, end_date, meg_filters, group_sel, focus_code):
    sd, ed = pd.to_datetime(start_date), pd.to_datetime(end_date)
    base_codes = macro_codes + ls_codes
    ativos_allowed = get_allowed_activos(meg_filters or [])
    cols_all = [c for c in df_ret_full.columns
                if any(c.startswith(code) for code in base_codes) and c in ativos_allowed]

    if not cols_all:
        return _empty_fig(f'PL × Volatilidade (cor = ρ) — {sd.date()} a {ed.date()}',
                          "Nenhum fundo atende aos filtros MEG.")

    def is_macro(c): return any(c.startswith(code) for code in macro_codes)
    def is_ls(c):    return any(c.startswith(code) for code in ls_codes)
    if group_sel == 'macro':
        cols = [c for c in cols_all if is_macro(c)]
    elif group_sel == 'ls':
        cols = [c for c in cols_all if is_ls(c)]
    else:
        cols = cols_all

    if len(cols) < 2:
        return _empty_fig(f'PL × Volatilidade (cor = ρ) — {sd.date()} a {ed.date()}',
                          "Selecione pelo menos 2 fundos no grupo/MEG e aumente o período.")

    df_ret = df_ret_full.loc[sd:ed, cols]
    if df_ret.dropna(how='all').empty:
        return _empty_fig(f'PL × Volatilidade (cor = ρ) — {sd.date()} a {ed.date()}',
                          "Sem retornos no período selecionado.")

    # Correlação média excluindo diagonal
    corr = df_ret.corr(min_periods=30)
    if corr.shape[0] < 2 or corr.isna().all().all():
        return _empty_fig(f'PL × Volatilidade (cor = ρ) — {sd.date()} a {ed.date()}',
                          "Dados insuficientes para correlação (tente ampliar o período).")
    corr.values[np.arange(len(corr)), np.arange(len(corr))] = np.nan
    denom = (~corr.isna()).sum(axis=1).replace(0, np.nan)
    avg_corr = (corr.sum(axis=1, skipna=True) / denom).replace([np.inf, -np.inf], np.nan)

    # Vol média (fallback se rolling MINP=126 zerar tudo)
    df_vol = df_vol_full.loc[sd:ed, cols]
    avg_vol = df_vol.mean()
    if avg_vol.isna().all():
        avg_vol = (df_ret.std() * (252 ** 0.5) * 100)

    # PL
    df_pl_port = df_all.set_index('Ativo')['Patrimônio|||em milhares'].astype(float).mul(1_000)

    df_plot = (pd.DataFrame({
        'Code':       avg_corr.index,
        'AvgCorr':    avg_corr.values,
        'Volatility': avg_vol.reindex(avg_corr.index).values,
        'PL':         df_pl_port.reindex(avg_corr.index).values
    }).dropna(subset=['AvgCorr', 'Volatility', 'PL']))

    if df_plot.empty:
        return _empty_fig(f'PL × Volatilidade (cor = ρ) — {sd.date()} a {ed.date()}',
                          "Sem dados após limpeza (NaNs). Ajuste período/filtros.")

    df_plot['Fund'] = df_plot['Code'].map(mapping).fillna(df_plot['Code'])

    fig = px.scatter(
        df_plot, x='Volatility', y='PL', size='PL', color='AvgCorr',
        hover_name='Fund',
        hover_data={'Code': True, 'Volatility': ':.2f', 'PL': ':,.0f', 'AvgCorr': ':.2f'},
        size_max=90, labels={'Volatility':'Volatilidade (%)','PL':'Patrimônio Líquido (R$)','AvgCorr':'Média de ρ'},
        title=f'PL × Volatilidade (cor = Correlação Média) — {sd.date()} a {ed.date()}',
        template='plotly_white', color_continuous_scale='Viridis', range_color=(-1, 1)
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='black'), sizemin=10))
    fig.update_coloraxes(showscale=True, cmin=-1, cmax=1, colorbar=dict(title='ρ'))
    fig.update_layout(margin=dict(r=120, t=60, l=40, b=40))

    if focus_code in set(df_plot['Code']):
        row = df_plot[df_plot['Code'] == focus_code].iloc[0]
        fig.add_trace(go.Scatter(
            x=[row['Volatility']], y=[row['PL']], mode='markers+text',
            text=[f"★ {row['Fund']}"], textposition='top center',
            name=f"Destaque: {row['Fund']}",
            marker=dict(size=18, symbol='circle-open-dot', line=dict(width=2, color='black'),
                        color='rgba(0,0,0,0)'), showlegend=False
        ))
    return fig






@app.callback(
    Output("bubble3d", "figure"),
    Input("bubble-date-picker", "start_date"),
    Input("bubble-date-picker", "end_date"),
    Input("bubble-funds-select", "value"),
)
def update_bubble3d(start_date, end_date, sel_codes):
    if not sel_codes:
        return _annot_fig('Bubble Chart 3D', 'Selecione ao menos um fundo.')

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    df_ret = df_ret_full.reindex(columns=sel_codes).loc[sd:ed]
    df_vol = df_vol_full.reindex(columns=sel_codes).loc[sd:ed]

    if df_ret.dropna(how='all').empty:
        return _annot_fig('Bubble Chart 3D', 'Nenhum dado disponível neste período.')

    # Correlação média por fundo (exclui a diagonal). Fallback = 0.0
    corr = df_ret.corr(min_periods=30)
    if corr.shape[0] > 0:
        corr = corr.copy()
        np.fill_diagonal(corr.values, np.nan)
        avg_corr = corr.mean(axis=1)
    else:
        avg_corr = pd.Series(index=sel_codes, dtype='float64')
    avg_corr = avg_corr.reindex(sel_codes).fillna(0.0)

    # Vol média anualizada; fallback p/ std anualizada caso NaN
    avg_vol = df_vol.mean().reindex(sel_codes)
    fallback_vol = (df_ret.std() * (252 ** 0.5) * 100).reindex(sel_codes)
    avg_vol = avg_vol.fillna(fallback_vol)

    # Retorno acumulado Base 100 (robusto a NaN no início)
    ret_end = pd.Series(index=sel_codes, dtype='float64')
    for code in sel_codes:
        s = df_ret[code].dropna()
        if s.empty:
            continue
        cum = (1.0 + s).cumprod()
        ret_end[code] = (cum.iloc[-1] / cum.iloc[0]) * 100.0

    # Tamanho pela PL (robusto a ruído/NaN)
    pl_vals = pd.to_numeric(df_pl.reindex(sel_codes), errors="coerce")
    valid_pl = pl_vals.dropna()
    max_pl = valid_pl.max() if (not valid_pl.empty and valid_pl.max() > 0) else 1.0
    sizes = ((pl_vals.fillna(0.0) / max_pl) * 40.0 + 5.0)

    fig = go.Figure()
    plotted = 0
    for code in sel_codes:
        x = float(avg_corr.get(code, np.nan))
        y = float(avg_vol.get(code, np.nan))
        z = float(ret_end.get(code, np.nan))
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z], mode='markers',
            name=mapping.get(code, code),
            marker=dict(
                size=float(sizes.get(code, 10.0)),
                line=dict(width=1, color='black'),
                opacity=0.85
            )
        ))
        plotted += 1

    if plotted == 0:
        return _annot_fig('Bubble Chart 3D', 'Sem pontos após a limpeza. Amplie o período ou selecione outros fundos.')

    fig.update_layout(
        scene=dict(
            xaxis_title="Média de Correlação (ρ)",
            yaxis_title="Volatilidade (%)",
            zaxis_title="Retorno Acumulado (Base 100)"
        ),
        template='plotly_white',
        title=f"Período: {sd.date()} → {ed.date()}",
        height=600,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    return fig




# ─── Bloco: Callback para o PCA ───
@app.callback(
    Output("pca-graph", "figure"),
    Input("pca-fundos", "value"),
    Input("pca-date-range", "start_date"),
    Input("pca-date-range", "end_date"),
)
def update_pca(fund_list, start_date, end_date):
    import plotly.express as px

    if not fund_list:
        return _annot_fig("PCA 2D", "Selecione ao menos um fundo.")

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    cols = [c for c in fund_list if c in df_ret_full.columns]
    if not cols:
        return _annot_fig("PCA 2D", "Fundos inválidos para o período.")

    df_ret = df_ret_full.reindex(columns=cols).loc[sd:ed]
    df_vol = df_vol_full.reindex(columns=cols).loc[sd:ed]

    if df_ret.dropna(how='all').empty:
        return _annot_fig("PCA 2D", "Nenhum dado disponível neste período.")

    # Métricas robustas
    corr = df_ret.corr(min_periods=30)
    if corr.shape[0] > 0:
        corr = corr.copy()
        np.fill_diagonal(corr.values, np.nan)
        avg_corr = corr.mean(axis=1)
    else:
        avg_corr = pd.Series(index=cols, dtype='float64')
    avg_corr = avg_corr.reindex(cols).fillna(0.0)

    avg_vol  = df_vol.mean().reindex(cols)
    fallback_vol = (df_ret.std() * (252 ** 0.5) * 100).reindex(cols)
    avg_vol  = avg_vol.fillna(fallback_vol)

    # Retorno Base 100 (fim do range), robusto a NaN no início
    ret_end = pd.Series(index=cols, dtype='float64')
    for code in cols:
        s = df_ret[code].dropna()
        if s.empty:
            continue
        cum = (1.0 + s).cumprod()
        ret_end[code] = (cum.iloc[-1] / cum.iloc[0]) * 100.0

    pl_vals = pd.to_numeric(df_pl.reindex(cols), errors="coerce")

    features = (
        pd.DataFrame({
            'AvgCorr':    avg_corr,
            'Volatility': avg_vol,
            'RetBase100': ret_end,
            'PL':         pl_vals
        })
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if features.shape[0] < 2:
        return _annot_fig("PCA 2D", "Dados insuficientes após limpeza (precisa ≥ 2 fundos).")

    try:
        X = features[['AvgCorr', 'Volatility', 'RetBase100']].values
        pcs = PCA(n_components=2).fit_transform(X)
    except Exception as e:
        return _annot_fig("PCA 2D", f"Erro no PCA: {e}")

    df_pca = pd.DataFrame(pcs, index=features.index, columns=['PC1', 'PC2'])
    df_pca['Fund'] = [mapping.get(code, code) for code in df_pca.index]  # ← fixa fillna(Index)
    df_pca['PL']   = features['PL'].astype(float)

    fig = px.scatter(
        df_pca,
        x='PC1', y='PC2',
        size='PL', color='Fund',
        hover_name='Fund',
        hover_data={'PL': ':,.0f', 'PC1': ':.2f', 'PC2': ':.2f'},
        size_max=60,
        template='plotly_white',
        title=f"PCA 2D ({sd.date()} → {ed.date()})"
    )
    return fig


@app.callback(
    Output('dendo-graph', 'figure'),
    Input('dendo-vol-range', 'value'),
    Input('dendo-funds-select', 'value'),
)
def update_dendrogram(vol_range, selected_codes):
    from plotly.figure_factory import create_dendrogram
    from scipy.cluster import hierarchy as sch
    from scipy.spatial import distance as ssd

    # Helper: garante escala em %
    def to_percent(s):
        s = pd.Series(s, dtype="float64")
        return s * 100.0 if s.max() <= 1.5 else s  # até ~150% assume decimal

    vmin, vmax = (vol_range or [0, 60])  # faixas em %
    sd = START_FILTER
    ed = df_ret_full.index.max()

    fund_pool = [
        c for c in (selected_codes or [])
        if any(c.startswith(code) for code in codes) and c in df_ret_full.columns
    ]
    if len(fund_pool) < 2:
        return go.Figure(layout_title_text="Selecione ao menos 2 fundos.")

    # Vol anualizada média no período -> converte para %
    vol_mean_raw = df_vol_full.loc[sd:ed, fund_pool].mean().dropna()
    vol_mean_pct = to_percent(vol_mean_raw)

    # Filtra pela faixa (em %)
    eligible = vol_mean_pct[(vol_mean_pct >= vmin) & (vol_mean_pct <= vmax)].index.tolist()
    if len(eligible) < 2:
        return go.Figure(layout_title_text="Faixa de volatilidade muito restrita: selecione ao menos 2 fundos.")

    # Retornos (preenche NaN p/ robustez)
    df_ret = df_ret_full.loc[sd:ed, eligible].fillna(0.0)
    X = df_ret.T.values  # (n_fundos, n_datas)

    # Distância 1 - correlação (fallback euclidiana)
    try:
        D = ssd.pdist(X, metric='correlation')
        Z = sch.linkage(D, method='average')
        distfun = lambda x: ssd.pdist(x, 'correlation')
    except Exception:
        Z = sch.linkage(X, method='average', metric='euclidean')
        distfun = lambda x: ssd.pdist(x, 'euclidean')

    # Rótulos com nome + vol%
    labels = [
        f"{mapping.get(f, f)} — {vol_mean_pct[f]:.2f}%"
        for f in df_ret.columns
    ]

    fig = create_dendrogram(
        X,
        labels=labels,
        orientation='left',
        distfun=distfun,
        linkagefun=lambda x: sch.linkage(x, method='average')
    )

    n = len(eligible)
    fig.update_layout(
        title=(f"Dendrograma (distância = 1 − correlação) — "
               f"Faixa de Vol: {vmin:.1f}% a {vmax:.1f}% | {n} fundos"),
        template='plotly_white',
        height=max(300, 30 * n + 200),
        margin=dict(l=260, r=20, t=70, b=20)
    )
    fig.update_xaxes(title_text="Distância")

    return fig


@app.callback(
    Output('mortality-chart', 'figure'),
    Input('mort-date-picker', 'start_date'),
    Input('mort-date-picker', 'end_date'),
    Input('mort-fund-select', 'value'),
)
def update_mortality_norm_with_bubbles(start_date, end_date, fund_values):
    # 1) pegar df de vida sem usar "or" em DataFrame
    life_df = globals().get('df_life_all', None)
    if life_df is None:
        life_df = globals().get('df_life', None)
    if life_df is None or not isinstance(life_df, pd.DataFrame) or life_df.empty:
        return _annot_fig('Mortalidade (anual)', 'Planilha inicio_fim.xlsx vazia ou não carregada.')

    code_col  = 'CodigoFinal' if 'CodigoFinal' in life_df.columns else 'Codigo'
    ini_col   = 'DataInicio'
    fim_col   = 'DataFim'

    df = life_df.copy()
    df[code_col] = df[code_col].astype(str).str.strip()
    df[ini_col]  = pd.to_datetime(df[ini_col], errors='coerce')
    df[fim_col]  = pd.to_datetime(df[fim_col], errors='coerce')

    # 2) intervalo
    sd = pd.to_datetime(start_date) if start_date else globals().get('MORT_MIN', pd.Timestamp('2000-01-01'))
    ed = pd.to_datetime(end_date)   if end_date   else globals().get('MORT_MAX', pd.Timestamp.today().normalize())
    if sd >= ed:
        return _annot_fig('Mortalidade (anual)', 'Intervalo de datas inválido.')

    years = np.arange(sd.year, ed.year + 1, dtype=int)

    # 3) mortes/total por ano (SÓ inicio_fim.xlsx)
    deaths, totals = [], []
    for y in years:
        y0 = pd.Timestamp(y, 1, 1)
        y1 = pd.Timestamp(y, 12, 31)

        # mortes: códigos distintos com DataFim dentro do ano
        d = df.loc[df[fim_col].notna() & (df[fim_col] >= y0) & (df[fim_col] <= y1), code_col].nunique()

        # total de fundos vivos em QUALQUER ponto do ano (iniciou até 31/12 e não acabou antes de 01/01)
        alive_mask = (df[ini_col].notna()) & (df[ini_col] <= y1) & (df[fim_col].isna() | (df[fim_col] >= y0))
        t = df.loc[alive_mask, code_col].nunique()

        deaths.append(int(d))
        totals.append(int(t))

    ratio = np.array([d / t if t > 0 else np.nan for d, t in zip(deaths, totals)], dtype=float)

    # 4) figura base (pontos = anos, y = mortes/total)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=ratio, mode='markers',
        name='Mortes / Total (ano)',
        marker=dict(size=9, line=dict(width=1, color='black'), opacity=0.9),
        customdata=np.c_[deaths, totals],
        hovertemplate='Ano: %{x}<br>Mortes: %{customdata[0]}<br>Total: %{customdata[1]}<br>Taxa: %{y:.4f}<extra></extra>'
    ))

    # 5) linha de média do período
    mean_val = float(np.nanmean(ratio)) if np.isfinite(ratio).any() else np.nan
    if np.isfinite(mean_val):
        fig.add_trace(go.Scatter(
            x=years, y=[mean_val] * len(years), mode='lines',
            name='Média no período', line=dict(width=3)
        ))

    # 6) bolhas de INÍCIO e FIM para os fundos selecionados (multi)
    if isinstance(fund_values, list):
        selected = [str(v) for v in fund_values]
    elif fund_values:
        selected = [str(fund_values)]
    else:
        selected = []

    if selected:
        ratio_by_year = {int(y): r for y, r in zip(years, ratio)}

        # INÍCIO
        xs_i, ys_i, labels_i = [], [], []
        # FIM
        xs_f, ys_f, labels_f = [], [], []

        for code in selected:
            rows = df.loc[df[code_col].astype(str) == code]
            if rows.empty:
                continue

            di = rows[ini_col].dropna()
            dfim = rows[fim_col].dropna()

            if not di.empty:
                yi = int(di.iloc[0].year)
                if years[0] <= yi <= years[-1]:
                    val = ratio_by_year.get(yi, np.nan)
                    if pd.notna(val):
                        xs_i.append(yi); ys_i.append(val); labels_i.append(code)

            if not dfim.empty:
                yf = int(dfim.iloc[0].year)
                if years[0] <= yf <= years[-1]:
                    val = ratio_by_year.get(yf, np.nan)
                    if pd.notna(val):
                        xs_f.append(yf); ys_f.append(val); labels_f.append(code)

        if xs_i:
            fig.add_trace(go.Scatter(
                x=xs_i, y=ys_i, mode='markers+text',
                name='Início (selecionados)',
                text=labels_i, textposition='top center',
                marker=dict(symbol='circle', size=14, line=dict(width=2, color='black'))
            ))
        if xs_f:
            fig.add_trace(go.Scatter(
                x=xs_f, y=ys_f, mode='markers+text',
                name='Fim (selecionados)',
                text=labels_f, textposition='bottom center',
                marker=dict(symbol='diamond', size=14, line=dict(width=2, color='black'))
            ))

    # 7) layout
    fig.update_layout(
        title='Mortes / Total de fundos por ano (com bolhas de Início/Fim dos selecionados)',
        template='plotly_white',
        xaxis_title='Ano',
        yaxis_title='Mortes / Total',
        height=440,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    fig.update_xaxes(dtick=1, tickformat='d')
    fig.update_yaxes(tickformat='.4f')

    return fig








# no bloco de execução local:
if __name__ == "__main__":
    import os
    PORT = int(os.environ.get("PORT") or 8080)  # usa 8080 se PORT estiver vazia
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

