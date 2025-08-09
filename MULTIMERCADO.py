#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from datetime import datetime, timedelta
import requests
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
from plotly.figure_factory import create_dendrogram
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance as ssd
from dash import Dash

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server



load_dotenv()
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================
# PARÂMETROS GERAIS
# ======================
START_FILTER = pd.Timestamp("2021-06-30")
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

# ======================
# CARREGA SÉRIES DE PREÇO E MAPPING BÁSICO
# ======================
import requests
import pandas as pd

# ————————————————————————————————
# Série de preços dos fundos e mapping
# ————————————————————————————————
df_p       = pd.read_parquet("df_p.parquet")
mapping_df = pd.read_parquet("mapping.parquet")  # Ativo → Nome
mapping    = dict(zip(mapping_df["Ativo"], mapping_df["Nome"]))
df_p = df_p.sort_index()

# Lista base de fundos que têm nome no mapping
funds = [c for c in df_p.columns if c in mapping]

# ————————————————————————————————
# Benchmarks: CDI e IHFA via API, e Meta Atuarial do Excel
# ————————————————————————————————
# CDI
cdi_url = "https://api.maisretorno.com/v3/indexes/quotes/cdi"
resp    = requests.get(cdi_url); resp.raise_for_status()
dados   = resp.json()
if isinstance(dados, dict) and "quotes" in dados:
    df_q = pd.DataFrame(dados["quotes"])
else:
    df_q = pd.DataFrame(pd.DataFrame(dados)["quotes"].tolist())
df_q["date"]  = pd.to_datetime(df_q["d"], unit="ms")
df_q["value"] = df_q["c"]
cdi_idx = df_q.set_index("date")["value"].sort_index()

# IHFA
ihfa_url = "https://api.maisretorno.com/v3/indexes/quotes/ihfa"
resp     = requests.get(ihfa_url); resp.raise_for_status()
dados    = resp.json()
if isinstance(dados, dict) and "quotes" in dados:
    df_q = pd.DataFrame(dados["quotes"])
else:
    df_q = pd.DataFrame(pd.DataFrame(dados)["quotes"].tolist())
df_q["date"]  = pd.to_datetime(df_q["d"], unit="ms")
df_q["value"] = df_q["c"]
ihfa_idx = df_q.set_index("date")["value"].sort_index()

# Meta Atuarial (Excel)
try:
    df_atu = pd.read_excel("serie_historica_atuarial_30_06.xlsx", parse_dates=["Data"]).set_index("Data")
    atu_series = df_atu["Cota"].sort_index()
except Exception:
    atu_series = pd.Series(dtype="float64")


# ======================
# CÁLCULOS GLOBAIS (sem filtros)
# ======================
# Retornos diários dos fundos
df_ret_full   = df_p.pct_change(fill_method=None)

# CDI diário reindexado ao calendário dos fundos
cdi_daily_idx = cdi_idx.reindex(df_ret_full.index, method="ffill")
cdi_ret_daily = cdi_daily_idx.pct_change().fillna(0)

# Vol anualizada em %
df_vol_full = (
    df_ret_full
    .rolling(WINDOW, min_periods=MINP)
    .std()
    .replace(0, pd.NA)
    * (WINDOW ** 0.5) * 100
)

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

# Mapear para colunas reais presentes nas séries
macro_cols   = [c for c in df_ret_full.columns if any(c.startswith(code) for code in macro_codes)]
ls_cols      = [c for c in df_ret_full.columns if any(c.startswith(code) for code in ls_codes)]
macro_labels = [mapping.get(c, c) for c in macro_cols]
ls_labels    = [mapping.get(c, c) for c in ls_cols]

# Fundos usados nos controles (apenas os que casam com os "codes")
fund_codes   = [c for c in df_ret_full.columns if any(c.startswith(code) for code in codes)]
fund_options = [{"label": mapping.get(c, c), "value": c} for c in fund_codes]



# ======================
# CARREGA mapping_all PARA FILTROS MEG-143 E LIMPEZA
# ======================
df_all = pd.read_parquet("mapping_all.parquet")
df_pl = df_all.set_index("Ativo")["Patrimônio|||em milhares"].fillna(0) * 1000



# Ajustes iniciais e normalizações
df_all['CNPJ'] = (
    df_all['CNPJ']
    .astype(str)
    .str.replace(r'\D', '', regex=True)
    .str.zfill(14)
)
df_all['Gestora|'] = df_all['Gestora|'].astype(str).str.strip()
df_all['Patrimônio|||em milhares'] = pd.to_numeric(df_all['Patrimônio|||em milhares'], errors='coerce')
df_all['Média de|Cotistas|1 mês|em unidades'] = pd.to_numeric(
    df_all['Média de|Cotistas|1 mês|em unidades'], errors='coerce'
)
df_all['Data do|Início da Série'] = pd.to_datetime(
    df_all['Data do|Início da Série'], errors='coerce'
)
df_all['Comp carteira|3m Antes Ult|em %|Ihfa'] = pd.to_numeric(
    df_all['Comp carteira|3m Antes Ult|em %|Ihfa'], errors='coerce'
)

# Identificar gestoras com histórico no IHFA
gestoras_com_ihfa = df_all.loc[
    df_all['Comp carteira|3m Antes Ult|em %|Ihfa'].notna(),
    'Gestora|'
].unique()
df_all['Gestora com histórico IHFA'] = df_all['Gestora|'].apply(
    lambda g: 'SIM' if g in gestoras_com_ihfa else 'NÃO'
)

# Data de corte para 4 anos de atuação
data_corte = pd.Timestamp(datetime.today()) - pd.DateOffset(years=4)


# depois dos to_numeric / to_datetime acima
df_pl = (
    df_all
      .set_index("Ativo")["Patrimônio|||em milhares"]
      .astype("float64")
      .mul(1_000)          # vira R$ (era em milhares)
      .rename("PL")
)



# Dicionário de funções de filtro
filter_funcs = {
    'patrimonio': lambda d: d['Patrimônio|||em milhares'] > 300_000,
    'atuacao':     lambda d: d['Data do|Início da Série'] <= data_corte,
    'cotistas':    lambda d: d['Média de|Cotistas|1 mês|em unidades'] > 100,
    'multigestor': lambda d: ~d['Multigestor'].isin(['Multigestor', 'Espelho']),
    'exclusivo':   lambda d: d['Fundo|exclusivo'] != 'Sim',
    'condominio':  lambda d: d['Forma de|condomínio'] != 'Fechado',
    'exterior':    lambda d: ~d['Investimento|no Exterior'].isin(['', 'Até 100%']),
    'restrito':    lambda d: d['Restrito'] != 'Sim',
    'ihfa':        lambda d: d['Gestora com histórico IHFA'] == 'SIM',
}

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



from datetime import datetime, timedelta
from dash import html, dcc

def serve_layout():
    today = datetime.today().date()
    ten_years_ago = today - timedelta(days=3650)

    return html.Div(
        [
            html.H1('Painel FI Multimercado', style={'textAlign': 'center'}),

            # Filtros MEG 143
            html.H2('Filtros MEG 143'),
            dcc.Checklist(
                id='meg-filters',
                options=filter_options,
                value=[],
                inputStyle={'margin-right': '5px'}
            ),

            # Benchmarks CDI, IHFA & Meta Atuarial
            html.H2('Benchmarks: CDI, IHFA & Meta Atuarial'),
            dcc.Checklist(
                id='benchmarks-selector',
                options=[
                    {'label': 'CDI', 'value': 'cdi'},
                    {'label': 'IHFA', 'value': 'ihfa'},
                    {'label': 'Meta Atuarial', 'value': 'atu'},
                ],
                value=['cdi', 'ihfa', 'atu'],
                inputStyle={'margin-right': '5px'}
            ),

            # Índice Acumulado (Base 100 no range)
            html.H2('Índice Acumulado (Base 100 no range)'),
            dcc.DatePickerRange(
                id='cdi-date-picker',
                start_date=max(cdi_idx.index.min().date(), ten_years_ago),
                end_date=today,
                min_date_allowed=cdi_idx.index.min().date(),
                max_date_allowed=today,
                display_format='YYYY-MM-DD'
            ),
            dcc.Graph(id='cdi-chart', style={'marginTop': '20px'}),

            # Médias da Indústria (Robustas, Base 100 no range)
            html.H2('Médias da Indústria (Robustas, Base 100 no range)'),
            dcc.DatePickerRange(
                id='ind-date-picker',
                start_date=ret_ind.index.min().date(),
                end_date=ret_ind.index.max().date(),
                min_date_allowed=ret_ind.index.min().date(),
                max_date_allowed=ret_ind.index.max().date(),
                display_format='YYYY-MM-DD'
            ),
            # ✅ agora é 1 métrica por vez
            dcc.RadioItems(
                id='ind-metrics-selector',
                options=[
                    {'label': 'Retorno Acumulado (Base 100)', 'value': 'ret'},
                    {'label': 'Volatilidade', 'value': 'vol'},
                    {'label': 'Sharpe', 'value': 'sharpe'},
                ],
                value='ret',
                inline=True,
                style={'marginTop': '10px'},
                inputStyle={'margin-right': '6px'}
            ),
            # ✅ comparar com fundos (pelos nomes)
            dcc.Dropdown(
                id='ind-funds-compare',
                options=[{'label': mapping.get(a, a), 'value': a} for a in funds],
                value=[],
                multi=True,
                placeholder='Selecione fundos para comparar com a média da indústria',
                style={'width': '60%', 'marginTop': '10px'}
            ),
            dcc.Graph(id='ind-comp-chart', style={'marginTop': '20px'}),

            html.Div(
                [
                    dcc.Graph(id='ret-ind-chart'),
                    dcc.Graph(id='vol-ind-chart'),
                    dcc.Graph(id='sh-ind-chart'),
                    dcc.Graph(id='ativos-chart'),
                ],
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '20px',
                    'marginTop': '40px'
                }
            ),

            # Correlação de Retornos Diários
            html.H2('Correlação de Retornos Diários'),
            dcc.DatePickerRange(
                id='corr-date-picker',
                start_date=df_ret_full.index.min().date(),
                end_date=df_ret_full.index.max().date(),
                min_date_allowed=df_ret_full.index.min().date(),
                max_date_allowed=df_ret_full.index.max().date(),
                display_format='YYYY-MM-DD',
                style={'marginTop': '20px'}
            ),
            html.Div(
                [
                    dcc.Graph(id='macro-corr-heatmap', style={'width': '100%', 'marginTop': '40px'}),
                    dcc.Graph(id='ls-corr-heatmap', style={'width': '100%', 'marginTop': '40px'}),
                ],
                style={'display': 'flex', 'flexDirection': 'column', 'gap': '40px'}
            ),

            # Rolling Correlation
            html.H2('Correlação Rolling (ρ)'),
            dcc.DatePickerRange(
                id='rolling-corr-date-picker',
                start_date=df_ret_full.index.min().date(),
                end_date=df_ret_full.index.max().date(),
                min_date_allowed=df_ret_full.index.min().date(),
                max_date_allowed=df_ret_full.index.max().date(),
                display_format='YYYY-MM-DD',
                style={'marginBottom': '10px'}
            ),
            html.Label('Selecione até dois fundos:'),
            dcc.Dropdown(
                id='fund-selector',
                options=[{'label': mapping.get(a, a), 'value': a} for a in funds],
                value=funds[:2],
                multi=True,
                searchable=True,
                placeholder='Digite e selecione fundos...',
                style={'width': '60%', 'marginBottom': '20px'}
            ),
            dcc.Graph(id='rolling-corr-graph', style={'marginBottom': '40px'}),

            # Correlação Média x PL x Volatilidade
            html.H2('Correlação Média x PL x Volatilidade'),
            html.Div(
                [
            dcc.DatePickerRange(
                id='corrpl-date-picker',
                start_date=df_ret_full.index.min().date(),
                end_date=df_ret_full.index.max().date(),
                min_date_allowed=df_ret_full.index.min().date(),
                max_date_allowed=df_ret_full.index.max().date(),
                display_format='YYYY-MM-DD',
            ),
            dcc.RadioItems(
                id='corrpl-group',
                options=[
                {'label': 'Todos',              'value': 'all'},
                {'label': 'Todos - Macro',      'value': 'macro'},
                {'label': 'Todos - Long/Short', 'value': 'ls'},
            ],
            value='all',
            inline=True,
            style={'marginLeft': '16px'}
        ),
    ],
    style={
        'display': 'flex', 'alignItems': 'center', 'gap': '16px',
        'flexWrap': 'wrap', 'marginBottom': '8px'
    }
),
dcc.Dropdown(
    id='corrpl-highlight',
    options=[{'label': f"{mapping.get(a,a)} ({a})", 'value': a}
             for a in funds if any(a.startswith(code) for code in (macro_codes + ls_codes))],
    value=None,
    placeholder='Destaque um fundo (opcional)…',
    style={'width': '40%', 'marginBottom': '8px'}
),
dcc.Graph(id='corrpl-scatter', style={'marginBottom': '40px'}),


            # Bubble Chart 3D: Correlação × Volatilidade × Retorno
            html.H2('Bubble Chart 3D: Correlação × Volatilidade × Retorno'),
            html.Div(
                [
                    dcc.DatePickerRange(
                        id='bubble-date-picker',
                        start_date=START_FILTER.date(),
                        end_date=today,
                        min_date_allowed=START_FILTER.date(),
                        max_date_allowed=today,
                        display_format='YYYY-MM-DD',
                        style={'margin-right': '20px'}
                    ),
                    dcc.Dropdown(
                        id='bubble-funds-select',
                        options=[
                            {'label': mapping.get(a, a), 'value': a}
                            for a in funds
                            if any(a.startswith(code) for code in macro_codes + ls_codes)
                        ],
                        multi=True,
                        placeholder='Selecione fundos...',
                        style={'width': '400px'}
                    ),
                ],
                style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}
            ),
            dcc.Graph(id='bubble3d', style={'marginBottom': '40px'}),

            # ─── Bloco PCA 2D ───
            html.H2('PCA 2D de Fundos Multimercado'),
            html.Div(
                [
                    dcc.Dropdown(
                        id='pca-fundos',
                        options=fund_options,
                        value=fund_codes[:5],
                        multi=True,
                        placeholder='Selecione fundos...',
                        style={'width': '300px', 'margin-right': '20px'}
                    ),
                    dcc.DatePickerRange(
                        id='pca-date-range',
                        start_date=START_FILTER.date(),
                        end_date=today,
                        display_format='YYYY-MM-DD'
                    ),
                ],
                style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}
            ),
            dcc.Graph(id='pca-graph', style={'height': '600px'}),

            # ─── Dendrograma por Faixa de Volatilidade ───
            html.H2('Dendrograma por Faixa de Volatilidade'),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label('Faixa de Volatilidade (anualizada, %)'),
                            dcc.RangeSlider(
                                id='dendo-vol-range',
                                min=0, max=60, step=0.5,
                                value=[5, 25],
                                allowCross=False,
                                marks={i: f'{i}%' for i in range(0, 61, 5)},
                                tooltip={'always_visible': False, 'placement': 'bottom'}
                            ),
                        ],
                        style={
                            'width': '48%',
                            'display': 'inline-block',
                            'verticalAlign': 'top',
                            'paddingRight': '10px'
                        }
                    ),
                    html.Div(
                        [
                            html.Label('Selecione os Fundos'),
                            dcc.Dropdown(
                                id='dendo-funds-select',
                                options=[
                                    {'label': mapping.get(c, c), 'value': c}
                                    for c in funds
                                    if any(c.startswith(code) for code in codes)
                                ],
                                value=[
                                    c for c in funds
                                    if any(c.startswith(code) for code in codes)
                                ][:10],
                                multi=True,
                                placeholder='Escolha os fundos para o dendrograma...'
                            ),
                        ],
                        style={
                            'width': '48%',
                            'display': 'inline-block',
                            'verticalAlign': 'top'
                        }
                    ),
                ],
                style={'marginBottom': '10px'}
            ),
            dcc.Graph(id='dendo-graph', style={'height': '700px'}),

            # Análise por Fundo (Base 100 no range)
            html.H2('Análise por Fundo (Base 100 no range)'),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id=f'fund{i}-dropdown',
                                options=[{'label': mapping.get(a, a), 'value': a} for a in funds],
                                placeholder=f'Fundo {i}'
                            )
                            for i in range(1, 6)
                        ],
                        style={
                            'display': 'grid',
                            'gridTemplateColumns': 'repeat(5, 1fr)',
                            'gap': '10px'
                        }
                    ),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[
                            {'label': 'Retorno Acumulado', 'value': 'retorno'},
                            {'label': 'Volatilidade', 'value': 'volatilidade'},
                            {'label': 'Sharpe', 'value': 'sharpe'}
                        ],
                        value='retorno',
                        style={'width': '30%', 'marginTop': '20px'}
                    ),
                    dcc.Checklist(
                        id='fund-benchmarks-selector',
                        options=[
                            {'label': 'CDI', 'value': 'cdi'},
                            {'label': 'IHFA', 'value': 'ihfa'},
                            {'label': 'Meta Atuarial', 'value': 'atu'},
                        ],
                        value=['cdi', 'ihfa', 'atu'],
                        inputStyle={'margin-right': '5px'},
                        style={'marginTop': '20px'}
                    ),
                    dcc.DatePickerRange(
                        id='fund-date-picker',
                        start_date=df_ret_full.index.min().date(),
                        end_date=df_ret_full.index.max().date(),
                        min_date_allowed=df_ret_full.index.min().date(),
                        max_date_allowed=df_ret_full.index.max().date(),
                        display_format='YYYY-MM-DD',
                        style={'marginTop': '20px'}
                    ),
                    dcc.Graph(id='fund-metric-chart', style={'marginTop': '20px'}),
                ],
                style={'marginTop': '20px'}
            ),
        ],
        style={'maxWidth': '1600px', 'width': '95%', 'margin': '0 auto', 'padding': '20px'}
    )



app.layout = serve_layout

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

    # MEG ativo? (há ao menos 1 filtro)
    meg_active = bool(selected_filters)
    # universo MEG: se não houver filtros, usa TODOS os fundos
    ativos = get_allowed_activos(selected_filters or [])
    if not meg_active:
        ativos = funds[:]  # todos

    # ======================
    # RETORNO — Indústria (MEG) + (opcional) Estática
    # ======================
    # MEG
    df_ret_meg = df_ret_full[ativos].loc[sd:ed].dropna(how='all')
    if not df_ret_meg.empty:
        ret_cs_meg     = robust_cross_section_mean(df_ret_meg)
        ret_smooth_meg = ret_cs_meg.rolling(ROLL_MM, min_periods=1).mean().dropna()
        ret_slice      = base100_from(((1 + ret_smooth_meg).cumprod()).dropna())
    else:
        ret_slice = pd.Series(dtype='float64')

    # Estática (todos os fundos), só quando MEG está ativo
    if meg_active:
        df_ret_all = df_ret_full[funds].loc[sd:ed].dropna(how='all')
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
    # VOL — Indústria (MEG) + (opcional) Estática
    # ======================
    df_vol_meg = df_vol_full[ativos].loc[sd:ed].dropna(how='all')
    if not df_vol_meg.empty:
        vol_cs_meg = df_vol_meg.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
    else:
        vol_cs_meg = pd.Series(dtype='float64')

    if meg_active:
        df_vol_all = df_vol_full[funds].loc[sd:ed].dropna(how='all')
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
    # SHARPE — Indústria (MEG) + (opcional) Estática
    # ======================
    df_sh_meg = sharpe_full[ativos].loc[sd:ed].dropna(how='all')
    if not df_sh_meg.empty:
        sh_cs_meg = df_sh_meg.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
    else:
        sh_cs_meg = pd.Series(dtype='float64')

    if meg_active:
        df_sh_all = sharpe_full[funds].loc[sd:ed].dropna(how='all')
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
    # Nº de fundos por ano (sempre baseado no universo mostrado)
    # ======================
    if ativos:
        ativos_por_ano_f = (
            df_p[ativos].loc[sd:ed]
            .stack(dropna=True)
            .reset_index(level=1, name='Preco')
            .assign(Ano=lambda df: df.index.year)
            .groupby('Ano')['Ativo'].nunique()
        )
    else:
        ativos_por_ano_f = pd.Series(dtype='float64')

    anos = list(range(sd.year, ed.year + 1))
    ativos_por_ano_f = ativos_por_ano_f.reindex(anos, fill_value=0)

    fig_ativ = go.Figure(go.Scatter(
        x=ativos_por_ano_f.index, y=ativos_por_ano_f.values,
        mode='markers+lines', name='Nº de Fundos (universo exibido)'
    ))
    fig_ativ.update_layout(
        title='Evolução Anual do Nº de FI Multimercado',
        xaxis_title='Ano', yaxis_title='Número de Fundos',
        template='plotly_white'
    )

    return fig_ret, fig_vol, fig_sh, fig_ativ






from dash.dependencies import Input, Output, State

# --- opções do dropdown "ind-funds-compare" sensíveis ao MEG ---
from dash.dependencies import Input, Output, State

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

    ativos = get_allowed_activos(selected_filters or [])
    if not ativos:
        return _empty_fig("Média da Indústria — sem dados", "Nenhum fundo atende aos filtros do MEG selecionados.")

    # 2) série da indústria (apenas fundos filtrados)
    if selected_metric == 'ret':
        df_ret = df_ret_full[ativos].loc[sd:ed]
        if df_ret.empty:
            return _empty_fig("Média da Indústria — Retorno (Base 100)", "Sem dados disponíveis no período.")
        ret_cs     = robust_cross_section_mean(df_ret)
        ret_smooth = ret_cs.rolling(ROLL_MM, min_periods=1).mean()
        ret_cum    = (1 + ret_smooth).cumprod().dropna()
        if ret_cum.empty:
            return _empty_fig("Média da Indústria — Retorno (Base 100)", "Sem dados válidos após limpeza.")
        ret_cum    = start_at_first_valid(ret_cum)
        ind_series = (ret_cum / ret_cum.iloc[0]) * 100
        y_title, metric_label = 'Retorno (Base 100)', 'Retorno (Base 100)'

    elif selected_metric == 'vol':
        df_vol = df_vol_full[ativos].loc[sd:ed]
        if df_vol.empty:
            return _empty_fig("Média da Indústria — Volatilidade", "Sem dados disponíveis no período.")
        vol_cs     = df_vol.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
        if vol_cs.empty:
            return _empty_fig("Média da Indústria — Volatilidade", "Sem dados válidos após limpeza.")
        ind_series = vol_cs
        y_title, metric_label = 'Volatilidade (%)', 'Volatilidade'

    else:  # 'sharpe'
        df_sh = sharpe_full[ativos].loc[sd:ed]
        if df_sh.empty:
            return _empty_fig("Média da Indústria — Sharpe", "Sem dados disponíveis no período.")
        sh_cs = df_sh.apply(trim_mean_series, axis=1).rolling(ROLL_MM, min_periods=1).mean().dropna()
        if sh_cs.empty:
            return _empty_fig("Média da Indústria — Sharpe", "Sem dados válidos após limpeza.")
        ind_series = sh_cs
        y_title, metric_label = 'Sharpe', 'Sharpe'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ind_series.index, y=ind_series.values, mode='lines',
                             name='Média da Indústria', line=dict(width=3)))

    # 4) fundos selecionados (restritos ao universo MEG)
    sel = [f for f in (selected_funds or []) if f in ativos]
    for f in sel:
        if selected_metric == 'ret':
            s = (1 + df_ret_full[f].loc[sd:ed].fillna(0.0)).cumprod().dropna()
            if s.empty: 
                continue
            s = start_at_first_valid(s)
            s = (s / s.iloc[0]) * 100
        elif selected_metric == 'vol':
            s = df_vol_full[f].loc[sd:ed].rolling(ROLL_MM, min_periods=1).mean().dropna()
        else:
            s = sharpe_full[f].loc[sd:ed].rolling(ROLL_MM, min_periods=1).mean().dropna()

        if not s.empty:
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=mapping.get(f, f), opacity=0.9))

    fig.update_layout(
        title=f"Média da Indústria — {metric_label} (MEG aplicados: {len(ativos)} fundos)",
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
    import numpy as np

    if not sel_codes:
        return go.Figure(layout_title_text="Selecione ao menos um fundo.")

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    # Slices brutos
    df_ret = df_ret_full.reindex(columns=sel_codes).loc[sd:ed]
    df_vol = df_vol_full.reindex(columns=sel_codes).loc[sd:ed]

    if df_ret.dropna(how="all").empty:
        return go.Figure(layout_title_text="Nenhum dado disponível neste período.")

    # --- helpers robustos ---
    MIN_OBS = 30  # mínimo de observações de retorno por fundo

    def ret_base100_end(col: pd.Series) -> float:
        s = col.dropna()
        if s.size < 2:
            return np.nan
        cp = (1 + s).cumprod()
        return float((cp.iloc[-1] / cp.iloc[0]) * 100)

    def vol_mean_pct(code: str) -> float:
        # tenta média da vol anualizada já calculada; se NaN, calcula da série de retorno
        v = df_vol[code].dropna().mean()
        if pd.isna(v):
            r = df_ret[code].dropna()
            if r.size >= 2:
                v = float(r.std() * (252 ** 0.5) * 100)
        return v

    # --- filtra fundos com dados mínimos de retorno ---
    enough_obs = {c: df_ret[c].dropna().size for c in sel_codes}
    valid_for_ret = [c for c, n in enough_obs.items() if n >= MIN_OBS]

    if not valid_for_ret:
        msg = "Todos os fundos têm menos de {MIN_OBS} observações de retorno no período."
        return go.Figure(layout_title_text=msg)

    # --- métricas por fundo ---
    ret_b100 = pd.Series({c: ret_base100_end(df_ret[c]) for c in valid_for_ret})
    vol_avg  = pd.Series({c: vol_mean_pct(c) for c in valid_for_ret})

    # correlação média: se só 1, define 0; senão, média por linha sem diagonal
    if len(valid_for_ret) == 1:
        avg_corr = pd.Series({valid_for_ret[0]: 0.0})
    else:
        C = df_ret[valid_for_ret].corr(min_periods=MIN_OBS)
        np.fill_diagonal(C.values, np.nan)
        avg_corr = C.mean(skipna=True)

    # PL seguro (float)
    pl_vals = pd.to_numeric(df_pl.reindex(valid_for_ret), errors="coerce").fillna(0.0)
    max_pl  = float(pl_vals.max()) if np.isfinite(pl_vals.max()) and pl_vals.max() > 0 else 1.0
    sizes   = ((pl_vals / max_pl) * 40 + 5).astype(float)

    # monta DF e remove linhas sem métrica essencial
    df_plot = (
        pd.DataFrame({
            "Code":       valid_for_ret,
            "AvgCorr":    avg_corr.reindex(valid_for_ret).fillna(0.0).values,  # fallback 0
            "Volatility": vol_avg.reindex(valid_for_ret).values,
            "RetBase100": ret_b100.reindex(valid_for_ret).values,
            "PL":         pl_vals.reindex(valid_for_ret).values,
            "Size":       sizes.reindex(valid_for_ret).values,
            "Fund":       [mapping.get(c, c) for c in valid_for_ret],
        })
        .dropna(subset=["Volatility", "RetBase100"])
    )

    if df_plot.empty:
        # Diagnóstico: explica quem foi descartado e por quê
        reasons = []
        for c in valid_for_ret:
            r = []
            if pd.isna(ret_b100.get(c)):
                r.append("RetBase100")
            if pd.isna(vol_avg.get(c)):
                r.append("Volatilidade")
            if not r:
                r.append("desconhecido")
            reasons.append(f"{mapping.get(c,c)} ({c}): {', '.join(r)}")
        msg = "Sem dados suficientes após limpeza:\n" + "\n".join(reasons)
        return go.Figure(layout_title_text=msg)

    # Plot
    fig = go.Figure()
    for _, row in df_plot.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row["AvgCorr"]],
            y=[row["Volatility"]],
            z=[row["RetBase100"]],
            mode="markers",
            name=row["Fund"],
            marker=dict(
                size=float(row["Size"]),
                color=float(row["Volatility"]),
                sizemode="diameter",
                line=dict(width=1, color="black"),
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "ρ médio: %{x:.2f}<br>"
                "Vol (%): %{y:.2f}<br>"
                "Ret (Base 100): %{z:.2f}<br>"
                "PL: R$ %{customdata:,.0f}<extra></extra>"
            ),
            text=row["Fund"],
            customdata=[row["PL"]],
        ))

    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis_title="Média de Correlação (ρ)",
            yaxis_title="Volatilidade (%)",
            zaxis_title="Retorno Acumulado (Base 100)"
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        title=f"Período: {sd.date()} → {ed.date()}",
        template="plotly_white"
    )
    fig.update_scenes(xaxis=dict(range=[-1, 1]))
    return fig



# ─── Bloco: Callback para o PCA ───

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
        return px.scatter(title="Selecione ao menos um fundo.")

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    df_ret = df_ret_full[fund_list].loc[sd:ed]
    if df_ret.empty:
        return px.scatter(title="Nenhum dado disponível neste período.")
    df_vol = df_vol_full[fund_list].loc[sd:ed]

    avg_corr    = df_ret.corr().mean()
    avg_vol     = df_vol.mean()
    ret_cum_end = (1 + df_ret).cumprod().iloc[-1] * 100
    pl_vals     = df_pl.reindex(fund_list)

    features = pd.DataFrame({
        'AvgCorr':    avg_corr,
        'Volatility': avg_vol,
        'RetBase100': ret_cum_end,
        'PL':         pl_vals
    }).dropna()

    # PCA
    n = features.shape[0]
    if n >= 2:
        pcs = PCA(n_components=2).fit_transform(features[['AvgCorr','Volatility','RetBase100']])
        df_pca = pd.DataFrame(pcs, index=features.index, columns=['PC1','PC2'])
    else:
        df_pca = pd.DataFrame({
            'PC1': features['AvgCorr'],
            'PC2': features['Volatility']
        }, index=features.index)

    df_pca['Fund'] = df_pca.index.map(mapping)
    df_plot = df_pca.join(features['PL'].astype(float))

    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Fund',
        size='PL',
        hover_name='Fund',
        title=f"PCA 2D ({sd.date()} → {ed.date()})",
        labels={'PC1':'Componente 1','PC2':'Componente 2'},
        size_max=60,
        color_discrete_sequence=px.colors.qualitative.Dark24
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




if __name__ == "__main__":
    # Local
    app.run_server(host="0.0.0.0", port=8050, debug=True)



# In[8]:


get_ipython().run_line_magic('cd', '"C:/Users/otavi/Documents/MM_Dash"')


# In[ ]:




