from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Health Risk App", layout="wide")
st.sidebar.success("Escolha uma página acima ☝️")
st.title("📊 Health Risk Dashboard")
st.caption("Explore e visualize o conjunto de dados clínicos por diferentes perspectivas.")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_CANDIDATES = [
    "Cell output 23 [DW].csv",   # preferido (otimizado)
    "Health_Risk_Dataset.csv",   # fallback
]
def find_default_csv() -> Path:
    for name in DEFAULT_CSV_CANDIDATES:
        p = ROOT / name
        if p.exists():
            return p
    return ROOT / DEFAULT_CSV_CANDIDATES[0]

DEFAULT_RELATIVE = Path(__file__).resolve().parent.parent / "Health_Risk_Dataset.csv"
NUMERIC_COLS = [
    "Respiratory_Rate",
    "Oxygen_Saturation",
    "O2_Scale",
    "Systolic_BP",
    "Heart_Rate",
    "Temperature",
    "On_Oxygen",
]
CATEG_COLS = ["Consciousness", "Risk_Level"]

@st.cache_data(show_spinner=False)
def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    for col in ["On_Oxygen", "O2_Scale"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in CATEG_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# === Sidebar: Fonte do arquivo ===
st.sidebar.header("⚙️ Configurações")
file_mode = st.sidebar.radio(
    "Fonte do CSV",
    ("Usar arquivo local do projeto", "Enviar um CSV"),
    index=0,
)

if file_mode == "Enviar um CSV":
    up = st.sidebar.file_uploader("Envie o arquivo CSV", type=["csv"]) 
    if up is not None:
        df = pd.read_csv(up)
    else:
        st.info("Envie um arquivo CSV para começar ou mude para 'Usar arquivo local do projeto'.")
        st.stop()
else:
    path_text = st.sidebar.text_input(
        "Caminho do CSV (opcional)",
        value=str(DEFAULT_RELATIVE),
        help="Por padrão, procura por 'Cell output 23 [DW].csv' na raiz do app.",
    )
    csv_path = Path(path_text).expanduser()
    if not csv_path.exists():
        st.error(f"CSV não encontrado em: {csv_path}")
        st.stop()
    df = load_data(csv_path)

st.success(f"Dados carregados: {len(df):,} linhas × {len(df.columns)} colunas")

# === Sidebar: Filtros ===
st.sidebar.subheader("Filtros")
use_filters = st.sidebar.checkbox("Ativar filtros", value=False)
filtered = df.copy()

if use_filters:
    if "Risk_Level" in df.columns:
        opts = sorted(df["Risk_Level"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Risk Level", opts, default=opts)
        filtered = filtered[filtered["Risk_Level"].isin(sel)]

    if "Consciousness" in df.columns:
        opts = sorted(df["Consciousness"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Consciousness", opts, default=opts)
        filtered = filtered[filtered["Consciousness"].isin(sel)]

    if "On_Oxygen" in df.columns:
        opts = sorted(filtered["On_Oxygen"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("On Oxygen (0/1)", opts, default=opts)
        filtered = filtered[filtered["On_Oxygen"].isin(sel)]

    for col in NUMERIC_COLS:
        if col in filtered.columns:
            cmin = float(filtered[col].min())
            cmax = float(filtered[col].max())
            if np.isfinite(cmin) and np.isfinite(cmax):
                vmin, vmax = st.sidebar.slider(col, cmin, cmax, (cmin, cmax))
                filtered = filtered[(filtered[col] >= vmin) & (filtered[col] <= vmax)]

st.caption(f"Após filtros: {len(filtered):,} linhas")

# === Seletor de análise ===
analysis = st.sidebar.selectbox(
    "Análise",
    [
        "Resumo",
        "Distribuição de risco",
        "Boxplots por risco",
        "Correlação (heatmap)",
        "Dispersões (scatter)",
        "Distribuições (histogramas)",
    ],
)

# === RESUMO ===
if analysis == "Resumo":
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Colunas")
        st.write(pd.DataFrame({
            "coluna": df.columns,
            "tipo": [str(df[c].dtype) for c in df.columns],
            "n_na": [int(df[c].isna().sum()) for c in df.columns],
        }))
    with c2:
        st.subheader("Amostras (head)")
        st.write(filtered.head(20))

    st.subheader("Contagem por nível de risco")
    if "Risk_Level" in filtered.columns:
        counts = filtered["Risk_Level"].value_counts().sort_index()
        st.bar_chart(counts)
    else:
        st.info("Coluna 'Risk_Level' não encontrada no dataset.")

elif analysis == "Distribuição de risco":
    st.subheader("Distribuição de pacientes por nível de risco")
    if "Risk_Level" not in filtered.columns:
        st.error("Coluna 'Risk_Level' não encontrada.")
    else:
        counts = filtered["Risk_Level"].value_counts().sort_index()
        c1, c2 = st.columns([1,1])
        with c1:
            st.bar_chart(counts)
        with c2:
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            ax.set_title("Proporção por risco")
            st.pyplot(fig)

elif analysis == "Boxplots por risco":
    if "Risk_Level" not in filtered.columns:
        st.error("Coluna 'Risk_Level' não encontrada.")
    else:
        st.subheader("Boxplots de variáveis contínuas por nível de risco")
        vars_disp = [c for c in NUMERIC_COLS if c in filtered.columns and c != "On_Oxygen"]
        var_sel = st.selectbox("Escolha a variável", vars_disp, index=0)
        groups = [g[var_sel].dropna().values for _, g in filtered.groupby("Risk_Level")]
        labels = [str(k) for k, _ in filtered.groupby("Risk_Level")]
        if len(groups) == 0:
            st.info("Sem dados suficientes para o boxplot.")
        else:
            fig, ax = plt.subplots()
            ax.boxplot(groups, labels=labels, showmeans=True)
            ax.set_xlabel("Risk Level")
            ax.set_ylabel(var_sel)
            ax.set_title(f"{var_sel} por nível de risco")
            st.pyplot(fig)

elif analysis == "Correlação (heatmap)":
    st.subheader("Matriz de correlação (variáveis numéricas)")
    num_cols_present = [c for c in NUMERIC_COLS if c in filtered.columns]
    corr = filtered[num_cols_present].corr(numeric_only=True)
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlação entre variáveis")
    st.pyplot(fig)

elif analysis == "Distribuições (histogramas)":
    st.subheader("Histogramas por variável")
    vars_disp = [c for c in NUMERIC_COLS if c in filtered.columns and c != "On_Oxygen"]
    var_sel = st.selectbox("Escolha a variável", vars_disp, index=0)
    overlay_oxygen = st.checkbox("Separar por On_Oxygen (0/1)", value=True)

    fig, ax = plt.subplots()
    data = filtered[var_sel].dropna()
    if overlay_oxygen and "On_Oxygen" in filtered.columns:
        for val in sorted(filtered["On_Oxygen"].dropna().unique()):
            subset = filtered[filtered["On_Oxygen"] == val][var_sel].dropna()
            ax.hist(subset, bins=20, alpha=0.6, label=f"On_Oxygen={val}")
        ax.legend()
    else:
        ax.hist(data, bins=20)
    ax.set_xlabel(var_sel)
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição de {var_sel}")
    st.pyplot(fig)

elif analysis == "Dispersões (scatter)":
    st.subheader("Relação entre variáveis (scatter)")
    vars_disp = [c for c in NUMERIC_COLS if c in filtered.columns]
    x = st.selectbox("Eixo X", vars_disp, index=0)
    y = st.selectbox("Eixo Y", vars_disp, index=min(1, len(vars_disp)-1))
    color_by_risk = st.checkbox("Colorir por Risk_Level", value=True)

    fig, ax = plt.subplots()
    if color_by_risk and "Risk_Level" in filtered.columns:
        for lvl, g in filtered.groupby("Risk_Level"):
            ax.scatter(g[x], g[y], alpha=0.7, label=str(lvl))
        ax.legend(title="Risk_Level")
    else:
        ax.scatter(filtered[x], filtered[y], alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    st.pyplot(fig)

st.markdown("---")
st.caption("Dica: ative os filtros na barra lateral para refinar as visualizações.")
