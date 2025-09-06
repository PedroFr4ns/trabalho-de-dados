# pages/2_modelos.py — PT-BR + Métricas + Gráficos

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, adjusted_rand_score
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Health Risk App", layout="wide")
st.sidebar.success("Escolha uma página acima ☝️")

st.title("🤖 Modelos — Regressão Linear & K-Means")
st.caption("Treina os dois modelos, exibe métricas, gráficos e permite predição com inputs manuais.")

# ===== colunas do dataset =====
FEATURE_COLS = [
    "Respiratory_Rate",
    "Oxygen_Saturation",
    "O2_Scale",
    "Systolic_BP",
    "Heart_Rate",
    "Temperature",
    "Consciousness",
    "On_Oxygen",
]
TARGET_COL = "Risk_Level"

# labels PT-BR para inputs
LABEL = {
    "Respiratory_Rate": "Frequência respiratória (irpm)",
    "Oxygen_Saturation": "Saturação de O₂ (%)",
    "O2_Scale": "Escala de O₂ (2 = verdadeiro)",
    "Systolic_BP": "Pressão sistólica (mmHg)",
    "Heart_Rate": "Frequência cardíaca (bpm)",
    "Temperature": "Temperatura (°C)",
    "Consciousness": "Nível de consciência (A=1, P=2, V=3, U=4, C=5)",
    "On_Oxygen": "Em oxigênio suplementar?",
}

def fmt(x: float, casas: int = 3) -> str:
    return f"{x:.{casas}f}".replace(".", ",")

# === CSV padrão: prefere o arquivo otimizado ===
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

# ---------- limpeza/conversões ----------
def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # booleanos 0/1
    for bcol in ["O2_Scale", "On_Oxygen"]:
        if bcol in df.columns:
            if df[bcol].dtype == "bool":
                df[bcol] = df[bcol].astype(int)
            else:
                df[bcol] = (
                    df[bcol]
                    .replace({"True": 1, "False": 0, True: 1, False: 0})
                    .astype("Int64")
                ).fillna(0).astype(int)

    # Consciousness: A,P,V,U,C -> 1..5 se vier texto
    if "Consciousness" in df.columns and df["Consciousness"].dtype == "object":
        map_cons = {"A": 1, "P": 2, "V": 3, "U": 4, "C": 5}
        df["Consciousness"] = df["Consciousness"].str.strip().map(map_cons).astype("Int64")

    # Risk_Level: Normal,Low,Medium,High -> 1..4 se vier texto
    if "Risk_Level" in df.columns and df["Risk_Level"].dtype == "object":
        map_risk = {"Normal": 1, "Low": 2, "Medium": 3, "High": 4}
        df["Risk_Level"] = df["Risk_Level"].str.strip().map(map_risk).astype("Int64")

    # numéricos principais
    for c in ["Respiratory_Rate", "Oxygen_Saturation", "Systolic_BP", "Heart_Rate", "Temperature"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # finalizar tipos e remover NaNs
    if "Consciousness" in df.columns:
        df["Consciousness"] = df["Consciousness"].astype("Int64")
    if "Risk_Level" in df.columns:
        df["Risk_Level"] = df["Risk_Level"].astype("Int64")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    df["Consciousness"] = df["Consciousness"].astype(int)
    df["Risk_Level"] = df["Risk_Level"].astype(int)
    return df

# --------- carregar CSV (widgets FORA de cache) ---------
st.sidebar.header("⚙️ Dados")
mode = st.sidebar.radio("Fonte do CSV", ("Usar arquivo local do projeto", "Enviar um CSV"), index=0)

df = None
if mode == "Enviar um CSV":
    up = st.sidebar.file_uploader("Envie o arquivo CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
else:
    default_path = find_default_csv()
    path_text = st.sidebar.text_input(
        "Caminho do CSV (opcional)",
        value=str(default_path),
        help="Por padrão, tenta o arquivo otimizado 'Cell output 23 [DW].csv' na raiz do app.",
    )
    csv_path = Path(path_text).expanduser()

    @st.cache_data(show_spinner=False)
    def load_csv_from_disk(p: str):
        return pd.read_csv(p)

    if csv_path.exists():
        df = load_csv_from_disk(str(csv_path))
    else:
        st.warning(f"Arquivo não encontrado: {csv_path}")

if df is None:
    st.warning("Envie ou selecione um CSV válido para continuar.")
    st.stop()

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"O CSV está faltando colunas: {missing}")
    st.stop()

# --------- treino + artefatos p/ gráficos ---------
@st.cache_resource(show_spinner=True)
def train_models(df_in: pd.DataFrame, random_state: int = 42):
    df_clean = ensure_types(df_in)

    X = df_clean[FEATURE_COLS].copy()
    y = df_clean[TARGET_COL].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Regressão Linear
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)
    linreg = LinearRegression()
    linreg.fit(X_tr, y_tr)
    y_pred = linreg.predict(X_te)
    mse = float(mean_squared_error(y_te, y_pred))
    r2 = float(r2_score(y_te, y_pred))
    residuals = (y_te - y_pred)

    # K-Means
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)
    ari = float(adjusted_rand_score(y, clusters))
    inertia = float(kmeans.inertia_)  # SSE

    # PCA para projeção 2D (para gráficos)
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)  # shape (n_samples, 2)

    # cluster -> rótulo por maioria
    mapping = {}
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        majority = Counter(y.iloc[idx]).most_common(1)[0][0]
        mapping[int(c)] = int(majority)

    metrics = {
        "linreg_mse": mse,
        "linreg_r2": r2,
        "kmeans_ari": ari,
        "kmeans_inertia": inertia,
    }
    artifacts = {
        "scaler": scaler,
        "linreg": linreg,
        "kmeans": kmeans,
        "cluster_to_label": mapping,
        "X_test": X_te,
        "y_test": y_te,
        "y_pred": y_pred,
        "residuals": residuals,
        "X_pca": X_pca,
        "clusters": clusters,
        "y_all": y.values,
    }
    return artifacts, metrics

artifacts, metrics = train_models(df)

# ===== métricas =====
st.subheader("📊 Métricas do Treino")
c1, c2, c3 = st.columns(3)
c1.metric("Regressão — EQM", fmt(metrics["linreg_mse"]))
c2.metric("Regressão — R²", fmt(metrics["linreg_r2"]))
c3.metric("K-Means — ARI", fmt(metrics["kmeans_ari"]))
st.caption(f"Inércia (K-Means / SSE): {fmt(metrics['kmeans_inertia'])}")

# ===== GRÁFICOS =====
st.subheader("📈 Gráficos")

# ---- Regressão: y_real vs y_previsto + resíduos ----
st.markdown("**Regressão Linear**")
gc1, gc2 = st.columns(2)

with gc1:
    y_test = artifacts["y_test"]
    y_pred = artifacts["y_pred"]
    # dispersão y_real vs y_previsto
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    # linha y=x
    lo = min(min(y_test), min(y_pred))
    hi = max(max(y_test), max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Valor real (Risk_Level)")
    ax.set_ylabel("Valor previsto")
    ax.set_title("Dispersão: real vs previsto")
    st.pyplot(fig)

with gc2:
    residuals = artifacts["residuals"]
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=20)
    ax.set_xlabel("Resíduo (real − previsto)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição dos resíduos")
    st.pyplot(fig)

# ---- K-Means: PCA 2D por cluster e por rótulo real ----
st.markdown("**K-Means (visualização em 2D via PCA)**")
kc1, kc2 = st.columns(2)
X_pca = artifacts["X_pca"]
clusters = artifacts["clusters"]
y_all = artifacts["y_all"]

with kc1:
    fig, ax = plt.subplots()
    for c in np.unique(clusters):
        pts = X_pca[clusters == c]
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.7, label=f"Cluster {c}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Amostras coloridas por cluster")
    ax.legend()
    st.pyplot(fig)

with kc2:
    fig, ax = plt.subplots()
    for lbl in np.unique(y_all):
        pts = X_pca[y_all == lbl]
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.7, label=f"Risco {lbl}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Amostras coloridas por nível de risco real")
    ax.legend()
    st.pyplot(fig)

# tabela de contingência (risk × cluster)
st.markdown("**Contingência (Risco real × Cluster)**")
ct = pd.crosstab(pd.Series(y_all, name="Risco real"), pd.Series(clusters, name="Cluster"))
st.dataframe(ct, use_container_width=True)

# ===== PREVISÃO COM INPUTS =====
st.subheader("🧮 Prever com novos dados")
c1, c2 = st.columns(2)
with c1:
    resp_rate = st.number_input(LABEL["Respiratory_Rate"], min_value=5, max_value=60, value=20, step=1)
    oxy_sat   = st.number_input(LABEL["Oxygen_Saturation"],  min_value=50, max_value=100, value=96, step=1)
    o2_scale  = st.checkbox(LABEL["O2_Scale"], value=False)
    systolic  = st.number_input(LABEL["Systolic_BP"],        min_value=70, max_value=220, value=110, step=1)
with c2:
    heart = st.number_input(LABEL["Heart_Rate"], min_value=30, max_value=220, value=90, step=1)
    temp  = st.number_input(LABEL["Temperature"], min_value=30.0, max_value=43.0, value=37.2, step=0.1, format="%.1f")
    conc  = st.slider(LABEL["Consciousness"], min_value=1, max_value=5, value=1, step=1)
    on_oxy = st.checkbox(LABEL["On_Oxygen"], value=False)

row = pd.DataFrame([{
    "Respiratory_Rate": int(resp_rate),
    "Oxygen_Saturation": int(oxy_sat),
    "O2_Scale": int(o2_scale),
    "Systolic_BP": int(systolic),
    "Heart_Rate": int(heart),
    "Temperature": float(temp),
    "Consciousness": int(conc),
    "On_Oxygen": int(on_oxy),
}])

if st.button("Calcular predições"):
    scaler = artifacts["scaler"]
    linreg = artifacts["linreg"]
    kmeans = artifacts["kmeans"]
    c2l = artifacts["cluster_to_label"]

    row_prep = ensure_types(row)[FEATURE_COLS]
    row_scaled = scaler.transform(row_prep)

    y_cont = float(linreg.predict(row_scaled)[0])
    y_round = int(np.clip(np.rint(y_cont), 1, 4))

    cluster = int(kmeans.predict(row_scaled)[0])
    estimated_label = c2l.get(cluster, None)

    st.write("### Resultado")
    st.write(f"- **Regressão Linear — nível de risco (contínuo):** `{fmt(y_cont)}`  → **arredondado:** `{y_round}`")
    st.write(f"- **K-Means — cluster:** `{cluster}`  → **nível de risco estimado (maioria do treino):** `{estimated_label}`")
    st.info("Observação: K-Means é não supervisionado; o rótulo estimado é uma heurística (maioria no treino).")
