# streamlit_app.py
import re
import pandas as pd
import streamlit as st

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Tenta usar webdriver-manager (facilita localmente)
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WDM = True
except Exception:
    USE_WDM = False

# ---------------------------
# Helpers
# ---------------------------
def is_valid_url(u: str) -> bool:
    return bool(re.match(r"^https?://", (u or "").strip()))

def build_driver():
    """Cria um Chrome headless robusto (server-friendly)."""
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--window-size=1920,1080")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--disable-extensions")
    chrome_opts.add_argument("--start-maximized")
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option("useAutomationExtension", False)

    if USE_WDM:
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_opts)
    else:
        return webdriver.Chrome(options=chrome_opts)

def safe_get_text(driver, by, selector, wait: WebDriverWait, log, attr=None):
    """
    Busca um elemento com espera explícita e retorna .text ou o atributo solicitado.
    Em caso de falha, loga aviso e retorna string vazia (não quebra o fluxo).
    """
    try:
        elem = wait.until(EC.presence_of_element_located((by, selector)))
        if attr:
            val = elem.get_attribute(attr)
            if val:
                return val.strip()
        return elem.text.strip()
    except Exception as e:
        log.append(f"[WARN] Não encontrei '{selector}' ({by}). Detalhe: {e}")
        return ""

def scrape_post(url: str, log: list) -> dict:
    driver = build_driver()
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        # ===== Ajuste os seletores conforme o HTML real do site-alvo =====
        titulo = safe_get_text(driver, By.CSS_SELECTOR, "h1", wait, log)

        autor = ""
        for sel in [".author", "[itemprop='author']", "a[rel='author']"]:
            if not autor:
                autor = safe_get_text(driver, By.CSS_SELECTOR, sel, wait, log)

        data = safe_get_text(driver, By.CSS_SELECTOR, "time", wait, log, attr="datetime")
        if not data:
            data = safe_get_text(driver, By.CSS_SELECTOR, "time, .date, .posted-on", wait, log)

        views = ""
        for sel in [".views", "[data-views]", "[class*='view']"]:
            if not views:
                views = safe_get_text(driver, By.CSS_SELECTOR, sel, wait, log)

        likes = ""
        for sel in [".likes", "[data-likes]", "[class*='like']"]:
            if not likes:
                likes = safe_get_text(driver, By.CSS_SELECTOR, sel, wait, log)

        comentarios = ""
        for sel in [".comments-count", "[data-comments]", "[class*='comment']"]:
            if not comentarios:
                comentarios = safe_get_text(driver, By.CSS_SELECTOR, sel, wait, log)

        # >>> Novo campo: Compartilhamentos <<<
        compartilhametos = ""  # nome interno sem acento evita confusões
        for sel in [".shares", "[data-shares]", "[class*='share']"]:
            if not compartilhametos:
                compartilhametos = safe_get_text(driver, By.CSS_SELECTOR, sel, wait, log)

        return {
            "URL": url,
            "Título": titulo,
            "Autor": autor,
            "Data": data,
            "Views": views,
            "Likes": likes,
            "Comentários": comentarios,
            "Compartilhamentos": compartilhametos
        }
    finally:
        driver.quit()

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Scraper de Métricas", page_icon="📊", layout="centered")
st.title("📊 Scraper de Métricas de Post por URL")
st.caption("Cole o link do post para extrair Título, Autor, Data, Views, Likes, Comentários e Compartilhamentos.")

url = st.text_input("URL do post", placeholder="https://exemplo.com/post/123")
colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("Extrair métricas")
with colB:
    show_logs = st.toggle("Mostrar logs de debug", value=False)

if run_btn:
    if not is_valid_url(url):
        st.error("Informe uma URL válida (começando com http:// ou https://).")
        st.stop()

    logs = []
    with st.spinner("Coletando dados…"):
        try:
            data = scrape_post(url, logs)
        except Exception as e:
            st.error(f"Falha ao extrair dados: {e}")
            if show_logs:
                st.code("\n".join(logs) or "(sem logs)", language="bash")
            st.stop()

    # Verificação mínima: pelo menos um dos campos de métrica
    if not any([data.get("Título"), data.get("Autor"), data.get("Views"),
                data.get("Likes"), data.get("Comentários"), data.get("Compartilhamentos")]):
        st.warning("Consegui abrir a página, mas não identifiquei os seletores padrão. "
                   "Ajuste os seletores no código para o HTML do site alvo.")
        if show_logs:
            st.code("\n".join(logs) or "(sem logs)", language="bash")
        st.stop()

    st.success("Extração concluída!")
    df = pd.DataFrame([data])
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Baixar CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="post_metrics.csv",
        mime="text/csv"
    )

    if show_logs:
        st.subheader("Logs de debug")
        st.code("\n".join(logs) or "(sem logs)", language="bash")

st.markdown("---")
