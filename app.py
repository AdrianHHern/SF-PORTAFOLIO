import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize


st.markdown("""
<style>
h2 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #2b2d42 !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Análisis de Portafolios",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f9fafc;
}

section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

div[data-testid="stSlider"] > div > div > div:nth-child(2) > div {
    background-color: #1f77b4 !important;
}


div[data-testid="stSlider"] div[role="slider"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}


div[data-testid="stRadio"] > label > div:first-child {
    border: 2px solid #7a3db8 !important;
}
div[data-testid="stRadio"] > label > div:first-child:hover {
    border-color: #a86dd8 !important;
}
div[data-testid="stRadio"] > label > div[aria-checked="true"] {
    background-color: #7a3db8 !important;
    border-color: #7a3db8 !important;
}

div.stButton > button {
    background-color: #7a3db8 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    border: none !important;
    font-weight: 600 !important;
}
div.stButton > button:hover {
    background-color: #5c2c91 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.stApp {
    background-color: #f9fafc;
}

section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

div[data-testid="stSlider"] > div > div > div:nth-child(2) > div {
    background-color: #1f77b4 !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-testid="stRadio"] > label > div:first-child {
    border: 2px solid #7a3db8 !important;
}

div[data-testid="stRadio"] > label > div:first-child:hover {
    border-color: #a86dd8 !important;
}

div[data-testid="stRadio"] > label > div[aria-checked="true"] {
    background-color: #7a3db8 !important;
    border-color: #7a3db8 !important;
}

</style>
""", unsafe_allow_html=True)






# 1. Datos de los Tickers por Regiones y Sectores
# Se muestran los diversos ETF de Regiones como EUA, Cánada, Europa, entre otros.
# Se muestran los de Sectores como los de Comunicaciones, consumo discrecional, consumo básico, energía, etc.

TICKERS_REGIONES = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]

TICKERS_SECTORES = [
    "XLC","XLY","XLP","XLE","XLF",
    "XLV","XLI","XLB","XLRE","XLK","XLU"
]
#Se tienen los pesos por sector y región, datos que se dan en el pdf del proyecto.

PESOS_REGIONES = {
    "SPLG":0.7062,
    "EWC":0.0323,
    "IEUR":0.1176,
    "EEM":0.0902,
    "EWJ":0.0537
}
PESOS_SECTORES = {
    "XLC": 0.0999,
    "XLY": 0.1025,
    "XLP": 0.0482,
    "XLE":  0.0295,
    "XLF":  0.1307,
    "XLV":  0.0958,
    "XLI":  0.0809,
    "XLB":  0.0166,
    "XLRE": 0.0187,
    "XLK":  0.3535,
    "XLU":  0.0237
}

# 2.Se presentan dos tipos de funciones

# 2.1. Funciones de las métricas vistas en clase

# Beta: sensibilidad respecto al benchmark
def beta(port, benchmark):
    cov = np.cov(port, benchmark)[0, 1]   # Covarianza portafolio vs benchmark
    var = np.var(benchmark)   # Varianza benchmark
    return cov / var

# Media de los retornos
def media(r):
    return r.mean()

# Volatilidad
def volatilidad(r):
    return r.std()

# Ratio Sharpe anualizado
def sharpe(r, rf=0.0):
    excess = r - rf/252   # Retorno excedente diario
    return np.sqrt(252) * excess.mean() / excess.std()

# Ratio de Sortino
def sortino(r, rf=0.0):
    excess = r - rf/252 # Retorno excedente
    downside = excess[excess < 0].std()  # Solo desviación de retornos negativos
    return np.sqrt(252) * excess.mean() / downside

# Máxima caída desde un pico
def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()   # Nivel máximo alcanzado hasta cada punto
    dd = (cum - peak) / peak   # Drawdown relativo
    return dd.min()   # Máximo valor más bajo

# Valor en Riesgo al 95%
def var_95(r):
    return np.percentile(r, 5)   # Percentil 5

# CVaR o Expected Shortfall 95%
def cvar_95(r):
    v = var_95(r)
    return r[r <= v].mean()   # Promedio de pérdidas más extremas

# Sesgo de distribución
def sesgo(r):
    return skew(r)

# Kurtosis
def curtosis(r):
    return kurtosis(r)

# Función que organiza todas las métricas en un diccionario
def calcular_metricas(serie, rf=0.05):
    return {
        "Media diaria": media(serie),
        "Volatilidad diaria": volatilidad(serie),
        "Sharpe (5% rf)": sharpe(serie, rf=rf),
        "Sortino (5% rf)": sortino(serie, rf=rf),
        "Max Drawdown": max_drawdown(serie),
        "VaR 95%": var_95(serie),
        "CVaR 95%": cvar_95(serie),
        "Skew": sesgo(serie),
        "Kurtosis": curtosis(serie),
    }


# 2.2. Funciones que se encargan de:
#  Descargar precios, Calcular rendimientos, Construir portafolios a partir de pesos
#  Obtener medias y covarianzas.


def descargar_precios(tickers, years=4):
    data = yf.download(tickers, period=f"{years}y")["Close"]
    return data

# Construye un portafolio usando retornos ya calculados y pesos arbitrarios.


def construir_portafolio_arbitrario(retornos, pesos_dict):
    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    r = retornos[columnas]
    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()
    portafolio = (r * pesos).sum(axis=1)
    return portafolio

# Obtener medias y covarianzas
def obtener_mu_cov(retornos):
    mu = retornos.mean()          # media diaria
    cov = retornos.cov()          # covarianza diaria
    return mu, cov

#  Calcula rendimientos y portafolio dada una tabla de precios y pesos (diccionario).

def construir_portafolio(data_precios, pesos_dict):
    retornos = data_precios.pct_change().dropna()

    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    retornos = retornos[columnas]

    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()

    portafolio = (retornos * pesos).sum(axis=1)
    return retornos, portafolio



# 4. Optimización de portafolio

# Volatilidad del portafolio
def port_vol(w, cov):
    w = np.array(w)
    return np.sqrt(w.T @ cov.values @ w)

# Retorno del portafolio
def port_ret(w, mu):
    w = np.array(w)
    return w @ mu.values

# Portafolio de mínima varianza
def min_var_portfolio(mu, cov):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    )

    def obj(w):
        return w.T @ cov.values @ w

    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

# Portafolio con máximo Sharpe
def max_sharpe_portfolio(mu, cov, rf_anual):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    rf_diario = rf_anual / 252.0

    def neg_sharpe(w):
        r_p = port_ret(w, mu)
        v_p = port_vol(w, cov)
        if v_p == 0:
            return 1e6
        return - (r_p - rf_diario) / v_p

    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    )

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

# Markowitz con retorno objetivo
def markowitz_target_portfolio(mu, cov, target_anual):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    target_diario = target_anual / 252.0

    def obj(w):
        return w.T @ cov.values @ w

    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: port_ret(w, mu) - target_diario},
    )

    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None


# 5. App. INTERFAZ PRINCIPAL DE LA APLICACIÓN

def main():
    st.markdown("## 📈 Cálculo de Métricas de Portafolios")
   st.markdown("""
<div style="
    background-color:#ffffff;
    padding:18px;
    border-radius:10px;
    border-left:4px solid #7a3db8;
    box-shadow:0px 2px 6px rgba(0,0,0,0.05);
    margin-bottom:20px;">
    <p style="font-size:1rem; color:#333;">
        Aplicación para analizar portafolios de <b>Regiones</b> y <b>Sectores</b>:
    </p>
    <ul style="font-size:1rem; color:#333; line-height:1.6;">
        <li>Benchmark (pesos dados)</li>
        <li>Portafolio arbitrario (definido por el usuario)</li>
        <li>Portafolios optimizados: mínima varianza, máximo Sharpe y Markowitz con rendimiento objetivo.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


    # Sidebar: parámetros generales
    estrategia = st.sidebar.selectbox("Estrategia", ["Regiones", "Sectores"])
    years = st.sidebar.slider("Años de datos históricos", 1, 10, 4)
    rf_anual = st.sidebar.number_input("Tasa libre de riesgo anual (rf)", 0.0, 0.20, 0.05, step=0.005)

    modo = st.sidebar.radio(
        "Portafios a calcular",
        ["Solo benchmark", "Solo arbitrario", "Benchmark y arbitrario", "Optimización"],
        index=2
    )

    if estrategia == "Regiones":
        tickers = TICKERS_REGIONES
        pesos_bench = PESOS_REGIONES
    else:
        tickers = TICKERS_SECTORES
        pesos_bench = PESOS_SECTORES

    # Pesos arbitrarios (si aplica)
    pesos_arbitrarios = {}
    if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
        st.sidebar.markdown("### Pesos portafolio arbitrario")
        st.sidebar.caption("Introduce los pesos (se normalizan automáticamente para sumar 1).")
        for t in tickers:
            default = float(pesos_bench.get(t, 0.0))
            w = st.sidebar.number_input(
                f"Peso {t}",
                min_value=0.0,
                max_value=1.0,
                value=default,
                step=0.01
            )
            pesos_arbitrarios[t] = w

    # Parámetro para Markowitz si estamos en optimización
    target_anual = None
    if modo == "Optimización":
        st.sidebar.markdown("### Markowitz – Rendimiento objetivo")
        target_anual = st.sidebar.number_input(
            "Rendimiento objetivo anual (en decimal, ej. 0.10 = 10%)",
            min_value=0.0,
            max_value=0.5,
            value=0.10,
            step=0.01
        )

    st.subheader(f"Estrategia seleccionada: {estrategia}")

    if st.button("Calcular métricas"):
        with st.spinner("Descargando datos y calculando…"):
            data = descargar_precios(tickers, years)
            retornos, portafolio_bench = construir_portafolio(data, pesos_bench)

            mu, cov = obtener_mu_cov(retornos)

            # Portafolio arbitrario (si aplica)
            portafolio_arbi = None
            if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
                if sum(pesos_arbitrarios.values()) == 0:
                    st.error("Los pesos del portafolio arbitrario no pueden ser todos cero.")
                    return
                portafolio_arbi = construir_portafolio_arbitrario(retornos, pesos_arbitrarios)

            # Portafolios optimizados (si aplica)
            w_minvar = w_maxsharpe = w_markowitz = None
            port_minvar = port_maxsharpe = port_markowitz = None

            if modo == "Optimización":
                # Mínima varianza
                w_minvar = min_var_portfolio(mu, cov)
                # Máximo Sharpe
                w_maxsharpe = max_sharpe_portfolio(mu, cov, rf_anual)
                # Markowitz con retorno objetivo
                w_markowitz = markowitz_target_portfolio(mu, cov, target_anual)

                if w_minvar is None or w_maxsharpe is None or w_markowitz is None:
                    st.error("No se pudo encontrar una solución óptima para alguna de las optimizaciones.")
                    return

                cols = retornos.columns
                # Series de retornos de cada portafolio optimizado
                port_minvar = (retornos[cols] * w_minvar).sum(axis=1)
                port_maxsharpe = (retornos[cols] * w_maxsharpe).sum(axis=1)
                port_markowitz = (retornos[cols] * w_markowitz).sum(axis=1)

        # ----------------------------
        # Mostrar datos básicos
        # ----------------------------
        st.markdown("### Precios de cierre (últimos 10 registros)")
        st.dataframe(data.tail(10))

        st.markdown("### Retornos diarios (primeros 5 registros)")
        st.dataframe(retornos.head())

        # ----------------------------
        # Métricas
        # ----------------------------
        if modo != "Optimización":
            metrics_dict = {}

            if modo in ["Solo benchmark", "Benchmark y arbitrario"]:
                metrics_dict["Benchmark"] = calcular_metricas(portafolio_bench, rf=rf_anual)

            if modo in ["Solo arbitrario", "Benchmark y arbitrario"] and portafolio_arbi is not None:
                metrics_dict["Arbitrario"] = calcular_metricas(portafolio_arbi, rf=rf_anual)

            df_metrics = pd.DataFrame(metrics_dict)
            st.markdown("### Métricas de portafolios")
            st.dataframe(df_metrics.style.format("{:.6f}"))

            # Rendimiento acumulado
            st.markdown("### Rendimiento acumulado")
            df_cum = pd.DataFrame()
            if "Benchmark" in metrics_dict:
                df_cum["Benchmark"] = (1 + portafolio_bench).cumprod()
            if "Arbitrario" in metrics_dict and portafolio_arbi is not None:
                df_cum["Arbitrario"] = (1 + portafolio_arbi).cumprod()
            st.line_chart(df_cum)

        else:
            # Métricas de los portafolios optimizados
            metrics_opt = {
                "MinVar": calcular_metricas(port_minvar, rf=rf_anual),
                "MaxSharpe": calcular_metricas(port_maxsharpe, rf=rf_anual),
                "Markowitz": calcular_metricas(port_markowitz, rf=rf_anual),
            }

            df_metrics_opt = pd.DataFrame(metrics_opt)
            st.markdown("### Métricas de portafolios optimizados")
            st.dataframe(df_metrics_opt.style.format("{:.6f}"))

            # Pesos de cada portafolio
            weights_df = pd.DataFrame(
                {
                    "MinVar": w_minvar,
                    "MaxSharpe": w_maxsharpe,
                    "Markowitz": w_markowitz,
                },
                index=retornos.columns
            )
            st.markdown("### Pesos de los portafolios optimizados")
            st.dataframe(weights_df.style.format("{:.4f}"))

            # Rendimiento acumulado de optimizados
            st.markdown("### Rendimiento acumulado – Portafolios optimizados")
            df_cum_opt = pd.DataFrame({
                "MinVar": (1 + port_minvar).cumprod(),
                "MaxSharpe": (1 + port_maxsharpe).cumprod(),
                "Markowitz": (1 + port_markowitz).cumprod(),
            })
            st.line_chart(df_cum_opt)

if __name__ == "__main__":
    main()






