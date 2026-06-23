import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Configuración inicial de la página web
st.set_page_config(page_title="Sistema Predictivo de Riesgo", page_icon="🎓", layout="wide")

# ==========================================
# 1. CARGA DE MODELOS (Caché para velocidad)
# ==========================================
@st.cache_resource
def cargar_modelos():
    with open('modelo_xgboost.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('modelo_random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    return xgb, rf

modelo_xgb, modelo_rf = cargar_modelos()

# ==========================================
# 2. INTERFAZ: MENÚ LATERAL (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Configuración del Sistema")
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el Algoritmo Predictivo:",
    ("XGBoost", "Random Forest")
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulación de Datos del Estudiante")

# Sliders LIMPIOS para el profesor (solo lo que es fácil de entender)
intentos_previos = st.sidebar.number_input("Intentos Previos en el Curso", min_value=0, max_value=5, value=0)
inasistencias_4w = st.sidebar.slider("Faltas Acumuladas (Semanas 1-4)", min_value=0, max_value=10, value=1)
notas_sem1_4w = st.sidebar.slider("Nota del Primer Control (0 - 100)", min_value=0.0, max_value=100.0, value=75.0)

# ==========================================
# 3. LÓGICA OCULTA (El "truco" de ingeniería)
# ==========================================
# Aquí traducimos las faltas al porcentaje que espera el modelo, sin molestar al usuario.
if inasistencias_4w == 0:
    porcentaje_calculado = 0.0
elif inasistencias_4w == 1:
    porcentaje_calculado = 0.75  # La regla exacta que me indicaste
else:
    # Si tiene 2 o más faltas, asumimos el tope máximo de riesgo de inasistencia (1.0)
    # Puedes modificar esta matemática después si descubres otra regla en tu Excel
    porcentaje_calculado = 1.0 

# Creamos el DataFrame con los 4 "ingredientes" que exige el modelo
datos_entrada = pd.DataFrame({
    'num_of_prev_attempts': [intentos_previos],
    'Porcentaje_Inasistencias_S4': [porcentaje_calculado],
    'notas_sem1_4w': [notas_sem1_4w],
    'inasistencias_4w': [inasistencias_4w]
})

# ==========================================
# 4. PROCESAMIENTO Y PREDICCIÓN
# ==========================================
modelo_activo = modelo_xgb if modelo_seleccionado == "XGBoost" else modelo_rf

prediccion = modelo_activo.predict(datos_entrada)[0]
probabilidad = modelo_activo.predict_proba(datos_entrada)[0][0] * 100 # Índice [0] asumiendo que 0 es Riesgo

# ==========================================
# 5. INTERFAZ: PANEL PRINCIPAL
# ==========================================
st.title("📊 Panel de Alerta Temprana Estudiantil")
st.markdown(f"**Modelo en ejecución:** `{modelo_seleccionado}`")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado Predictivo")
    # Lógica invertida corregida: 0 = Riesgo
    if prediccion == 0:
        st.error("⚠️ ALTO RIESGO DE DESERCIÓN")
    else:
        st.success("✅ ESTUDIANTE FUERA DE RIESGO")

with col2:
    st.subheader("Nivel de Probabilidad")
    st.metric(label="Probabilidad de Riesgo Calculada", value=f"{probabilidad:.1f}%")

st.markdown("---")

# ==========================================
# 6. MOTOR XAI (SHAP) - Lógica a prueba de balas
# ==========================================
st.subheader("🧠 Justificación del Algoritmo (XAI)")
st.write("El siguiente gráfico explica cómo las variables ingresadas empujaron la decisión del modelo:")

explainer = shap.TreeExplainer(modelo_activo)
shap_values = explainer.shap_values(datos_entrada)

# Escáner inteligente de dimensiones (Resuelve errores de versión de SHAP)
if isinstance(shap_values, list):
    # Comportamiento clásico de RF (Lista de matrices)
    valores_grafico = shap_values[0][0]
    valor_base = explainer.expected_value[0]
else:
    # Comportamiento tipo Numpy Array (XGBoost o SHAP moderno)
    if len(shap_values.shape) == 3:
        # Formato 3D: (muestras, variables, clases) -> Extraemos muestra 0, clase 0
        valores_grafico = shap_values[0, :, 0]
        valor_base = explainer.expected_value[0]
    elif len(shap_values.shape) == 2:
        # Formato 2D: (muestras, variables) -> Extraemos muestra 0
        valores_grafico = shap_values[0]
        valor_base = explainer.expected_value
        if isinstance(valor_base, (list, np.ndarray)):
            valor_base = valor_base[0]
    else:
        # Formato 1D (Muy raro, pero nos cubrimos)
        valores_grafico = shap_values
        valor_base = explainer.expected_value

# Crear el gráfico con los datos quirúrgicamente extraídos
fig, ax = plt.subplots(figsize=(8, 4))
shap.decision_plot(valor_base, 
                   valores_grafico, 
                   datos_entrada.columns.tolist(), 
                   show=False)
st.pyplot(fig)