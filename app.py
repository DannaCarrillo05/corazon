import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from io import BytesIO
from PIL import Image

# Cargar el modelo y el scaler
modelo = joblib.load('modelo_knn.bin')
scaler = joblib.load('esclador.bin')

# Título y autor
st.title("Asistente Cardiaco")
st.subheader("Autor: Danna Carrillo")

# Instrucciones
st.markdown("""
### Instrucciones de uso:
1. En la pestaña de "Ingreso de Datos", utiliza los deslizadores para ingresar tu edad y nivel de colesterol.
2. En la pestaña de "Predicción", verás el resultado si tienes o no problemas cardíacos, basado en los datos que proporcionaste.
3. Si tienes algún problema cardíaco, se mostrará una imagen relacionada, de lo contrario, te indicaremos que no hay problemas.
""")

# Pestañas para Ingreso de Datos y Predicción
tab1, tab2 = st.tabs(["Ingreso de Datos", "Predicción"])

with tab1:
    # Captura de datos del usuario
    edad = st.slider('Edad', 18, 80, 25)
    colesterol = st.slider('Colesterol', 100, 600, 200)
    
    # Crear el DataFrame con los valores
    df_input = pd.DataFrame([[edad, colesterol]], columns=["Edad", "Colesterol"])
    
    # Normalización de los datos con el escalador cargado
    df_normalizado = scaler.transform(df_input)

with tab2:
    # Predicción
    if st.button("Predecir"):
        # Realizar la predicción
        prediccion = modelo.predict(df_normalizado)
        
        if prediccion == 1:
            # Problemas cardíacos
            st.subheader("¡Atención! Tienes problemas cardíacos.")
            # Mostrar imagen de problemas cardíacos
            img_url = "https://cloudfront-us-east-1.images.arcpublishing.com/prisaradiomx/DDH7MV74XVI5TDU7PMRENKRXUI.jpg"
        else:
            # No tiene problemas cardíacos
            st.subheader("¡Felicidades! No tienes problemas cardíacos.")
            # Mostrar imagen de no problemas cardíacos
            img_url = "https://diariocorreo.pe/resizer/_xtlzJ1UjrpkV5FJ9CWvcAGshvI=/580x330/smart/filters:format(jpeg):quality(75)/cloudfront-us-east-1.images.arcpublishing.com/elcomercio/FDSBP33QPZG3NNEPXEKS5T4P4I.jpg"
        
        # Mostrar la imagen correspondiente
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Resultado de la predicción", use_column_width=True)
