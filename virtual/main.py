from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

# Definir el esquema de entrada con listas
class PredictionInput(BaseModel):
    store_nbr: List[int]
    item_nbr: List[int]
    months: List[int]
    years: List[int]

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo (función de ejemplo)
def load_model(filename):
    model = joblib.load(filename)
    return model

@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Crear todas las combinaciones posibles de tienda, producto, mes y año
    combinations = pd.MultiIndex.from_product(
        [input_data.store_nbr, input_data.item_nbr, input_data.months, input_data.years], 
        names=['store_nbr', 'item_nbr', 'month_name', 'year']
    ).to_frame(index=False)

    try:
        # Cargar el modelo (puedes cambiar por el modelo que estés utilizando)
        model = load_model("./models/decision_tree_model.pkl")

        # Hacer predicciones
        predictions = model.predict(combinations)

        # Añadir la columna de predicciones
        combinations['predicted_sales'] = predictions

        # Convertir el DataFrame a una lista de listas para la respuesta
        result = combinations.values.tolist()

        # Devolver el resultado en el formato deseado
        return {"predictions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al hacer predicción: {e}")
