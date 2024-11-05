from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os

# Definir el esquema de entrada con listas y el nombre del modelo
class PredictionInput(BaseModel):
    store_nbr: List[int]
    item_nbr: List[int]
    months: List[int]
    years: List[int]
    forecast_model: str  # Nombre del modelo proporcionado por el front-end

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo (función general para todos los modelos)
def load_model(filename):
    model = joblib.load(filename)
    return model

@app.get('/')
def hello_world():
    return "Hello,World"


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)


@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Crear todas las combinaciones posibles de tienda, producto, mes y año
    combinations = pd.MultiIndex.from_product(
        [input_data.store_nbr, input_data.item_nbr, input_data.months, input_data.years], 
        names=['store_nbr', 'item_nbr', 'month_name', 'year']
    ).to_frame(index=False)

    # Crear la ruta del archivo para el modelo
    model_filename = f"./models/{input_data.forecast_model}.pkl"
    
    # Verificar si el archivo del modelo existe
    if not os.path.isfile(model_filename):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    try:
        # Cargar el modelo especificado por el front-end
        model = load_model(model_filename)

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
