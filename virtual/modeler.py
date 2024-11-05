from model_utils import load_data, clean_data, prepare_data, save_all_models, get_models
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Cargar los datos
df = load_data()

# Limpiar los datos
df = clean_data(df)

# Preparar los datos para el entrenamiento
X_train, X_test, y_train, y_test = prepare_data(df)

# Entrenar y evaluar todos los modelos
if __name__ == "__main__":
    models = get_models()
    results = {}

    # Ajustar y evaluar cada modelo
    for name, model in models.items():
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Evaluar el modelo en el conjunto de prueba
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Guardar los resultados de la evaluación
        results[name] = {"MAE": mae, "RMSE": rmse}
        
        # Imprimir los resultados de cada modelo
        print(f"Modelo: {name}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print("-" * 30)

    # Guardar los modelos entrenados en archivos
    save_all_models(models)
