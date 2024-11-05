#import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import joblib
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Reemplaza estos valores con tu URL y clave de API
def connect_db():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)
    
    return supabase


# Obtener datos de Supabase
def load_data():
    supabase = connect_db()
    try:
        # Realizar la consulta a la tabla en Supabase
        response = (
            supabase
            .table("sales")  # Nombre de la tabla en Supabase
            .select("store_nbr, item_nbr, total_sales, month_name, year")  # Columnas específicas
            #.gte("year", 2013)  # Filtro de año >= 2013
            #.lte("year", 2014)  # Filtro de año <= 2017
            .execute()
        )
        
        # Verificar si la consulta devolvió datos
        if response.data:
            df = pd.DataFrame(response.data)
            return df
        else:
            print("No se encontraron datos.")
            return None

    except Exception as e:
        # Manejar cualquier error en la consulta
        print("Error al realizar la consulta:", e)
        return None



'''
def connect_db():
    engine = create_engine('mysql+mysqlconnector://root:@localhost/forecasting_app')
    return engine

# Obtener datos de MySQL
def load_data():
    engine = connect_db()
    query = """
    SELECT
    store_nbr,
    item_nbr,
    total_sales,
    month_name,
    year
        FROM train_aggregated
        WHERE year BETWEEN 2013 AND 2017;
    """
    df = pd.read_sql(query, engine)
    return df
'''


# Limpiar los datos
def clean_data(df):
    # Asumiendo que la columna 'month_name' contiene el nombre del mes
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 
                 'May': 5, 'June': 6, 'July': 7, 'August': 8, 
                 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    # Reemplazar los nombres de los meses con sus valores numéricos correspondientes
    df['month_name'] = df['month_name'].map(month_mapping)
    return df

# Preparar los datos para el entrenamiento
def prepare_data(df):
    X = df[['store_nbr', 'item_nbr', 'month_name', 'year']]
    y = df['total_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para entrenar y evaluar un modelo (separada para multiprocesamiento)
def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test, return_dict):
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    predictions = model.predict(X_test)
    
    # Evaluar el modelo
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Guardar el resultado en un diccionario compartido entre procesos
    return_dict[name] = {"MAE": mae, "RMSE": rmse}
    
    # Imprimir los resultados del modelo
    print(f"Modelo: {name}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print("-" * 30)
    

# Función para entrenar y evaluar todos los modelos en paralelo
def train_and_evaluate_models_parallel(X_train, y_train, X_test, y_test):
    models = {
        #"Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()  # Diccionario compartido entre procesos
    jobs = []

    # Crear un proceso para cada modelo
    for name, model in models.items():
        p = multiprocessing.Process(target=train_and_evaluate_model, args=(name, model, X_train, y_train, X_test, y_test, return_dict))
        jobs.append(p)
        p.start()  # Iniciar el proceso

    # Esperar a que todos los procesos terminen
    for proc in jobs:
        proc.join()

    # Convertir los resultados en un dict
    results = dict(return_dict)
    
    return results



# Visualizar los resultados
def plot_results(results):
    names = list(results.keys())
    mae_values = [results[name]["MAE"] for name in names]
    rmse_values = [results[name]["RMSE"] for name in names]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(x=names, y=mae_values, ax=ax[0])
    ax[0].set_title("MAE por modelo")
    ax[0].set_ylabel("MAE")

    sns.barplot(x=names, y=rmse_values, ax=ax[1])
    ax[1].set_title("RMSE por modelo")
    ax[1].set_ylabel("RMSE")

    plt.show()
    
    
# Función para guardar un modelo en un archivo con compresión
def save_model(model, filename, compress_level=3):
    # Guardar el modelo utilizando compresión para reducir tamaño
    joblib.dump(model, filename, compress=compress_level)
    print(f"Modelo guardado en: {filename}")

    
    
# Guardar todos los modelos, exceptuando Random Forest
def save_all_models(models, directory="./models/"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterar sobre los modelos y guardar solo los que no son Random Forest
    for name, model in models.items():
            # Crear el nombre del archivo en función del nombre del modelo
            filename = f"{directory}{name.lower().replace(' ', '_')}_model.pkl"
            # Guardar el modelo con un nivel de compresión para reducir el tamaño
            save_model(model, filename, compress_level=3)  # Ajuste aquí para usar compress_level
        
        
# Función para cargar un modelo desde un archivo
def load_model(filename):
    model = joblib.load(filename)
    print(f"Modelo cargado desde: {filename}")
    return model

# Cargar el modelo seleccionado por el usuario
def select_and_load_model():
    print("Elige un modelo para cargar:")
    print("1. Random Forest")
    print("2. Linear Regression")
    print("3. Decision Tree")
    print("4. XGBoost")
    
    choice = input("Ingresa el número del modelo: ")

    if choice == "1":
        return load_model("./models/random_forest_model.pkl")
    elif choice == "2":
        return load_model("./models/linear_regression_model.pkl")
    elif choice == "3":
        return load_model("./decision_tree_model.pkl")
    elif choice == "4":
        return load_model("./models/xgboost_model.pkl")
    else:
        print("Selección inválida")
        return None
    

# Guardar las predicciones en un archivo CSV
def save_predictions(predictions, filename="predictions.csv"):
    pd.DataFrame(predictions, columns=["predicted_sales"]).to_csv(filename, index=False)
    print(f"Predicciones guardadas en: {filename}")
    
    
# Función que devuelve los modelos predefinidos
def get_models():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }
    
 