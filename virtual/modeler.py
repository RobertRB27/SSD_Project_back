from model_utils import load_data, clean_data, prepare_data, save_all_models, get_models

# Cargar los datos
df = load_data()

# Limpiar los datos
df = clean_data(df)

# Preparar los datos para el entrenamiento
X_train, X_test, y_train, y_test = prepare_data(df)

# Entrenar todos los modelos
if __name__ == "__main__":
    models = get_models()

    # Ajustar los modelos a los datos de entrenamiento
    for name, model in models.items():
        model.fit(X_train, y_train)

    # Guardar los modelos entrenados en archivos
    save_all_models(models)