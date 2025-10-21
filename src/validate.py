import joblib
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import os

THRESHOLD = 5000.0  # umbral máximo aceptable de error

print("🔍 Validando modelo...")

# Cargar datos
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar modelo entrenado
model_path = os.path.join(os.getcwd(), "src/model.pkl")
if not os.path.exists(model_path):
    print(f"❌ No se encontró el archivo del modelo en {model_path}")
    sys.exit(1)

model = joblib.load(model_path)

# Evaluar desempeño
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"📊 MSE obtenido: {mse:.2f} (umbral: {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ Modelo válido. Cumple el criterio de calidad.")
    sys.exit(0)
else:
    print("❌ Modelo no cumple con el umbral de calidad.")
    sys.exit(1)
