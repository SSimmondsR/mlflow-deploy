import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# ========== CONFIGURACIÓN BÁSICA ==========
print("🚀 Iniciando entrenamiento con MLflow...")

# Directorio actual
workspace_dir = os.getcwd()

# Carpeta donde MLflow guardará los resultados
mlruns_dir = os.path.join(workspace_dir, "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow (versión compatible con Windows y Linux) ---
from pathlib import Path

mlruns_dir = Path(os.getcwd()) / "mlruns"
mlruns_dir.mkdir(exist_ok=True)

# Genera una URI universal válida (e.g. file:///C:/Users/Santiago/mlruns)
tracking_uri = mlruns_dir.as_uri()

mlflow.set_tracking_uri(tracking_uri)
print(f"✅ Tracking URI configurado en: {tracking_uri}")

# Crear o establecer experimento
experiment_name = "CI-CD-Lab2"
mlflow.set_experiment(experiment_name)
print(f"✅ Experimento activo: {experiment_name}")



# Crear o usar experimento llamado "CI-CD-Lab2"
experiment_name = "CI-CD-Lab2"
mlflow.set_experiment(experiment_name)

# ========== ENTRENAMIENTO ==========
print("📦 Cargando dataset de diabetes...")
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Entrenando modelo...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones y métrica
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"✅ Entrenamiento completo. MSE: {mse:.4f}")

# ========== REGISTRO EN MLFLOW ==========
with mlflow.start_run() as run:
    mlflow.log_param("modelo", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    print(f"📝 Run guardado en: {run.info.run_id}")

# ========== GUARDAR MODELO LOCALMENTE ==========
joblib.dump(model, "src/model.pkl")
print("💾 Modelo guardado localmente en src/model.pkl")
