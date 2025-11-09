# ML Time Series API

API de Machine Learning para análisis de series de tiempo usando modelo Perceptrón. Optimizada para despliegue en Render.com con integración MongoDB.

## 🚀 Características

- **FastAPI**: API REST moderna y rápida
- **Modelo Perceptrón**: Optimizado para series de tiempo
- **MongoDB**: Integración asíncrona con Motor
- **Docker**: Containerización completa
- **Variables de entorno**: Configuración segura
- **Métricas**: MAE, MSE, RMSE incluidas
- **Escalable**: Diseñado para servicios cloud gratuitos

## 📁 Estructura del Proyecto

```
Proyecto-ml/
├── main.py                 # API principal FastAPI
├── requirements.txt        # Dependencias Python
├── Dockerfile             # Configuración Docker
├── docker-compose.yml     # Docker Compose para desarrollo
├── .env.example           # Ejemplo de variables de entorno
├── .gitignore            # Archivos ignorados por Git
├── .dockerignore         # Archivos ignorados por Docker
├── models/
│   ├── __init__.py
│   └── perceptron_model.py # Modelo Perceptrón para series de tiempo
├── database/
│   ├── __init__.py
│   └── mongodb_client.py   # Cliente MongoDB asíncrono
└── README.md              # Esta documentación
```

## 🛠️ Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd Proyecto-ml
```

### 2. Configurar variables de entorno

Copiar `.env.example` a `.env` y configurar:

```bash
# MongoDB Configuration
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=tu_base_de_datos

# API Configuration
PORT=8000
ENVIRONMENT=production

# Model Configuration
DEFAULT_WINDOW_SIZE=10
DEFAULT_EPOCHS=100
MODEL_SAVE_PATH=./models/saved_model.pkl
```

### 3. Desarrollo local con Docker

```bash
# Construir y ejecutar
docker-compose up --build

# Solo ejecutar (si ya está construido)
docker-compose up

# Ejecutar en background
docker-compose up -d
```

### 4. Desarrollo local sin Docker

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🌐 Despliegue en Render.com

### 1. Preparar repositorio

- Asegurar que todos los archivos estén en el repositorio
- El `Dockerfile` está optimizado para Render
- Variables de entorno configuradas en Render

### 2. Configurar en Render

1. **Crear nuevo Web Service**
2. **Conectar repositorio GitHub**
3. **Configurar build settings:**
   - Build Command: `docker build -t ml-api .`
   - Start Command: `docker run -p $PORT:8000 ml-api`

### 3. Variables de entorno en Render

Configurar en el dashboard de Render:
- `MONGODB_URL`
- `MONGODB_DATABASE`
- `PORT` (automático)
- `ENVIRONMENT=production`

## 📖 Uso de la API

### Endpoints Principales

#### 1. Health Check
```bash
GET /health
```

Respuesta:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": false
}
```

#### 2. Entrenar Modelo
```bash
POST /train
Content-Type: application/json

{
  "collection_name": "mi_coleccion",
  "window_size": 10,
  "epochs": 100
}
```

#### 3. Hacer Predicción
```bash
POST /predict
Content-Type: application/json

{
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "value": 100.5
    },
    {
      "timestamp": "2024-01-01T01:00:00Z", 
      "value": 102.3
    }
  ],
  "window_size": 10
}
```

#### 4. Información del Modelo
```bash
GET /model/info
```

### Formato de Datos MongoDB

Los documentos en MongoDB deben tener la estructura:

```json
{
  "_id": "ObjectId",
  "timestamp": "2024-01-01T00:00:00Z",
  "value": 100.5,
  "additional_features": {
    "sensor_id": "sensor_1",
    "location": "office"
  }
}
```

**Campos requeridos:**
- `value`: Valor numérico de la serie de tiempo
- `timestamp`: Fecha y hora del punto de datos

## 🔧 Ejemplos de Uso

### Python Client

```python
import requests
import json
from datetime import datetime

# URL de tu API
API_URL = "https://tu-app.onrender.com"

# 1. Verificar salud
response = requests.get(f"{API_URL}/health")
print(response.json())

# 2. Entrenar modelo
train_data = {
    "collection_name": "sensor_data",
    "window_size": 10,
    "epochs": 50
}
response = requests.post(f"{API_URL}/train", json=train_data)
print(response.json())

# 3. Hacer predicción
prediction_data = {
    "data": [
        {"timestamp": "2024-01-01T00:00:00Z", "value": 100.0},
        {"timestamp": "2024-01-01T01:00:00Z", "value": 102.0},
        {"timestamp": "2024-01-01T02:00:00Z", "value": 104.0},
        # ... más datos hasta completar window_size
    ],
    "window_size": 10
}
response = requests.post(f"{API_URL}/predict", json=prediction_data)
print(response.json())
```

### JavaScript Client

```javascript
// 1. Entrenar modelo
const trainModel = async () => {
  const response = await fetch('https://tu-app.onrender.com/train', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      collection_name: 'sensor_data',
      window_size: 10,
      epochs: 50
    })
  });
  
  const result = await response.json();
  console.log(result);
};

// 2. Hacer predicción
const makePrediction = async (timeSeries) => {
  const response = await fetch('https://tu-app.onrender.com/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      data: timeSeries,
      window_size: 10
    })
  });
  
  const result = await response.json();
  return result.prediction;
};
```

## 🧪 Testing

### Probar endpoints localmente

```bash
# Health check
curl http://localhost:8000/health

# Información del modelo
curl http://localhost:8000/model/info

# Entrenar (reemplazar con tu colección)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "tu_coleccion", "window_size": 10, "epochs": 50}'
```

## 📊 Métricas del Modelo

El modelo proporciona las siguientes métricas:

- **MAE** (Mean Absolute Error): Error absoluto medio
- **MSE** (Mean Squared Error): Error cuadrático medio  
- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático medio
- **Final Loss**: Pérdida final del entrenamiento
- **Training Samples**: Número de muestras de entrenamiento

## 🔒 Seguridad

- Variables de entorno para credenciales sensibles
- Usuario no-root en Docker
- Timeouts configurados para MongoDB
- Validación de datos de entrada
- Logging para auditoría

## 🐛 Troubleshooting

### Error: "Import could not be resolved"
- Normal durante desarrollo, se resuelve al instalar dependencias

### Error de conexión MongoDB
- Verificar `MONGODB_URL` en variables de entorno
- Asegurar que la IP esté en whitelist de MongoDB Atlas
- Verificar que el usuario tenga permisos de lectura

### Error de memoria en Render
- Reducir `epochs` en entrenamiento
- Usar `window_size` más pequeño
- Considerar entrenar offline y cargar modelo

### API no responde en Render
- Verificar que `PORT` esté configurado correctamente
- Revisar logs en dashboard de Render
- Verificar que Docker exponga el puerto correcto

## 📞 Soporte

Para problemas o preguntas:
1. Revisar logs de la aplicación
2. Verificar configuración de variables de entorno
3. Comprobar conectividad con MongoDB
4. Revisar métricas en Render dashboard

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
