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

## 🌐 Despliegue en la Nube

### Variables de Entorno Requeridas

Todas las plataformas necesitan estas variables:
```bash
MONGODB_URL=mongodb+srv://user:password@cluster.mongodb.net/
MONGODB_DATABASE=nombre_base_datos
PORT=8000
ENVIRONMENT=production
```

### 1. 🚀 **Render.com** (Gratis con limitaciones)

**✅ Pros:** Fácil, gratis  
**❌ Contras:** Se duerme tras 15min inactividad (delay 50s)

```bash
# Build Command
docker build -t ml-api .

# Start Command  
docker run -p $PORT:8000 ml-api
```

1. Conectar repositorio GitHub
2. Seleccionar "Web Service"
3. Configurar variables de entorno
4. Deploy automático

---

### 2. ⚡ **Railway.app** (Recomendado)

**✅ Pros:** $5/mes gratis, sin sleep, mejor rendimiento  
**❌ Contras:** Requiere tarjeta de crédito

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login y deploy
railway login
railway project create
railway up
```

**O desde GitHub:**
1. Conectar repositorio en railway.app
2. Configurar variables de entorno
3. Deploy automático

---

### 3. ☁️ **Google Cloud Run** (Escalable)

**✅ Pros:** $300 iniciales, pay-per-use, profesional  
**❌ Contras:** Más complejo

```bash
# Instalar Google Cloud CLI
gcloud auth login
gcloud config set project TU_PROJECT_ID

# Deploy
gcloud run deploy ml-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

---

### 4. 🌊 **Fly.io** (Global)

**✅ Pros:** 3 apps gratis, servidores globales  
**❌ Contras:** Configuración inicial

```toml
# fly.toml
app = "tu-ml-api"

[build]
  dockerfile = "Dockerfile"

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

```bash
# Instalar Fly CLI
flyctl launch
flyctl deploy
```

---

### 5. 🔮 **DigitalOcean App Platform**

**✅ Pros:** $200 iniciales, infraestructura sólida  
**❌ Contras:** ~$5/mes después de créditos

1. Conectar repositorio GitHub
2. Seleccionar "Docker" como build pack
3. Configurar variables de entorno
4. Deploy

---

### 6. ⚡ **Vercel** (Solo APIs rápidas)

**⚠️ Limitación:** Timeout 10s (no ideal para ML)

```json
// vercel.json
{
  "builds": [
    {
      "src": "main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "main.py"
    }
  ]
}
```

---

## 🎯 **Recomendaciones por Uso**

| Escenario | Plataforma Recomendada | Motivo |
|-----------|----------------------|---------|
| **Desarrollo/Testing** | Render | Gratis, fácil setup |
| **Producción Pequeña** | Railway | Sin sleep, confiable |
| **Startup/Empresa** | Google Cloud Run | Escalable, profesional |
| **App Global** | Fly.io | Múltiples regiones |
| **Presupuesto Flexible** | DigitalOcean | Balance precio/calidad |

## ⚖️ **Estrategia de Redundancia** (Recomendado)

Para máxima disponibilidad, desplegar en 2 plataformas:

```javascript
// Cliente con fallback automático
class MLAPIClient {
  constructor() {
    this.apis = [
      'https://tu-api-railway.up.railway.app',  // Principal
      'https://tu-api-render.onrender.com'      // Backup
    ];
    this.currentAPI = 0;
  }

  async makeRequest(endpoint, data, timeout = 30000) {
    for (let i = 0; i < this.apis.length; i++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        const response = await fetch(`${this.apis[this.currentAPI]}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
          signal: controller.signal
        });

        clearTimeout(timeoutId);
        
        if (response.ok) {
          return await response.json();
        }
      } catch (error) {
        console.log(`API ${this.currentAPI} failed, trying backup...`);
        this.currentAPI = (this.currentAPI + 1) % this.apis.length;
      }
    }
    
    throw new Error('All ML APIs are unavailable');
  }
}
```

## 🚨 **Manejo de Cold Starts**

Si usas plataformas con sleep (Render), implementa:

```javascript
// Keep-alive para evitar cold starts
const keepAPIAlive = async () => {
  try {
    await fetch('https://tu-api.onrender.com/health');
  } catch (error) {
    console.log('API warming up...');
  }
};

// Ping cada 10 minutos
setInterval(keepAPIAlive, 10 * 60 * 1000);
```

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

### 🔧 Errores Comunes de Desarrollo

#### "Import could not be resolved"
- **Causa:** Normal durante desarrollo sin dependencias instaladas
- **Solución:** Se resuelve automáticamente en Docker/producción

#### Error de conexión MongoDB
```bash
# Verificar conectividad
curl -X GET https://tu-api.com/database/test
```
- Verificar `MONGODB_URL` en variables de entorno
- Asegurar IP en whitelist de MongoDB Atlas  
- Confirmar permisos de usuario (lectura mínimo)
- Probar conexión directa con MongoDB Compass

### 🚀 Errores de Despliegue

#### Error de memoria/recursos
```yaml
# Síntomas: 
# - Timeouts durante entrenamiento
# - Out of memory errors
# - API no responde

# Soluciones:
```
- **Reducir epochs:** `epochs: 20` → `epochs: 10`
- **Window size menor:** `window_size: 10` → `window_size: 5`  
- **Menos datos:** Limitar dataset de entrenamiento
- **Migrar a plataforma** con más recursos (Railway, GCP)

#### API no responde
```bash
# Diagnosticar
curl -v https://tu-api.com/health

# Revisar logs (ejemplo Railway)
railway logs

# Revisar logs (ejemplo Render)  
# Ver dashboard → Logs
```

**Soluciones por plataforma:**
- **Render:** Verificar que no esté en sleep mode
- **Railway:** Revisar variables de entorno
- **Google Cloud:** Verificar quotas y billing
- **Fly.io:** Revisar health checks

#### Docker build fails
```bash
# Error común: dependencias
# Solución: Verificar requirements.txt

# Error común: permisos
# Solución: Verificar Dockerfile permisos de usuario

# Error común: puertos
# Solución: Exponer puerto correcto (8000)
```

### ⏱️ Problemas de Performance

#### Cold Start lento (>30s)
```javascript
// Implementar pre-calentamiento
const warmupAPI = async () => {
  try {
    await fetch('https://tu-api.com/health', { 
      timeout: 60000 
    });
  } catch (e) {
    console.log('Cold start detected');
  }
};
```

#### Predicciones lentas
- **Verificar:** Tamaño del modelo (window_size)
- **Reducir:** Epochs de entrenamiento
- **Cache:** Implementar cache de predicciones
- **Batch:** Agrupar múltiples predicciones

### 🔍 Debugging

#### Ver logs en tiempo real
```bash
# Railway
railway logs --tail

# Render  
# Dashboard → Logs tab

# Google Cloud Run
gcloud logging read "resource.type=cloud_run_revision"

# Fly.io
flyctl logs

# DigitalOcean
doctl apps logs <app-id> --tail
```

#### Probar endpoints localmente
```bash
# Health check
curl http://localhost:8000/health

# Test con datos reales
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [...], "window_size": 5}'
```

#### Variables de entorno
```bash
# Verificar que estén configuradas
curl https://tu-api.com/model/info

# Debería mostrar conexión DB exitosa
```

## � Monitoreo y Alertas

### Health Checks Automatizados
```javascript
// Implementar en tu app principal
const healthMonitor = {
  async checkAPI() {
    try {
      const response = await fetch('https://tu-api.com/health');
      const data = await response.json();
      
      if (data.status !== 'healthy') {
        this.sendAlert('API unhealthy', data);
      }
      
      return data;
    } catch (error) {
      this.sendAlert('API down', error);
      throw error;
    }
  },

  sendAlert(type, details) {
    // Integrar con tu sistema de alertas
    console.error(`🚨 ${type}:`, details);
  }
};

// Verificar cada 5 minutos
setInterval(() => healthMonitor.checkAPI(), 5 * 60 * 1000);
```

## 📞 Soporte

### 1. 🔍 Diagnóstico Automático
```bash
# Script de diagnóstico
curl -s https://tu-api.com/health | jq '.'
curl -s https://tu-api.com/model/info | jq '.'
curl -s https://tu-api.com/database/test | jq '.'
```

### 2. 📋 Checklist de Problemas
- [ ] Variables de entorno configuradas
- [ ] MongoDB accesible desde la plataforma  
- [ ] Puerto expuesto correctamente (8000)
- [ ] Dockerfile building sin errores
- [ ] Logs muestran conexión DB exitosa
- [ ] Health endpoint responde 200

### 3. 🆘 Escalación
Si persisten problemas:
1. **Revisar logs específicos** de la plataforma
2. **Probar localmente** con Docker
3. **Verificar quotas/límites** de la plataforma
4. **Considerar migración** a otra plataforma

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
