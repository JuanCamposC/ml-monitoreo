from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging
import numpy as np
from datetime import datetime
import time
import asyncio

from models.perceptron_model import PerceptronTimeSeries
from database.mongodb_client import MongoDBClient

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Time Series API",
    description="API para análisis de series de tiempo usando Perceptrón",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
mongo_client = MongoDBClient()
perceptron_model = PerceptronTimeSeries()

# Modelos Pydantic para request/response
class TimeSeriesData(BaseModel):
    timestamp: datetime
    value: float
    additional_features: Optional[dict] = None

class PredictionRequest(BaseModel):
    parameter: str  # temperatura, ph, oxigeno
    collection_name: Optional[str] = "datos"
    window_size: Optional[int] = None  # Si no se especifica, usa el del entrenamiento

class PredictionResponse(BaseModel):
    prediction: float
    parameter: str
    confidence: float
    data_used: dict
    model_info: dict
    timestamp: str

class TrainingRequest(BaseModel):
    collection_name: str
    parameter: Optional[str] = "temperatura"  # temperatura, ph, oxigeno
    window_size: Optional[int] = 10
    epochs: Optional[int] = 100
    limit: Optional[int] = 1000  # Nuevo campo para límite de datos

class TrainAllRequest(BaseModel):
    collection_name: str = "datos"
    window_size: Optional[int] = 5
    epochs: Optional[int] = 20
    limit: Optional[int] = None

class PredictAllRequest(BaseModel):
    collection_name: str = "datos"

@app.on_event("startup")
async def startup_event():
    """Inicializar conexiones al arrancar"""
    max_retries = 5
    retry_delay = 10  # segundos
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Intento de conexión MongoDB #{attempt + 1}/{max_retries}")
            await mongo_client.connect()
            logger.info("✅ Conexión MongoDB exitosa")
            return
            
        except Exception as e:
            logger.error(f"❌ Intento #{attempt + 1} falló: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"⏳ Esperando {retry_delay}s antes del siguiente intento...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("❌ Todos los intentos de conexión fallaron")
                # No lanzar excepción para permitir que la API arranque
                logger.warning("⚠️ API iniciada sin conexión MongoDB")

@app.on_event("shutdown")
async def shutdown_event():
    """Cerrar conexiones al apagar la API"""
    logger.info("Cerrando conexiones...")
    await mongo_client.close()

@app.get("/")
async def root():
    """Endpoint de salud de la API"""
    return {
        "message": "ML Time Series API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Verificar el estado de la API y sus dependencias"""
    try:
        db_status = await mongo_client.ping()
        return {
            "status": "healthy",
            "database": "connected" if db_status else "disconnected",
            "model_loaded": perceptron_model.is_trained
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/train", response_model=dict)
async def train_model(request: TrainingRequest):
    """Entrenar el modelo con datos de MongoDB"""
    try:
        logger.info(f"Iniciando entrenamiento con colección: {request.collection_name}, parámetro: {request.parameter}")
        
        # Validar parámetro
        valid_parameters = ['temperatura', 'ph', 'oxigeno']
        if request.parameter not in valid_parameters:
            raise HTTPException(
                status_code=400, 
                detail=f"Parámetro inválido. Use uno de: {valid_parameters}"
            )
        
        # Obtener datos de MongoDB
        data = await mongo_client.get_time_series_data(
            collection_name=request.collection_name,
            parameter=request.parameter,
            limit=request.limit  # Usar el nuevo campo limit
        )
        
        if not data:
            raise HTTPException(status_code=404, detail="No se encontraron datos")
        
        if len(data) < 5:  # Mínimo requerido para entrenamiento
            raise HTTPException(
                status_code=400, 
                detail=f"Se necesitan al menos 5 registros para entrenar. Encontrados: {len(data)}"
            )
        
        logger.info(f"Datos obtenidos: {len(data)} registros")
        
        # Entrenar modelo
        metrics = perceptron_model.train(
            data=data,
            window_size=request.window_size,
            epochs=request.epochs
        )
        
        logger.info("Entrenamiento completado")
        return {
            "status": "success",
            "message": f"Modelo entrenado exitosamente para {request.parameter}",
            "parameter": request.parameter,
            "metrics": metrics,
            "data_points": len(data)
        }
        
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Hacer predicción automática: La API obtiene los datos de MongoDB
    
    Request: {
        "parameter": "temperatura",
        "collection_name": "datos",
        "window_size": 5 (opcional)
    }
    """
    try:
        # Validar parámetro
        if request.parameter not in ["temperatura", "ph", "oxigeno"]:
            raise HTTPException(
                status_code=400, 
                detail="Parámetro debe ser: temperatura, ph, o oxigeno"
            )
        
        # Verificar que el modelo esté entrenado para este parámetro
        if not perceptron_model.is_trained:
            raise HTTPException(
                status_code=400, 
                detail=f"El modelo para '{request.parameter}' no está entrenado. Usa /train primero."
            )
        
        # Determinar window_size (usar el del entrenamiento si no se especifica)
        window_size = request.window_size if request.window_size else 5
        
        logger.info(f"Obteniendo últimos {window_size} registros para {request.parameter}")
        
        # Obtener automáticamente los últimos datos de MongoDB
        recent_data = await mongo_client.get_time_series_data(
            collection_name=request.collection_name,
            parameter=request.parameter,
            limit=window_size * 2  # Obtener más datos para asegurar que tenemos suficientes
        )
        
        if not recent_data:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron datos para {request.parameter} en {request.collection_name}"
            )
        
        if len(recent_data) < window_size:
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan {window_size} registros, solo se encontraron {len(recent_data)}"
            )
        
        # Tomar exactamente los últimos window_size registros
        prediction_data = recent_data[-window_size:]
        
        logger.info(f"Usando {len(prediction_data)} registros para predicción de {request.parameter}")
        
        # Hacer predicción
        values = [float(item['value']) for item in prediction_data]
        
        prediction, confidence = perceptron_model.predict(
            values,
            return_confidence=True
        )
        
        # Calcular estadísticas de los datos usados
        values_used = [item['value'] for item in prediction_data]
        
        return PredictionResponse(
            prediction=round(float(prediction), 2),
            parameter=request.parameter,
            confidence=round(float(confidence), 3),
            data_used={
                "count": len(prediction_data),
                "values": values_used,
                "avg": round(sum(values_used) / len(values_used), 2),
                "min": min(values_used),
                "max": max(values_used),
                "latest_timestamp": str(prediction_data[-1]['timestamp']),
                "oldest_timestamp": str(prediction_data[0]['timestamp'])
            },
            model_info={
                "window_size": window_size,
                "metrics": perceptron_model.get_metrics()
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error durante predicción automática: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Obtener información del modelo actual"""
    return {
        "is_trained": perceptron_model.is_trained,
        "metrics": perceptron_model.get_metrics(),
        "parameters": perceptron_model.get_parameters()
    }

@app.post("/model/reset")
async def reset_model():
    """Reiniciar el modelo"""
    perceptron_model.reset()
    return {"status": "success", "message": "Modelo reiniciado"}

@app.get("/collections")
async def list_collections():
    """Listar todas las colecciones disponibles en MongoDB"""
    try:
        collections = await mongo_client.list_collections()
        return {
            "status": "success",
            "collections": collections,
            "count": len(collections)
        }
    except Exception as e:
        logger.error(f"Error listando colecciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Obtener información detallada de una colección"""
    try:
        info = await mongo_client.get_collection_info(collection_name)
        return {
            "status": "success",
            "info": info
        }
    except Exception as e:
        logger.error(f"Error obteniendo info de colección: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/test")
async def test_database_connection():
    """Probar la conexión a la base de datos"""
    try:
        result = await mongo_client.test_connection()
        return result
    except Exception as e:
        logger.error(f"Error en test de base de datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/sample/{collection_name}")
async def get_sample_data(collection_name: str, limit: int = 10):
    """Obtener datos de muestra de una colección para visualizar estructura"""
    try:
        # Obtener muestra de cada parámetro
        sample_data = {}
        parameters = ['temperatura', 'ph', 'oxigeno']
        
        for param in parameters:
            data = await mongo_client.get_time_series_data(
                collection_name=collection_name,
                parameter=param,
                limit=limit
            )
            sample_data[param] = data
        
        return {
            "status": "success",
            "collection": collection_name,
            "sample_size": limit,
            "data": sample_data,
            "total_records": {param: len(data) for param, data in sample_data.items()}
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo muestra: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/all-parameters")
async def train_all_parameters(request: TrainAllRequest):
    """Entrenar modelos para todos los parámetros disponibles"""
    try:
        start_time = time.time()
        
        # Verificar conexión
        if not await mongo_client.ping():
            raise HTTPException(status_code=500, detail="No se puede conectar a MongoDB")
        
        # Obtener datos para todos los parámetros
        all_data = await mongo_client.get_all_parameters_data(
            collection_name=request.collection_name,
            limit=request.limit
        )
        
        if not all_data:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron datos en la colección '{request.collection_name}'"
            )
        
        results = {}
        parameters = ["temperatura", "ph", "oxigeno"]  # Parámetros de tu sistema
        
        for parameter in parameters:
            try:
                if parameter in all_data and len(all_data[parameter]) >= 5:
                    logger.info(f"Entrenando modelo para {parameter} con {len(all_data[parameter])} registros")
                    
                    metrics = perceptron_model.train(
                        data=all_data[parameter],
                        window_size=request.window_size,
                        epochs=request.epochs,
                        parameter=parameter
                    )
                    
                    results[parameter] = {
                        "success": True,
                        "data_points": len(all_data[parameter]),
                        "final_metrics": metrics,
                        "training_time": round(time.time() - start_time, 2)
                    }
                else:
                    results[parameter] = {
                        "success": False,
                        "error": f"Datos insuficientes: {len(all_data.get(parameter, []))} < 5 requeridos",
                        "data_points": len(all_data.get(parameter, []))
                    }
                    
            except Exception as e:
                results[parameter] = {
                    "success": False,
                    "error": str(e)
                }
        
        total_time = round(time.time() - start_time, 2)
        successful_trainings = sum(1 for r in results.values() if r.get("success"))
        
        return {
            "message": f"Entrenamiento completado: {successful_trainings}/{len(parameters)} modelos exitosos",
            "results": results,
            "total_training_time": total_time,
            "collection_used": request.collection_name,
            "total_parameters": len(parameters)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error durante entrenamiento múltiple: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error durante entrenamiento: {str(e)}")

@app.post("/predict/all")
async def predict_all_parameters(request: PredictAllRequest):
    """
    Predicción automática para todos los parámetros
    
    Request: {
        "collection_name": "datos"
    }
    """
    try:
        parameters = ["temperatura", "ph", "oxigeno"]
        predictions = {}
        
        for parameter in parameters:
            try:
                # Verificar que el modelo esté entrenado
                if not perceptron_model.is_trained:
                    predictions[parameter] = {
                        "success": False,
                        "error": f"Modelo no entrenado para {parameter}",
                        "prediction": None
                    }
                    continue
                
                # Obtener datos automáticamente (usando window_size por defecto de 5)
                window_size = 5
                recent_data = await mongo_client.get_time_series_data(
                    collection_name=request.collection_name,
                    parameter=parameter,
                    limit=window_size * 2
                )
                
                if len(recent_data) < window_size:
                    predictions[parameter] = {
                        "success": False,
                        "error": f"Datos insuficientes: {len(recent_data)}/{window_size}",
                        "prediction": None
                    }
                    continue
                
                # Hacer predicción
                prediction_data = recent_data[-window_size:]
                values = [float(item['value']) for item in prediction_data]
                
                prediction, confidence = perceptron_model.predict(
                    values,
                    return_confidence=True
                )
                
                current_avg = sum(values) / len(values)
                trend = "up" if prediction > values[-1] else "down" if prediction < values[-1] else "stable"
                change = round(prediction - values[-1], 2)
                
                predictions[parameter] = {
                    "success": True,
                    "prediction": round(float(prediction), 2),
                    "confidence": round(float(confidence), 3),
                    "current_avg": round(current_avg, 2),
                    "latest_value": values[-1],
                    "trend": trend,
                    "change": change,
                    "data_points_used": len(values)
                }
                
            except Exception as e:
                predictions[parameter] = {
                    "success": False,
                    "error": str(e),
                    "prediction": None
                }
        
        successful_predictions = sum(1 for p in predictions.values() if p.get("success"))
        
        return {
            "predictions": predictions,
            "summary": {
                "successful": successful_predictions,
                "total": len(parameters),
                "collection_used": request.collection_name,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error en predicción múltiple automática: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)