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

# Modelos MULTIVARIABLES separados para cada parámetro objetivo
# Cada modelo usa las 3 variables (temperatura, pH, oxígeno) para predecir su target
models = {
    'temperatura': PerceptronTimeSeries(target_parameter='temperatura'),
    'ph': PerceptronTimeSeries(target_parameter='ph'),
    'oxigeno': PerceptronTimeSeries(target_parameter='oxigeno')
}

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
            "models_loaded": {param: model.is_trained for param, model in models.items()}
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
        
        if len(data) < 10:  # Mínimo requerido para entrenamiento
            raise HTTPException(
                status_code=400, 
                detail=f"Se necesitan al menos 10 registros para entrenar. Encontrados: {len(data)}"
            )
        
        logger.info(f"Datos obtenidos: {len(data)} registros")
        
        # Entrenar modelo específico para este parámetro
        if request.parameter not in models:
            raise HTTPException(status_code=400, detail=f"Parámetro no válido: {request.parameter}")
            
        metrics = models[request.parameter].train(
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
    Hacer predicción MULTIVARIABLE automática: usa las 3 variables (temperatura, pH, oxígeno) para predecir
    
    Request: {
        "parameter": "temperatura",  # Parámetro a predecir
        "collection_name": "datos",
        "window_size": 10 (opcional)
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
        if not models[request.parameter].is_trained:
            raise HTTPException(
                status_code=400, 
                detail=f"El modelo para '{request.parameter}' no está entrenado. Usa /train primero."
            )
        
        # Determinar window_size (usar el del entrenamiento si no se especifica)
        window_size = request.window_size if request.window_size else 5
        
        logger.info(f"Obteniendo últimos {window_size} registros de TODAS las variables para predecir {request.parameter}")
        
        # Obtener datos de las 3 variables para predicción multivariable
        all_recent_data = await mongo_client.get_all_parameters_data(
            collection_name=request.collection_name,
            limit=window_size
        )
        
        if not all_recent_data:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron datos en {request.collection_name}"
            )
        
        # Verificar que tengamos suficientes datos de cada variable
        required_params = ["temperatura", "ph", "oxigeno"]
        for param in required_params:
            if param not in all_recent_data or len(all_recent_data[param]) < window_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Se necesitan {window_size} registros de {param}, solo se encontraron {len(all_recent_data.get(param, []))}"
                )
        
        logger.info(f"Usando datos multivariables: {window_size} registros de cada variable")
        
        # Preparar datos para predicción multivariable
        multivariate_sequence = {
            'temperatura': [float(item['value']) for item in all_recent_data['temperatura']],
            'ph': [float(item['value']) for item in all_recent_data['ph']],
            'oxigeno': [float(item['value']) for item in all_recent_data['oxigeno']]
        }
        
        # Hacer predicción multivariable
        prediction, confidence = models[request.parameter].predict(
            multivariate_sequence,
            return_confidence=True,
            multivariate=True
        )
        
        # Estadísticas del parámetro objetivo
        target_values = multivariate_sequence[request.parameter]
        
        return PredictionResponse(
            prediction=round(float(prediction), 2),
            parameter=request.parameter,
            confidence=round(float(confidence), 3),
            data_used={
                "count": len(target_values),
                "values": target_values,
                "avg": round(sum(target_values) / len(target_values), 2),
                "min": min(target_values),
                "max": max(target_values),
                "latest_timestamp": str(all_recent_data[request.parameter][-1]['timestamp']),
                "oldest_timestamp": str(all_recent_data[request.parameter][0]['timestamp']),
                "multivariate_info": {
                    "uses_all_variables": True,
                    "temperatura_range": f"{min(multivariate_sequence['temperatura']):.2f} - {max(multivariate_sequence['temperatura']):.2f}",
                    "ph_range": f"{min(multivariate_sequence['ph']):.2f} - {max(multivariate_sequence['ph']):.2f}",
                    "oxigeno_range": f"{min(multivariate_sequence['oxigeno']):.2f} - {max(multivariate_sequence['oxigeno']):.2f}"
                }
            },
            model_info={
                "window_size": window_size,
                "metrics": models[request.parameter].get_metrics(),
                "model_type": "Multivariable (usa temperatura + pH + oxígeno)"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error durante predicción multivariable: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Obtener información detallada de todos los modelos"""
    trained_models = sum(1 for model in models.values() if model.is_trained)
    total_models = len(models)
    
    model_details = {}
    for param, model in models.items():
        if model.is_trained:
            metrics = model.get_metrics()
            parameters = model.get_parameters()
            
            model_details[param] = {
                "status": "trained",
                "performance": {
                    "rmse": metrics.get("rmse", "N/A"),
                    "mae": metrics.get("mae", "N/A"),
                    "mse": metrics.get("mse", "N/A"),
                    "final_loss": metrics.get("final_loss", "N/A")
                },
                "training_info": {
                    "epochs_trained": metrics.get("epochs_trained", "N/A"),
                    "window_size": metrics.get("window_size", "N/A"),
                    "training_samples": metrics.get("training_samples", "N/A")
                },
                "model_config": {
                    "learning_rate": parameters.get("learning_rate", "N/A"),
                    "n_weights": parameters.get("n_weights", "N/A"),
                    "bias": parameters.get("bias", "N/A")
                },
                "ready_for_prediction": True
            }
        else:
            model_details[param] = {
                "status": "not_trained",
                "performance": None,
                "training_info": None,
                "model_config": None,
                "ready_for_prediction": False
            }
    
    return {
        "overview": {
            "total_models": total_models,
            "trained_models": trained_models,
            "untrained_models": total_models - trained_models,
            "readiness_percentage": round((trained_models / total_models) * 100, 1),
            "all_models_ready": trained_models == total_models
        },
        "model_details": model_details,
        "api_status": {
            "prediction_available": trained_models > 0,
            "full_prediction_available": trained_models == total_models,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.post("/model/reset")
async def reset_model():
    """Reiniciar todos los modelos"""
    for model in models.values():
        model.reset()
    return {"status": "success", "message": "Todos los modelos reiniciados"}

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

@app.get("/data/raw/{collection_name}")
async def get_raw_mongodb_data(collection_name: str, limit: int = 5):
    """Ver documentos RAW directos de MongoDB incluyendo el campo fecha original"""
    try:
        if mongo_client.database is None:
            raise HTTPException(status_code=500, detail="No hay conexión a la base de datos")
        
        collection = mongo_client.database[collection_name]
        
        # Obtener por fecha descendente
        docs_fecha = await collection.find().sort('fecha', -1).limit(limit).to_list(length=limit)
        # Obtener por _id descendente
        docs_id = await collection.find().sort('_id', -1).limit(limit).to_list(length=limit)
        
        # Convertir ObjectId a string para JSON
        for doc in docs_fecha:
            doc['_id'] = str(doc['_id'])
        for doc in docs_id:
            doc['_id'] = str(doc['_id'])
        
        return {
            "sorted_by_fecha_desc": docs_fecha,
            "sorted_by_id_desc": docs_id,
            "note": "Muestra documentos RAW de MongoDB con ambos ordenamientos"
        }
    except Exception as e:
        logger.error(f"Error obteniendo datos raw: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/sample/{collection_name}")
async def get_sample_data(collection_name: str, limit: int = 10):
    """Obtener muestra de los últimos 10 datos (exactamente los que usa para entrenamiento)"""
    try:
        # Forzar limit a 10 para mostrar exactamente los datos que usa el entrenamiento
        training_limit = 10
        
        # Obtener exactamente los últimos 10 datos de cada parámetro
        sample_data = {}
        parameters = ['temperatura', 'ph', 'oxigeno']
        
        for param in parameters:
            data = await mongo_client.get_time_series_data(
                collection_name=collection_name,
                parameter=param,
                limit=training_limit
            )
            sample_data[param] = data
        
        return {
            "status": "success",
            "collection": collection_name,
            "sample_size": training_limit,
            "data": sample_data,
            "total_records": {param: len(data) for param, data in sample_data.items()},
            "note": "Estos son exactamente los últimos 10 datos que usa el entrenamiento (mínimo requerido: 10)",
            "training_info": {
                "data_used_for_training": training_limit,
                "minimum_data_required": 10,
                "window_size_default": 5,
                "sequences_created_per_param": max(0, training_limit - 5)
            }
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo muestra: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/all-parameters")
async def train_all_parameters(request: TrainAllRequest):
    """Entrenar modelos MULTIVARIABLES para todos los parámetros (usa las 3 variables para cada predicción)"""
    try:
        start_time = time.time()
        
        # Verificar conexión
        if not await mongo_client.ping():
            raise HTTPException(status_code=500, detail="No se puede conectar a MongoDB")
        
        # Obtener solo los últimos 10 datos de TODAS las variables para entrenamiento multivariable
        all_data = await mongo_client.get_all_parameters_data(
            collection_name=request.collection_name,
            limit=10  # Usar solo los últimos 10 registros
        )
        
        if not all_data:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron datos en la colección '{request.collection_name}'"
            )
        
        # Verificar que tengamos datos de las 3 variables
        parameters = ["temperatura", "ph", "oxigeno"]
        for param in parameters:
            if param not in all_data or len(all_data[param]) < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Se requieren al menos 10 registros de cada variable. {param}: {len(all_data.get(param, []))}"
                )
        
        results = {}
        
        # Entrenar un modelo para cada parámetro objetivo (pero cada uno usa las 3 variables)
        for parameter in parameters:
            param_start_time = time.time()
            try:
                logger.info(f"Entrenando modelo MULTIVARIABLE para predecir {parameter}")
                logger.info(f"  Usando {len(all_data['temperatura'])} registros de temperatura")
                logger.info(f"  Usando {len(all_data['ph'])} registros de pH")
                logger.info(f"  Usando {len(all_data['oxigeno'])} registros de oxígeno")
                
                # Obtener estadísticas del parámetro objetivo
                values = [float(item['value']) for item in all_data[parameter]]
                data_stats = {
                    "count": len(values),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2), 
                    "mean": round(sum(values) / len(values), 2),
                    "std": round(np.std(values), 2)
                }
                
                # Entrenar modelo MULTIVARIABLE (usa las 3 variables)
                metrics = models[parameter].train(
                    data=all_data,  # Pasar TODAS las variables
                    window_size=request.window_size,
                    epochs=request.epochs,
                    multivariate=True  # Activar modo multivariable
                )
                
                param_training_time = round(time.time() - param_start_time, 2)
                
                results[parameter] = {
                    "success": True,
                    "data_stats": data_stats,
                    "training_data_used": all_data[parameter],  # Mostrar datos del target
                    "multivariate_info": {
                        "uses_all_variables": True,
                        "variables": ["temperatura", "pH", "oxígeno"],
                        "total_features": metrics.get("n_features", 0),
                        "description": f"Modelo usa las 3 variables simultáneamente para predecir {parameter}"
                    },
                    "training_config": {
                        "window_size": request.window_size,
                        "epochs": request.epochs,
                        "sequences_created": metrics.get("training_samples", 0),
                        "limit_applied": 10,
                        "multivariate": True
                    },
                    "performance_metrics": {
                        "mae": metrics.get("mae", 0),
                        "mse": metrics.get("mse", 0), 
                        "rmse": metrics.get("rmse", 0),
                        "final_loss": metrics.get("final_loss", 0)
                    },
                    "training_time_seconds": param_training_time,
                    "model_status": "trained_successfully",
                    "timestamp": datetime.now().isoformat()
                }
                    
            except Exception as e:
                logger.error(f"Error entrenando {parameter}: {e}")
                results[parameter] = {
                    "success": False,
                    "error": str(e)
                }
        
        total_time = round(time.time() - start_time, 2)
        successful_trainings = sum(1 for r in results.values() if r.get("success"))
        failed_trainings = len(parameters) - successful_trainings
        
        # Calcular estadísticas agregadas de rendimiento
        successful_results = [r for r in results.values() if r.get("success")]
        if successful_results:
            avg_rmse = round(np.mean([r["performance_metrics"]["rmse"] for r in successful_results]), 4)
            best_rmse = round(min([r["performance_metrics"]["rmse"] for r in successful_results]), 4)
            worst_rmse = round(max([r["performance_metrics"]["rmse"] for r in successful_results]), 4)
        else:
            avg_rmse = best_rmse = worst_rmse = None
        
        return {
            "status": "completed",
            "message": f"Entrenamiento MULTIVARIABLE completado: {successful_trainings}/{len(parameters)} modelos exitosos",
            "summary": {
                "successful_models": successful_trainings,
                "failed_models": failed_trainings,
                "total_models": len(parameters),
                "success_rate": round((successful_trainings / len(parameters)) * 100, 1),
                "collection_used": request.collection_name,
                "model_type": "Multivariable Perceptron (usa temperatura + pH + oxígeno)",
                "total_training_time": total_time,
                "avg_training_time_per_model": round(total_time / len(parameters), 2)
            },
            "performance_overview": {
                "average_rmse": avg_rmse,
                "best_rmse": best_rmse,
                "worst_rmse": worst_rmse,
                "models_ready_for_prediction": successful_trainings
            },
            "detailed_results": results,
            "training_config": {
                "window_size": request.window_size,
                "epochs": request.epochs,
                "limit": request.limit
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error durante entrenamiento múltiple: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error durante entrenamiento: {str(e)}")

@app.post("/predict/all")
async def predict_all_parameters(request: PredictAllRequest):
    """
    Predicción MULTIVARIABLE automática para todos los parámetros.
    Cada modelo usa las 3 variables (temperatura, pH, oxígeno) para hacer su predicción.
    
    Request: {
        "collection_name": "datos"
    }
    """
    try:
        parameters = ["temperatura", "ph", "oxigeno"]
        predictions = {}
        window_size = 5
        
        # Obtener datos de TODAS las variables una sola vez
        all_recent_data = await mongo_client.get_all_parameters_data(
            collection_name=request.collection_name,
            limit=window_size
        )
        
        # Verificar que tengamos datos de todas las variables
        data_available = all(
            param in all_recent_data and len(all_recent_data[param]) >= window_size
            for param in parameters
        )
        
        if not data_available:
            missing_info = {
                param: len(all_recent_data.get(param, []))
                for param in parameters
            }
            raise HTTPException(
                status_code=400,
                detail=f"Datos insuficientes para predicción multivariable. Necesarios: {window_size}, Disponibles: {missing_info}"
            )
        
        # Preparar secuencia multivariable (misma para todos los modelos)
        multivariate_sequence = {
            'temperatura': [float(item['value']) for item in all_recent_data['temperatura']],
            'ph': [float(item['value']) for item in all_recent_data['ph']],
            'oxigeno': [float(item['value']) for item in all_recent_data['oxigeno']]
        }
        
        for parameter in parameters:
            try:
                # Verificar que el modelo esté entrenado
                if not models[parameter].is_trained:
                    predictions[parameter] = {
                        "success": False,
                        "error": f"Modelo no entrenado para {parameter}",
                        "prediction": None
                    }
                    continue
                
                # Hacer predicción multivariable
                prediction, confidence = models[parameter].predict(
                    multivariate_sequence,
                    return_confidence=True,
                    multivariate=True
                )
                
                # Estadísticas del parámetro objetivo
                target_values = multivariate_sequence[parameter]
                current_avg = sum(target_values) / len(target_values)
                current_min = min(target_values)
                current_max = max(target_values)
                current_std = np.std(target_values)
                
                # Análisis de tendencia
                trend = "up" if prediction > target_values[-1] else "down" if prediction < target_values[-1] else "stable"
                change = round(prediction - target_values[-1], 2)
                change_percent = round((change / target_values[-1]) * 100, 2) if target_values[-1] != 0 else 0
                
                # Obtener métricas del modelo
                model_metrics = models[parameter].get_metrics()
                
                # Calcular volatilidad
                volatility = "high" if current_std > current_avg * 0.1 else "medium" if current_std > current_avg * 0.05 else "low"
                
                predictions[parameter] = {
                    "success": True,
                    "prediction": {
                        "value": round(float(prediction), 2),
                        "confidence": round(float(confidence), 3),
                        "confidence_level": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
                        "last_value": target_values[-1]
                    },
                    "trend_analysis": {
                        "direction": trend,
                        "change": change,
                        "change_percent": change_percent,
                        "volatility": volatility
                    },
                    "data_analysis": {
                        "values_used": target_values,
                        "data_points": len(target_values),
                        "current_avg": round(current_avg, 2),
                        "current_min": round(current_min, 2),
                        "current_max": round(current_max, 2),
                        "current_std": round(current_std, 2),
                        "latest_value": target_values[-1],
                        "oldest_value": target_values[0],
                        "data_range": round(current_max - current_min, 2)
                    },
                    "multivariate_context": {
                        "uses_all_variables": True,
                        "temperatura": f"avg={np.mean(multivariate_sequence['temperatura']):.2f}",
                        "ph": f"avg={np.mean(multivariate_sequence['ph']):.2f}",
                        "oxigeno": f"avg={np.mean(multivariate_sequence['oxigeno']):.2f}"
                    },
                    "model_performance": {
                        "rmse": model_metrics.get("rmse", "N/A"),
                        "mae": model_metrics.get("mae", "N/A"),
                        "mse": model_metrics.get("mse", "N/A"),
                        "training_samples": model_metrics.get("training_samples", "N/A")
                    },
                    "timestamps": {
                        "oldest": str(all_recent_data[parameter][0]['timestamp']),
                        "newest": str(all_recent_data[parameter][-1]['timestamp'])
                    }
                }
                
            except Exception as e:
                logger.error(f"Error prediciendo {parameter}: {e}")
                predictions[parameter] = {
                    "success": False,
                    "error": str(e),
                    "prediction": None
                }
        
        successful_predictions = sum(1 for p in predictions.values() if p.get("success"))
        failed_predictions = len(parameters) - successful_predictions
        
        # Calcular estadísticas generales de las predicciones exitosas
        successful_preds = [p for p in predictions.values() if p.get("success")]
        if successful_preds:
            avg_confidence = round(np.mean([p["prediction"]["confidence"] for p in successful_preds]), 3)
            min_confidence = round(min([p["prediction"]["confidence"] for p in successful_preds]), 3)
            max_confidence = round(max([p["prediction"]["confidence"] for p in successful_preds]), 3)
            
            # Análisis de tendencias
            trends = [p["trend_analysis"]["direction"] for p in successful_preds]
            trend_summary = {
                "up": trends.count("up"),
                "down": trends.count("down"), 
                "stable": trends.count("stable")
            }
        else:
            avg_confidence = min_confidence = max_confidence = None
            trend_summary = {"up": 0, "down": 0, "stable": 0}
        
        return {
            "status": "completed",
            "predictions": predictions,
            "summary": {
                "prediction_results": {
                    "successful": successful_predictions,
                    "failed": failed_predictions,
                    "total": len(parameters),
                    "success_rate": round((successful_predictions / len(parameters)) * 100, 1)
                },
                "confidence_analysis": {
                    "average": avg_confidence,
                    "minimum": min_confidence,
                    "maximum": max_confidence
                },
                "trend_overview": trend_summary,
                "data_source": {
                    "collection_used": request.collection_name,
                    "window_size": 5,
                    "prediction_timestamp": datetime.now().isoformat()
                }
            },
            "next_actions": {
                "retrain_needed": failed_predictions > 0,
                "models_to_retrain": [param for param, pred in predictions.items() if not pred.get("success")],
                "ready_for_monitoring": successful_predictions > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error en predicción múltiple automática: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)