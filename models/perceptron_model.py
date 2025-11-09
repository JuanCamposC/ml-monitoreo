import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class PerceptronTimeSeries:
    """
    Modelo Perceptrón optimizado para análisis de series de tiempo.
    Incluye preprocesamiento automático y métricas de evaluación.
    """
    
    def __init__(self, learning_rate: float = 0.01, random_state: int = 42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}
        
        # Configurar semilla para reproducibilidad
        np.random.seed(random_state)
    
    def _create_sequences(self, data: List[float], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear secuencias de entrada y salida para entrenamiento de series de tiempo.
        
        Args:
            data: Lista de valores de la serie de tiempo
            window_size: Tamaño de la ventana deslizante
            
        Returns:
            Tupla (X, y) donde X son las secuencias de entrada y y los valores objetivo
        """
        if len(data) <= window_size:
            raise ValueError(f"Los datos deben tener más de {window_size} puntos")
        
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        
        return np.array(X), np.array(y)
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Función de activación (lineal para regresión)"""
        return x
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Paso hacia adelante del perceptrón"""
        return self._activation_function(np.dot(X, self.weights) + self.bias)
    
    def train(self, data: List[Dict], window_size: int = 10, epochs: int = 100) -> Dict:
        """
        Entrenar el modelo Perceptrón con datos de series de tiempo.
        
        Args:
            data: Lista de documentos con campos 'timestamp' y 'value'
            window_size: Tamaño de la ventana deslizante
            epochs: Número de épocas de entrenamiento
            
        Returns:
            Diccionario con métricas del entrenamiento
        """
        try:
            logger.info(f"Iniciando entrenamiento con {len(data)} puntos de datos")
            
            # Extraer valores y ordenar por timestamp
            # Manejar diferentes tipos de timestamp (datetime, int, str)
            def get_sort_key(item):
                ts = item.get('timestamp')
                if ts is not None:
                    if isinstance(ts, str):
                        try:
                            return int(ts)
                        except:
                            return 0
                    return ts
                # Si no hay timestamp, usar el _id como fallback
                id_val = item.get('_id', '0')
                if isinstance(id_val, str):
                    return 0  # Para ObjectIds como string, usar 0 como fallback
                return id_val
            
            sorted_data = sorted(data, key=get_sort_key)
            values = [float(item['value']) for item in sorted_data]
            
            # Crear secuencias
            X, y = self._create_sequences(values, window_size)
            logger.info(f"Creadas {len(X)} secuencias de entrenamiento")
            
            # Normalizar datos
            X_scaled = self.scaler.fit_transform(X)
            
            # Inicializar pesos y bias
            n_features = X_scaled.shape[1]
            self.weights = np.random.normal(0, 0.01, n_features)
            self.bias = 0.0
            
            # Entrenamiento
            losses = []
            for epoch in range(epochs):
                # Forward pass
                predictions = self._forward_pass(X_scaled)
                
                # Calcular error
                error = predictions - y
                loss = np.mean(error ** 2)
                losses.append(loss)
                
                # Backward pass (gradiente descendente)
                dw = (2 / len(y)) * np.dot(X_scaled.T, error)
                db = (2 / len(y)) * np.sum(error)
                
                # Actualizar parámetros
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Log cada 20 épocas
                if epoch % 20 == 0:
                    logger.info(f"Época {epoch}/{epochs}, Loss: {loss:.6f}")
            
            # Calcular métricas finales
            final_predictions = self._forward_pass(X_scaled)
            mae = mean_absolute_error(y, final_predictions)
            mse = mean_squared_error(y, final_predictions)
            rmse = np.sqrt(mse)
            
            self.training_metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'final_loss': float(losses[-1]),
                'epochs_trained': epochs,
                'window_size': window_size,
                'training_samples': len(X)
            }
            
            self.is_trained = True
            logger.info(f"Entrenamiento completado. RMSE: {rmse:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            raise
    
    def predict(self, sequence: List[float], return_confidence: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Hacer predicción para una secuencia de valores.
        
        Args:
            sequence: Lista de valores de entrada
            return_confidence: Si retornar también la confianza
            
        Returns:
            Predicción o tupla (predicción, confianza)
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        if len(sequence) != len(self.weights):
            raise ValueError(f"La secuencia debe tener {len(self.weights)} valores")
        
        # Normalizar entrada
        sequence_array = np.array(sequence).reshape(1, -1)
        sequence_scaled = self.scaler.transform(sequence_array)
        
        # Hacer predicción
        prediction = self._forward_pass(sequence_scaled)[0]
        
        if return_confidence:
            # Confianza simple basada en la varianza de los pesos
            confidence = 1.0 / (1.0 + np.std(self.weights))
            return float(prediction), float(confidence)
        
        return float(prediction)
    
    def get_metrics(self) -> Dict:
        """Obtener métricas del modelo actual"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return self.training_metrics.copy()
    
    def get_parameters(self) -> Dict:
        """Obtener parámetros del modelo"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "learning_rate": self.learning_rate,
            "n_weights": len(self.weights) if self.weights is not None else 0,
            "bias": float(self.bias) if self.bias is not None else None,
            "weights_mean": float(np.mean(self.weights)) if self.weights is not None else None,
            "weights_std": float(np.std(self.weights)) if self.weights is not None else None
        }
    
    def save_model(self, filepath: str):
        """Guardar modelo entrenado"""
        if not self.is_trained:
            raise ValueError("No hay modelo entrenado para guardar")
        
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'scaler': self.scaler,
            'training_metrics': self.training_metrics,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath: str):
        """Cargar modelo guardado"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.scaler = model_data['scaler']
        self.training_metrics = model_data['training_metrics']
        self.learning_rate = model_data['learning_rate']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        logger.info(f"Modelo cargado desde {filepath}")
    
    def reset(self):
        """Reiniciar el modelo"""
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}
        logger.info("Modelo reiniciado")

class PerceptronTimeSeriesModel:
    def __init__(self):
        self.models = {}  # Diccionario para múltiples parámetros
        self.scalers = {}  # Scaler por parámetro
        self.is_trained = {}  # Estado por parámetro
        self.training_history = {}  # Historial por parámetro
        
    def train(self, data: List[Dict], window_size: int = 10, epochs: int = 100, parameter: str = "default"):
        """
        Entrenar modelo para un parámetro específico
        
        Args:
            data: Lista de diccionarios con timestamp y value
            window_size: Tamaño de la ventana de predicción
            epochs: Número de épocas de entrenamiento
            parameter: Nombre del parámetro (temperatura, ph, oxigeno)
        """
        try:
            if len(data) < window_size + 1:
                raise ValueError(f"Se necesitan al menos {window_size + 1} puntos de datos para window_size={window_size}")
            
            # Preparar datos
            values = [float(item['value']) for item in data]
            
            # Normalizar datos específicos para este parámetro
            if parameter not in self.scalers:
                self.scalers[parameter] = StandardScaler()
            
            values_scaled = self.scalers[parameter].fit_transform(np.array(values).reshape(-1, 1)).flatten()
            
            # Crear ventanas de entrenamiento
            X, y = self._create_windows(values_scaled, window_size)
            
            # Inicializar modelo para este parámetro
            self.models[parameter] = Perceptron(random_state=42, max_iter=epochs)
            
            # Entrenar
            self.models[parameter].fit(X, y)
            
            # Predecir para calcular métricas
            predictions = self.models[parameter].predict(X)
            
            # Desnormalizar para métricas reales
            y_real = self.scalers[parameter].inverse_transform(y.reshape(-1, 1)).flatten()
            pred_real = self.scalers[parameter].inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            # Calcular métricas
            mae = mean_absolute_error(y_real, pred_real)
            mse = mean_squared_error(y_real, pred_real) 
            rmse = np.sqrt(mse)
            
            # Guardar estado
            self.is_trained[parameter] = True
            self.training_history[parameter] = {
                'window_size': window_size,
                'epochs': epochs,
                'data_points': len(data),
                'mae': float(mae),
                'mse': float(mse), 
                'rmse': float(rmse),
                'trained_at': datetime.now().isoformat()
            }
            
            logger.info(f"Modelo {parameter} entrenado - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse)
            }
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {parameter}: {str(e)}")
            raise

    def predict(self, data: List[Dict], window_size: int, parameter: str = "default"):
        """
        Hacer predicción para un parámetro específico
        """
        if parameter not in self.models or not self.is_trained.get(parameter, False):
            raise ValueError(f"El modelo para '{parameter}' no ha sido entrenado")
        
        if len(data) != window_size:
            raise ValueError(f"Se requieren exactamente {window_size} puntos de datos")
        
        try:
            # Preparar datos
            values = [float(item['value']) for item in data]
            
            # Normalizar con el scaler específico
            values_scaled = self.scalers[parameter].transform(np.array(values).reshape(-1, 1)).flatten()
            
            # Hacer predicción
            prediction_scaled = self.models[parameter].predict([values_scaled])[0]
            
            # Desnormalizar
            prediction = self.scalers[parameter].inverse_transform([[prediction_scaled]])[0][0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error prediciendo {parameter}: {str(e)}")
            raise

    def get_info(self):
        """Información de todos los modelos"""
        return {
            'model_type': 'Perceptron',
            'parameters_trained': list(self.is_trained.keys()),
            'is_trained': any(self.is_trained.values()),
            'training_history': self.training_history
        }

    def reset(self):
        """Reiniciar todos los modelos"""
        self.models.clear()
        self.scalers.clear() 
        self.is_trained.clear()
        self.training_history.clear()
        logger.info("Todos los modelos reiniciados")