import os
import logging
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    Cliente MongoDB asíncrono para operaciones de series de tiempo.
    Optimizado para usar con FastAPI y variables de entorno.
    """
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.mongodb_url = os.getenv("MONGODB_URL")
        self.database_name = os.getenv("MONGODB_DATABASE")
        
        if not self.mongodb_url:
            raise ValueError("MONGODB_URL no está configurada en las variables de entorno")
        
        if not self.database_name:
            raise ValueError("MONGODB_DATABASE no está configurada en las variables de entorno")
    
    async def connect(self):
        """Establecer conexión con MongoDB"""
        try:
            logger.info(f"Conectando a MongoDB: {self.mongodb_url[:50]}...")
            
            # Aumentar timeouts para servicios cloud
            self.client = AsyncIOMotorClient(
                self.mongodb_url,
                serverSelectionTimeoutMS=30000,  # 30 segundos
                connectTimeoutMS=30000,          # 30 segundos  
                socketTimeoutMS=30000,           # 30 segundos
                maxPoolSize=10,                  # Pool de conexiones
                retryWrites=True,                # Reintentar escrituras
                retryReads=True                  # Reintentar lecturas
            )
            
            # Verificar conexión con timeout extendido
            await asyncio.wait_for(
                self.client.admin.command('ping'), 
                timeout=30.0
            )
            
            self.database = self.client[self.database_name]
            logger.info(f"✅ Conectado exitosamente a MongoDB: {self.database_name}")
            
        except asyncio.TimeoutError:
            logger.error("❌ Timeout conectando a MongoDB (30s)")
            raise ConnectionFailure("Timeout de conexión a MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Error conectando a MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error inesperado conectando a MongoDB: {e}")
            raise
    
    async def close(self):
        """Cerrar conexión con MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Conexión MongoDB cerrada")
    
    async def ping(self) -> bool:
        """Verificar que la conexión esté activa"""
        try:
            if not self.client:
                return False
                
            # Ping con timeout extendido
            await asyncio.wait_for(
                self.client.admin.command('ping'),
                timeout=20.0
            )
            return True
            
        except asyncio.TimeoutError:
            logger.error("Timeout en ping a MongoDB")
            return False
        except Exception as e:
            logger.error(f"Error en ping a MongoDB: {e}")
            return False
    
    async def get_time_series_data(
        self, 
        collection_name: str, 
        parameter: str = "temperatura",
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_field: str = "fecha"
    ) -> List[Dict]:
        """
        Obtener datos de series de tiempo de una colección para un parámetro específico.
        
        Args:
            collection_name: Nombre de la colección
            parameter: Parámetro a extraer (temperatura, ph, oxigeno)
            limit: Límite máximo de documentos a retornar
            start_date: Fecha inicial para filtrar
            end_date: Fecha final para filtrar
            sort_field: Campo por el cual ordenar (por defecto fecha)
            
        Returns:
            Lista de documentos transformados con timestamp y value
        """
        try:
            if self.database is None:
                raise ValueError("No hay conexión a la base de datos")
            
            collection = self.database[collection_name]
            
            # Construir filtro de fechas
            date_filter = {}
            if start_date or end_date:
                date_filter[sort_field] = {}
                if start_date:
                    date_filter[sort_field]["$gte"] = start_date
                if end_date:
                    date_filter[sort_field]["$lte"] = end_date
            
            # Construir query
            cursor = collection.find(date_filter).sort(sort_field, -1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            # Ejecutar query
            documents = await cursor.to_list(length=None)
            
            logger.info(f"Obtenidos {len(documents)} documentos de {collection_name}")
            
            # Transformar documentos al formato esperado por el modelo
            validated_docs = []
            for doc in documents:
                if parameter in doc and 'fecha' in doc:
                    try:
                        # Normalizar el timestamp - convertir todo a número
                        timestamp_value = doc['fecha']
                        if isinstance(timestamp_value, str):
                            # Si es string datetime, convertir a timestamp
                            from datetime import datetime
                            try:
                                dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                                timestamp_value = int(dt.timestamp() * 1000)  # milliseconds
                            except:
                                # Si no se puede parsear, usar timestamp actual
                                timestamp_value = int(datetime.now().timestamp() * 1000)
                        elif hasattr(timestamp_value, 'timestamp'):
                            # Si es un objeto datetime
                            timestamp_value = int(timestamp_value.timestamp() * 1000)
                        
                        # Crear documento en formato esperado por el modelo
                        transformed_doc = {
                            '_id': str(doc.get('_id')),
                            'timestamp': timestamp_value,
                            'value': float(doc[parameter]),
                            'original_data': {
                                'temperatura': doc.get('temperatura'),
                                'ph': doc.get('ph'),
                                'oxigeno': doc.get('oxigeno')
                            }
                        }
                        validated_docs.append(transformed_doc)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Documento con {parameter} no numérico omitido: {doc.get('_id')} - {e}")
                else:
                    logger.warning(f"Documento sin campo '{parameter}' o 'fecha' omitido: {doc.get('_id')}")
            
            logger.info(f"Validados {len(validated_docs)} documentos para parámetro {parameter}")
            return validated_docs
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de {collection_name}: {e}")
            raise
    
    async def get_all_parameters_data(
        self,
        collection_name: str,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[Dict]]:
        """
        Obtener datos de todos los parámetros (temperatura, ph, oxigeno) de una vez.
        
        Returns:
            Diccionario con listas de datos para cada parámetro
        """
        try:
            parameters = ['temperatura', 'ph', 'oxigeno']
            all_data = {}
            
            for param in parameters:
                data = await self.get_time_series_data(
                    collection_name=collection_name,
                    parameter=param,
                    limit=limit,
                    start_date=start_date,
                    end_date=end_date
                )
                all_data[param] = data
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de todos los parámetros: {e}")
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict:
        """
        Obtener información sobre una colección.
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Diccionario con información de la colección
        """
        try:
            if self.database is None:
                raise ValueError("No hay conexión a la base de datos")
            
            collection = self.database[collection_name]
            
            # Obtener estadísticas básicas
            count = await collection.count_documents({})
            
            # Obtener un documento ejemplo
            sample_doc = await collection.find_one()
            
            # Obtener fechas min y max si existe timestamp
            date_range = None
            if sample_doc and 'timestamp' in sample_doc:
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "min_date": {"$min": "$timestamp"},
                            "max_date": {"$max": "$timestamp"}
                        }
                    }
                ]
                
                async for result in collection.aggregate(pipeline):
                    date_range = {
                        "start": result["min_date"],
                        "end": result["max_date"]
                    }
                    break
            
            # Convertir ObjectId a string si existe
            if sample_doc and '_id' in sample_doc:
                sample_doc['_id'] = str(sample_doc['_id'])
            
            return {
                "collection_name": collection_name,
                "document_count": count,
                "sample_document": sample_doc,
                "date_range": date_range
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo info de {collection_name}: {e}")
            raise
    
    async def list_collections(self) -> List[str]:
        """Listar todas las colecciones disponibles"""
        try:
            if self.database is None:
                raise ValueError("No hay conexión a la base de datos")
            
            collections = await self.database.list_collection_names()
            return collections
            
        except Exception as e:
            logger.error(f"Error listando colecciones: {e}")
            raise
    
    async def test_connection(self) -> Dict:
        """
        Probar la conexión y retornar información de la base de datos.
        
        Returns:
            Diccionario con información de la conexión
        """
        try:
            if not self.client:
                await self.connect()
            
            # Obtener información del servidor
            server_info = await self.client.server_info()
            
            # Listar colecciones
            collections = await self.list_collections()
            
            return {
                "status": "connected",
                "database": self.database_name,
                "server_version": server_info.get("version"),
                "collections": collections,
                "collection_count": len(collections)
            }
            
        except Exception as e:
            logger.error(f"Error en test de conexión: {e}")
            return {
                "status": "error",
                "error": str(e)
            }