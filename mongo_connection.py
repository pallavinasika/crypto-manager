import os
import asyncio
import json
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
DATABASE_NAME = "crypto_intelligence_platform"
MOCK_DATA_FILE = "database/offline_storage.json"
# We define these as helpers but will fetch them fresh in connect_to_mongo
def get_mongodb_url():
    return os.getenv("MONGODB_URL", "mongodb://localhost:27017")

def get_use_offline_storage():
    return os.getenv("USE_OFFLINE_STORAGE", "false").lower() == "true"

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def json_decoder(dct):
    for k, v in dct.items():
        if isinstance(v, str):
            # Try parsing ObjectId
            if len(v) == 24 and all(c in "0123456789abcdef" for c in v):
                try: dct[k] = ObjectId(v)
                except: pass
            # Try parsing datetime
            elif "T" in v and len(v) >= 19:
                try: dct[k] = datetime.fromisoformat(v)
                except: pass
    return dct

class AsyncIter:
    def __init__(self, data, parent_col, projection=None):
        self.data = data
        self.index = 0
        self.parent_col = parent_col
        self.projection = projection
    def __aiter__(self):
        return self
    async def __anext__(self):
        if self.index < len(self.data):
            val = self.data[self.index].copy()
            if self.projection and self.projection.get("_id") == 0:
                val.pop("_id", None)
            self.index += 1
            return val
        raise StopAsyncIteration
    def limit(self, n):
        self.data = self.data[:n]
        return self
    def sort(self, field, direction=1):
        self.data = sorted(self.data, key=lambda x: x.get(field, datetime.min), reverse=(direction == -1))
        return self
    async def to_list(self, length=None):
        data_to_return = self.data[:length] if length else self.data
        if self.projection and self.projection.get("_id") == 0:
            return [{k: v for k, v in d.items() if k != "_id"} for d in data_to_return]
        return data_to_return

class MemoryCollection:
    """Mock MongoDB collection for demonstration if real DB is down."""
    def __init__(self, name, parent_db=None):
        self.name = name
        self.data = []
        self.parent_db = parent_db

    async def _save(self):
        if self.parent_db:
            await self.parent_db.save_to_file()

    async def find_one(self, query=None, projection=None, sort=None):
        if query is None:
            query = {}
        
        filtered_data = [item for item in self.data if self._match_document(item, query)]

        if not filtered_data:
            return None

        if sort:
            sort_field, direction = sort[0]
            filtered_data = sorted(filtered_data, key=lambda x: x.get(sort_field, datetime.min), reverse=(direction == -1))

        if projection:
            result = filtered_data[0].copy()
            if projection.get("_id") == 0:
                result.pop("_id", None)
            return result
            
        return filtered_data[0].copy()


    def _match_document(self, item, query):
        if not query: return True
        
        # Special case for $or
        if "$or" in query:
            for option in query["$or"]:
                if self._match_document(item, option):
                    return True
            return False

        for k, v in query.items():
            item_val = item.get(k)
            if isinstance(v, dict):
                # Handle basic $gte, $lte
                try:
                    if "$gte" in v and (item_val is None or item_val < v["$gte"]): return False
                    if "$lte" in v and (item_val is None or item_val > v["$lte"]): return False
                except: return False
            # Handle ObjectId vs string comparison for any field
            elif isinstance(v, (ObjectId, str)) and isinstance(item_val, (ObjectId, str)) and (isinstance(v, ObjectId) or isinstance(item_val, ObjectId)):
                if str(v) != str(item_val): return False
            elif item_val != v:
                return False
        return True

    async def insert_one(self, document):
        if "_id" not in document:
            document["_id"] = ObjectId()
        self.data.append(document)
        await self._save()
        # Add a dummy result object with inserted_id
        from unittest.mock import MagicMock
        res = MagicMock()
        res.inserted_id = document["_id"]
        return res

    def find(self, query=None, projection=None, sort=None):
        if query is None: query = {}
        filtered_data = [item for item in self.data if self._match_document(item, query)]
        return AsyncIter(filtered_data, self, projection)

    async def update_one(self, query, update, upsert=False):
        item = await self.find_one(query)
        if item:
            idx = next((i for i, d in enumerate(self.data) if str(d.get("_id")) == str(item.get("_id"))), -1)
            if idx != -1:
                target = self.data[idx]
                if "$set" in update: target.update(update["$set"])
                if "$push" in update:
                    for k, v in update["$push"].items():
                        if k not in target: target[k] = []
                        target[k].append(v)
                if "$pull" in update:
                    for k, v in update["$pull"].items():
                        if k in target and isinstance(target[k], list):
                            if isinstance(v, dict):
                                target[k] = [x for x in target[k] if not all(x.get(sk) == sv for sk, sv in v.items())]
                            else:
                                target[k] = [x for x in target[k] if x != v]
                await self._save()
        elif upsert:
            new_item = query.copy()
            if "$set" in update: new_item.update(update["$set"])
            await self.insert_one(new_item)
        return True

    async def insert_many(self, documents):
        for doc in documents:
            if "_id" not in doc: doc["_id"] = ObjectId()
            self.data.append(doc)
        await self._save()
        return True

    async def delete_one(self, query):
        item = await self.find_one(query)
        if item:
            idx = next((i for i, d in enumerate(self.data) if str(d.get("_id")) == str(item.get("_id"))), -1)
            if idx != -1:
                self.data.pop(idx)
                await self._save()
                return type('obj', (object,), {'deleted_count': 1})
        return type('obj', (object,), {'deleted_count': 0})

    async def delete_many(self, query):
        if query is None: query = {}
        initial_count = len(self.data)
        self.data = [item for item in self.data if not self._match_document(item, query)]
        deleted_count = initial_count - len(self.data)
        if deleted_count > 0:
            await self._save()
        return type('obj', (object,), {'deleted_count': deleted_count})

    async def count_documents(self, query=None):
        if query is None: return len(self.data)
        return sum(1 for item in self.data if self._match_document(item, query))

class MemoryDatabase:
    """Mock MongoDB database for demonstration."""
    def __init__(self):
        self.collections = {}
        self.is_mock = True
        self._save_task = None

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MemoryCollection(name, self)
        return self.collections[name]

    def get_storage_name(self):
        return "Offline JSON Storage"

    async def save_to_file(self):
        """Debounced save to file to prevent excessive disk I/O."""
        if self._save_task:
            self._save_task.cancel()
        
        async def _do_save():
            await asyncio.sleep(0.5) # Wait 500ms for more changes
            try:
                os.makedirs(os.path.dirname(MOCK_DATA_FILE), exist_ok=True)
                serializable_data = {name: col.data for name, col in self.collections.items()}
                # Use a temp file for safer writing
                temp_file = MOCK_DATA_FILE + ".tmp"
                with open(temp_file, "w") as f:
                    json.dump(serializable_data, f, cls=JSONEncoder, indent=2)
                os.replace(temp_file, MOCK_DATA_FILE)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Failed to save offline storage: {e}")
            finally:
                self._save_task = None

        self._save_task = asyncio.create_task(_do_save())

    async def load_from_file(self):
        if os.path.exists(MOCK_DATA_FILE):
            try:
                with open(MOCK_DATA_FILE, "r") as f:
                    loaded_data = json.load(f, object_hook=json_decoder)
                    for name, data in loaded_data.items():
                        col = self[name]
                        col.data = data
                print(f"✅ Loaded persistent offline data from {MOCK_DATA_FILE}")
            except Exception as e:
                print(f"⚠️  Failed to load offline storage: {e}")

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None
    is_mock = False

    def get_storage_name(self):
        return "Offline JSON Storage" if self.is_mock else "MongoDB"

db = MongoDB()

async def connect_to_mongo():
    mongodb_url = get_mongodb_url()
    use_offline = get_use_offline_storage()
    
    # Check if running on Render but no MongoDB URL is provided (defaults to localhost)
    is_render = os.getenv("RENDER", "false").lower() == "true"
    is_default_url = "localhost" in mongodb_url
    
    if use_offline or (is_render and is_default_url):
        if is_render and is_default_url:
            print("💡 Render environment detected: No MONGODB_URL provided.")
        print("💡 Using Dynamic JSON Offline Storage.")
        db.db = MemoryDatabase()
        db.is_mock = True
        await db.db.load_from_file()
        return

    try:
        print(f"🔌 Connecting to MongoDB Atlas: {mongodb_url[:20]}...")
        # Robust connection with pooling and timeouts
        db.client = AsyncIOMotorClient(
            mongodb_url, 
            serverSelectionTimeoutMS=5000,
            maxPoolSize=100,
            minPoolSize=10
        )
        print("📡 Sending ping to MongoDB...")
        # Verify connection with a ping
        await db.client.admin.command('ping')
        print("✅ Ping successful!")
        db.db = db.client[DATABASE_NAME]
        db.is_mock = False
        print(f"✅ Successfully connected to MongoDB at {mongodb_url[:20]}...")
        
        # Create necessary indexes
        print("📂 Verifying database indexes...")
        await create_indexes()
        print("🚀 Startup process complete.")
        
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("💡 Switching to Dynamic JSON Offline Storage...")
        db.db = MemoryDatabase()
        db.is_mock = True
        await db.db.load_from_file()

async def create_indexes():
    """Create indexes for optimized performance."""
    if db.is_mock:
        return
        
    try:
        # Users indexes
        await db.db["users"].create_index("email", unique=True)
        
        # Market data indexes
        await db.db["market_data"].create_index([("coin_id", 1), ("timestamp", -1)])
        await db.db["market_data"].create_index("timestamp")
        
        # Portfolios indexes
        await db.db["portfolios"].create_index("user_id")
        
        # Alerts indexes
        await db.db["alerts"].create_index("user_id")
        await db.db["alerts"].create_index("is_active")
        
        print("🚀 Database indexes verified/created successfully")
    except Exception as e:
        print(f"⚠️  Error creating indexes: {e}")

async def close_mongo_connection():
    if db.client:
        db.client.close()
    print("Closed database connection")

def get_database():
    return db.db

def get_storage_name():
    """Returns the name of the currently active storage system."""
    if db.is_mock and db.db is not None:
        return db.db.get_storage_name()
    return "MongoDB" if not db.is_mock else "Offline JSON Storage"

