import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

JSON_PATH = "output/loksatta_complete.json"

COLLECTION_NAME = "loksatta"

model = SentenceTransformer(
    "BAAI/bge-m3",
    trust_remote_code=True
)

client = QdrantClient(path="./qdrant_db")

with open(JSON_PATH,"r",encoding="utf-8") as f:
    data=json.load(f)

pages=data["pages"]

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

points=[]

idx=0

for page in pages:

    text=page["text"]

    embedding=model.encode(
        text,
        normalize_embeddings=True
    )

    points.append(
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "page":page["page_number"],
                "text":text
            }
        )
    )

    idx+=1

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("Completed")