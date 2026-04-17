import os
import uuid
import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Clients
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

azure_client = AsyncOpenAI(
    api_key=azure_api_key,
    base_url=azure_endpoint
)

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    check_compatibility=False
)

COLLECTION_NAME = "councilx_audit_nodes"

async def init_qdrant():
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            # Azure text-embedding-3-small is usually 1536 dims
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )
            # Create payload index for session_id to allow filtering
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="session_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print(f"Created Qdrant Collection and Payload Index: {COLLECTION_NAME}")
        else:
            # For robustness, try creating the index if it was missed before
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="session_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except:
                pass # Already exists or other non-critical error
    except Exception as e:
        print(f"Qdrant Init Error: {e}")

async def get_embeddings(text: str):
    try:
        response = await azure_client.embeddings.create(
            input=text,
            model=embedding_deployment
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

async def ingest_pdf(file_path: str, session_id: str):
    await init_qdrant()
    
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    
    # Simple chunking for forensic audit
    chunks = [text[i:i+1500] for i in range(0, len(text), 1200)]
    
    points = []
    for chunk in chunks:
        if not chunk.strip(): continue
        vector = await get_embeddings(chunk)
        if vector:
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "session_id": session_id,
                    "source": os.path.basename(file_path)
                }
            ))
    
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Ingested {len(points)} chunks into Qdrant for session {session_id}")
    return len(points)

async def search_context(query: str, session_id: str, limit: int = 3):
    await init_qdrant()
    vector = await get_embeddings(query)
    if not vector: return ""
    
    from qdrant_client.http.exceptions import UnexpectedResponse
    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            ),
            limit=limit
        ).points
        
        context = "\n\n".join([r.payload["text"] for r in results])
        return context
    except UnexpectedResponse as e:
        if e.status_code == 404:
            print(f"Qdrant collection {COLLECTION_NAME} not found. Returning empty context.")
            return ""
        raise e
    except Exception as e:
        print(f"Search Context Error: {e}")
        return ""
