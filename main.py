import os
import json
import uuid
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Council X Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://councilx.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Safety net: force CORS headers on every response (Azure App Service sometimes strips them)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

# Note: CORSMiddleware is usually enough. ForceCORSMiddleware can sometimes interfere with streaming.
# We'll keep it but ensure it doesn't buffer.


app.add_middleware(ForceCORSMiddleware)

# Models
class ChatQuery(BaseModel):
    uid: str
    prompt: str
    history: List[dict]
    modelsL1: List[str]
    modelL2: str
    session_id: Optional[str] = None

class TitleRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-4o"

class SaveChatRequest(BaseModel):
    session_id: str
    title: str
    messages: list

class FeedbackRequest(BaseModel):
    session_id: str
    message_idx: int
    feedback: str

class AuthProfileRequest(BaseModel):
    uid: str
    email: str
    display_name: str

class ProfileUpdateRequest(BaseModel):
    uid: str
    display_name: Optional[str] = None
    custom_instructions: Optional[str] = None
    has_premium: Optional[bool] = None

# Logic Imports
from graph import get_app, route_query

@app.post("/ask/stream_graph")
async def stream_graph(query: ChatQuery):
    graph = get_app()
    
    async def event_generator():
        config = {"configurable": {"thread_id": query.session_id or str(uuid.uuid4())}}
        initial_state = {
            "uid": query.uid,
            "session_id": query.session_id,
            "question": query.prompt,
            "history": query.history,
            "models_l1": query.modelsL1,
            "model_l2": query.modelL2,
            "iterations": 0,
            "context": ""
        }
        
        try:
            # Send initial event to confirm connection
            yield f"data: {json.dumps({'node': 'start', 'state': {}})}\n\n"
            
            async for event in graph.astream(initial_state, config=config):
                for node_name, state_update in event.items():
                    try:
                        clean_state = jsonable_encoder(state_update)
                        yield f"data: {json.dumps({'node': node_name, 'state': clean_state})}\n\n"
                    except Exception as e:
                        print(f"Error encoding state: {e}")
                        yield f"data: {json.dumps({'node': 'error_node', 'state': {'error': str(e)}})}\n\n"
        except Exception as e:
            print(f"Graph execution error: {e}")
            yield f"data: {json.dumps({'node': 'error_node', 'state': {'error': str(e)}})}\n\n"
                
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering for Azure/Nginx
            "Access-Control-Allow-Origin": "*", # Redundant but safe
        }
    )

@app.post("/api/ingest")
async def ingest_file_api(session_id: str = Form(...), file: UploadFile = File(...)):
    from rag_service import ingest_pdf
    
    # Save temp file
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        chunks = await ingest_pdf(file_path, session_id)
        return {"status": "success", "chunks": chunks, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Chat Management Routes
@app.post("/api/chats/generate_title")
async def generate_title_api(req: TitleRequest):
    sys_prompt = "Generate a 3-4 word clinical title for the inquiry. No quotes."
    model_to_use = req.model if req.model else "gpt-5.4-mini"
    resp = await route_query(model_to_use, sys_prompt, req.prompt)
    return {"status": "success", "title": resp.response}

@app.post("/api/auth/profile")
async def auth_profile_api(req: AuthProfileRequest):
    from db import create_or_update_user
    user_data = await create_or_update_user(req.uid, req.email, req.display_name)
    if "_id" in user_data: user_data["_id"] = str(user_data["_id"])
    return {"status": "success", "user": user_data}

@app.post("/api/auth/update_profile")
async def update_profile_api(req: ProfileUpdateRequest):
    from db import update_user_profile
    data = {}
    if req.display_name is not None: data["display_name"] = req.display_name
    if req.custom_instructions is not None: data["custom_instructions"] = req.custom_instructions
    if req.has_premium is not None: data["has_premium"] = req.has_premium
    user_data = await update_user_profile(req.uid, data)
    if not user_data: raise HTTPException(status_code=404, detail="User not found")
    if "_id" in user_data: user_data["_id"] = str(user_data["_id"])
    return {"status": "success", "user": user_data}

@app.get("/api/chats/{uid}")
async def get_chats_api(uid: str):
    from db import get_user_chats
    chats = await get_user_chats(uid)
    return {"status": "success", "chats": chats}

@app.post("/api/chats/{uid}")
async def save_chat_api(uid: str, req: SaveChatRequest):
    from db import save_chat_session
    await save_chat_session(uid, req.session_id, req.title, req.messages)
    return {"status": "success"}

@app.get("/api/chats/shared/{session_id}")
async def get_shared_chat(session_id: str):
    from db import get_chat_by_id
    chat = await get_chat_by_id(session_id)
    if not chat: raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "success", "chat": chat}

@app.post("/api/chats/{uid}/feedback")
async def save_feedback(uid: str, req: FeedbackRequest):
    from db import save_message_feedback
    await save_message_feedback(req.session_id, req.message_idx, req.feedback)
    return {"status": "success"}

@app.delete("/api/chats/{uid}/{session_id}")
async def delete_chat_api(uid: str, session_id: str):
    from db import delete_chat_session
    await delete_chat_session(uid, session_id)
    return {"status": "success"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
