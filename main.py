import os
import json
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Council X Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatQuery(BaseModel):
    uid: str
    prompt: str
    history: List[dict]
    modelsL1: List[str]
    modelL2: str
    modelL3: str
    modelL4: str

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

# Logic Imports
from graph import get_app, route_query

@app.post("/ask/stream_graph")
async def stream_graph(query: ChatQuery):
    from graph import get_app
    graph = get_app()
    
    async def event_generator():
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = {
            "question": query.prompt,
            "history": query.history,
            "models_l1": query.modelsL1,
            "model_l2": query.modelL2,
            "model_l3": query.modelL3,
            "model_l4": query.modelL4,
            "iterations": 0
        }
        print(f"DEBUG: Initial State -> {initial_state}")
        
        try:
            async for event in graph.astream(initial_state, config=config):
                for node_name, state_update in event.items():
                    try:
                        clean_state = jsonable_encoder(state_update)
                        yield f"data: {json.dumps({'node': node_name, 'state': clean_state})}\n\n"
                    except Exception as e:
                        print(f"CRITICAL SERIALIZATION ERROR in node {node_name}: {str(e)}")
                        yield f"data: {json.dumps({'node': 'error_node', 'state': {'error': str(e)}})}\n\n"
        except Exception as e:
            print(f"CRITICAL GRAPH EXECUTION ERROR: {str(e)}")
            yield f"data: {json.dumps({'node': 'error_node', 'state': {'error': str(e)}})}\n\n"
                
    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Chat Management Routes
@app.post("/api/chats/generate_title")
async def generate_title_api(req: TitleRequest):
    sys_prompt = "You are a forensic taxonomist. Generate a 3-4 word clinical title for the user's inquiry. DO NOT use quotes or periods. Just the words."
    model_to_use = req.model if req.model else "gpt-4o"
    resp = await route_query(model_to_use, sys_prompt, req.prompt)
    return {"status": "success", "title": resp.response}

@app.get("/api/chats/{uid}")
async def get_chats_api(uid: str):
    from db import get_user_chats
    chats = await get_user_chats(uid)
    return {"status": "success", "chats": chats}

@app.post("/api/chats/{uid}")
async def save_chat(uid: str, req: SaveChatRequest):
    from db import save_chat_session
    await save_chat_session(uid, req.session_id, req.title, req.messages)
    return {"status": "success"}

@app.get("/api/chats/shared/{session_id}")
async def get_shared_chat(session_id: str):
    from db import get_chat_by_id
    chat = await get_chat_by_id(session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
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
