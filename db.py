import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI or "<db_password>" in MONGO_URI:
    print("Warning: MONGODB_URI is misconfigured or missing real password.")

client = AsyncIOMotorClient(MONGO_URI)
db = client.parliament

users_collection = db.users
chats_collection = db.chats

async def get_user_data(uid: str):
    return await users_collection.find_one({"uid": uid})

async def create_or_update_user(uid: str, email: str, display_name: str, has_premium: bool = False):
    user = await users_collection.find_one({"uid": uid})
    if not user:
        new_user = {
            "uid": uid,
            "email": email,
            "display_name": display_name,
            "prompts_used": 0,
            "has_premium": has_premium,
            "role": "user"
        }
        await users_collection.insert_one(new_user)
        new_user["isNewUser"] = True
        return new_user
    user["isNewUser"] = False
    return user

async def inc_user_prompts(uid: str):
    await users_collection.update_one({"uid": uid}, {"$inc": {"prompts_used": 1}})

async def save_chat_session(uid: str, session_id: str, title: str, messages: list):
    await chats_collection.update_one(
        {"session_id": session_id},
        {"$set": {"uid": uid, "title": title, "messages": messages}},
        upsert=True
    )

async def get_user_chats(uid: str):
    cursor = chats_collection.find({"uid": uid}).sort("_id", -1)
    chats = await cursor.to_list(length=100)
    for c in chats:
        c["_id"] = str(c["_id"])
    return chats

async def get_chat_by_id(session_id: str):
    chat = await chats_collection.find_one({"session_id": session_id})
    if chat:
        chat["_id"] = str(chat["_id"])
    return chat

async def save_message_feedback(session_id: str, message_idx: int, feedback: str):
    field = f"messages.{message_idx}.feedback"
    await chats_collection.update_one(
        {"session_id": session_id},
        {"$set": {field: feedback}}
    )

async def delete_chat_session(uid: str, session_id: str):
    await chats_collection.delete_one({"uid": uid, "session_id": session_id})
