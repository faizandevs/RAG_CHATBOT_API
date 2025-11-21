from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Store messages in memory (for testing - will reset when server restarts)
messages_storage = {}

# ============================================================
# DATA MODELS
# ============================================================

class Message(BaseModel):
    id: int
    user_id: str
    text: str
    sender: str
    timestamp: int

class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    user_message: str
    bot_response: str
    timestamp: int

class MessagesResponse(BaseModel):
    messages: List[Message]

# ============================================================
# CHATBOT FUNCTION
# ============================================================

def get_bot_response(question: str) -> str:
    """Query Gemini API directly for response"""
    try:
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"You are an English learning assistant. Answer this question about English grammar and usage: {question}"
                        }
                    ]
                }
            ]
        }
        
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"]
        
        return "I couldn't generate a response. Please try again."
    
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Chatbot API running"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main endpoint:
    1. Save user message to memory
    2. Get bot response from Gemini
    3. Save bot message to memory
    4. Return both messages
    """
    try:
        timestamp = int(time.time() * 1000)
        
        # Initialize user storage if needed
        if request.user_id not in messages_storage:
            messages_storage[request.user_id] = []
        
        msg_id = len(messages_storage[request.user_id]) + 1
        
        # 1. Save user message
        user_msg = {
            "id": msg_id,
            "user_id": request.user_id,
            "text": request.question,
            "sender": "user",
            "timestamp": timestamp
        }
        messages_storage[request.user_id].append(user_msg)
        print(f"✅ User message saved: {request.question[:50]}...")
        
        # 2. Get bot response
        response_text = get_bot_response(request.question)
        
        # 3. Save bot message
        msg_id += 1
        bot_msg = {
            "id": msg_id,
            "user_id": request.user_id,
            "text": response_text,
            "sender": "bot",
            "timestamp": timestamp + 1
        }
        messages_storage[request.user_id].append(bot_msg)
        print(f"✅ Bot response saved")
        
        return ChatResponse(
            user_message=request.question,
            bot_response=response_text,
            timestamp=timestamp
        )
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/messages/{user_id}")
async def get_messages(user_id: str):
    """Retrieve all messages for a user"""
    try:
        if user_id not in messages_storage:
            messages_storage[user_id] = []
        
        messages = messages_storage[user_id]
        print(f"✅ Retrieved {len(messages)} messages for user {user_id}")
        
        return MessagesResponse(messages=messages)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/messages/{user_id}")
async def clear_messages(user_id: str):
    """Delete all messages for a user"""
    try:
        count = len(messages_storage.get(user_id, []))
        if user_id in messages_storage:
            del messages_storage[user_id]
        
        print(f"✅ Deleted {count} messages for user {user_id}")
        
        return {
            "status": "success",
            "message": f"Deleted {count} messages"
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 STARTING FASTAPI CHATBOT SERVER")
    print("="*80)
    print("📍 API running at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)