from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(
    title="Todo API",
    description="Simple Todo API for Next.js frontend",
    version="1.0.0"
)

# Configure CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # Local development
        "https://*.vercel.app",            # All Vercel preview deployments
        "https://vercel.app",              # Vercel production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (will reset on Render restart)
todos_db = []
next_id = 1


class TodoCreate(BaseModel):
    task: str


@app.get("/")
async def root():
    return {
        "message": "✅ Todo API is running on Render!",
        "status": "healthy",
        "endpoints": {
            "GET /todos": "Get all todos",
            "POST /todos": "Create a new todo",
            "DELETE /todos/{id}": "Delete a todo",
            "GET /health": "Health check"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_todos": len(todos_db),
        "service": "render"
    }


@app.get("/todos")
async def get_todos():
    """Get all todos"""
    return {"todos": todos_db}


@app.post("/todos")
async def create_todo(todo: TodoCreate):
    """Create a new todo"""
    global next_id

    if not todo.task or not todo.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    new_todo = {
        "id": next_id,
        "task": todo.task.strip()
    }

    todos_db.append(new_todo)
    next_id += 1

    return {
        "todos": todos_db,
        "message": "Todo created successfully",
        "created": new_todo
    }


@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int):
    """Delete a todo by ID"""
    global todos_db

    initial_length = len(todos_db)
    todos_db = [todo for todo in todos_db if todo["id"] != todo_id]

    if len(todos_db) == initial_length:
        raise HTTPException(status_code=404, detail="Todo not found")

    return {
        "todos": todos_db,
        "message": "Todo deleted successfully"
    }

# For Render deployment
if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
