from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
import uvicorn

import cv2

# === Connect to SQLite DB ===
db = SQLDatabase.from_uri(
    "sqlite:///data/tracking.db"
)
# db.table_info = {
#     "people_tracks": "Tracks each unique person detected in the scene, with their entry and exit times.",
#     "positions": "Stores timestamped (x, y) positions of each person, including optional zone_id."
# }

# === Create LangChain SQL Agent ===
llm = ChatOpenAI(temperature=0)  # Requires OPENAI_API_KEY env variable
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# === Pydantic Models ===
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

class ErrorResponse(BaseModel):
    error: str

# === FastAPI App ===
app = FastAPI(
    title="SQL Agent API",
    description="Ask natural language questions about your tracking database",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def gen_frames():
    cap = cv2.VideoCapture(0)  # or use a file
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")


@app.get("/video/1")
def video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/ask", response_model=QuestionResponse, responses={400: {"model": ErrorResponse}})
async def ask(request: QuestionRequest):
    """
    Ask a natural language question about the tracking database.
    
    The agent can query information about:
    - people_tracks: Entry/exit times for detected people
    - positions: Timestamped positions and zones for each person
    """
    try:
        answer = agent.run(request.question)
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Removed reload=True
