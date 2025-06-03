import os
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

from fastapi import Path, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from digitaltwin.streamers import CameraStreamer, DirectoryStreamer
from digitaltwin.pydantic import QuestionRequest, QuestionResponse, ErrorResponse


BASE_IMAGE_DIR = "data/processed/Wildtrack_dataset/Image_subsets"
FRAME_INTERVAL = 1.0 / 20.0  # ~20 FPS

# What streams should we use
streams = [
    CameraStreamer(cam_id = 1),
    DirectoryStreamer(images_dir=os.path.join(BASE_IMAGE_DIR, "C1")),
    DirectoryStreamer(images_dir=os.path.join(BASE_IMAGE_DIR, "C2")),
    DirectoryStreamer(images_dir=os.path.join(BASE_IMAGE_DIR, "C3")),
]
assert len(streams) <= 4, "Error: for now only 4 streams - dont wanna do dynamic html shit"


# === Connect to SQLite DB ===
db = SQLDatabase.from_uri(
    "sqlite:///data/tracking.db",
    include_tables=["people_tracks", "positions"],
    sample_rows_in_table_info=2,
    # custom_table_info={
    #     "people_tracks": (
    #         "Table `people_tracks` tracks each unique person detected in the scene. "
    #         "Columns:\n"
    #         "- id: unique ID of the person\n"
    #         "- first_seen: timestamp of first detection\n"
    #         "- last_seen: timestamp of last detection"
    #     ),
    #     "positions": (
    #         "Table `positions` stores timestamped (x, y) coordinates of each person. "
    #         "Columns:\n"
    #         "- id: unique position ID\n"
    #         "- track_id: foreign key to people_tracks.id\n"
    #         "- timestamp: time of observation\n"
    #         "- x: x-coordinate\n"
    #         "- y: y-coordinate"
    #     )
    # }
)
print(f"Agent context: {db.table_info}")

# === Create LangChain SQL Agent ===
llm = ChatOpenAI(temperature=0)  # Requires OPENAI_API_KEY env variable
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# === FastAPI App ===
app = FastAPI(
    title="SQL Agent API",
    description="Ask natural language questions about your tracking database",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # could be "http://localhost:8000" idk
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/video/{camera_id}")
def video(camera_id: int = Path(..., ge=1, le=6)):
    try:
        return StreamingResponse(
            streams[camera_id - 1],
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


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

app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # reload=True
