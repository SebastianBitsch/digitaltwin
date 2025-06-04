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

from ultralytics import YOLO

from digitaltwin.streamers import CameraStreamer, DirectoryStreamer, VideoStreamer, YOLOStreamer, HeatmapStreamer
from digitaltwin.pydantic import QuestionRequest, QuestionResponse, ErrorResponse
from digitaltwin.utils import to_bytes
from digitaltwin.database.db_logger import DatabaseLogger
from digitaltwin.objects import Camera

CLEAR_DB = True
DB_PATH = "data/tracking.db"

if CLEAR_DB and os.path.exists(DB_PATH):
    print("Cleared DB")
    os.remove(DB_PATH)

BASE_IMAGE_DIR = "data/processed/Wildtrack_dataset/Image_subsets"
FRAME_INTERVAL = 1.0 / 20.0  # ~20 FPS

model = YOLO("models/yolo11m.pt")
logger = DatabaseLogger(DB_PATH)

# Live demo
streams = [
    YOLOStreamer(CameraStreamer(id=0, cam_id=4), model=model, 
        zones=[[(150,150), (1130,150), (1130, 570), (150, 570)]],
        camera=Camera.from_json("data/ps3eye.json")
    ),
    # YOLOStreamer(CameraStreamer(id=1, cam_id=0), model=model),
    HeatmapStreamer(id=1, db_path = DB_PATH, im_size=(640, 480)),
]

# warehouse
# BASE_IMAGE_DIR = "data/processed/warehouse"
# streams = [
#     YOLOStreamer(VideoStreamer(id=0, video_path=os.path.join(BASE_IMAGE_DIR, "Camera_0000.mp4")), model=model),
#     YOLOStreamer(VideoStreamer(id=1, video_path=os.path.join(BASE_IMAGE_DIR, "Camera_0001.mp4")), model=model),
#     YOLOStreamer(VideoStreamer(id=2, video_path=os.path.join(BASE_IMAGE_DIR, "Camera_0002.mp4")), model=model),
#     HeatmapStreamer(id=1, db_path = DB_PATH, im_size=(1920, 1280)),
# ]

# zurich
# BASE_IMAGE_DIR = "data/processed/EPFL-RLC_dataset/frames"
# streams = [
#     YOLOStreamer(DirectoryStreamer(id=0,images_dir=os.path.join(BASE_IMAGE_DIR, "cam0")), model=model),
#     YOLOStreamer(DirectoryStreamer(id=1,images_dir=os.path.join(BASE_IMAGE_DIR, "cam1")), model=model),
#     YOLOStreamer(DirectoryStreamer(id=2,images_dir=os.path.join(BASE_IMAGE_DIR, "cam2")), model=model),
#     HeatmapStreamer(id=3, db_path = DB_PATH, im_size = (480, 270)),
# ]



# Only some streamers want to log data
for s in streams:
    if getattr(s, "add_listener", None) is None:
        continue
    print("adding listener")
    s.add_listener(logger)


assert len(streams) <= 4, "Error: for now only 4 streams - dont wanna do dynamic html shit"


# === Connect to SQLite DB ===
db = SQLDatabase.from_uri(
    "sqlite:///data/tracking.db",
    include_tables=["people_tracks", "positions"],
    sample_rows_in_table_info=2,
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
def video(camera_id: int = Path(..., ge=0, le=4)):
    if len(streams) <= camera_id:
        return
    
    try:
        return StreamingResponse(
            to_bytes(streams[camera_id]),
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
