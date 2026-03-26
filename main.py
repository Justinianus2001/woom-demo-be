from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import zipfile
import logging
import base64
import json
from typing import List
# v2/v3 remain in processor.py for rollback, but the API now exposes only the unified v1 pipeline.
from processor import mix_audio_v1, adjust_bpm, preprocess_shared

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Woom Audio Mixer API")

# Define the tracks directory
TRACKS_DIR = os.path.join(os.path.dirname(__file__), "tracks")
os.makedirs(TRACKS_DIR, exist_ok=True)

app = FastAPI(title="Woom Audio Mixer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_temp(temp_dir: str):
    """Safely remove the temporary directory."""
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.get("/tracks/{track_name}")
def get_track(track_name: str):
    """Serve a specific background music track file."""
    try:
        file_path = os.path.join(TRACKS_DIR, track_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Track not found")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error serving track {track_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tracks")
def list_tracks():
    """Return list of available background music tracks."""
    try:
        if not os.path.exists(TRACKS_DIR):
            return {"tracks": []}
        files = [f for f in os.listdir(TRACKS_DIR) if f.endswith(('.mp3', '.wav', '.m4a'))]
        return {"tracks": sorted(files)}
    except Exception as e:
        logger.error(f"Error listing tracks: {e}")
        return {"tracks": []}

@app.post("/mix-all")
def mix_all(
    background_tasks: BackgroundTasks,
    picked: UploadFile = File(...),
    track_name: str = Form(...)
):
    """
    Mix heartbeat (picked) with a pre-selected background music track.
    Streams results progressively as JSON Lines (one object per completed version).
    Each line contains: {version, status, progress, data (base64 audio)}.
    """
    logger.info(f"==========> [/mix-all] NEW REQUEST STARTED. Track: '{track_name}', UploadFile: '{picked.filename}' (content_type: {picked.content_type})")
    
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp, temp_dir)
    
    try:
        # Validate track exists
        asset_path = os.path.join(TRACKS_DIR, track_name)
        if not os.path.exists(asset_path):
            logger.error(f"Track not found: {track_name}")
            raise HTTPException(status_code=400, detail=f"Track '{track_name}' not found.")
        
        logger.info(f"Received mix-all request: picked_file='{picked.filename}', track_name='{track_name}'")
        
        # Save heartbeat file
        picked_filename = "".join([c for c in picked.filename if c.isalnum() or c in "._-"])
        picked_path = os.path.join(temp_dir, f"picked_{picked_filename}")
        logger.info(f"[/mix-all] Starting to write user uploaded file to local temp disk: {picked_path}")
        with open(picked_path, "wb") as buffer:
            shutil.copyfileobj(picked.file, buffer)
            
        file_size = os.path.getsize(picked_path)
        logger.info(f"[/mix-all] Finished uploading and saving user file to disk. Size: {file_size} bytes.")
        
        def generate_results():
            """Generator that yields step-by-step JSON lines for the unified v1 pipeline."""
            version_name = "v1"
            out_path = os.path.join(temp_dir, f"{version_name}_mixed.flac")
            total_steps = 7

            def emit(step: int, status: str, message: str, data: str = None, error: str = None):
                payload = {
                    "version": version_name,
                    "status": status,
                    "progress": f"{step}/{total_steps}",
                    "message": message,
                }
                if data is not None:
                    payload["data"] = data
                if error is not None:
                    payload["error"] = error
                return json.dumps(payload) + "\n"

            try:
                from processor import calculate_duration_from_analysis, detect_tempo

                logger.info("[mix-all] Step 1/7 - analyzing heartbeat tempo")
                yield emit(1, "progress", "Analyzing heartbeat tempo...")
                heart_duration, heart_tempo = calculate_duration_from_analysis(picked_path)

                logger.info("[mix-all] Step 2/7 - analyzing track tempo")
                yield emit(2, "progress", "Analyzing track tempo...")
                music_tempo = detect_tempo(asset_path)

                logger.info("[mix-all] Step 3/7 - shared preprocessing")
                yield emit(3, "progress", "Preprocessing shared audio assets...")
                shared_data = preprocess_shared(asset_path, picked_path, temp_dir)
                if not shared_data.get('success'):
                    logger.error("Shared preprocessing failed — using local fallback in mix_audio_v1")
                    shared_data = None

                logger.info("[mix-all] Step 4/7 - running unified v1 mix")
                yield emit(4, "progress", "Mixing heartbeat with track...")
                mix_audio_v1(
                    asset_path,
                    picked_path,
                    out_path,
                    heart_duration=heart_duration,
                    heart_tempo=heart_tempo,
                    music_tempo=music_tempo,
                    shared_data=shared_data,
                )

                logger.info("[mix-all] Step 5/7 - validating output file")
                yield emit(5, "progress", "Validating mixed output...")
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    logger.info("[mix-all] Step 6/7 - encoding output")
                    yield emit(6, "progress", "Encoding output audio...")
                    with open(out_path, "rb") as f:
                        audio_data = base64.b64encode(f.read()).decode('utf-8')
                    logger.info("[mix-all] Step 7/7 - done")
                    yield emit(7, "done", "Unified mix ready.", data=audio_data)
                    logger.info("Streamed unified v1 result (Progress: 7/7)")
                else:
                    logger.warning("Unified v1 output was not created or is empty.")
                    yield emit(7, "failed", "Mixing failed: output file was not created.", error="Output file not created")
            except Exception as e:
                import traceback
                logger.error(f"Error creating unified v1 output: {e}\n{traceback.format_exc()}")
                yield emit(7, "failed", "Mixing failed due to server error.", error=str(e))
        
        return StreamingResponse(generate_results(), media_type="application/x-ndjson")

    except Exception as e:
        import traceback
        logger.error(f"Global error in /mix-all: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adjust-bpm")
def adjust_bpm_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    speeds: List[str] = Form(...)
):
    """
    Accept a single audio file and one or more speed modes, then return a ZIP
    containing files adjusted by the requested tempo factors. The client will
    send the mix generated previously along with either `Slow`, `Normal` or
    `Fast` (or any custom factor) values.
    """
    logger.info(f"==========> [/adjust-bpm] NEW REQUEST STARTED. UploadFile: '{file.filename}', speeds: {speeds}")
    
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp, temp_dir)

    logger.info(f"[/adjust-bpm] Starting to write user input file to local temp disk...")

    # save incoming file
    input_path = os.path.join(temp_dir, "input_mix.mp3")
    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    file_size = os.path.getsize(input_path)
    logger.info(f"[/adjust-bpm] Finished uploading and saving user file. Saved input file to disk. Size: {file_size} bytes.")

    zip_path = os.path.join(temp_dir, "bpm_adjusted.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for speed in speeds:
            safe = "".join(c for c in speed if c.isalnum() or c in "._-")
            if not safe:
                continue
            out_name = f"{safe}.flac"
            out_path = os.path.join(temp_dir, out_name)
            try:
                adjust_bpm(input_path, out_path, speed)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    zipf.write(out_path, out_name)
                    logger.info(f"Created adjusted file for {speed}")
                else:
                    logger.warning(f"adjust_bpm produced no output for {speed}")
            except Exception as e:
                import traceback
                logger.error(f"Error adjusting bpm ({speed}): {e}\n{traceback.format_exc()}")
    return FileResponse(zip_path, media_type="application/zip", filename="bpm_adjusted.zip")


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Woom Audio Mixer API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
