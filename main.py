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
from processor import mix_audio_v1, mix_audio_v2, mix_audio_v3, mix_audio_v4, adjust_bpm, adjust_bpm

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
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp, temp_dir)
    
    try:
        # Validate track exists
        asset_path = os.path.join(TRACKS_DIR, track_name)
        if not os.path.exists(asset_path):
            raise HTTPException(status_code=400, detail=f"Track '{track_name}' not found.")
        
        # Save heartbeat file
        picked_filename = "".join([c for c in picked.filename if c.isalnum() or c in "._-"])
        picked_path = os.path.join(temp_dir, f"picked_{picked_filename}")
        with open(picked_path, "wb") as buffer:
            shutil.copyfileobj(picked.file, buffer)
        
        # Pre-calculate audio features to share across versions (saves roughly 15-20s total)
        from processor import calculate_duration_from_analysis, detect_tempo
        logger.info("Analyzing audio tracks...")
        heart_duration, heart_tempo = calculate_duration_from_analysis(picked_path)
        music_tempo = detect_tempo(asset_path)
        
        # Define mixing functions and implementations
        # Each version now receives the shared analysis results
        versions = [
            ("v1", mix_audio_v1, {"heart_duration": heart_duration}),
            ("v2", mix_audio_v2, {"heart_duration": heart_duration}),
            ("v3", mix_audio_v3, {"heart_duration": heart_duration, "heart_tempo": heart_tempo, "music_tempo": music_tempo}),
            ("v4", mix_audio_v4, {"heart_duration": heart_duration, "heart_tempo": heart_tempo, "music_tempo": music_tempo}),
        ]
        
        def generate_results():
            """Generator that yields JSON lines as each version completes."""
            import concurrent.futures
            import threading
            
            # Atomic counter to track finished tasks correctly
            finished_lock = threading.Lock()
            finished_count = 0

            # Increase max_workers to 4 to process all versions in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for version_name, mix_func, extra_args in versions:
                    out_path = os.path.join(temp_dir, f"{version_name}_mixed.flac")
                    logger.info(f"Submitting {version_name}...")
                    futures[executor.submit(mix_func, asset_path, picked_path, out_path, **extra_args)] = version_name
                
                for future in concurrent.futures.as_completed(futures):
                    version_name = futures[future]
                    with finished_lock:
                        finished_count += 1
                        current_progress = finished_count

                    try:
                        future.result()
                        out_path = os.path.join(temp_dir, f"{version_name}_mixed.flac")
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                            # Read and encode audio as base64
                            with open(out_path, "rb") as f:
                                audio_data = base64.b64encode(f.read()).decode('utf-8')
                            result = {
                                "version": version_name,
                                "status": "done",
                                "progress": f"{current_progress}/4",
                                "data": audio_data
                            }
                            yield json.dumps(result) + "\n"
                            logger.info(f"Streamed {version_name} (Progress: {current_progress}/4)")
                        else:
                            logger.warning(f"File {version_name} was not created or is empty.")
                            result = {
                                "version": version_name,
                                "status": "failed",
                                "progress": f"{current_progress}/4",
                                "error": "Output file not created"
                            }
                            yield json.dumps(result) + "\n"
                    except Exception as e:
                        logger.error(f"Error creating {version_name}: {e}")
                        result = {
                            "version": version_name,
                            "status": "failed",
                            "progress": f"{current_progress}/4",
                            "error": str(e)
                        }
                        yield json.dumps(result) + "\n"
        
        return StreamingResponse(generate_results(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Global error in /mix-all: {e}")
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
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp, temp_dir)

    # save incoming file
    input_path = os.path.join(temp_dir, "input_mix.mp3")
    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

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
                logger.error(f"Error adjusting bpm ({speed}): {e}")
    return FileResponse(zip_path, media_type="application/zip", filename="bpm_adjusted.zip")


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Woom Audio Mixer API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
