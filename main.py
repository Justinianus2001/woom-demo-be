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
import mimetypes
import urllib.parse
import urllib.request
import urllib.error
from typing import List
# v2/v3 remain in processor.py for rollback, but the API now exposes only the unified v1 pipeline.
from processor import mix_audio_v1, adjust_bpm, preprocess_shared

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

R2_PUBLIC_BASE_URL = os.getenv(
    "R2_PUBLIC_BASE_URL",
    "https://pub-be426cc47866401c9c6513aa344cb2f0.r2.dev",
).rstrip("/")

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


def sanitize_track_name(track_name: str) -> str:
    """Allow only plain filenames to avoid path traversal from user input."""
    safe_name = os.path.basename((track_name or "").strip())
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid track name")
    if safe_name != track_name.strip():
        raise HTTPException(status_code=400, detail="Invalid track name")
    return safe_name


def build_r2_track_url(track_name: str) -> str:
    safe_name = sanitize_track_name(track_name)
    encoded_name = urllib.parse.quote(safe_name)
    return f"{R2_PUBLIC_BASE_URL}/{encoded_name}"


def download_track_from_r2(track_name: str, temp_dir: str) -> str:
    """Download selected track from public R2 into temp directory for processing."""
    safe_name = sanitize_track_name(track_name)
    track_url = build_r2_track_url(safe_name)
    local_path = os.path.join(temp_dir, f"r2_{safe_name}")

    logger.info(f"Downloading track from R2: {track_url}")
    req = urllib.request.Request(track_url, headers={"User-Agent": "woom-mixer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp, open(local_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.warning(f"Track not found on R2: {safe_name}")
            raise HTTPException(status_code=404, detail=f"Track '{safe_name}' not found")
        logger.error(f"R2 HTTP error while downloading '{safe_name}': {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch track from R2")
    except urllib.error.URLError as e:
        logger.error(f"R2 URL error while downloading '{safe_name}': {e}")
        raise HTTPException(status_code=502, detail="Cannot connect to R2")

    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        raise HTTPException(status_code=502, detail="Downloaded track is empty")

    logger.info(f"Downloaded R2 track to temp path: {local_path}")
    return local_path

@app.get("/tracks/{track_name}")
def get_track(track_name: str):
    """Proxy and stream a specific track from Cloudflare R2 by name."""
    try:
        safe_name = sanitize_track_name(track_name)
        track_url = build_r2_track_url(safe_name)
        req = urllib.request.Request(track_url, headers={"User-Agent": "woom-mixer/1.0"})

        try:
            upstream = urllib.request.urlopen(req, timeout=45)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise HTTPException(status_code=404, detail="Track not found")
            logger.error(f"R2 HTTP error while proxying '{safe_name}': {e}")
            raise HTTPException(status_code=502, detail="Failed to fetch track from R2")
        except urllib.error.URLError as e:
            logger.error(f"R2 URL error while proxying '{safe_name}': {e}")
            raise HTTPException(status_code=502, detail="Cannot connect to R2")

        def stream_from_r2():
            try:
                with upstream as resp:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                logger.error(f"Error while streaming proxied track '{safe_name}': {e}")

        media_type = upstream.headers.get_content_type() or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        return StreamingResponse(
            stream_from_r2(),
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving track {track_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/tracks")
# def list_tracks():
#     """Deprecated: frontend now uses fixed track names and fetches files via /tracks/{name}."""
#     return {"tracks": []}

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
        # Download selected track from R2 into request temp directory.
        asset_path = download_track_from_r2(track_name, temp_dir)
        
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

    except HTTPException:
        raise
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
