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
from functools import lru_cache
from typing import List
# v2/v3 remain in processor.py for rollback, but the API now exposes only the unified v1 pipeline.
from processor import mix_audio_v1, adjust_bpm, preprocess_shared

try:
    import boto3
    from botocore.client import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # pragma: no cover - keep API running even if boto3 is absent.
    boto3 = None
    BotoConfig = None
    BotoCoreError = Exception
    ClientError = Exception

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_local_env_file() -> None:
    """Load a local .env file when running in Docker or from the repo root.

    This keeps the MVP runnable with `docker run -p 8000:8000 ...` while still
    allowing real environment variables to take precedence.
    """
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue

        try:
            loaded = 0
            with open(env_path, "r", encoding="utf-8") as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key:
                        continue

                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    os.environ.setdefault(key, value)
                    loaded += 1

            logger.info(f"Loaded {loaded} values from local env file: {env_path}")
            return
        except Exception as exc:
            logger.warning(f"Failed to load local env file {env_path}: {exc}")


load_local_env_file()

R2_PUBLIC_BASE_URL = os.getenv(
    "R2_PUBLIC_BASE_URL",
    "https://pub-be426cc47866401c9c6513aa344cb2f0.r2.dev",
).rstrip("/")
R2_S3_BUCKET = os.getenv("R2_S3_BUCKET", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
R2_S3_ENDPOINT = os.getenv("R2_S3_ENDPOINT", "").strip()
ALLOWED_TRACK_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}

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


def resolve_r2_s3_endpoint() -> str:
    if R2_S3_ENDPOINT:
        return R2_S3_ENDPOINT.rstrip("/")
    if R2_ACCOUNT_ID:
        return f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return ""


def is_r2_s3_ready() -> bool:
    endpoint = resolve_r2_s3_endpoint()
    return bool(
        boto3
        and endpoint
        and R2_S3_BUCKET
        and R2_ACCESS_KEY_ID
        and R2_SECRET_ACCESS_KEY
    )


@lru_cache(maxsize=1)
def get_r2_s3_client():
    if not is_r2_s3_ready():
        return None

    endpoint = resolve_r2_s3_endpoint()
    kwargs = {
        "service_name": "s3",
        "endpoint_url": endpoint,
        "aws_access_key_id": R2_ACCESS_KEY_ID,
        "aws_secret_access_key": R2_SECRET_ACCESS_KEY,
        "region_name": "auto",
    }
    if BotoConfig is not None:
        kwargs["config"] = BotoConfig(signature_version="s3v4")

    return boto3.client(**kwargs)


def guess_track_file_type(track_name: str) -> str:
    lower_name = (track_name or "").lower()
    if "heartbeat" in lower_name:
        return "heartbeat"
    return "trackbeat"


def list_tracks_from_r2() -> List[str]:
    """List supported audio files from Cloudflare R2 bucket (S3 API)."""
    if not is_r2_s3_ready():
        logger.warning("R2 S3 listing is not ready. Missing config or boto3 package.")
        return []

    s3_client = get_r2_s3_client()
    if s3_client is None:
        return []

    keys: List[str] = []
    continuation_token = None

    try:
        while True:
            params = {"Bucket": R2_S3_BUCKET, "MaxKeys": 1000}
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            response = s3_client.list_objects_v2(**params)
            for item in response.get("Contents", []) or []:
                raw_key = str(item.get("Key") or "").strip()
                key_name = os.path.basename(raw_key)
                if not key_name:
                    continue
                ext = os.path.splitext(key_name)[1].lower()
                if ext in ALLOWED_TRACK_EXTENSIONS:
                    keys.append(key_name)

            if not response.get("IsTruncated"):
                break

            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break
    except ClientError as e:
        logger.error(f"R2 S3 list_objects_v2 failed: {e}")
        return []
    except BotoCoreError as e:
        logger.error(f"R2 S3 list failed due to botocore error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error while listing R2 tracks: {e}")
        return []

    deduped = list(dict.fromkeys(keys))
    deduped.sort(key=lambda item: item.lower())
    return deduped


def download_track_from_r2(track_name: str, temp_dir: str) -> str:
    """Download selected track from public R2 into temp directory for processing."""
    safe_name = sanitize_track_name(track_name)
    local_path = os.path.join(temp_dir, f"r2_{safe_name}")

    # Prefer signed S3 API when credentials are provided. Fallback to public URL.
    if is_r2_s3_ready():
        s3_client = get_r2_s3_client()
        if s3_client is not None:
            logger.info(f"Downloading track from R2 S3 API: {safe_name}")
            try:
                s3_client.download_file(R2_S3_BUCKET, safe_name, local_path)
                if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                    raise HTTPException(status_code=502, detail="Downloaded track is empty")
                logger.info(f"Downloaded R2 track to temp path via S3: {local_path}")
                return local_path
            except ClientError as e:
                code = str((e.response or {}).get("Error", {}).get("Code") or "")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    raise HTTPException(status_code=404, detail=f"Track '{safe_name}' not found")
                logger.warning(f"R2 S3 download failed, fallback to public URL. error={e}")
            except BotoCoreError as e:
                logger.warning(f"R2 S3 download botocore error, fallback to public URL: {e}")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"R2 S3 download unexpected error, fallback to public URL: {e}")

    track_url = build_r2_track_url(safe_name)

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


def stream_track_from_r2(track_name: str, as_attachment: bool = False):
    """Stream track bytes from public R2 URL for preview/playback."""
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

    media_type = (
        upstream.headers.get_content_type()
        or mimetypes.guess_type(safe_name)[0]
        or "application/octet-stream"
    )
    headers = {"Content-Disposition": f'inline; filename="{safe_name}"'}
    if as_attachment:
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    return StreamingResponse(stream_from_r2(), media_type=media_type, headers=headers)


def generate_mix_results(asset_path: str, picked_path: str, temp_dir: str, endpoint_name: str):
    """Generator for unified v1 mix pipeline (shared by /mix-all and /mix-file)."""
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

        logger.info(f"[{endpoint_name}] Step 1/7 - analyzing heartbeat tempo")
        yield emit(1, "progress", "Analyzing heartbeat tempo...")
        heart_duration, heart_tempo = calculate_duration_from_analysis(picked_path)

        logger.info(f"[{endpoint_name}] Step 2/7 - analyzing track tempo")
        yield emit(2, "progress", "Analyzing track tempo...")
        music_tempo = detect_tempo(asset_path)

        logger.info(f"[{endpoint_name}] Step 3/7 - shared preprocessing")
        yield emit(3, "progress", "Preprocessing shared audio assets...")
        shared_data = preprocess_shared(asset_path, picked_path, temp_dir)
        if not shared_data.get("success"):
            logger.error(f"[{endpoint_name}] Shared preprocessing failed; fallback inside mix_audio_v1")
            shared_data = None

        logger.info(f"[{endpoint_name}] Step 4/7 - running unified v1 mix")
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

        logger.info(f"[{endpoint_name}] Step 5/7 - validating output")
        yield emit(5, "progress", "Validating mixed output...")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            logger.info(f"[{endpoint_name}] Step 6/7 - encoding output")
            yield emit(6, "progress", "Encoding output audio...")
            with open(out_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
            logger.info(f"[{endpoint_name}] Step 7/7 - done")
            yield emit(7, "done", "Unified mix ready.", data=audio_data)
            return

        logger.warning(f"[{endpoint_name}] Unified v1 output missing or empty")
        yield emit(7, "failed", "Mixing failed: output file was not created.", error="Output file not created")
    except Exception as e:
        import traceback

        logger.error(f"[{endpoint_name}] Error creating unified v1 output: {e}\n{traceback.format_exc()}")
        yield emit(7, "failed", "Mixing failed due to server error.", error=str(e))


@app.get("/tracks")
def list_tracks():
    """List available audio tracks from Cloudflare R2 bucket for frontend library."""
    tracks = list_tracks_from_r2()
    payload = [
        {
            "track_name": name,
            "file_type": guess_track_file_type(name),
            "file_url": build_r2_track_url(name),
            "source": "r2",
        }
        for name in tracks
    ]

    logger.info(f"Returning {len(payload)} tracks from R2 library")
    return {"tracks": payload}

@app.get("/tracks/{track_name}")
def get_track(track_name: str):
    """Proxy and stream a specific track from Cloudflare R2 by name."""
    try:
        return stream_track_from_r2(track_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving track {track_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tracks/audio/{track_name}")
def get_track_audio(track_name: str):
    """Alias endpoint for frontend preview with explicit '/audio' segment."""
    try:
        return stream_track_from_r2(track_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving track audio {track_name}: {e}")
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
        picked_name = picked.filename or "picked_audio.wav"
        picked_filename = "".join([c for c in picked_name if c.isalnum() or c in "._-"])
        if not picked_filename:
            picked_filename = "picked_audio.wav"
        picked_path = os.path.join(temp_dir, f"picked_{picked_filename}")
        logger.info(f"[/mix-all] Starting to write user uploaded file to local temp disk: {picked_path}")
        with open(picked_path, "wb") as buffer:
            shutil.copyfileobj(picked.file, buffer)
            
        file_size = os.path.getsize(picked_path)
        logger.info(f"[/mix-all] Finished uploading and saving user file to disk. Size: {file_size} bytes.")

        return StreamingResponse(
            generate_mix_results(asset_path, picked_path, temp_dir, endpoint_name="mix-all"),
            media_type="application/x-ndjson",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Global error in /mix-all: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mix-file")
def mix_file(
    background_tasks: BackgroundTasks,
    track_name: str = Form(...),
    heartbeat_name: str = Form(...),
):
    """Mix two existing files from R2: one trackbeat and one heartbeat from library."""
    logger.info(
        f"==========> [/mix-file] NEW REQUEST STARTED. track_name='{track_name}', heartbeat_name='{heartbeat_name}'"
    )

    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp, temp_dir)

    try:
        asset_path = download_track_from_r2(track_name, temp_dir)
        heartbeat_path = download_track_from_r2(heartbeat_name, temp_dir)

        logger.info(
            "[/mix-file] Downloaded both files from R2. asset=%s bytes, heartbeat=%s bytes",
            os.path.getsize(asset_path),
            os.path.getsize(heartbeat_path),
        )

        return StreamingResponse(
            generate_mix_results(asset_path, heartbeat_path, temp_dir, endpoint_name="mix-file"),
            media_type="application/x-ndjson",
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Global error in /mix-file: {e}\n{traceback.format_exc()}")
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
