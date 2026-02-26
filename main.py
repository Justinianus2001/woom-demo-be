from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import zipfile
import logging
from processor import mix_audio_v1, mix_audio_v2, mix_audio_v3, mix_audio_v4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.post("/mix-all")
def mix_all(
    background_tasks: BackgroundTasks,
    asset: UploadFile = File(...),
    picked: UploadFile = File(...)
):
    """
    Inputs two audio files (asset and picked) and returns a ZIP file with 4 mixed versions.
    """
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp, temp_dir)
    
    try:
        # Sanitize filenames for path joining
        asset_filename = "".join([c for c in asset.filename if c.isalnum() or c in "._-"])
        picked_filename = "".join([c for c in picked.filename if c.isalnum() or c in "._-"])
        
        asset_path = os.path.join(temp_dir, f"asset_{asset_filename}")
        picked_path = os.path.join(temp_dir, f"picked_{picked_filename}")

        # Save uploaded files
        with open(asset_path, "wb") as buffer:
            shutil.copyfileobj(asset.file, buffer)
        with open(picked_path, "wb") as buffer:
            shutil.copyfileobj(picked.file, buffer)

        # Define output paths
        outputs = {
            "v1_mixed.mp3": mix_audio_v1,
            "v2_mixed.mp3": mix_audio_v2,
            "v3_mixed.mp3": mix_audio_v3,
            "v4_mixed.mp3": mix_audio_v4
        }
        
        # Results storage
        zip_path = os.path.join(temp_dir, "mixed_versions.zip")
        any_success = False

        import concurrent.futures
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Using ThreadPoolExecutor with lower workers prevents Out-Of-Memory (OOM) 
            # and fork/OpenBLAS segment faults that abruptly kill processes.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                for filename, mix_func in outputs.items():
                    out_path = os.path.join(temp_dir, filename)
                    logger.info(f"Dispatching {filename}...")
                    futures[executor.submit(mix_func, asset_path, picked_path, out_path)] = filename
                
                for future in concurrent.futures.as_completed(futures):
                    filename = futures[future]
                    out_path = os.path.join(temp_dir, filename)
                    try:
                        future.result()
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                            zipf.write(out_path, filename)
                            any_success = True
                            logger.info(f"Successfully added {filename} to zip.")
                        else:
                            logger.warning(f"File {filename} was not created or is empty.")
                    except Exception as e:
                        logger.error(f"Error creating {filename}: {e}")

        if not any_success:
            raise HTTPException(status_code=500, detail="All mixing methods failed. Please check if FFmpeg is installed and files are valid.")

        return FileResponse(
            zip_path, 
            media_type="application/zip", 
            filename="mixed_versions.zip"
        )

    except Exception as e:
        logger.error(f"Global error in /mix-all: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Woom Audio Mixer API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
