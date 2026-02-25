from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import tempfile
import zipfile
from processor import mix_audio_v1, mix_audio_v2, mix_audio_v3, mix_audio_v4

app = FastAPI(title="Woom Audio Mixer API")

@app.post("/mix-all")
async def mix_all(
    asset: UploadFile = File(...),
    picked: UploadFile = File(...)
):
    """
    Inputs two audio files (asset and picked) and returns a ZIP file with 4 mixed versions.
    """
    # Create a temporary directory to store files
    temp_dir = tempfile.mkdtemp()
    try:
        asset_path = os.path.join(temp_dir, asset.filename)
        picked_path = os.path.join(temp_dir, picked.filename)

        # Save uploaded files
        with open(asset_path, "wb") as buffer:
            shutil.copyfileobj(asset.file, buffer)
        with open(picked_path, "wb") as buffer:
            shutil.copyfileobj(picked.file, buffer)

        # Define output paths
        v1_out = os.path.join(temp_dir, "v1_mixed.mp3")
        v2_out = os.path.join(temp_dir, "v2_mixed.mp3")
        v3_out = os.path.join(temp_dir, "v3_mixed.mp3")
        v4_out = os.path.join(temp_dir, "v4_mixed.mp3")

        # Run the 4 mixing methods
        try:
            mix_audio_v1(asset_path, picked_path, v1_out)
        except Exception as e:
            print(f"Error in v1: {e}")
            
        try:
            mix_audio_v2(asset_path, picked_path, v2_out)
        except Exception as e:
            print(f"Error in v2: {e}")
            
        try:
            mix_audio_v3(asset_path, picked_path, v3_out)
        except Exception as e:
            print(f"Error in v3: {e}")
            
        try:
            mix_audio_v4(asset_path, picked_path, v4_out)
        except Exception as e:
            print(f"Error in v4: {e}")

        # Zip the results
        zip_path = os.path.join(temp_dir, "mixed_versions.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for out_file in [v1_out, v2_out, v3_out, v4_out]:
                if os.path.exists(out_file):
                    zipf.write(out_file, os.path.basename(out_file))

        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate any mixed files.")

        return FileResponse(zip_path, media_type="application/zip", filename="mixed_versions.zip")

    except Exception as e:
        # Cleanup on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Woom Audio Mixer API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
