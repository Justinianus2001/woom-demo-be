# Woom Audio Mixer API

**Build:**

```bash
docker build -t woom-mixer .
```

**Run:**

```bash
docker run -p 8000:8000 woom-mixer
docker run -d -p 8000:8000 woom-mixer (background)
```

---

## API Endpoints

### `GET /`

Health check endpoint. Returns status if the service is running.

### `GET /tracks`

List available tracks from Cloudflare R2. Returns:

```json
{
  "tracks": [
    {
      "track_name": "WooM-Twinkle_132.wav",
      "file_type": "trackbeat",
      "file_url": "https://...r2.dev/WooM-Twinkle_132.wav",
      "source": "r2"
    },
    {
      "track_name": "heartbeat_demo.wav",
      "file_type": "heartbeat",
      "file_url": "https://...r2.dev/heartbeat_demo.wav",
      "source": "r2"
    }
  ]
}
```

### `GET /tracks/audio/{track_name}`

Stream a single track by filename (frontend preview/player use this endpoint).

### `POST /mix-all`

Streams mixed audio versions in real-time using NDJSON (newline-delimited JSON).

**Parameters:**

- `picked` (file, required) – Heartbeat audio file to be mixed
- `track_name` (string, required) – Name of background music track from `/tracks`

### `POST /mix-file`

Mix two files already in Cloudflare R2 (no local heartbeat upload).

**Parameters:**

- `track_name` (string, required) – Background track (`file_type = trackbeat`)
- `heartbeat_name` (string, required) – Heartbeat track (`file_type = heartbeat`)

**Response:** Streaming NDJSON where each line is a completed version:

```json
{"version": "v1", "status": "progress", "progress": "1/7", "message": "Analyzing heartbeat tempo..."}
{"version": "v1", "status": "done", "progress": "7/7", "data": "<base64-encoded-audio>"}
```

Each line contains:

- `version` – `v1` (unified pipeline)
- `status` – `progress`, `done`, or `failed`
- `progress` – Current progress like `1/7`, `2/7`, etc.
- `data` (on success) – Base64-encoded audio file
- `error` (on failure) – Error message

### `POST /adjust-bpm`

Adjust playback speed of an audio file.

**Parameters:**

- `file` (file, required) – Audio file
- `speeds` (list, required) – One or more tempos: "Slow", "Normal", "Fast"

**Response:** ZIP file containing adjusted versions

---

## Progress Tracking

The `/mix-all` endpoint streams results as they complete, enabling:

- Real-time progress updates (1/4, 2/4, 3/4, 4/4)
- Immediate display of results without waiting for all versions
- Better user experience with transparent processing

---

## R2 Environment Variables

Set these in `.env` before running the backend:

- `R2_PUBLIC_BASE_URL`
- `R2_S3_BUCKET`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_ACCOUNT_ID`
- `R2_S3_ENDPOINT`
