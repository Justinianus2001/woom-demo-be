# Woom Audio Mixer API

**Build:**

```bash
docker build -t woom-mixer .
```

**Run:**

```bash
docker run -p 8000:8000 woom-mixer
```

---

## API Endpoints

### `GET /`

Health check endpoint. Returns status if the service is running.

### `GET /tracks`

List available background music tracks. Returns:

```json
{
  "tracks": ["track1.mp3", "track2.wav", ...]
}
```

### `POST /mix-all`

Streams mixed audio versions in real-time using NDJSON (newline-delimited JSON).

**Parameters:**

- `picked` (file, required) – Heartbeat audio file to be mixed
- `track_name` (string, required) – Name of background music track from `/tracks`

**Response:** Streaming NDJSON where each line is a completed version:

```json
{"version": "v1", "status": "done", "progress": "1/4", "data": "<base64-encoded-audio>"}
{"version": "v3", "status": "done", "progress": "2/4", "data": "<base64-encoded-audio>"}
...
```

Each line contains:

- `version` – v1, v2, v3, or v4
- `status` – "done" or "failed"
- `progress` – Current progress like "1/4", "2/4", etc.
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

## Tracks Directory

Place background music files in `./tracks/` directory. Supported formats:

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)

For Docker, mount it as a volume:

```bash
docker run -v $(pwd)/tracks:/app/tracks -p 8000:8000 woom-mixer
```

See `tracks/README.md` for details.
