# Background Music Tracks

Place your background music files in this directory. Supported formats:
- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)

Files added here will automatically be available for selection in the web UI via the `/tracks` endpoint.

Example:
```
tracks/
├── lullaby_1.mp3
├── lullaby_2.mp3
└── ambient_music.wav
```

**Note:** For Docker deployments, mount this directory as a volume to persist tracks between container restarts:
```bash
docker run -v /path/to/tracks:/app/tracks -p 8000:8000 woom-mixer
```
