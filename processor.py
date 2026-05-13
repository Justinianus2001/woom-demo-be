import shutil
import subprocess
import shlex
import signal
import os
import tempfile
import librosa
from pydub import AudioSegment
import numpy as np
from scipy import signal
import soundfile as sf
import logging
import traceback
import math
import hashlib
import functools
from pathlib import Path

def _check_lfs_pointer(path: str) -> bool:
    """Check if the file is actually a Git LFS text pointer instead of real audio."""
    try:
        with open(path, 'rb') as f:
            header = f.read(30)
            if header.startswith(b"version https://git-lfs"):
                return True
    except Exception:
        pass
    return False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disk Cache for Preprocessed Tracks (LRU, 500MB limit)
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'tracks')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_MAX_SIZE_MB = 500
def _get_cache_key(file_path: str, context: str = "") -> str:
    """Generate a cache key based on file path and modification time.

    Args:
        file_path: Path to the audio file
        context: Optional context string to differentiate cache entries
                 (e.g., 'loudnorm', 'hb-stereo', 'hb-mono', 'stretch-1.200')
    """
    try:
        stat = os.stat(file_path)
        key_material = f"{os.path.basename(file_path)}_{stat.st_size}_{stat.st_mtime}"
        if context:
            key_material += f"_{context}"
        return hashlib.md5(key_material.encode()).hexdigest()
    except Exception:
        # Fallback to filename hash
        return hashlib.md5((os.path.basename(file_path) + context).encode()).hexdigest()

def _check_cache(cache_key: str) -> str:
    """Check if file exists in cache, return path if exists."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path
    return ""

def _save_to_cache(cache_key: str, source_path: str) -> str:
    """Save file to cache, run LRU eviction if needed."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")
    try:
        shutil.copy2(source_path, cache_path)
        _cleanup_cache()
        return cache_path
    except Exception as e:
        logger.warning(f"Failed to save to cache: {e}")
        return source_path

def _restore_cached_heartbeat_variants(picked_audio: str, stereo_path: str, mono_path: str) -> bool:
    """Restore cached heartbeat WAV variants when the same source file is reused."""
    if not os.path.exists(picked_audio):
        return False

    stereo_cache = _check_cache(_get_cache_key(picked_audio, "hb-stereo"))
    mono_cache = _check_cache(_get_cache_key(picked_audio, "hb-mono"))
    if not (stereo_cache and mono_cache):
        return False

    try:
        shutil.copy2(stereo_cache, stereo_path)
        shutil.copy2(mono_cache, mono_path)
    except Exception as e:
        logger.warning(f"[preprocess_shared] Failed to restore cached heartbeat variants: {e}")
        return False

    if _is_valid_decoded_audio(mono_path) and _is_valid_decoded_audio(stereo_path):
        logger.info("[preprocess_shared] Cache hit for heartbeat stereo & mono")
        return True

    return False

def _cache_heartbeat_variants(picked_audio: str, stereo_path: str, mono_path: str) -> None:
    if not os.path.exists(picked_audio):
        return

    _save_to_cache(_get_cache_key(picked_audio, "hb-stereo"), stereo_path)
    _save_to_cache(_get_cache_key(picked_audio, "hb-mono"), mono_path)
    logger.info("[preprocess_shared] Cached heartbeat stereo & mono")

def cleanup_old_cache(directory: str, max_size_mb: int = CACHE_MAX_SIZE_MB) -> None:
    """LRU eviction: remove oldest files when directory exceeds max_size_mb."""
    try:
        if not os.path.exists(directory):
            return
        cache_files = []
        for f in os.listdir(directory):
            fpath = os.path.join(directory, f)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                cache_files.append((fpath, stat.st_mtime, stat.st_size))
        
        if not cache_files:
            return
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        total_size = sum(f[2] for f in cache_files)
        max_bytes = max_size_mb * 1024 * 1024
        
        while total_size > max_bytes and cache_files:
            oldest_path, _, oldest_size = cache_files.pop(0)
            try:
                os.remove(oldest_path)
                total_size -= oldest_size
                logger.info(f"Evicted from cache: {os.path.basename(oldest_path)}")
            except Exception as e:
                logger.warning(f"Failed to evict cache file: {e}")
    except Exception as e:
        logger.warning(f"Cache cleanup failed: {e}")

def _cleanup_cache(max_size_mb: int = CACHE_MAX_SIZE_MB) -> None:
    """LRU eviction for the global CACHE_DIR."""
    cleanup_old_cache(CACHE_DIR, max_size_mb)



def _get_duration_ffprobe(path: str) -> float:
    """Lấy duration (giây) của file audio bằng ffprobe.

    Hoạt động với mọi format (WAV, FLAC, MP3...).
    FLAC thường trả về N/A cho format=duration → dùng stream=duration hoặc
    tính từ nb_samples / sample_rate (đọc từ FLAC STREAMINFO block).
    """

    def _run(args):
        r = subprocess.run(
            args, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        return (r.stdout or b'').decode().strip()

    # Strategy 1: format=duration (works for WAV/MP3, sometimes N/A for FLAC)
    try:
        val = _run([
            'ffprobe', '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path,
        ])
        if val and val.upper() != 'N/A':
            return float(val)
    except Exception:
        pass

    # Strategy 2: stream=duration — reliable for FLAC (reads STREAMINFO)
    try:
        val = _run([
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path,
        ])
        if val and val.upper() != 'N/A':
            return float(val)
    except Exception:
        pass

    # Strategy 3: compute from nb_samples / sample_rate (FLAC STREAMINFO fallback)
    try:
        raw = _run([
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=nb_samples,sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path,
        ])
        d = {}
        for line in raw.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                d[k.strip()] = v.strip()
        nb = d.get('nb_samples', 'N/A')
        sr = d.get('sample_rate', 'N/A')
        if nb not in ('N/A', '') and sr not in ('N/A', ''):
            return float(nb) / float(sr)
    except Exception as e:
        logger.warning(f"[ffprobe] all duration strategies failed for '{path}': {e}")

    return 0.0


def _librosa_load_safe(audio_path: str, duration: float = 30.0):
    """Load audio via librosa, falling back to a temp WAV conversion if the
    file format is not recognised by libsndfile/audioread."""
    if _check_lfs_pointer(audio_path):
        logger.error(f"❌ '{audio_path}' is a Git LFS pointer, not actual audio data! Run 'git lfs pull' on your server.")
        return np.array([]), 22050

    import tempfile as _tempfile
    # First, try direct load
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=duration)
        if len(y) > 0:
            return y, sr
    except Exception:
        pass  # fall through to ffmpeg conversion

    # Fallback: convert to standard PCM WAV via ffmpeg then load
    tmp = _tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        # Try with multiple demuxers
        converted = False
        strategies = [
            "-probesize 50M -analyzeduration 100M",
            "-f mp3",
            "-f mp4",
            "-f flac",
            "-f w64",
            "",
            "-f wav"
        ]
        for extra in strategies:
            cmd = f'ffmpeg -y {extra} -i "{audio_path}" -ar 44100 -ac 1 -sample_fmt s16 "{tmp_path}"'
            # collapse multiple spaces
            cmd = ' '.join(cmd.split())
            import subprocess as _sp, shlex as _shlex
            try:
                result = _sp.run(_shlex.split(cmd), stdin=_sp.DEVNULL,
                                 stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=ANALYSIS_FFMPEG_TIMEOUT_SECONDS)
                if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    converted = True
                    break
            except Exception:
                pass
        if converted:
            y, sr = librosa.load(tmp_path, sr=None, duration=duration)
            return y, sr
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return np.array([]), 22050


def calculate_duration_from_analysis(picked_audio, num_beats=4):
    """Phân tích file để lấy duration chính xác cho N nhịp tim."""
    try:
        y, sr = _librosa_load_safe(picked_audio, duration=HEARTBEAT_ANALYSIS_SECONDS)
        if len(y) == 0:
            return None, 120.0
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"): # Handle cases where tempo might be an array
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        if tempo <= 0: tempo = 120.0
        if len(beats) >= num_beats + 1:
            duration = librosa.frames_to_time(beats[num_beats] - beats[0], sr=sr)
            return duration, tempo
    except Exception as e:
        logger.error(f"❌ Phân tích thất bại: {e}\n{traceback.format_exc()}")
    return None, 120.0

def detect_tempo(audio_path):
    """Tự detect tempo của file audio dùng Librosa."""
    try:
        y, sr = _librosa_load_safe(audio_path, duration=TRACK_ANALYSIS_SECONDS)
        if len(y) == 0:
            return 120.0
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        if tempo <= 0: tempo = 120.0
        return tempo
    except Exception as e:
        logger.error(f"❌ Detect tempo thất bại: {e}\n{traceback.format_exc()}")
        return 120.0

FFMPEG_TIMEOUT = 60  # seconds – kill ffmpeg if it runs longer than this (optimized from 120s)
INTRO_DELAY_MS = 5000
INTRO_SECONDS = INTRO_DELAY_MS / 1000.0
FADE_IN_SECONDS = 0.0
FADE_OUT_SECONDS = 8.0
HEARTBEAT_SILENT_LEAD_SECONDS = 0.5
HEARTBEAT_VOLUME_RAMP_SECONDS = 1.5
HEARTBEAT_MIN_VALID_SECONDS = 0.1
MIN_REASONABLE_MIX_SECONDS = 8.0
MIN_DURATION_RATIO_VS_ASSET = 0.55
SILENT_DBFS_THRESHOLD = -70.0
MIN_PRECONVERT_ASSET_SECONDS = 1.0
HEARTBEAT_ANALYSIS_SECONDS = 16.0
TRACK_ANALYSIS_SECONDS = 24.0
ANALYSIS_FFMPEG_TIMEOUT_SECONDS = 18

HEARTBEAT_INPUT_STRATEGIES = [
    ("auto_large_probe", "-probesize 50M -analyzeduration 100M"),
    ("auto", ""),
    ("wav", "-f wav"),
    ("w64", "-f w64"),
    ("flac", "-f flac"),
    ("mp4", "-f mp4"),
    ("mp3", "-f mp3"),
]

MAX_BPM_STRETCH = 0.15
AMBIENT_TRACK_BPM_THRESHOLD = 60.0
AMBIENT_SYNC_DEVIATION_THRESHOLD = 0.34
BPM_SYNC_APPLY_EPS = 0.02
AMBIENT_HEARTBEAT_WEIGHT = 0.30
STANDARD_HEARTBEAT_WEIGHT = 0.55
MEET_IN_MIDDLE_MAX_STRETCH = 0.15
BIDIRECTIONAL_SYNC_RATIO_THRESHOLD = 0.08
AFFTDN_NF_MIN_DB = -80.0
AFFTDN_NF_MAX_DB = -20.0  # FFmpeg afftdn nf valid range is [-80, -20]
STANDARD_AFFTDN_NF_DB = -20.0
AMBIENT_AFFTDN_NF_DB = -24.0
HEARTBEAT_LOOP_CROSSFADE_MS = 180
HEARTBEAT_LOOP_TRIM_CHUNK_MS = 10
HEARTBEAT_LOOP_TRIM_PRE_ROLL_MS = 80
HEARTBEAT_LOOP_TRIM_TAIL_MS = 260
HEARTBEAT_LOOP_INTRO_SILENCE_MS = int(HEARTBEAT_SILENT_LEAD_SECONDS * 1000)
HEARTBEAT_LOOP_INTRO_RAMP_MS = int(HEARTBEAT_VOLUME_RAMP_SECONDS * 1000)


def _build_optimized_mix_filter(params: dict, quality_info: dict = None) -> str:
    """Build optimized FFmpeg filter chain combining mix + fade + 432Hz.

    Adaptive filter chain based on input quality:
    - Clean: wider bandpass (40-420Hz), lighter compression (ratio=1.3), milder denoise (nf=-16)
    - Moderate: balanced settings (50-380Hz, ratio=1.4, nf=-18)
    - Noisy: aggressive settings (60-350Hz, ratio=1.5, nf=-20) - same as original

    Only works when loop_bed_ready=True (finite length heartbeat bed).
    This merges:
    - Asset filter (adelay, equalizer, volume)
    - Picked filter (highpass, lowpass, bass, volume, acompressor, stereowiden, afftdn)
    - Mix (amix with weights and limiter)
    - Fade in (if fade_in_s > 0)
    - Fade out (if fade_out_s > 0, requires fade_out_start)
    - 432Hz tuning (asetrate + aresample + atempo)

    Args:
        params: dict with keys:
            - intro_delay_ms, volume_asset, volume_picked
            - fade_in_s, fade_out_start, fade_out_s
            - bpm_mode, use_loop_bed, heart_len_s, sr
            - lowpass_f (optional, default based on quality)
            - heart_ramp_end_s, heartbeat_intro_envelope (for aloop case)
        quality_info: dict from detect_input_quality() (optional)

    Returns:
        Filter chain string ready for -filter_complex
    """
    # Determine filter params based on quality
    quality = 'moderate'
    if quality_info:
        quality = quality_info.get('quality', 'moderate')

    if quality == 'clean':
        hp_freq = 40
        lp_freq = 420
        afftdn_nf = AFFTDN_NF_MAX_DB  # Now -20.0, valid for afftdn nf param
        comp_ratio = 1.3
    elif quality == 'noisy':
        hp_freq = 60
        lp_freq = params.get('lowpass_f', 350)
        afftdn_nf = STANDARD_AFFTDN_NF_DB
        comp_ratio = 1.5
    else:  # moderate
        hp_freq = 50
        lp_freq = params.get('lowpass_f', 380)
        afftdn_nf = -18
        comp_ratio = 1.4

    asset_weight = float(params.get('asset_weight', 0.45))
    heart_weight = float(params.get('heart_weight', 0.55))

    # Asset filter
    asset_f = (
        f"[0:a]"
        f"adelay={params['intro_delay_ms']}|{params['intro_delay_ms']},"
        f"equalizer=f=100:width_type=o:width=2:g=-5,"
        f"volume={safe_db(params['volume_asset'])}dB"
        f"[a0];"
    )

    # Picked filter
    if params.get('use_loop_bed'):
        picked_f = (
            f"[1:a]"
            f"highpass=f={hp_freq},lowpass=f={lp_freq},"
            f"bass=g=4:f=80,"
            f"volume={safe_db(params['volume_picked'])},"
            f"acompressor=threshold=-18dB:ratio={comp_ratio}:attack=8:release=100,"
            f"stereowiden=delay=5,"
            f"afftdn=nf={safe_afftdn_nf(afftdn_nf):.1f}"
            f"[a1];"
        )
    else:
        heart_ramp_end_s = params.get('heart_ramp_end_s', HEARTBEAT_SILENT_LEAD_SECONDS + HEARTBEAT_VOLUME_RAMP_SECONDS)
        heartbeat_intro_envelope = (
            f"if(lt(t,{HEARTBEAT_SILENT_LEAD_SECONDS:.2f}),0,"
            f"if(lt(t,{heart_ramp_end_s:.2f}),"
            f"(t-{HEARTBEAT_SILENT_LEAD_SECONDS:.2f})/{HEARTBEAT_VOLUME_RAMP_SECONDS:.2f},1))"
        )
        picked_f = (
            f"[1:a]"
            f"highpass=f={hp_freq},lowpass=f={lp_freq},"
            f"bass=g=4:f=80,"
            f"volume='{heartbeat_intro_envelope}':eval=frame,"
            f"volume={safe_db(params['volume_picked'])},"
            f"acompressor=threshold=-18dB:ratio={comp_ratio}:attack=8:release=100,"
            f"stereowiden=delay=5,"
            f"afftdn=nf={safe_afftdn_nf(afftdn_nf):.1f},"
            f"aloop=loop=-1:size={int(params.get('heart_len_s', 10) * params.get('sr', 44100))}"
            f"[a1];"
        )

    # Mix + Fade + 432Hz (when loop bed is ready)
    if params.get('use_loop_bed'):
        mix_chain = (
            f"{asset_f}{picked_f}"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=3"
            f":weights={asset_weight:.3f} {heart_weight:.3f},"
        )

        # Fade in
        fade_in_s = params.get('fade_in_s', 0)
        if fade_in_s > 0.01:
            mix_chain += f"afade=t=in:st=0:d={fade_in_s:.2f},"

        # Fade out
        fade_out_s = params.get('fade_out_s', 0)
        fade_out_start = params.get('fade_out_start', 0)
        if fade_out_s > 0.01:
            mix_chain += f"afade=t=out:st={fade_out_start:.2f}:d={fade_out_s:.2f},"

        # 432Hz tuning
        mix_chain += (
            f"asetrate=44100*432/440,"
            f"aresample=44100,"
            f"atempo=1.0185185185185186,"
        )

        # Limiter
        mix_chain += f"alimiter=limit=0.9[a]"
    else:
        # Keep separate for aloop case (can't fade/432Hz infinite stream)
        mix_chain = (
            f"{asset_f}{picked_f}"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=3"
            f":weights={asset_weight:.3f} {heart_weight:.3f},"
            f"alimiter=limit=0.9"
            f"[a]"
        )

    return mix_chain


def _clamp_tempo_rate(rate: float, max_stretch: float = MAX_BPM_STRETCH) -> float:
    return max(1.0 - max_stretch, min(1.0 + max_stretch, rate))


def _normalize_music_tempo_for_sync(music_tempo: float, heart_tempo: float) -> tuple[float, int]:
    """Normalize music tempo by octaves so comparison against heartbeat is meaningful."""
    normalized = float(music_tempo)
    shift = 0
    if normalized <= 0 or heart_tempo <= 0:
        return normalized, shift

    while normalized / heart_tempo > 1.415 and shift > -4:
        normalized /= 2.0
        shift -= 1
    while normalized / heart_tempo < 0.707 and shift < 4:
        normalized *= 2.0
        shift += 1
    return normalized, shift


def _plan_bpm_sync_adjustments(heart_tempo: float, music_tempo: float, max_stretch: float = MAX_BPM_STRETCH):
    """Pick a BPM sync plan with bidirectional meet-in-the-middle for large gaps.

    - Small gap (<=8%): light-sync -- only heartbeat stretched, max +/-15%
    - Large gap (>8%): meet-in-middle -- both sides move toward geometric midpoint,
      each capped at MEET_IN_MIDDLE_MAX_STRETCH (15%)
    - Ambient/low-BPM: no stretch, reduce heartbeat prominence instead
    """
    heart_tempo = float(heart_tempo or 120.0)
    music_tempo = float(music_tempo or 120.0)

    normalized_music_tempo, music_octave_shift = _normalize_music_tempo_for_sync(music_tempo, heart_tempo)
    exact_ratio = normalized_music_tempo / max(heart_tempo, 1e-9)
    ratio_deviation = abs(math.log(max(exact_ratio, 1e-9)))
    ambient_mode = normalized_music_tempo <= AMBIENT_TRACK_BPM_THRESHOLD or ratio_deviation >= AMBIENT_SYNC_DEVIATION_THRESHOLD

    base_plan = {
        "music_tempo": normalized_music_tempo,
        "music_octave_shift": music_octave_shift,
        "heart_rate": 1.0,
        "asset_rate": 1.0,
        "adjusted_heart_tempo": heart_tempo,
        "adjusted_music_tempo": music_tempo,
        "residual_ratio": exact_ratio,
        "exact_ratio": exact_ratio,
        "policy_mode": "ambient-texture" if ambient_mode else "light-sync",
        "heart_limit": 0.0 if ambient_mode else max_stretch,
        "asset_limit": 0.0,
        "heart_weight": AMBIENT_HEARTBEAT_WEIGHT if ambient_mode else STANDARD_HEARTBEAT_WEIGHT,
        "asset_weight": 1.0 - (AMBIENT_HEARTBEAT_WEIGHT if ambient_mode else STANDARD_HEARTBEAT_WEIGHT),
    }

    if heart_tempo <= 0 or music_tempo <= 0:
        return base_plan

    if ambient_mode:
        return {
            **base_plan,
            "policy_mode": "ambient-texture",
            "heart_rate": 1.0,
            "adjusted_heart_tempo": heart_tempo,
            "adjusted_music_tempo": normalized_music_tempo,
            "heart_limit": 0.0,
            "asset_limit": 0.0,
            "residual_ratio": exact_ratio,
            "heart_weight": AMBIENT_HEARTBEAT_WEIGHT,
            "asset_weight": 1.0 - AMBIENT_HEARTBEAT_WEIGHT,
        }

    ratio_gap = abs(1.0 - exact_ratio)

    # --- Bidirectional "meet in the middle" for large gaps ---
    if ratio_gap > BIDIRECTIONAL_SYNC_RATIO_THRESHOLD:
        target_ratio = math.sqrt(exact_ratio)

        heart_rate = _clamp_tempo_rate(target_ratio, MEET_IN_MIDDLE_MAX_STRETCH)
        asset_rate = _clamp_tempo_rate(1.0 / target_ratio, MEET_IN_MIDDLE_MAX_STRETCH)

        adjusted_heart = heart_tempo * heart_rate
        adjusted_music = music_tempo * asset_rate

        return {
            **base_plan,
            "heart_rate": heart_rate,
            "asset_rate": asset_rate,
            "adjusted_heart_tempo": adjusted_heart,
            "adjusted_music_tempo": adjusted_music,
            "residual_ratio": adjusted_music / max(adjusted_heart, 1e-9),
            "policy_mode": "meet-in-middle",
            "heart_limit": MEET_IN_MIDDLE_MAX_STRETCH,
            "asset_limit": MEET_IN_MIDDLE_MAX_STRETCH,
            "heart_weight": STANDARD_HEARTBEAT_WEIGHT,
            "asset_weight": 1.0 - STANDARD_HEARTBEAT_WEIGHT,
        }

    # --- Original light-sync for small gaps ---
    heart_rate = _clamp_tempo_rate(exact_ratio, max_stretch=max_stretch)
    return {
        **base_plan,
        "heart_rate": heart_rate,
        "adjusted_heart_tempo": heart_tempo * heart_rate,
        "residual_ratio": normalized_music_tempo / max(heart_tempo * heart_rate, 1e-9),
        "policy_mode": "light-sync",
        "heart_limit": max_stretch,
        "asset_limit": 0.0,
        "heart_weight": STANDARD_HEARTBEAT_WEIGHT,
        "asset_weight": 1.0 - STANDARD_HEARTBEAT_WEIGHT,
    }

def preconvert_asset(asset_audio: str, output_path: str) -> bool:
    """Try multiple FFmpeg strategies to convert an asset audio file to a
    standard 44100Hz stereo PCM WAV.
    
    The track files might use Wave64 (w64) format — a 64-bit extension of RIFF WAV
    that uses a GUID-based header. This explains the ffmpeg error:
        [wav] "invalid start code vers in RIFF header"

    We try strategies in order of likelihood:
    1. auto_large_probe → Increase probesize for huge ID3 tags
    2. mp3/mp4/flac → Extremely common mislabeled formats (M4A/MP3 -> .wav)
    3. w64      → Wave64 demuxer
    4. Auto     → let FFmpeg probe normally
    5. wav      → explicit WAV demuxer

    Raw PCM strategies (f32le, s16le, etc.) are intentionally EXCLUDED —
    they return rc=0 but produce silent/garbage audio because they misinterpret
    the headers as raw sample data.

    Returns True if any strategy succeeded and passed basic sanity checks.
    """
    if _check_lfs_pointer(asset_audio):
        logger.error(f"❌ '{asset_audio}' is a Git LFS pointer, not actual audio data! Run 'git lfs pull' on your server.")
        return False
    strategies = [
        # (label, extra input flags before -i)
        ("auto_large_probe", "-probesize 50M -analyzeduration 100M"),
        ("auto",  ""),
        ("wav",   "-f wav"),
        ("w64",   "-f w64"),
        ("flac",  "-f flac"),
        ("mp4",   "-f mp4"),
        ("mp3",   "-f mp3"),
    ]

    best_candidate = None
    best_score = -1.0

    for label, extra in strategies:
        candidate_path = f"{output_path}.{label}.wav"
        cmd = f'ffmpeg -y {extra} -i "{asset_audio}" -ar 44100 -ac 2 -sample_fmt s16 "{candidate_path}"'.strip()
        # Collapse double spaces that appear when extra == ""
        cmd = ' '.join(cmd.split())
        if run_ffmpeg(cmd):
            if not (os.path.exists(candidate_path) and os.path.getsize(candidate_path) > 0):
                logger.warning(f"[preconvert_asset] Strategy '{label}' produced empty file, skipping.")
                _try_unlink(candidate_path)
                continue
            # Validate: make sure the audio actually has signal (not all-zero silence)
            try:
                info = sf.info(candidate_path)
                decoded_duration = float(getattr(info, 'duration', 0.0) or 0.0)

                # Duration is the most reliable signal here. Some tracks start with
                # silence, so sampling only the first 8192 frames can be misleading.
                if decoded_duration >= MIN_PRECONVERT_ASSET_SECONDS:
                    if best_candidate and best_candidate != candidate_path:
                        _try_unlink(best_candidate)
                    os.replace(candidate_path, output_path)
                    logger.info(
                        f"[preconvert_asset] Success with strategy '{label}' "
                        f"(duration={decoded_duration:.1f}s)"
                    )
                    return True

                data, _ = sf.read(candidate_path, frames=8192)
                rms = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0.0
                if rms < 1e-6:
                    logger.warning(
                        f"[preconvert_asset] Strategy '{label}' produced silent audio "
                        f"(RMS={rms:.2e}), likely wrong format — skipping."
                    )
                    _try_unlink(candidate_path)
                    continue

                score = decoded_duration * max(rms, 1e-6)
                if score > best_score:
                    if best_candidate:
                        _try_unlink(best_candidate)
                    best_candidate = candidate_path
                    best_score = score
                else:
                    _try_unlink(candidate_path)

                logger.warning(
                    f"[preconvert_asset] Strategy '{label}' decoded too short "
                    f"({decoded_duration:.2f}s) — trying next strategy."
                )
            except Exception as val_err:
                logger.warning(f"[preconvert_asset] Validation failed for strategy '{label}': {val_err}")
                _try_unlink(candidate_path)
                continue
        else:
            _try_unlink(candidate_path)

    if best_candidate and os.path.exists(best_candidate):
        try:
            info = sf.info(best_candidate)
            decoded_duration = float(getattr(info, 'duration', 0.0) or 0.0)
            if decoded_duration >= MIN_PRECONVERT_ASSET_SECONDS:
                os.replace(best_candidate, output_path)
                logger.warning(
                    f"[preconvert_asset] Using best candidate after all strategies "
                    f"(duration={decoded_duration:.2f}s)."
                )
                return True
            logger.error(
                f"[preconvert_asset] Best decoded candidate still too short "
                f"({decoded_duration:.2f}s < {MIN_PRECONVERT_ASSET_SECONDS:.2f}s)."
            )
            _try_unlink(best_candidate)
            return False
        except Exception as e:
            logger.error(f"[preconvert_asset] Cannot finalize best candidate: {e}")
            _try_unlink(best_candidate)

    logger.error(f"[preconvert_asset] All strategies failed for '{asset_audio}'")
    return False


def _try_unlink(path: str) -> None:
    """Silently remove a file if it exists."""
    if os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _is_valid_decoded_audio(path: str, min_duration: float = HEARTBEAT_MIN_VALID_SECONDS) -> bool:
    """Return True if decoded audio exists, is non-empty, and libsndfile can read it."""
    if not (path and os.path.exists(path) and os.path.getsize(path) > 0):
        return False
    try:
        duration = float(getattr(sf.info(path), 'duration', 0.0) or 0.0)
    except Exception:
        return False
    return duration >= min_duration


def _ffmpeg_convert_heartbeat_variants(picked_audio: str, stereo_out: str, mono_out: str) -> bool:
    """Convert uploaded heartbeat to PCM WAV stereo+mono with robust demuxer fallbacks."""
    if _check_lfs_pointer(picked_audio):
        logger.error(
            f"❌ '{picked_audio}' is a Git LFS pointer, not actual audio data! Run 'git lfs pull' on your server."
        )
        return False

    _try_unlink(stereo_out)
    _try_unlink(mono_out)

    # Fast path: single ffmpeg process writes both outputs.
    if run_ffmpeg(
        f'ffmpeg -y -i "{picked_audio}" '
        f'-t 30 -ar 44100 -ac 2 -sample_fmt s16 "{stereo_out}" '
        f'-t 30 -ar 44100 -ac 1 -sample_fmt s16 "{mono_out}"'
    ):
        if _is_valid_decoded_audio(stereo_out) and _is_valid_decoded_audio(mono_out):
            return True
        logger.warning("[heartbeat_convert] Fast-path decode produced invalid outputs, trying fallbacks")

    _try_unlink(stereo_out)
    _try_unlink(mono_out)

    for label, extra in HEARTBEAT_INPUT_STRATEGIES:
        stereo_cmd = (
            f'ffmpeg -y {extra} -i "{picked_audio}" '
            f'-t 30 -ar 44100 -ac 2 -sample_fmt s16 "{stereo_out}"'
        )
        mono_cmd = (
            f'ffmpeg -y {extra} -i "{picked_audio}" '
            f'-t 30 -ar 44100 -ac 1 -sample_fmt s16 "{mono_out}"'
        )
        stereo_cmd = ' '.join(stereo_cmd.split())
        mono_cmd = ' '.join(mono_cmd.split())

        stereo_ok = run_ffmpeg(stereo_cmd)
        mono_ok = run_ffmpeg(mono_cmd)
        if stereo_ok and mono_ok and _is_valid_decoded_audio(stereo_out) and _is_valid_decoded_audio(mono_out):
            logger.info(f"[heartbeat_convert] Success with strategy '{label}'")
            return True

        _try_unlink(stereo_out)
        _try_unlink(mono_out)

    logger.error(f"[heartbeat_convert] All decode strategies failed for '{picked_audio}'")
    return False


def safe_db(value: float, fallback: float = 0.0, limit: float = 40.0) -> float:
    """Return a finite, non-extreme dB value safe to use in an FFmpeg filter.

    Protects against:
    - math.inf / float('inf')  => would produce 'volume=infdB' (invalid)
    - float('nan')             => would produce 'volume=nandB'  (invalid)
    - Values > +limit dB       => excessive gain, likely a measurement mistake
    """
    import math
    if value is None or math.isnan(value) or math.isinf(value):
        return fallback
    return max(-limit, min(limit, value))


def safe_afftdn_nf(value: float, fallback: float = STANDARD_AFFTDN_NF_DB) -> float:
    """Clamp afftdn `nf` to FFmpeg's valid range [-80, -20] dB."""
    if value is None or math.isnan(value) or math.isinf(value):
        value = fallback
    return max(AFFTDN_NF_MIN_DB, min(AFFTDN_NF_MAX_DB, value))

def safe_ffmpeg_load(path, timeout=FFMPEG_TIMEOUT):
    """Load audio file as AudioSegment by converting to WAV via our controlled
    run_ffmpeg() first, then using pydub's fast WAV path (no internal subprocess).
    This avoids pydub's internal Popen.communicate() which has no timeout."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        # WAV files can be loaded directly via the fast path
        return AudioSegment.from_file(path, format='wav')
    # Convert to WAV first using our controlled subprocess
    wav_path = path + '.safe_load.wav'
    try:
        if not run_ffmpeg(f'ffmpeg -y -i "{path}" -f wav "{wav_path}"', timeout=timeout):
            raise RuntimeError(f"FFmpeg conversion to WAV failed for {path}")
        return AudioSegment.from_file(wav_path, format='wav')
    finally:
        if os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass

def get_mean_volume(audio_path):
    """Đo mean volume (dBFS) dùng PyDub."""
    try:
        audio = safe_ffmpeg_load(audio_path)
        return audio.dBFS
    except Exception as e:
        logger.error(f"❌ Đo volume thất bại: {e}\n{traceback.format_exc()}")
        return -16.0


def fast_mean_volume(wav_path: str) -> float:
    """Đo mean volume (dBFS) trực tiếp bằng soundfile + numpy.

    KHÔNG spawn subprocess — nhanh hơn get_mean_volume() ~10-50x cho WAV files.
    Chỉ dùng cho file WAV chuẩn (PCM) đã được pre-convert.
    """
    try:
        data, _sr = sf.read(wav_path, dtype='float32')
        if len(data) == 0:
            return -16.0
        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms <= 0:
            return -96.0  # effective silence
        import math
        return 20.0 * math.log10(rms)
    except Exception as e:
        logger.error(f"❌ fast_mean_volume thất bại: {e}\n{traceback.format_exc()}")
        return -16.0


def quick_mean_volume(audio_path: str, max_seconds: float = 8.0) -> float:
    """Estimate dBFS from a short sample window to reduce CPU and I/O."""
    try:
        with sf.SoundFile(audio_path) as audio_file:
            total_frames = len(audio_file)
            if total_frames <= 0:
                return -96.0

            sample_frames = int(max(1, max_seconds * max(1, audio_file.samplerate)))
            sample_frames = min(sample_frames, total_frames)
            data = audio_file.read(sample_frames, dtype='float32')
            if len(data) == 0:
                return -96.0

            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms <= 0:
                return -96.0
            return 20.0 * math.log10(rms)
    except Exception as e:
        logger.warning(f"[quick_mean_volume] fallback to full read for '{audio_path}': {e}")
        return fast_mean_volume(audio_path)


def evaluate_mixed_output(mix_path: str, expected_asset_duration: float = 0.0):
    """Validate mixed audio duration and loudness to catch silent/short regressions.

    Returns:
        tuple(bool, str, float, float):
            (is_healthy, reason, measured_duration_s, measured_dbfs)
    """
    if not os.path.exists(mix_path) or os.path.getsize(mix_path) == 0:
        return False, "missing-or-empty", 0.0, -120.0

    measured_duration = 0.0
    try:
        measured_duration = float(sf.info(mix_path).duration)
    except Exception as info_err:
        logger.warning(f"[mix] Cannot read mix duration via soundfile: {info_err}")

    measured_dbfs = quick_mean_volume(mix_path)

    if measured_duration > 0 and expected_asset_duration and expected_asset_duration > 0:
        min_expected = max(
            MIN_REASONABLE_MIX_SECONDS,
            expected_asset_duration * MIN_DURATION_RATIO_VS_ASSET,
        )
        if measured_duration < min_expected:
            return False, f"too-short:{measured_duration:.2f}s<{min_expected:.2f}s", measured_duration, measured_dbfs

    if measured_dbfs <= SILENT_DBFS_THRESHOLD:
        return False, f"too-silent:{measured_dbfs:.2f}dBFS", measured_duration, measured_dbfs

    return True, "ok", measured_duration, measured_dbfs


def _trim_heartbeat_loop_source(
    source: AudioSegment,
    chunk_ms: int = HEARTBEAT_LOOP_TRIM_CHUNK_MS,
    pre_roll_ms: int = HEARTBEAT_LOOP_TRIM_PRE_ROLL_MS,
    tail_ms: int = HEARTBEAT_LOOP_TRIM_TAIL_MS,
) -> AudioSegment:
    """Trim leading/trailing silence so loop joins happen near real heartbeat events."""
    if len(source) <= 0:
        return source

    chunk_ms = max(5, int(chunk_ms))
    pre_roll_ms = max(0, int(pre_roll_ms))
    tail_ms = max(0, int(tail_ms))

    chunks = []
    for start in range(0, len(source), chunk_ms):
        chunk = source[start:start + chunk_ms]
        if len(chunk) > 0:
            chunks.append((start, chunk.rms))

    if not chunks:
        return source

    rms_values = [rms for _, rms in chunks]
    max_rms = max(rms_values)
    if max_rms <= 0:
        return source

    sorted_rms = sorted(rms_values)
    noise_floor = sorted_rms[max(0, int(len(sorted_rms) * 0.60) - 1)]
    threshold = max(1, int(max_rms * 0.08), int(noise_floor * 3.0))
    active = [start for start, rms in chunks if rms >= threshold]
    if not active:
        return source

    trim_start = max(0, active[0] - pre_roll_ms)
    trim_end = min(len(source), active[-1] + chunk_ms + tail_ms)
    trimmed_len = trim_end - trim_start
    min_trimmed_len = max(HEARTBEAT_LOOP_CROSSFADE_MS * 2, 250)

    if trimmed_len <= min_trimmed_len:
        logger.info(
            f"[mix] Loop trim skipped: source too short after trim ({trimmed_len}ms <= {min_trimmed_len}ms)"
        )
        return source

    if trim_start == 0 and trim_end == len(source):
        return source

    logger.info(
        f"[mix] Loop source trimmed: {len(source)}ms -> {trimmed_len}ms "
        f"(start={trim_start}ms, end={trim_end}ms, rms_threshold={threshold})"
    )
    return source[trim_start:trim_end]


def _build_looped_heartbeat_bed(
    source_path: str,
    output_path: str,
    target_duration_s: float,
    crossfade_ms: int = HEARTBEAT_LOOP_CROSSFADE_MS,
    intro_silence_ms: int = HEARTBEAT_LOOP_INTRO_SILENCE_MS,
    intro_ramp_ms: int = HEARTBEAT_LOOP_INTRO_RAMP_MS,
) -> bool:
    """Render a finite heartbeat bed with one-time intro and crossfaded loop joins."""
    try:
        source = AudioSegment.from_file(source_path, format='wav')
    except Exception as e:
        logger.warning(f"[mix] Cannot load loop bed source '{source_path}': {e}")
        return False

    if len(source) == 0 or target_duration_s <= 0:
        return False

    crossfade_ms = max(0, min(int(crossfade_ms), len(source) // 4, 250))
    intro_silence_ms = max(0, int(intro_silence_ms))
    intro_ramp_ms = max(0, int(intro_ramp_ms))

    source = _trim_heartbeat_loop_source(source)
    crossfade_ms = max(0, min(int(crossfade_ms), len(source) // 4, 250))

    bed = AudioSegment.silent(duration=intro_silence_ms)
    first_segment = source.fade_in(intro_ramp_ms) if intro_ramp_ms > 0 else source
    bed += first_segment

    target_ms = max(int(target_duration_s * 1000.0), len(bed))
    remaining_ms = target_ms - len(bed)
    if remaining_ms > 0:
        loop_block = source
        target_block_ms = remaining_ms + crossfade_ms
        while len(loop_block) < target_block_ms:
            if crossfade_ms > 0 and len(loop_block) > crossfade_ms:
                loop_block = loop_block.append(loop_block, crossfade=crossfade_ms)
            else:
                loop_block += loop_block

        if crossfade_ms > 0 and len(loop_block) > crossfade_ms:
            bed = bed.append(loop_block, crossfade=crossfade_ms)
        else:
            bed += loop_block

    if len(bed) > target_ms + crossfade_ms:
        bed = bed[:target_ms]

    try:
        bed.export(output_path, format='wav')
    except Exception as e:
        logger.warning(f"[mix] Failed to export loop bed '{output_path}': {e}")
        return False

    return _is_valid_decoded_audio(output_path)


def preprocess_shared(asset_audio: str, picked_audio: str, work_dir: str):
    """Tiền xử lý chung cho pipeline v1 — chỉ chạy MỘT LẦN.

    Thực hiện:
    1. Pre-convert asset audio → PCM WAV chuẩn (handles Wave64/RF64/float/m4a/mp3...)
    2. Loudnorm asset → -16 LUFS
    3. Convert picked (heartbeat) audio → PCM WAV 44100Hz mono+stereo
       (mono cho HPSS, stereo cho v1 low-pass)
    4. Đo volume asset & picked (bằng numpy, KHÔNG subprocess)

    Returns:
        dict with keys:
        - 'normalized_asset_path': str – asset WAV đã loudnorm
        - 'picked_wav_stereo': str – heartbeat WAV 44100Hz stereo (cho v1)
        - 'picked_wav_mono': str – heartbeat WAV 44100Hz mono (cho HPSS)
        - 'asset_volume': float – mean volume dBFS
        - 'error': str | None – machine-readable preprocessing error code
        - 'success': bool

    Raises RuntimeError if asset cannot be decoded.
    """
    logger.info(f"[preprocess_shared] Starting shared preprocessing...")

    raw_asset_path = os.path.join(work_dir, 'shared_asset_raw.wav')
    normalized_asset_path = os.path.join(work_dir, 'shared_asset_normalized.wav')
    picked_wav_stereo = os.path.join(work_dir, 'shared_picked_stereo.wav')
    picked_wav_mono = os.path.join(work_dir, 'shared_picked_mono.wav')

    # 0) Check cache for normalized asset (skip preconvert + loudnorm if hit)
    if os.path.exists(asset_audio):
        asset_cache_key = _get_cache_key(asset_audio, "loudnorm")
        cached_asset = _check_cache(asset_cache_key)
        if cached_asset:
            logger.info(f"[preprocess_shared] Cache hit for normalized asset: {os.path.basename(cached_asset)}")
            shutil.copy2(cached_asset, normalized_asset_path)
            asset_volume = fast_mean_volume(normalized_asset_path)
            heartbeat_ready = _restore_cached_heartbeat_variants(
                picked_audio,
                picked_wav_stereo,
                picked_wav_mono,
            )
            if not heartbeat_ready and not _ffmpeg_convert_heartbeat_variants(picked_audio, picked_wav_stereo, picked_wav_mono):
                logger.error(f"[preprocess_shared] Cannot decode heartbeat upload '{picked_audio}'")
                return {'success': False, 'error': 'heartbeat-decode-failed'}
            if not heartbeat_ready:
                _cache_heartbeat_variants(picked_audio, picked_wav_stereo, picked_wav_mono)
            logger.info(f"[preprocess_shared] Done (from cache). asset_vol={asset_volume:.1f}dB")
            return {
                'success': True,
                'normalized_asset_path': normalized_asset_path,
                'picked_wav_stereo': picked_wav_stereo,
                'picked_wav_mono': picked_wav_mono,
                'asset_volume': asset_volume,
            }

    # 1) Pre-convert asset (worst case tries 7 strategies — but only ONCE)
    if not preconvert_asset(asset_audio, raw_asset_path):
        logger.error(f"[preprocess_shared] Cannot decode asset audio '{asset_audio}'")
        return {'success': False, 'error': 'asset-decode-failed'}

    # 2) Loudnorm asset → chuẩn -16 LUFS
    if not run_ffmpeg(
        f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 '
        f'-af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"'
    ):
        logger.error("[preprocess_shared] Asset loudnorm failed")
        return {'success': False, 'error': 'asset-loudnorm-failed'}
    _try_unlink(raw_asset_path)  # free disk space early

    # Save to cache
    if os.path.exists(asset_audio):
        asset_cache_key = _get_cache_key(asset_audio, "loudnorm")
        _save_to_cache(asset_cache_key, normalized_asset_path)
        logger.info(f"[preprocess_shared] Saved normalized asset to cache")

    # 3) Convert picked → WAV stereo và mono, có fallback demuxer.
    heartbeat_ready = _restore_cached_heartbeat_variants(
        picked_audio,
        picked_wav_stereo,
        picked_wav_mono,
    )
    if not heartbeat_ready and not _ffmpeg_convert_heartbeat_variants(picked_audio, picked_wav_stereo, picked_wav_mono):
        logger.error(f"[preprocess_shared] Cannot decode heartbeat upload '{picked_audio}'")
        return {'success': False, 'error': 'heartbeat-decode-failed'}

    if not heartbeat_ready:
        _cache_heartbeat_variants(picked_audio, picked_wav_stereo, picked_wav_mono)

    # 4) Đo volume asset bằng numpy (0 subprocess)
    asset_volume = fast_mean_volume(normalized_asset_path)

    logger.info(
        f"[preprocess_shared] Done. asset_vol={asset_volume:.1f}dB"
    )

    return {
        'success': True,
        'normalized_asset_path': normalized_asset_path,
        'picked_wav_stereo': picked_wav_stereo,
        'picked_wav_mono': picked_wav_mono,
        'asset_volume': asset_volume,
    }


def run_ffmpeg(command, timeout=FFMPEG_TIMEOUT):
    """Chạy FFmpeg command với Popen và proper timeout.
    
    KHÔNG dùng shell=True (tránh orphan process).
    KHÔNG dùng start_new_session=True (gây deadlock khi fork trong multi-thread).
    stdin=DEVNULL để tránh ffmpeg chờ input.
    """
    logger.info(f"Running ffmpeg command: {command}")
    cmd_list = shlex.split(command)
    # Thêm -nostdin nếu là ffmpeg command (chặn ffmpeg đọc stdin hoàn toàn)
    if cmd_list and cmd_list[0].endswith('ffmpeg') and '-nostdin' not in cmd_list:
        cmd_list.insert(1, '-nostdin')
    process = None
    try:
        logger.info(f"[run_ffmpeg] Spawning process...")
        process = subprocess.Popen(
            cmd_list,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"[run_ffmpeg] Process spawned (pid={process.pid}), waiting for completion...")
        _stdout, stderr = process.communicate(timeout=timeout)
        success = process.returncode == 0
        if not success:
            logger.error(
                f"❌ FFmpeg failed (code {process.returncode}): "
                f"{stderr.decode(errors='replace')}\nCommand: {command}"
            )
        logger.info(f"[run_ffmpeg] Process completed (pid={process.pid}, rc={process.returncode})")
        return success
    except subprocess.TimeoutExpired:
        logger.error(f"❌ FFmpeg TIMEOUT after {timeout}s – killing.\nCommand: {command}")
        if process is not None:
            process.kill()
            process.wait()
        return False
    except Exception as e:
        logger.error(f"❌ Exception running FFmpeg: {e}\n{traceback.format_exc()}\nCommand: {command}")
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        return False


def adjust_bpm(input_path, output_path, speed_mode):
    """Adjust playback speed of an audio file using FFmpeg's atempo filter.

    The `speed_mode` may be one of the named presets or a numeric factor (as
    a string). Presets are:
        * "Slow"  -> 0.8
        * "Normal" -> 1.0
        * "Fast"  -> 1.2

    Any other value will be parsed as a float and clipped to a sane range.
    """
    speed_map = {
        'Slow': 0.8,
        'Normal': 1.0,
        'Fast': 1.2,
    }

    # resolve factor
    try:
        speed = speed_map.get(speed_mode, float(speed_mode))
    except Exception:
        speed = 1.0

    # clamp to avoid crazy atempo values (FFmpeg allows 0.5-2.0 per filter, but
    # chaining is expensive; we allow a wider overall range here and let
    # FFmpeg decide internally.)
    if speed <= 0 or speed is None or isinstance(speed, complex):
        speed = 1.0
    speed = max(0.5, min(100.0, speed))

    logger.info(f"Adjusting BPM: Mode='{speed_mode}', Factor={speed}, Output={output_path}")
    # select codec based on output extension
    codec = ''
    if output_path.lower().endswith('.flac'):
        codec = ' -c:a flac'
    elif output_path.lower().endswith('.mp3'):
        codec = ' -c:a libmp3lame -q:a 2'

    atempo_str = get_atempo_filter(speed)
    cmd = f'ffmpeg -y -i "{input_path}" -af "{atempo_str}"{codec} "{output_path}"'
    if not run_ffmpeg(cmd):
        # copy through if atempo fails
        run_ffmpeg(f'ffmpeg -y -i "{input_path}"{codec} "{output_path}"')

def apply_noise_reduction(y, sr, denoise_level='auto', quality_info=None):
    """Adaptive HPSS: blend harmonic + percussive based on file quality.

    Args:
        y: numpy array audio
        sr: sample rate
        denoise_level: 'auto'|'none'|'mild'|'aggressive'
        quality_info: dict từ detect_input_quality() (optional)

    Returns:
        numpy array đã xử lý
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    if denoise_level == 'none':
        return y
    elif denoise_level == 'mild':
        return 0.7 * y_percussive + 0.3 * y_harmonic
    elif denoise_level == 'aggressive':
        return y_percussive

    # Auto mode - detect quality
    if quality_info is None:
        quality_info = detect_input_quality(y, sr)

    quality = quality_info['quality']

    if quality == 'clean':
        # Blend 60% percussive (attack) + 40% harmonic (warmth)
        logger.info("[HPSS] Clean file → blend 60% perc + 40% harm")
        return 0.6 * y_percussive + 0.4 * y_harmonic
    elif quality == 'moderate':
        logger.info("[HPSS] Moderate file → blend 75% perc + 25% harm")
        return 0.75 * y_percussive + 0.25 * y_harmonic
    else:  # noisy
        logger.info("[HPSS] Noisy file → percussive only (aggressive)")
        return y_percussive


def detect_input_quality(y: np.ndarray, sr: int) -> dict:
    """Phát hiện chất lượng file nhịp tim: 'clean', 'noisy', hoặc 'moderate'.

    Dựa trên HPSS harmonic/percussive ratio + SNR ước lượng:
    - harmonic_ratio < 0.6 và snr_db > 5 → clean (ít tạp âm tone)
    - harmonic_ratio > 1.2 hoặc snr_db < -5 → noisy (nhiều tạp âm)
    - Trường hợp khác → moderate

    Returns:
        dict với keys: 'quality', 'snr_db', 'harmonic_ratio', 'energy_variance'
    """
    if len(y) == 0:
        return {
            'quality': 'moderate',
            'snr_db': 0.0,
            'energy_variance': 1.0,
            'harmonic_ratio': 1.0
        }

    # HPSS decomposition
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Tính energy
    harm_energy = float(np.mean(y_harmonic ** 2))
    perc_energy = float(np.mean(y_percussive ** 2))

    # Harmonic ratio (cao = nhiều tiếng người/tone = noisy)
    harmonic_ratio = harm_energy / (perc_energy + 1e-10)

    # SNR ước lượng (dùng percussive làm signal, harmonic làm noise)
    if harm_energy > 0:
        snr_db = 10.0 * np.log10(perc_energy / (harm_energy + 1e-10))
    else:
        snr_db = 30.0  # Rất sạch

    # Energy variance (độ ổn định)
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    energy_var = float(np.std(rms) / (np.mean(rms) + 1e-10))

    # Phân loại
    if harmonic_ratio < 0.6 and snr_db > 5:
        quality = 'clean'
    elif harmonic_ratio > 1.2 or snr_db < -5:
        quality = 'noisy'
    else:
        quality = 'moderate'

    logger.info(f"[quality] Detected: {quality}, SNR={snr_db:.1f}dB, "
                f"harmonic_ratio={harmonic_ratio:.2f}, energy_var={energy_var:.2f}")

    return {
        'quality': quality,
        'snr_db': float(snr_db),
        'energy_variance': energy_var,
        'harmonic_ratio': float(harmonic_ratio)
    }


def codec_args(output_path: str):
    """Return codec arguments for ffmpeg based on file extension."""
    if output_path.lower().endswith('.flac'):
        # use flac codec with decent compression
        return '-c:a flac -compression_level 8'
    else:
        return '-c:a libmp3lame -q:a 2'

def tune_to_432hz(input_path, output_path):
    """Pitch shift toàn bộ audio xuống 432Hz tuning từ 440Hz dùng FFmpeg."""
    # asetrate changes pitch and speed, atempo corrects the speed back.
    # 432/440 = 0.981818... and 440/432 = 1.018518...
    # Pre-calculate asetrate value to avoid potential issues with expressions in some ffmpeg builds
    asetrate_val = int(44100 * 432 / 440)  # ~43209
    cmd = (
        f'ffmpeg -y -i "{input_path}" '
        f'-af "asetrate={asetrate_val},aresample=44100,atempo=1.0185185185185186" '
        f'{codec_args(output_path)} "{output_path}"'
    )
    logger.info(f"[tune_to_432hz] input={input_path}, output={output_path}, asetrate={asetrate_val}")
    result = run_ffmpeg(cmd)
    if result:
        logger.info(f"[tune_to_432hz] Success: {output_path} (size={os.path.getsize(output_path) if os.path.exists(output_path) else 'N/A'})")
    else:
        logger.error(f"[tune_to_432hz] Failed to create: {output_path}")
    return result

def get_atempo_filter(rate):
    """Helper to generate atempo filter string, chaining if rate is outside [0.5, 100]."""
    if rate <= 0: return "atempo=1.0"
    filters = []
    while rate < 0.5:
        filters.append("atempo=0.5")
        rate /= 0.5
    while rate > 100.0:
        filters.append("atempo=100.0")
        rate /= 100.0
    filters.append(f"atempo={rate}")
    return ",".join(filters)

def time_stretch_heartbeat(input_path, output_path, target_tempo, original_tempo):
    """Stretch nhịp tim dùng FFmpeg atempo."""
    if original_tempo <= 0 or target_tempo <= 0:
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')
        return

    rate = target_tempo / original_tempo
    if rate <= 0 or np.isinf(rate) or np.isnan(rate):
        rate = 1.0
    
    atempo_str = get_atempo_filter(rate)
    stretch_cmd = f'ffmpeg -y -i "{input_path}" -filter:a "{atempo_str}" "{output_path}"'
    if not run_ffmpeg(stretch_cmd):
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')

def extract_stable_heartbeat_segment(y: np.ndarray, sr: int,
                                     target_duration: float = 10.0,
                                     min_segment: float = 2.0,
                                     quality_info: dict = None) -> np.ndarray:
    """Adaptive segment selection dựa trên chất lượng file.

    Clean file: chọn 1 đoạn liền lạc (continuous) → tự nhiên hơn
    Noisy/moderate: giữ nguyên logic cũ (windowed selection với crossfade)
    """
    if len(y) == 0:
        return y

    # Detect quality nếu chưa có
    if quality_info is None:
        quality_info = detect_input_quality(y, sr)

    quality = quality_info.get('quality', 'moderate')

    if quality == 'clean':
        logger.info(f"[stable_seg] Clean file → selecting continuous segment")
        return _extract_continuous_segment(y, sr, target_duration)
    else:
        logger.info(f"[stable_seg] {quality} file → using windowed selection")
        return _extract_windowed_segment(y, sr, target_duration, min_segment)


def _extract_continuous_segment(y: np.ndarray, sr: int,
                                target_duration: float = 10.0) -> np.ndarray:
    """Chọn 1 đoạn liền lạc có RMS cao nhất.

    Không cắt ghép → giữ nhịp tự nhiên cho file sạch.
    """
    if len(y) == 0:
        return y

    target_samples = int(target_duration * sr)
    total_dur = len(y) / sr

    # Nếu file đủ ngắn, trả về toàn bộ
    if total_dur <= target_duration:
        logger.info(f"[continuous_seg] Audio ngắn hơn target ({total_dur:.1f}s) → dùng toàn bộ")
        return y

    # Chia thành các cửa sổ chồng lấp 50%
    win_samples = int(2.0 * sr)  # 2s windows
    hop_samples = win_samples // 2
    n_windows = (len(y) - win_samples) // hop_samples + 1

    if n_windows < 2:
        return y

    # Tính RMS cho mỗi cửa sổ
    rms_scores = []
    for i in range(n_windows):
        start = i * hop_samples
        end = start + win_samples
        if end > len(y):
            break
        window = y[start:end]
        rms = float(np.sqrt(np.mean(window ** 2)))
        rms_scores.append((rms, start, end))

    # Sắp xếp theo RMS giảm dần
    rms_scores.sort(key=lambda x: -x[0])

    # Chọn top cửa sổ, tìm đoạn liền lạc dài nhất chứa được target_duration
    best_start = 0
    best_score = 0

    for rms, start, end in rms_scores[:10]:  # Chỉ xét top 10
        # Tìm đoạn liền lạc xung quanh start có độ dài >= target_samples
        candidate_start = max(0, start - target_samples // 2)
        candidate_end = min(len(y), candidate_start + target_samples)
        candidate = y[candidate_start:candidate_end]

        # Tính RMS trung bình của đoạn này
        score = float(np.sqrt(np.mean(candidate ** 2))) * len(candidate)

        if score > best_score:
            best_score = score
            best_start = candidate_start

    # Trả về đoạn liền lạc
    result = y[best_start:best_start + target_samples]
    actual_dur = len(result) / sr
    logger.info(f"[continuous_seg] Selected 1 continuous segment: {actual_dur:.1f}s @ sample {best_start}")
    return result


def _extract_windowed_segment(y: np.ndarray, sr: int,
                               target_duration: float = 10.0,
                               min_segment: float = 2.0) -> np.ndarray:
    """Giữ nguyên logic cũ: chọn best windows với crossfade."""
    if len(y) == 0:
        return y

    total_dur = len(y) / sr
    if total_dur <= target_duration:
        logger.info(f"[windowed_seg] Audio ngắn hơn target ({total_dur:.1f}s) → dùng toàn bộ")
        return y

    win_samples = int(min_segment * sr)
    hop_samples = win_samples // 2  # 50% overlap
    n_frames = (len(y) - win_samples) // hop_samples + 1

    if n_frames < 2:
        logger.info(f"[windowed_seg] Không đủ frame để phân tích → dùng toàn bộ")
        return y

    # Tính điểm cho mỗi cửa sổ
    scores = []
    for i in range(n_frames):
        start = i * hop_samples
        end = start + win_samples
        frame = y[start:end]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        hop_inner = sr // 10  # 100ms hop
        energies = [
            np.mean(frame[j:j+hop_inner] ** 2)
            for j in range(0, len(frame) - hop_inner, hop_inner)
        ]
        energy_var = float(np.std(energies)) if len(energies) > 1 else 1.0
        score = rms / (energy_var + 1e-8)
        scores.append((score, i))

    scores.sort(key=lambda x: -x[0])

    target_samples = int(target_duration * sr)
    selected_starts = []
    selected_total = 0
    used_ranges = []

    for score, idx in scores:
        if selected_total >= target_samples:
            break
        start = idx * hop_samples
        end = start + win_samples
        overlap = False
        for (s, e) in used_ranges:
            overlap_len = max(0, min(end, e) - max(start, s))
            if overlap_len > win_samples * 0.5:
                overlap = True
                break
        if not overlap:
            selected_starts.append(start)
            used_ranges.append((start, end))
            selected_total += win_samples

    if not selected_starts:
        logger.warning("[windowed_seg] Không chọn được đoạn nào → dùng toàn bộ")
        return y

    selected_starts.sort()

    crossfade_samples = int(0.05 * sr)
    segments = []
    for start in selected_starts:
        end = min(start + win_samples, len(y))
        segments.append(y[start:end].copy())

    if len(segments) == 1:
        result = segments[0]
    else:
        result = segments[0]
        for seg in segments[1:]:
            if len(result) < crossfade_samples or len(seg) < crossfade_samples:
                result = np.concatenate([result, seg])
            else:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                result[-crossfade_samples:] = result[-crossfade_samples:] * fade_out
                seg_start = seg[:crossfade_samples] * fade_in
                blended = result[-crossfade_samples:] + seg_start
                result = np.concatenate([result[:-crossfade_samples], blended, seg[crossfade_samples:]])

    actual_dur = len(result) / sr
    logger.info(f"[windowed_seg] Chọn {len(selected_starts)} đoạn, tổng {actual_dur:.1f}s "
                f"(target {target_duration:.1f}s)")
    return result


def mix_audio_v1(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120, heart_duration=None, heart_tempo=None, music_tempo=None, shared_data=None):
    """Version duy nhất: HPSS denoising + BPM sync (±15%) + 432Hz + 4s intro heartbeat + fade in/out.

    Pipeline:
    1. Shared preprocessing (preconvert + loudnorm asset, convert picked mono)
    2. HPSS → tách percussive component (nhịp tim sạch)
    3. Tìm đoạn heartbeat ổn định nhất ~10s (extract_stable_heartbeat_segment)
    4. BPM sync: chỉ thay đổi heartbeat tempo tối đa ±15%, nếu delta > 15% giữ nguyên
    5. 4 giây đầu chỉ có heartbeat (adelay nhạc nền)
    6. Mix với nhạc nền → amix duration=first
    7b. Fade-in 4s / Fade-out 4s trên finite mixed file (KHÔNG dùng afade trong filter_complex
        vì aloop=-1 infinite stream → timestamp reset mỗi vòng → afade=t=out không hoạt động)
    8. 432Hz tuning → Output FLAC
    """
    # Log key paths for debugging "Output file not created" issues
    output_path = os.path.abspath(output_path)
    logger.info(f"[mix] === START mix_audio_v1 ===")
    logger.info(f"[mix] output_path (absolute): {output_path}")
    logger.info(f"[mix] output_path exists before start: {os.path.exists(output_path)}")
    if heart_tempo is None:
        _, heart_tempo = calculate_duration_from_analysis(picked_audio, num_beats=4)
    if heart_tempo <= 0:
        heart_tempo = 120.0

    if music_tempo is None:
        music_tempo = detect_tempo(asset_audio)
    if music_tempo <= 0:
        music_tempo = 120.0

    logger.info(f"[mix] Starting mix_audio: heart={heart_tempo:.0f}BPM, music={music_tempo:.0f}BPM")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        denoised_path       = os.path.join(temp_dir, 'picked_denoised.wav')
        stable_path         = os.path.join(temp_dir, 'picked_stable.wav')
        stretched_path      = os.path.join(temp_dir, 'picked_stretched.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        mixed_temp_path     = os.path.join(temp_dir, 'mixed_temp.flac')
 
        # ── 1. Shared preprocessing (with cache) ──────────────────────────────────
        use_shared = bool(shared_data and shared_data.get('success'))

        if use_shared:
            temp_wav_path = shared_data['picked_wav_mono']
            normalized_asset_path = shared_data['normalized_asset_path']
            vol_asset = shared_data['asset_volume']
            if _is_valid_decoded_audio(temp_wav_path):
                logger.info(f"[mix] Using shared preprocessed data")
            else:
                logger.warning("[mix] Shared heartbeat mono file missing/unreadable, fallback local preprocessing")
                use_shared = False

        if not use_shared:
            logger.info(f"[mix] No usable shared_data → running local preprocessing")
            temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
            temp_wav_stereo_path = os.path.join(temp_dir, 'picked_temp_stereo.wav')
            normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

            # Check cache for normalized asset
            use_cache = False
            if os.path.exists(asset_audio):
                asset_cache_key = _get_cache_key(asset_audio)
                logger.debug(f"[cache] Generated key {asset_cache_key} for asset {os.path.basename(asset_audio)}")
                cached_asset = _check_cache(asset_cache_key)
                if cached_asset:
                    logger.info(f"[mix] Cache hit for normalized asset: {os.path.basename(cached_asset)}")
                    shutil.copy2(cached_asset, normalized_asset_path)
                    vol_asset = fast_mean_volume(normalized_asset_path)
                    use_cache = True
                else:
                    logger.info(f"[mix] Cache miss for asset {os.path.basename(asset_audio)}")

            if not use_cache:
                if not _ffmpeg_convert_heartbeat_variants(picked_audio, temp_wav_stereo_path, temp_wav_path):
                    raise RuntimeError(
                        "Cannot decode heartbeat upload. Please re-export as PCM WAV, FLAC, or MP3 and try again."
                    )

                raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
                if not preconvert_asset(asset_audio, raw_asset_path):
                    raise RuntimeError("Cannot decode background track audio for mixing.")

                if not run_ffmpeg(
                    f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 '
                    f'-af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"'
                ):
                    raise RuntimeError("Failed to normalize background track audio for mixing.")

                # Save to cache and log details
                if os.path.exists(asset_audio):
                    asset_cache_key = _get_cache_key(asset_audio)
                    _save_to_cache(asset_cache_key, normalized_asset_path)
                    logger.info(f"[mix] Saved normalized asset to cache: {asset_cache_key}")

                vol_asset = fast_mean_volume(normalized_asset_path)

            # DEBUG: Check cache size and eviction
            try:
                from pathlib import Path
                cache_dir = Path(__file__).parent / 'cache'
                if cache_dir.exists():
                    total_mb = sum(p.stat().st_size for p in cache_dir.iterdir()) / 1e6
                    logger.debug(f"[cache] Current cache size: {total_mb:.2f} MB")
                    if total_mb > 500:
                        cleanup_old_cache(str(cache_dir), max_size_mb=500)
            except Exception as e:
                logger.debug(f"[cache] Cache size check failed: {e}")

        # ── 2. HPSS denoising ────────────────────────────────────────────────
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        logger.info(f"[mix] Audio loaded: {len(y)/sr:.1f}s @ {sr}Hz")
 

        # Auto-detect input quality
        quality_info = detect_input_quality(y, sr)
        logger.info(f"[mix] Auto-detected quality: {quality_info['quality']}")

        y_denoised = apply_noise_reduction(y, sr, quality_info=quality_info)
        logger.info(f"[mix] HPSS denoising done")
 
        # ── 3. Chọn đoạn heartbeat ổn định nhất ~10s ────────────────────────
        y_stable = extract_stable_heartbeat_segment(y_denoised, sr, target_duration=10.0, quality_info=quality_info)
        sf.write(stable_path, y_stable, sr)
        stable_dur = len(y_stable) / sr
        logger.info(f"[mix] Stable segment extracted: {stable_dur:.1f}s → {stable_path}")
 
        # ── 4. BPM sync: thử nhiều phương án trong biên ±15% rồi chọn phương án nghe tự nhiên nhất
        heart_tempo = max(40.0, min(180.0, heart_tempo))
        music_tempo = max(50.0, min(220.0, music_tempo))

        bpm_plan = _plan_bpm_sync_adjustments(heart_tempo, music_tempo)
        tempo_rate = bpm_plan["heart_rate"]
        bpm_mode = bpm_plan.get("policy_mode", "light-sync")

        asset_rate = bpm_plan.get("asset_rate", 1.0)
        logger.info(
            f"[mix] BPM sync plan: policy={bpm_plan.get('policy_mode', 'standard')}, "
            f"heart_stretch={tempo_rate:.3f}, asset_stretch={asset_rate:.3f}"
        )
        logger.info(
            f"[mix] BPM values: heartbeat_raw={heart_tempo:.1f} -> heartbeat_adjusted={bpm_plan['adjusted_heart_tempo']:.1f}, "
            f"track_raw={music_tempo:.1f}, track_adjusted={bpm_plan['adjusted_music_tempo']:.1f}"
        )

        if abs(tempo_rate - 1.0) > BPM_SYNC_APPLY_EPS:
            atempo_str = get_atempo_filter(tempo_rate)
            if not run_ffmpeg(
                f'ffmpeg -y -i "{stable_path}" -filter:a "{atempo_str}" "{stretched_path}"'
            ):
                logger.warning("[mix] atempo stretch failed, using original stable segment")
                stretched_path = stable_path
        else:
            logger.info(f"[mix] rate≈1.0 → skip stretch")
            stretched_path = stable_path

        # ── 4.5. Stretch asset tempo for bidirectional sync ─────────────
        if abs(asset_rate - 1.0) > BPM_SYNC_APPLY_EPS:
            stretched_asset_for_mix = os.path.join(temp_dir, 'asset_stretched.wav')
            atempo_asset_str = get_atempo_filter(asset_rate)
            if run_ffmpeg(
                f'ffmpeg -y -i "{normalized_asset_path}" '
                f'-filter:a "{atempo_asset_str}" "{stretched_asset_for_mix}"'
            ):
                logger.info(f"[mix] Asset tempo stretched: rate={asset_rate:.3f}")
            else:
                logger.warning("[mix] Asset atempo stretch failed, using original asset")
                stretched_asset_for_mix = normalized_asset_path
        else:
            stretched_asset_for_mix = normalized_asset_path
 
        # ── 5. Normalize heartbeat ───────────────────────────────────────────
        picked_seg = AudioSegment.from_file(stretched_path, format='wav').normalize()
        # RMS boost nếu signal sparse (HPSS percussive)
        target_rms_dbfs = -16.0 if bpm_mode == "ambient-texture" else -12.0
        if picked_seg.dBFS < target_rms_dbfs:
            boost_cap = 12.0 if bpm_mode == "ambient-texture" else 18.0
            boost = min(target_rms_dbfs - picked_seg.dBFS, boost_cap)
            picked_seg = picked_seg + boost
            logger.info(f"[mix] RMS boost: +{boost:.1f}dB")
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[mix] Normalized heartbeat: {heart_len_s:.1f}s, dBFS={picked_seg.dBFS:.1f}")
 
        # ── 6. Volume balance ─────────────────────────────────────────────────
        vol_picked = fast_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[mix] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB, diff={diff:.1f}dB")

        # ── 7. Prepare a finite heartbeat bed + mix ─────────────────────────
        # Heartbeat bed chỉ có intro silence/ramp ở lần đầu, các lần lặp dùng crossfade
        # để loại bỏ khe nghỉ và tránh cảm giác loop bị khựng.
        try:
            asset_dur_s = float(sf.info(stretched_asset_for_mix).duration)
            logger.info(f"[mix] Asset WAV duration: {asset_dur_s:.1f}s")
        except Exception as e:
            logger.warning(f"[mix] sf.info WAV failed ({e}), skip duration threshold check")
            asset_dur_s = 0.0

        heartbeat_bed_path = os.path.join(temp_dir, 'picked_loopbed.wav')
        heartbeat_bed_target_s = max(
            asset_dur_s + INTRO_SECONDS + FADE_OUT_SECONDS + 2.0,
            heart_len_s * 4.0,
            INTRO_SECONDS + 12.0,
        )
        loop_bed_ready = _build_looped_heartbeat_bed(
            normalized_picked_path,
            heartbeat_bed_path,
            heartbeat_bed_target_s,
        )
        if loop_bed_ready:
            picked_mix_input_path = heartbeat_bed_path
            logger.info(
                f"[mix] Loop bed ready: target={heartbeat_bed_target_s:.1f}s, "
                f"crossfade={HEARTBEAT_LOOP_CROSSFADE_MS}ms"
            )
        else:
            picked_mix_input_path = normalized_picked_path
            logger.warning("[mix] Loop bed build failed, falling back to legacy heartbeat input")

        # Build optimized mix filter chain
        fade_in_s = FADE_IN_SECONDS
        fade_out_s = FADE_OUT_SECONDS
        fade_out_start = max(0.0, (asset_dur_s + INTRO_SECONDS) - fade_out_s) if asset_dur_s > 0 else (INTRO_SECONDS + 12.0)

        mix_params = {
            'intro_delay_ms': INTRO_DELAY_MS,
            'volume_asset': safe_db(max(0, -diff) - (2 if bpm_mode == 'ambient-texture' else 3)),
            'volume_picked': safe_db(max(1, diff + 1) + (3 if bpm_mode == 'ambient-texture' else 6)),
            'fade_in_s': fade_in_s,
            'fade_out_s': fade_out_s,
            'fade_out_start': fade_out_start,
            'bpm_mode': bpm_mode,
            'use_loop_bed': loop_bed_ready,
            'asset_weight': bpm_plan.get('asset_weight', 1.0 - STANDARD_HEARTBEAT_WEIGHT),
            'heart_weight': bpm_plan.get('heart_weight', STANDARD_HEARTBEAT_WEIGHT),
            'heart_len_s': heart_len_s,
            'sr': sr,
            'lowpass_f': 420 if bpm_mode == "ambient-texture" else 350,
        }

        if not loop_bed_ready:
            heartbeat_ramp_end_s = HEARTBEAT_SILENT_LEAD_SECONDS + HEARTBEAT_VOLUME_RAMP_SECONDS
            heartbeat_intro_envelope = (
                f"if(lt(t,{HEARTBEAT_SILENT_LEAD_SECONDS:.2f}),0,"
                f"if(lt(t,{heartbeat_ramp_end_s:.2f}),"
                f"(t-{HEARTBEAT_SILENT_LEAD_SECONDS:.2f})/{HEARTBEAT_VOLUME_RAMP_SECONDS:.2f},1))"
            )
            mix_params['heart_ramp_end_s'] = heartbeat_ramp_end_s
            mix_params['heartbeat_intro_envelope'] = heartbeat_intro_envelope

        mix_filter = _build_optimized_mix_filter(mix_params, quality_info)

        enc = codec_args(mixed_temp_path)
        picked_mix_input_flag = f'"{picked_mix_input_path}"'
        primary_mix_ok = run_ffmpeg(
            f'ffmpeg -y -i "{stretched_asset_for_mix}" -i {picked_mix_input_flag} '
            f'-filter_complex "{mix_filter}" -map "[a]" {enc} "{mixed_temp_path}"'
        )
        if primary_mix_ok:
            if os.path.exists(mixed_temp_path) and os.path.getsize(mixed_temp_path) > 0:
                logger.info(f"[mix] ✅ mixed_temp_path created: {mixed_temp_path} (size={os.path.getsize(mixed_temp_path)})")
            else:
                logger.error(f"[mix] ❌ mixed_temp_path MISSING/EMPTY after successful ffmpeg: {mixed_temp_path}")
                primary_mix_ok = False
        else:
            logger.error(f"[mix] ❌ primary mix filter chain failed")

        # With loop_bed_ready: fade + 432Hz are already in the optimized filter chain
        if loop_bed_ready:
            logger.info("[mix] ✅ Fade + 432Hz included in optimized filter chain")

        # Nếu loop_bed_ready=True VÀ primary_mix_ok: fade + 432Hz đã nằm trong filter chain
        if loop_bed_ready and primary_mix_ok:
            if os.path.exists(mixed_temp_path) and os.path.getsize(mixed_temp_path) > 0:
                is_healthy, health_reason, measured_mix_dur, measured_mix_db = evaluate_mixed_output(
                    mixed_temp_path,
                    expected_asset_duration=asset_dur_s,
                )
                logger.info(
                    f"[mix] Loop bed health check: ok={is_healthy}, reason={health_reason}, "
                    f"duration={measured_mix_dur:.1f}s, dbfs={measured_mix_db:.1f}"
                )
                if is_healthy:
                    import shutil
                    shutil.copy2(mixed_temp_path, output_path)
                    logger.info(f"[mix] ✅ Mix completed (loop bed mode) → {output_path}")
                    return
                logger.warning(f"[mix] Loop bed primary output unhealthy ({health_reason}), retrying fallback")
                primary_mix_ok = False
            else:
                raise RuntimeError("[mix] loop_bed_ready but mixed_temp_path missing/empty, cannot create output")

        if not primary_mix_ok:
            logger.warning("[mix] Primary filter chain failed, retrying with safe fallback mix chain")
            asset_filter = (
                f"[0:a]"
                f"adelay={mix_params['intro_delay_ms']}|{mix_params['intro_delay_ms']},"
                f"equalizer=f=100:width_type=o:width=2:g=-5,"
                f"volume={safe_db(mix_params['volume_asset'])}dB"
                f"[a0];"
            )
            if loop_bed_ready:
                fallback_picked_filter = (
                    f"[1:a]"
                    f"highpass=f=55,lowpass=f=380,"
                    f"volume={safe_db(max(0, diff) + (1 if bpm_mode == 'ambient-texture' else 3))}dB,"
                    f"acompressor=threshold=-20dB:ratio=1.4:attack=10:release=120,"
                    f"afftdn=nf={safe_afftdn_nf(-24.0):.1f}"
                    f"[a1];"
                )
            else:
                fallback_picked_filter = (
                    f"[1:a]"
                    f"highpass=f=55,lowpass=f=380,"
                    f"volume='{heartbeat_intro_envelope}':eval=frame,"
                    f"volume={safe_db(max(0, diff) + (1 if bpm_mode == 'ambient-texture' else 3))}dB,"
                    f"acompressor=threshold=-20dB:ratio=1.4:attack=10:release=120,"
                    f"afftdn=nf={safe_afftdn_nf(-24.0):.1f},"
                    f"aloop=loop=-1:size={int(heart_len_s * sr)}"
                    f"[a1];"
                )
            fallback_mix_filter = (
                f"{asset_filter}{fallback_picked_filter}"
                f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2"
                f":weights={mix_params['asset_weight']:.3f} {mix_params['heart_weight']:.3f},"
                f"alimiter=limit=0.9"
                f"[a]"
            )
            fallback_input_flag = picked_mix_input_flag if loop_bed_ready else f'-i "{normalized_picked_path}"'
            primary_mix_ok = run_ffmpeg(
                f'ffmpeg -y -i "{stretched_asset_for_mix}" {fallback_input_flag} '
                f'-filter_complex "{fallback_mix_filter}" -map "[a]" {enc} "{mixed_temp_path}"'
            )

        if not primary_mix_ok:
            logger.error("[mix] Final mix FFmpeg call failed after safe fallback")
            raise RuntimeError("[mix] Primary mix and all fallbacks failed, cannot create output")


        # For loop_bed_ready=True: fade+432Hz already in filter chain
        # Skip health check and fade+432Hz section
        if loop_bed_ready:
            if os.path.exists(mixed_temp_path) and os.path.getsize(mixed_temp_path) > 0:
                is_healthy, health_reason, measured_mix_dur, measured_mix_db = evaluate_mixed_output(
                    mixed_temp_path,
                    expected_asset_duration=asset_dur_s,
                )
                logger.info(
                    f"[mix] Loop bed fallback health check: ok={is_healthy}, reason={health_reason}, "
                    f"duration={measured_mix_dur:.1f}s, dbfs={measured_mix_db:.1f}"
                )
                if not is_healthy:
                    raise RuntimeError(f"[mix] Loop bed output unhealthy: {health_reason}")
                import shutil
                shutil.copy2(mixed_temp_path, output_path)
                logger.info(f"[mix] ✅ Mix completed (loop bed mode) → {output_path}")
                return
            else:
                raise RuntimeError("[mix] loop_bed_ready but mixed_temp_path missing/empty")

        is_healthy, health_reason, measured_mix_dur, measured_mix_db = evaluate_mixed_output(
            mixed_temp_path,
            expected_asset_duration=asset_dur_s,
        )
        logger.info(
            f"[mix] Mixed health check: ok={is_healthy}, reason={health_reason}, "
            f"duration={measured_mix_dur:.1f}s, dbfs={measured_mix_db:.1f}"
        )

        if not is_healthy:
            fallback_mixed_path = os.path.join(temp_dir, 'mixed_temp_fallback.flac')
            logger.warning(f"[mix] Primary mix unhealthy ({health_reason}), running safe fallback chain")
            if loop_bed_ready:
                fallback_filter = (
                    f"[0:a]"
                    f"adelay={INTRO_DELAY_MS}|{INTRO_DELAY_MS},"
                    f"equalizer=f=100:width_type=o:width=2:g=-3,"
                    f"volume={safe_db(max(0, -diff) - 2)}dB"
                    f"[a0];"
                    f"[1:a]"
                    f"highpass=f=55,lowpass=f=360,"
                    f"volume={safe_db(max(1, diff + 1) + 4)}dB,"
                    f"acompressor=threshold=-20dB:ratio=2:attack=6:release=120"
                    f"[a1];"
                    f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=1:weights={mix_params['asset_weight']:.3f} {mix_params['heart_weight']:.3f},"
                    f"alimiter=limit=0.92"
                    f"[a]"
                )
            else:
                fallback_filter = (
                    f"[0:a]"
                    f"adelay={INTRO_DELAY_MS}|{INTRO_DELAY_MS},"
                    f"equalizer=f=100:width_type=o:width=2:g=-3,"
                    f"volume={safe_db(max(0, -diff) - 2)}dB"
                    f"[a0];"
                    f"[1:a]"
                    f"highpass=f=55,lowpass=f=360,"
                    f"volume='{heartbeat_intro_envelope}':eval=frame,"
                    f"volume={safe_db(max(1, diff + 1) + 4)}dB,"
                    f"acompressor=threshold=-20dB:ratio=2:attack=6:release=120"
                    f"[a1];"
                    f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=1:weights={mix_params['asset_weight']:.3f} {mix_params['heart_weight']:.3f},"
                    f"alimiter=limit=0.92"
                    f"[a]"
                )
            fallback_input_flag = picked_mix_input_flag if loop_bed_ready else f'-stream_loop -1 -i "{normalized_picked_path}"'
            fallback_ok = run_ffmpeg(
                f'ffmpeg -y -i "{stretched_asset_for_mix}" {fallback_input_flag} '
                f'-filter_complex "{fallback_filter}" -map "[a]" '
                f'-c:a flac -compression_level 5 "{fallback_mixed_path}"'
            )
            if not fallback_ok:
                logger.error("[mix] Safe fallback chain failed")
                raise RuntimeError("[mix] Safe fallback chain failed, cannot create output")

            fb_ok, fb_reason, fb_dur, fb_db = evaluate_mixed_output(
                fallback_mixed_path,
                expected_asset_duration=asset_dur_s,
            )
            logger.info(
                f"[mix] Fallback health check: ok={fb_ok}, reason={fb_reason}, "
                f"duration={fb_dur:.1f}s, dbfs={fb_db:.1f}"
            )
            if not fb_ok:
                logger.error(f"[mix] Fallback output still unhealthy: {fb_reason}")
                raise RuntimeError(f"[mix] Fallback output still unhealthy: {fb_reason}")
            mixed_temp_path = fallback_mixed_path

        # After all mix attempts: if loop_bed_ready, fade+432Hz are already applied
        # → copy mixed_temp_path to output_path and return
        if loop_bed_ready:
            if os.path.exists(mixed_temp_path) and os.path.getsize(mixed_temp_path) > 0:
                is_healthy, health_reason, measured_mix_dur, measured_mix_db = evaluate_mixed_output(
                    mixed_temp_path,
                    expected_asset_duration=asset_dur_s,
                )
                logger.info(
                    f"[mix] Loop bed final health check: ok={is_healthy}, reason={health_reason}, "
                    f"duration={measured_mix_dur:.1f}s, dbfs={measured_mix_db:.1f}"
                )
                if not is_healthy:
                    raise RuntimeError(f"[mix] Loop bed output unhealthy after all attempts: {health_reason}")
                import shutil
                shutil.copy2(mixed_temp_path, output_path)
                logger.info(f"[mix] ✅ Mix completed (loop bed mode) → {output_path}")
                return
            else:
                raise RuntimeError("[mix] loop_bed_ready but mixed_temp_path missing/empty after all attempts")

        # ── 8. Fade-in / Fade-out → sau đó 432Hz (2 bước riêng) ───────────────
        # Only runs when loop_bed_ready=False
        # When loop_bed_ready=True, the code returns above.
        # Dùng sf.info(stretched_asset_for_mix) — ĐÂY LÀ WAV → luôn đọc được.
        # KHÔNG dùng ffprobe hay sf.info(FLAC) vì cả hai đều fail trên Docker.
        #
        # Timing: mixed file duration ≈ asset_dur + 4s (do adelay=4000ms)
        # → fade_out_start = mixed_dur - fade_out_duration
        if not (os.path.exists(mixed_temp_path) and os.path.getsize(mixed_temp_path) > 0):
            logger.error("[mix] mixed_temp is empty/missing, cannot process")
            raise RuntimeError("[mix] mixed_temp is empty/missing, cannot create output")

        try:
            mixed_dur_s = float(sf.info(mixed_temp_path).duration)
        except Exception as mixed_info_err:
            logger.warning(f"[mix] Cannot read mixed duration via soundfile: {mixed_info_err}")
            mixed_dur_s = 0.0

        if mixed_dur_s <= 0:
            mixed_dur_s = (asset_dur_s + INTRO_SECONDS) if asset_dur_s > 0 else (INTRO_SECONDS + 12.0)

        fade_in_s      = FADE_IN_SECONDS
        fade_out_s     = FADE_OUT_SECONDS
        fade_out_start = max(0.0, mixed_dur_s - fade_out_s)
        logger.info(
            f"[mix] Fade: in 0→{fade_in_s}s | "
            f"out {fade_out_start:.1f}→{fade_out_start + fade_out_s:.1f}s"
        )

        # ── 8a. Apply afade trên finite FLAC mixed file ───────────────────────
        faded_mixed_path = os.path.join(temp_dir, 'mixed_faded.flac')
        fade_parts = []
        if fade_in_s > 0.01:
            fade_parts.append(f"afade=t=in:st=0:d={fade_in_s:.2f}")
        if fade_out_s > 0.01:
            fade_parts.append(f"afade=t=out:st={fade_out_start:.2f}:d={fade_out_s:.2f}")

        fade_ok = True
        if fade_parts:
            fade_filter = ",".join(fade_parts)
            fade_ok = run_ffmpeg(
                f'ffmpeg -y -i "{mixed_temp_path}" '
                f'-af "{fade_filter}" '
                f'-c:a flac -compression_level 5 "{faded_mixed_path}"'
            )
        else:
            fade_ok = run_ffmpeg(
                f'ffmpeg -y -i "{mixed_temp_path}" -c:a flac -compression_level 5 "{faded_mixed_path}"'
            )
        if fade_ok and os.path.exists(faded_mixed_path) and os.path.getsize(faded_mixed_path) > 0:
            logger.info("[mix] ✅ Fade-in/out applied successfully")
            src_for_432 = faded_mixed_path
        else:
            logger.warning("[mix] ⚠️ Fade step failed — applying 432Hz without fade")
            src_for_432 = mixed_temp_path

        # ── 8b. 432Hz tuning ──────────────────────────────────────────────────
        if not tune_to_432hz(src_for_432, output_path):
            logger.warning("[mix] 432Hz tuning failed, exporting original mixed source")
            if not run_ffmpeg(
                f'ffmpeg -y -i "{src_for_432}" {codec_args(output_path)} "{output_path}"'
            ):
                logger.error("[mix] Final export failed after 432Hz fallback")
                raise RuntimeError("[mix] Cannot create output: both 432Hz and fallback failed")
        logger.info(f"[mix] ✅ 432Hz tuning done → {output_path}")

 

        # ── 9. Final validation ───────────────────────────────
        if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
            logger.error(f"[mix] CRITICAL: output_path not created at end of function: {output_path}")
            raise RuntimeError(f"[mix] Output file not created: {output_path}")
        logger.info(f"[mix] Final output validated: {output_path} (size={os.path.getsize(output_path)})")

    except Exception as e:
        logger.error(f"[mix] Error: {e}\n{traceback.format_exc()}")
        logger.error(f"[mix] output_path exists at exception time: {os.path.exists(output_path)}")
        raise
    finally:
        logger.info(f"[mix] === END mix_audio_v1 ===")
        logger.info(f"[mix] output_path (absolute) at finally: {output_path}")
        logger.info(f"[mix] output_path exists at finally: {os.path.exists(output_path)}")
        if os.path.exists(output_path):
            logger.info(f"[mix] output_path size at finally: {os.path.getsize(output_path)} bytes")
        # Check if output_path is inside temp_dir_obj (should NOT be!)
        if output_path.startswith(temp_dir_obj.name):
            logger.error(f"[mix] ⚠️ WARNING: output_path is inside temp_dir_obj! Will be deleted by cleanup!")
        temp_dir_obj.cleanup()
 

