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
import scipy.signal
from scipy.signal import oaconvolve
import gc
import soundfile as sf
import logging
import traceback
import math
import hashlib
import functools
from pathlib import Path
import concurrent.futures


# ─── DSP HELPERS (Smart AI Mastering) ───────────────────────────────────────────
def normalize_rms(signal_arr: np.ndarray, target_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(signal_arr**2))
    if rms < 1e-9:
        return signal_arr
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    return signal_arr * gain_linear

def low_pass_filter(audio: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, cutoff_hz, btype='low', fs=sr, output='sos')
    if audio.ndim == 1:
        return signal.sosfilt(sos, audio).astype(np.float32)
    return np.stack([signal.sosfilt(sos, ch).astype(np.float32) for ch in audio], axis=0)

def smart_peak_limiter(signal_arr: np.ndarray, ceiling_db: float = -0.1) -> np.ndarray:
    ceil_lin = 10 ** (ceiling_db / 20.0)
    over = np.abs(signal_arr) / ceil_lin
    mask = over > 0.9  
    out = signal_arr.copy()
    out[mask] = np.sign(signal_arr[mask]) * ceil_lin * (0.9 + 0.1 * np.tanh((over[mask] - 0.9) / 0.1))
    return out

def find_best_window(y: np.ndarray, sr: int):
    win_size     = int(3.0 * sr)
    step         = int(0.05 * sr)
    boundary_ms  = 150                          
    bnd_size     = max(1, int(boundary_ms * sr / 1000))
    best_start   = 0
    best_score   = -1.0
    best_stats   = {}

    global_max = np.max(np.abs(y)) if len(y) > 0 else 1.0
    threshold  = 0.05 * global_max

    for start in range(0, len(y) - win_size, step):
        end      = start + win_size
        window_y = y[start:end]

        mean_abs  = float(np.mean(np.abs(window_y)))
        chunks    = np.array_split(window_y, 6)
        chunk_rms = [float(np.sqrt(np.mean(c**2))) for c in chunks]
        mean_rms  = float(np.mean(chunk_rms))
        std_rms   = float(np.std(chunk_rms))

        if mean_rms < 1e-5:
            continue

        cv           = std_rms / mean_rms
        stability    = 1.0 / (cv + 0.05)
        active_ratio = float(np.mean(np.abs(window_y) > threshold))

        rms_head = float(np.sqrt(np.mean(window_y[:bnd_size] ** 2)))
        rms_tail = float(np.sqrt(np.mean(window_y[-bnd_size:] ** 2)))
        head_ratio = rms_head / (mean_rms + 1e-9)
        tail_ratio = rms_tail / (mean_rms + 1e-9)
        boundary_factor = 1.0 / (1.0 + head_ratio ** 2 + tail_ratio ** 2)
        boundary_factor *= 4.0

        score = mean_abs * stability * active_ratio * boundary_factor

        if score > best_score:
            best_score = score
            best_start = start
            best_stats = {
                "mean_amplitude":   mean_abs,
                "energy_cv":        cv,
                "active_ratio":     active_ratio,
                "boundary_factor":  boundary_factor,
                "rms_head_ratio":   head_ratio,
                "rms_tail_ratio":   tail_ratio,
            }

    if best_score < 0:
        best_start = 0
        best_stats = {"fallback": True}

    return best_start, best_start + int(3.0 * sr), best_stats

def snap_to_zero_crossing(y: np.ndarray, sr: int, target: int, search_ms: int = 5, min_idx: int = 0, max_idx: int = None) -> int:
    radius = int(search_ms * sr / 1000)
    if max_idx is None:
        max_idx = len(y) - 1

    lo = max(min_idx, target - radius)
    hi = min(max_idx, target + radius)

    if hi <= lo: return target
    window = y[lo:hi]
    if len(window) >= 2:
        zc = np.where((window[:-1] <= 0) & (window[1:] > 0))[0]
        if len(zc) > 0:
            target_in_win = target - lo
            best_zc = zc[np.argmin(np.abs(zc - target_in_win))]
            return lo + int(best_zc)
    return target

def extract_continuous_stable_3s(y: np.ndarray, sr: int):
    orig_start, orig_end, _ = find_best_window(y, sr)
    # === FIND ALL VALLEYS + COST MINIMIZATION CHO S VA E ===
    env_smooth = int(0.05 * sr)
    full_env = np.sqrt(oaconvolve(y**2, np.ones(env_smooth, dtype=np.float32) / env_smooth, mode='same')).astype(np.float32)
    med_rms = float(np.median(full_env))
    
    # Bo distance de khong bo sot cac valley cuc bo
    valleys, _ = scipy.signal.find_peaks(-full_env)

    def find_best_valley(target_idx, search_lo, search_hi, alpha=5.0):
        best_cost = float('inf')
        best_v = target_idx
        for v in valleys:
            if search_lo <= v <= search_hi:
                rms = full_env[v]
                rms_norm = rms / (med_rms + 1e-6)
                dist_sec = abs(v - target_idx) / sr
                cost = rms_norm + alpha * dist_sec
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
        
        # Guard: Neu diem valley tim duoc khong cai thien qua 15% so voi diem target hien tai thi giu nguyen target
        rms_at_best = full_env[best_v]
        rms_at_target = full_env[min(target_idx, len(full_env) - 1)]
        if rms_at_target > 0 and rms_at_best >= 0.85 * rms_at_target:
            return target_idx
            
        return best_v

    # Tim s (ban kinh +- 1.5s quanh orig_start de quet rong hon)
    s = find_best_valley(orig_start, max(0, orig_start - int(1.5 * sr)), min(len(y) - 1, orig_start + int(1.5 * sr)))
    
    # Tim e (quet tu s + 1.5s den s + 4.5s de nam trong khoang 3s, uu tien vung gan orig_end)
    e = find_best_valley(orig_end, s + int(1.5 * sr), min(len(y) - 1, s + int(4.5 * sr)))

    del full_env
    gc.collect()

    # Gioi han min_duration neu file qua ngan
    min_duration = int(1.9 * sr)
    min_duration = min(min_duration, len(y) - s - int(0.1*sr))
    min_duration = max(min_duration, int(1.0 * sr)) 

    # Neu e bi chong cheo hoac ngan hon min_duration thi ep e dich ra xa
    if e < s + min_duration:
        e = min(s + min_duration, len(y) - 1)

    # Fine-snap +-5ms quanh diem ria de chot zero-crossing
    s = snap_to_zero_crossing(y, sr, s, search_ms=5)
    e = snap_to_zero_crossing(y, sr, e, search_ms=5, min_idx=s + min_duration)

    if s >= e or (e - s) < min_duration:
        e = s + min_duration
        if s >= e or e > len(y):
            s, e = orig_start, orig_end

    return s, e

def create_seamless_loop(y: np.ndarray, sr: int, s: int, e: int, target_duration: float = 30.0, crossfade_ms: float = 80.0) -> np.ndarray:
    """
    Tao file loop 30s.
    Phuong phap:
    1. Dung Cross-Correlation de tim chu ky (cycle length L) tron xoe nhat (tranh vấp nhip).
    2. Chi ap dung crossfade rat ngan (80ms) o diem cat L de khong bi chong cheo dai gay giam am luong (phasing/volume dip).
    """
    y_cut = y[s:e]
    
    # 1. Tinh Envelope de so khop nhip
    smooth_win = int(0.04 * sr)
    env = np.sqrt(oaconvolve(y_cut.astype(np.float32)**2, np.ones(smooth_win, dtype=np.float32) / smooth_win, mode='same')).astype(np.float32)
    
    # 2. Tim chu ky hoan hao bang Cross-Correlation
    min_overlap = int(0.1 * sr)
    max_overlap = int(1.2 * sr)
    max_overlap = min(max_overlap, len(y_cut) // 2)
    
    best_O = min_overlap
    best_score = -float('inf')
    
    if max_overlap > min_overlap:
        for O in range(min_overlap, max_overlap, int(0.01 * sr)): # Buoc nhay 10ms
            head = env[:O]
            tail = env[-O:]
            if np.std(head) > 0 and np.std(tail) > 0:
                score = np.corrcoef(head, tail)[0, 1]
            else:
                score = -1
                
            if score > best_score:
                best_score = score
                best_O = O
                
    del env
    gc.collect()
    
    # Fallback neu khong tim thay su tuong quan tot (< 0.4)
    if best_score > 0.4:
        L_samples = len(y_cut) - best_O
    else:
        # File khong co tinh chu ky ro rang, L_samples gan het file, de lai mot doan crossfade
        fade_len_default = int((crossfade_ms / 1000.0) * sr)
        L_samples = max(1, len(y_cut) - fade_len_default)
        
    # 3. Tao Loop chi voi 80ms crossfade
    fade_len = int((crossfade_ms / 1000.0) * sr)
    # Dam bao fade_len khong vuot qua phan am thanh con du
    fade_len = min(fade_len, len(y_cut) - L_samples)
    if fade_len <= 0: fade_len = 1
    
    loop_times = int(np.ceil((target_duration * sr) / L_samples))
    if loop_times < 1: loop_times = 1
    
    out_len = L_samples * loop_times + fade_len
    output = np.zeros(out_len, dtype=np.float32)
    
    # Dung Equal Power crossfade vi doan noi chi co 80ms
    t = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade_in = np.sqrt(t)
    fade_out = np.sqrt(1.0 - t)
    
    seg_base = y_cut[:L_samples + fade_len].astype(np.float32)
    
    for i in range(loop_times):
        pos = i * L_samples
        seg_win = seg_base.copy()
        
        if i > 0:
            seg_win[:fade_len] *= fade_in
        if i < loop_times - 1:
            seg_win[-fade_len:] *= fade_out
            
        output[pos : pos + len(seg_win)] += seg_win
        
    del seg_base, fade_in, fade_out, t
    gc.collect()
        
    # Cat dung do dai target
    target_samples_out = int(target_duration * sr)
    if len(output) > target_samples_out:
        output = output[:target_samples_out]
        
    return output.astype(y.dtype)
# ─────────────────────────────────────────────────────────────────────────────

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

MAX_BPM_STRETCH = 0.10
AMBIENT_TRACK_BPM_THRESHOLD = 60.0
BPM_SYNC_APPLY_EPS = 0.02
BPM_SYNC_RATIO_TOLERANCE = 0.10
AMBIENT_HEARTBEAT_WEIGHT = 0.30
STANDARD_HEARTBEAT_WEIGHT = 0.55
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
    """Pick a conservative BPM sync plan.

    - 1:1 and octave-equivalent 2:1 ratios are normalized before comparison.
    - Ambient/low-BPM: no stretch, reduce heartbeat prominence instead.
    - Close ratios are synced by stretching only the background track.
    - Larger gaps are left unsynced to avoid audible track warping.
    - Heartbeat tempo is never stretched so it stays natural.
    """
    heart_tempo = float(heart_tempo or 120.0)
    music_tempo = float(music_tempo or 120.0)

    normalized_music_tempo, music_octave_shift = _normalize_music_tempo_for_sync(music_tempo, heart_tempo)
    exact_ratio = normalized_music_tempo / max(heart_tempo, 1e-9)
    ratio_gap = abs(1.0 - exact_ratio)
    ambient_mode = normalized_music_tempo <= AMBIENT_TRACK_BPM_THRESHOLD

    base_plan = {
        "raw_music_tempo": music_tempo,
        "music_tempo": normalized_music_tempo,
        "music_octave_shift": music_octave_shift,
        "heart_rate": 1.0,
        "asset_rate": 1.0,
        "asset_rate_requested": 1.0,
        "adjusted_heart_tempo": heart_tempo,
        "adjusted_raw_music_tempo": music_tempo,
        "adjusted_music_tempo": normalized_music_tempo,
        "residual_ratio": exact_ratio,
        "exact_ratio": exact_ratio,
        "policy_mode": "ambient-texture" if ambient_mode else "natural-ratio",
        "heart_limit": 0.0,
        "asset_limit": 0.0,
        "heart_weight": AMBIENT_HEARTBEAT_WEIGHT if ambient_mode else STANDARD_HEARTBEAT_WEIGHT,
        "asset_weight": 1.0 - (AMBIENT_HEARTBEAT_WEIGHT if ambient_mode else STANDARD_HEARTBEAT_WEIGHT),
    }

    if heart_tempo <= 0 or music_tempo <= 0:
        return base_plan

    if ratio_gap <= BPM_SYNC_APPLY_EPS:
        return {
            **base_plan,
            "policy_mode": "natural-ratio",
            "heart_limit": 0.0,
            "asset_limit": 0.0,
            "heart_weight": STANDARD_HEARTBEAT_WEIGHT,
            "asset_weight": 1.0 - STANDARD_HEARTBEAT_WEIGHT,
        }

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

    if ratio_gap <= BPM_SYNC_RATIO_TOLERANCE + 1e-9:
        requested_asset_rate = 1.0 / exact_ratio
        asset_rate = _clamp_tempo_rate(requested_asset_rate, max_stretch=max_stretch)
        adjusted_raw_music = music_tempo * asset_rate
        adjusted_music = normalized_music_tempo * asset_rate
        return {
            **base_plan,
            "asset_rate": asset_rate,
            "asset_rate_requested": requested_asset_rate,
            "adjusted_raw_music_tempo": adjusted_raw_music,
            "adjusted_music_tempo": adjusted_music,
            "residual_ratio": adjusted_music / max(heart_tempo, 1e-9),
            "policy_mode": "track-sync",
            "heart_limit": 0.0,
            "asset_limit": max_stretch,
            "heart_weight": STANDARD_HEARTBEAT_WEIGHT,
            "asset_weight": 1.0 - STANDARD_HEARTBEAT_WEIGHT,
        }

    return {
        **base_plan,
        "policy_mode": "no-sync",
        "heart_limit": 0.0,
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


def _append_loop_crossfade(base: AudioSegment, segment: AudioSegment, crossfade_ms: int) -> AudioSegment:
    """Append a loop segment using overlap crossfade, falling back only when segments are too short."""
    crossfade_ms = max(0, int(crossfade_ms))
    if crossfade_ms <= 0:
        return base + segment

    safe_crossfade_ms = min(crossfade_ms, len(base) // 2, len(segment) // 2)
    if safe_crossfade_ms <= 0:
        return base + segment

    return base.append(segment, crossfade=safe_crossfade_ms)


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
    logger.info(
        f"[mix] Loop bed crossfade config: source={len(source)}ms, target={target_ms}ms, "
        f"crossfade={crossfade_ms}ms, intro_silence={intro_silence_ms}ms, "
        f"first_loop_ramp={intro_ramp_ms}ms"
    )
    remaining_ms = target_ms - len(bed)
    if remaining_ms > 0:
        loop_block = source
        target_block_ms = remaining_ms + crossfade_ms
        while len(loop_block) < target_block_ms:
            loop_block = _append_loop_crossfade(loop_block, loop_block, crossfade_ms)

        bed = _append_loop_crossfade(bed, loop_block, crossfade_ms)

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
    """Smart AI Mastering DSP Mix Pipeline"""
    output_path = os.path.abspath(output_path)
    logger.info(f"[mix] === START mix_audio_v1 ===")
    logger.info(f"[mix] output_path (absolute): {output_path}")

    # Tạo thư mục tạm
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        # Load asset track
        logger.info(f"[mix] Loading track: {asset_audio}")
        track_raw, track_sr = librosa.load(asset_audio, sr=None, mono=False)
        if track_raw.ndim == 1:
            track_raw = np.vstack([track_raw, track_raw]) 
        n_ch, track_samples = track_raw.shape
        
        track_mono = librosa.to_mono(track_raw)
        track_rms = np.sqrt(np.mean(track_mono**2))
        raw_rms_db = 20 * np.log10(track_rms) if track_rms > 0 else -100
        track_peak = np.max(np.abs(track_mono))
        raw_peak_db = 20 * np.log10(track_peak) if track_peak > 0 else -100
        crest_factor = raw_peak_db - raw_rms_db
        
        sample_len = min(len(track_mono), 30 * track_sr)
        S, _ = librosa.magphase(librosa.stft(track_mono[:sample_len]))
        centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=track_sr))
        
        logger.info(f"[mix] Original Track: RMS = {raw_rms_db:.2f} dBFS, Peak = {raw_peak_db:.2f} dBFS, Crest = {crest_factor:.1f} dB, Centroid = {centroid:.0f} Hz")
        
        # AI Logic: Leveling
        if raw_rms_db > -14.0:
            target_track_rms = -17.0
            track_norm = normalize_rms(track_raw, target_track_rms)
            logger.info(f"[mix] AI Action (Leveling): Track is LOUD. Reduced to {target_track_rms} dBFS.")
        else:
            target_track_rms = raw_rms_db
            track_norm = track_raw.copy()
            logger.info(f"[mix] AI Action (Leveling): Track is QUIET/AMBIENT. Preserved volume at {target_track_rms:.2f} dBFS.")

        # AI Logic: EQ
        if centroid > 1200:
            track_norm = low_pass_filter(track_norm, track_sr, 8000.0)
            logger.info(f"[mix] AI Action (EQ): Track is BRIGHT. Applied Low-Pass Filter @ 8000Hz.")
        elif centroid > 800:
            track_norm = low_pass_filter(track_norm, track_sr, 6000.0)
            logger.info(f"[mix] AI Action (EQ): Track is MEDIUM-BRIGHT. Applied Low-Pass Filter @ 6000Hz.")
        else:
            logger.info(f"[mix] AI Action (EQ): Track is DARK. NO Low-Pass Filter applied.")

        del track_raw
        gc.collect()

        # Load heartbeat for BPM Sync & Mix
        logger.info(f"[mix] Loading heartbeat: {picked_audio}")
        y_hb, hb_sr = librosa.load(picked_audio, sr=None, mono=True)

        # --- AI Logic: BPM Sync (New Forced Sync) ---
        try:
            if music_tempo and float(music_tempo) > 0:
                track_tempo_val = float(music_tempo)
                logger.info("[mix] Using provided music_tempo to skip analysis.")
            else:
                sample_len_beat = min(len(track_mono), 30 * track_sr)
                track_tempo_arr, _ = librosa.beat.beat_track(y=track_mono[:sample_len_beat], sr=track_sr)
                track_tempo_val = track_tempo_arr[0] if isinstance(track_tempo_arr, np.ndarray) else track_tempo_arr
            
            # Chuẩn hóa BPM nhạc nền (50 - 150)
            if track_tempo_val > 0:
                while track_tempo_val < 50:
                    track_tempo_val *= 2.0
                while track_tempo_val > 150:
                    track_tempo_val /= 2.0
                    
            if heart_tempo and float(heart_tempo) > 0:
                hb_tempo_val = float(heart_tempo)
                logger.info("[mix] Using provided heart_tempo to skip analysis.")
            else:
                hb_tempo_arr, _ = librosa.beat.beat_track(y=y_hb, sr=hb_sr)
                hb_tempo_val = hb_tempo_arr[0] if isinstance(hb_tempo_arr, np.ndarray) else hb_tempo_arr
            
            logger.info(f'[mix] Track BPM (Normalized): {track_tempo_val:.1f}, Heartbeat BPM: {hb_tempo_val:.1f}')
            
            if track_tempo_val > 0 and hb_tempo_val > 0:
                # Ép đồng bộ hoàn toàn
                rate = hb_tempo_val / track_tempo_val
                # Cơ chế an toàn (Safety Mechanism) - Tránh OOM & biến dạng quá mức
                clamped_rate = np.clip(rate, 0.5, 2.0)
                
                if abs(clamped_rate - 1.0) > 0.01:
                    logger.info(f'[mix] AI Action (BPM Sync): Forced sync without octave limits. Time-stretching track by {clamped_rate:.3f}.')
                    # Chạy tuần tự để tiết kiệm RAM, tránh OOM/Crash trên production server (đánh đổi chút CPU time)
                    stretched_channels = [librosa.effects.time_stretch(track_norm[i], rate=clamped_rate) for i in range(n_ch)]
                    del track_norm
                    gc.collect()
                    track_norm = np.stack(stretched_channels, axis=0)
                    del stretched_channels
                    gc.collect()
                    track_samples = track_norm.shape[1]
                else:
                    logger.info(f'[mix] AI Action (BPM Sync): Rate is ~1.0, no stretch applied.')
        except Exception as e:
            logger.error(f"[mix] Safety Fallback: BPM Sync time_stretch failed, using original track. Error: {e}")

        # Pad track (15.0s fade_duration padded)
        FADE_DURATION = 15.0
        fade_n = int(FADE_DURATION * track_sr)
        total_samples = fade_n + track_samples + fade_n
        
        track_padded = np.zeros((n_ch, total_samples), dtype=np.float32)
        track_padded[:, fade_n : fade_n + track_samples] = track_norm
        del track_norm
        gc.collect()
        
        logger.info(f"[mix] Extracting continuous stable 3s segment (Zero-Crossing + Boundary Check)...")
        s, e = extract_continuous_stable_3s(y_hb, hb_sr)
        
        logger.info(f"[mix] Creating SEAMLESS LOOP (Autocorrelation Beat-Sync)...")
        target_duration_sec = total_samples / track_sr
        hb_loop_raw = create_seamless_loop(y_hb, hb_sr, s, e, target_duration=target_duration_sec, crossfade_ms=80.0)
        
        if hb_sr != track_sr:
            hb_loop_raw = librosa.resample(hb_loop_raw, orig_sr=hb_sr, target_sr=track_sr)
            
        if len(hb_loop_raw) > total_samples:
            hb_loop_raw = hb_loop_raw[:total_samples]
        elif len(hb_loop_raw) < total_samples:
            pad_len = total_samples - len(hb_loop_raw)
            hb_loop_raw = np.pad(hb_loop_raw, (0, pad_len), mode='constant')
            
        # V3 AI Logic: Phân tích Peak/RMS cho tiếng tim
        hb_rms = np.sqrt(np.mean(hb_loop_raw**2) + 1e-9)
        hb_peak = np.max(np.abs(hb_loop_raw))
        hb_crest_factor = 20 * np.log10(hb_peak + 1e-9) - 20 * np.log10(hb_rms)

        # AI Logic: Bù trừ kép (Dual Crest-Factor Compensation)
        dynamic_hb_offset = -13.0 - (crest_factor - 11.0) + (hb_crest_factor - 14.5)
        dynamic_hb_offset = float(np.clip(dynamic_hb_offset, -30.0, -10.0))

        heartbeat_target_rms = target_track_rms + dynamic_hb_offset
        seg_norm = normalize_rms(hb_loop_raw, heartbeat_target_rms)
        del hb_loop_raw
        gc.collect()
        
        logger.info(f"[mix] Heartbeat synced at {heartbeat_target_rms:.2f} dBFS (Offset: {dynamic_hb_offset:.1f}dB, TrackCrest: {crest_factor:.1f}dB, HBCrest: {hb_crest_factor:.1f}dB).")
        
        seg_stereo = np.stack([seg_norm] * n_ch, axis=0).astype(np.float32)
        
        # Envelope fade (3s env_fade_duration)
        ENV_FADE_DURATION = 3.0
        env_fade_n = int(ENV_FADE_DURATION * track_sr)
        stable_env = np.ones(total_samples, dtype=np.float32)
        stable_env[:env_fade_n]  = np.linspace(0.0, 1.0, env_fade_n)
        stable_env[-env_fade_n:] = np.linspace(1.0, 0.0, env_fade_n)
        
        seg_mix = seg_stereo * stable_env[np.newaxis, :]
        del seg_stereo, stable_env
        gc.collect()

        logger.info(f"[mix] FINAL MIXING & PEAK PROTECTION...")
        
        # Mix (in-place cộng vào seg_mix)
        seg_mix += track_padded
        del track_padded
        gc.collect()
        
        # Master Peak Limiter
        final_out = smart_peak_limiter(seg_mix, ceiling_db=-0.1)
        del seg_mix
        gc.collect()

        temp_wav_out = os.path.join(temp_dir, 'final_mix.wav')
        sf.write(temp_wav_out, final_out.T, track_sr, subtype='PCM_16')
        
        logger.info(f"[mix] Saved temporary WAV. Converting to target format: {output_path}")
        
        # Convert qua output_path (FLAC/MP3) bằng ffmpeg
        import subprocess, shlex
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.flac':
            cmd = f'ffmpeg -y -i "{temp_wav_out}" -c:a flac -compression_level 5 "{output_path}"'
        elif ext == '.mp3':
            cmd = f'ffmpeg -y -i "{temp_wav_out}" -c:a libmp3lame -b:a 192k "{output_path}"'
        else:
            cmd = f'ffmpeg -y -i "{temp_wav_out}" "{output_path}"'

        res = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            logger.error(f"[mix] FFmpeg conversion failed: {res.stderr.decode()}")
            raise RuntimeError("FFmpeg conversion failed.")
        
        logger.info(f"[mix] ✅ Mix completed (Smart AI DSP) → {output_path}")

    except Exception as e:
        logger.error(f"[mix] Error in Smart AI Mastering DSP: {e}\n{traceback.format_exc()}")
        raise
    finally:
        logger.info(f"[mix] === END mix_audio_v1 ===")
        # keep the temp_dir cleaning by TemporaryDirectory auto cleanup
        logger.info(f"[mix] output_path (absolute) at finally: {output_path}")
        logger.info(f"[mix] output_path exists at finally: {os.path.exists(output_path)}")
        if os.path.exists(output_path):
            logger.info(f"[mix] output_path size at finally: {os.path.getsize(output_path)} bytes")
        # Check if output_path is inside temp_dir_obj (should NOT be!)
        if output_path.startswith(temp_dir_obj.name):
            logger.error(f"[mix] ⚠️ WARNING: output_path is inside temp_dir_obj! Will be deleted by cleanup!")
        temp_dir_obj.cleanup()
 

