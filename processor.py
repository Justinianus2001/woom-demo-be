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
                                 stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=60)
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
        y, sr = _librosa_load_safe(picked_audio, duration=30.0)
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
        y, sr = _librosa_load_safe(audio_path, duration=60.0)
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

FFMPEG_TIMEOUT = 120  # seconds – kill ffmpeg if it runs longer than this


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

    Returns True if any strategy succeeded AND the output has non-silent audio.
    """
    if _check_lfs_pointer(asset_audio):
        logger.error(f"❌ '{asset_audio}' is a Git LFS pointer, not actual audio data! Run 'git lfs pull' on your server.")
        return False
    strategies = [
        # (label, extra input flags before -i)
        ("auto_large_probe", "-probesize 50M -analyzeduration 100M"),
        ("mp3",   "-f mp3"),
        ("mp4",   "-f mp4"),
        ("flac",  "-f flac"),
        ("w64",   "-f w64"),
        ("auto",  ""),
        ("wav",   "-f wav"),
    ]
    for label, extra in strategies:
        cmd = f'ffmpeg -y {extra} -i "{asset_audio}" -ar 44100 -ac 2 -sample_fmt s16 "{output_path}"'.strip()
        # Collapse double spaces that appear when extra == ""
        cmd = ' '.join(cmd.split())
        if run_ffmpeg(cmd):
            if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
                logger.warning(f"[preconvert_asset] Strategy '{label}' produced empty file, skipping.")
                _try_unlink(output_path)
                continue
            # Validate: make sure the audio actually has signal (not all-zero silence)
            try:
                data, _ = sf.read(output_path, frames=8192)
                rms = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0.0
                if rms < 1e-6:
                    logger.warning(
                        f"[preconvert_asset] Strategy '{label}' produced silent audio "
                        f"(RMS={rms:.2e}), likely wrong format — skipping."
                    )
                    _try_unlink(output_path)
                    continue
            except Exception as val_err:
                logger.warning(f"[preconvert_asset] Validation failed for strategy '{label}': {val_err}")
                _try_unlink(output_path)
                continue
            logger.info(f"[preconvert_asset] Success with strategy '{label}' (RMS={rms:.4f})")
            return True
        else:
            _try_unlink(output_path)
    logger.error(f"[preconvert_asset] All strategies failed for '{asset_audio}'")
    return False


def _try_unlink(path: str) -> None:
    """Silently remove a file if it exists."""
    if os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


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


def preprocess_shared(asset_audio: str, picked_audio: str, work_dir: str):
    """Tiền xử lý chung cho cả 3 versions — chỉ chạy MỘT LẦN.

    Thực hiện:
    1. Pre-convert asset audio → PCM WAV chuẩn (handles Wave64/RF64/float/m4a/mp3...)
    2. Loudnorm asset → -16 LUFS
    3. Convert picked (heartbeat) audio → PCM WAV 44100Hz mono+stereo
       (mono cho v2/v3/v4 HPSS, stereo cho v1 low-pass)
    4. Đo volume asset & picked (bằng numpy, KHÔNG subprocess)

    Returns:
        dict with keys:
        - 'normalized_asset_path': str – asset WAV đã loudnorm
        - 'picked_wav_stereo': str – heartbeat WAV 44100Hz stereo (cho v1)
        - 'picked_wav_mono': str – heartbeat WAV 44100Hz mono (cho v2/v3/v4)
        - 'asset_volume': float – mean volume dBFS
        - 'success': bool

    Raises RuntimeError if asset cannot be decoded.
    """
    logger.info(f"[preprocess_shared] Starting shared preprocessing...")

    raw_asset_path = os.path.join(work_dir, 'shared_asset_raw.wav')
    normalized_asset_path = os.path.join(work_dir, 'shared_asset_normalized.wav')
    picked_wav_stereo = os.path.join(work_dir, 'shared_picked_stereo.wav')
    picked_wav_mono = os.path.join(work_dir, 'shared_picked_mono.wav')

    # 1) Pre-convert asset (worst case tries 7 strategies — but only ONCE)
    if not preconvert_asset(asset_audio, raw_asset_path):
        logger.error(f"[preprocess_shared] Cannot decode asset audio '{asset_audio}'")
        return {'success': False}

    # 2) Loudnorm asset → chuẩn -16 LUFS
    if not run_ffmpeg(
        f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 '
        f'-af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"'
    ):
        logger.error("[preprocess_shared] Asset loudnorm failed")
        return {'success': False}
    _try_unlink(raw_asset_path)  # free disk space early

    # 3) Convert picked → WAV stereo (v1) và mono (v2/v3/v4) trong 1 lần ffmpeg
    #    Dùng -t 30 để giới hạn 30s tránh OOM
    if not run_ffmpeg(
        f'ffmpeg -y -i "{picked_audio}" -t 30 -ar 44100 -ac 2 "{picked_wav_stereo}" '
        f'-t 30 -ar 44100 -ac 1 "{picked_wav_mono}"'
    ):
        # Fallback: convert riêng từng cái
        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ar 44100 -ac 2 "{picked_wav_stereo}"')
        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ar 44100 -ac 1 "{picked_wav_mono}"')

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

def apply_noise_reduction(y, sr):
    """Sử dụng HPSS từ Librosa để tách percussive (nhịp tim)."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_percussive


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
    cmd = f'ffmpeg -y -i "{input_path}" -af "asetrate=44100*432/440,aresample=44100,atempo=1.0185185185185186" "{output_path}"'
    run_ffmpeg(cmd)

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

def mix_audio_v1(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120, heart_duration=None, shared_data=None):
    """Version 1: Pure Natural Mix — Low-pass filter, heartbeat nổi bật nhất.

    - BỎ: Cắt 4 nhịp + aloop vô hạn → gây nhịp tim máy móc, lặp lại
    - BỎ: atempo trên nhạc nền → nhạc giữ nguyên tempo gốc
    - THÊM: Dùng toàn bộ heartbeat (≤30s), duration=shortest, fade-out cuối
    - Nhận shared_data từ preprocess_shared() → bỏ preconvert_asset, loudnorm, convert picked
    - Dùng fast_mean_volume (numpy) thay vì get_mean_volume (subprocess)
    - Giảm từ ~12 FFmpeg calls → ~2 calls (silence-remove + final-mix)
    """
    logger.info(f"[v1] Starting mix_audio_v1 for picked='{picked_audio}', asset='{asset_audio}', output='{output_path}'")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        filtered_path = os.path.join(temp_dir, 'picked_filtered.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')

        # === Lấy dữ liệu đã tiền xử lý hoặc fallback ===
        if shared_data and shared_data.get('success'):
            temp_wav_path = shared_data['picked_wav_stereo']
            normalized_asset_path = shared_data['normalized_asset_path']
            vol_asset = shared_data['asset_volume']
            logger.info(f"[v1] Using shared preprocessed data (0 FFmpeg calls for convert/preconvert)")
        else:
            # Fallback: xử lý riêng (backward compatible)
            temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
            normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
            run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 2 -ar 44100 "{temp_wav_path}"')
            raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
            if not preconvert_asset(asset_audio, raw_asset_path):
                logger.error(f"[v1] Cannot decode asset audio '{asset_audio}', aborting.")
                return
            run_ffmpeg(f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
            vol_asset = get_mean_volume(normalized_asset_path)

        # === HEARTBEAT: Giữ nguyên tự nhiên, chỉ lọc noise ===
        # [ĐẶC TRƯNG V1] Low-pass filter 1500Hz (Butterworth bậc 5)
        # Giữ tần số thấp nhịp tim (20-200Hz), loại bỏ noise cao tần
        y, sr = sf.read(temp_wav_path)
        if y.ndim == 1: y = y[:, np.newaxis]
        logger.info(f"[v1] Finished reading picked audio: {temp_wav_path}")
        nyq = 0.5 * sr
        low = 1500 / nyq  # Tăng lên 1500Hz (trước đây 500Hz) để loa ngoài điện thoại có thể phát được dải mid của nhịp tim
        b, a = signal.butter(5, low, btype='low')
        padlen = 3 * (max(len(b), len(a)) - 1)
        if y.shape[0] > padlen:
            y_filtered = signal.filtfilt(b, a, y, axis=0)
        else:
            y_filtered = y
        if len(y_filtered.shape) == 2 and y_filtered.shape[1] == 1:
            y_filtered = y_filtered.squeeze()
        sf.write(filtered_path, y_filtered, sr)
        logger.info(f"[v1] Applied low-pass filter: {filtered_path}")

        # Loại bỏ silence đầu file — FFmpeg call #1
        run_ffmpeg(f'ffmpeg -y -i "{filtered_path}" -af silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:detection=peak "{silenced_path}"')
        logger.info(f"[v1] Removed leading silence: {silenced_path}")

        # Normalize heartbeat — pydub direct WAV load (0 subprocess)
        picked_seg = AudioSegment.from_file(silenced_path, format='wav').normalize()
        if picked_seg.dBFS < -50: picked_seg += 10
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v1] Normalized heartbeat: duration={heart_len_s:.1f}s")

        # Đo volume — numpy direct (0 subprocess)
        vol_picked = fast_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v1] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]adelay=2000|2000,volume={safe_db(max(0, -diff) - 3)}dB[a0];"
        # Heartbeat: giữ nguyên tự nhiên, loop theo chiều dài nhạc nền
        # [FIX] Cấp size chính xác cho aloop để loop đúng duration của file
        picked_filter = f"[1:a]volume={safe_db(max(0, diff) + 4)}dB,aloop=loop=-1:size={int(heart_len_s * 44100)}[a1];"

        enc = codec_args(output_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            
            # KHÔNG compress mạnh, KHÔNG giảm volume — giữ nguyên cảm giác tự nhiên
            f"[a1]highpass=f=80,lowpass=f=250,afftdn=nf=-20[a1clean];"

            # Nhạc nền: giảm dải tần 60-200Hz (trùng với heartbeat) để heartbeat nổi lên
            f"[a0]equalizer=f=100:width_type=o:width=2:g=-4[a0clean];"

            # Mix: heartbeat 55% / nhạc nền 45% → heartbeat nghe rõ nhưng vẫn có nhạc nền
            f"[a0clean][a1clean]amix=inputs=2:duration=first:dropout_transition=2:weights=0.45 0.55,"
            f"alimiter=limit=0.9[a]"
        )
        # FFmpeg call #2 — final mix
        if run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{output_path}"'):
            logger.info(f"[v1] Finished successfully -> {output_path}")
        else:
            logger.error(f"[v1] mix_audio_v1 failed at the final step")
    except Exception as e:
        logger.error(f"[v1] Error during mix_audio_v1: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()

def mix_audio_v2(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120, heart_duration=None, shared_data=None):
    """Version 2: Clean Heartbeat + 432Hz — HPSS separation, ấm áp hơn.

    - HPSS: Tách percussive → heartbeat sạch, rõ từng nhịp (v1 dùng low-pass)
    - Dynamic silence threshold → loại noise chính xác hơn fixed -40dB của v1
    - 432Hz tuning → tần số ấm áp, thư giãn hơn (v1/v3 không có)
    - weights 0.65/0.35 → heartbeat rõ nhưng nhạc nổi hơn v1
    - BỎ: Cắt 4 nhịp + aloop vô hạn → gây nhịp tim máy móc
    - BỎ: Asset stretch (tempo_factor luôn = 1.0, code thừa)
    - THÊM: Dùng toàn bộ heartbeat (≤30s), duration=shortest, fade-out cuối
    - Nhận shared_data từ preprocess_shared() → bỏ preconvert_asset, loudnorm, convert picked
    - Dùng fast_mean_volume (numpy) thay vì get_mean_volume (subprocess)
    - Giảm từ ~13 FFmpeg calls → ~3 calls (silence-remove + mix + 432Hz)
    """

    logger.info(f"[v2] Starting mix_audio_v2 for picked='{picked_audio}', asset='{asset_audio}', output='{output_path}'")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')

        # === Lấy dữ liệu đã tiền xử lý hoặc fallback ===
        if shared_data and shared_data.get('success'):
            temp_wav_path = shared_data['picked_wav_mono']
            normalized_asset_path = shared_data['normalized_asset_path']
            vol_asset = shared_data['asset_volume']
            logger.info(f"[v2] Using shared preprocessed data (0 FFmpeg calls for convert/preconvert)")
        else:
            # Fallback: xử lý riêng (backward compatible)
            temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
            normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
            run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 2 -ar 44100 "{temp_wav_path}"')
            raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
            if not preconvert_asset(asset_audio, raw_asset_path):
                logger.error(f"[v2] Cannot decode asset audio '{asset_audio}', aborting.")
                return
            run_ffmpeg(f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
            vol_asset = get_mean_volume(normalized_asset_path)

        # [ĐẶC TRƯNG V2] HPSS — tách percussive (nhịp tim) khỏi harmonic (noise)
        # Cho heartbeat sạch hơn low-pass filter của v1
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        y_denoised = apply_noise_reduction(y, sr)
        logger.info(f"[v2] Finished denoising picked audio: {denoised_path}")
        sf.write(denoised_path, y_denoised, sr)

        # [ĐẶC TRƯNG V2] Dynamic threshold — tính từ peak, chính xác hơn fixed -40dB của v1
        max_val = np.max(np.abs(y_denoised)) if len(y_denoised) > 0 else 0
        if max_val > 0:
            peak_db = librosa.amplitude_to_db(max_val)
            threshold_db = max(-50, peak_db - 30)
        else:
            threshold_db = -50
        logger.info(f"[v2] Dynamic threshold: {threshold_db:.1f}dB")
        # FFmpeg call #1 — silence remove
        run_ffmpeg(f'ffmpeg -y -i "{denoised_path}" -af silenceremove=start_periods=1:start_duration=0:start_threshold={threshold_db}dB:detection=peak "{silenced_path}"')
        logger.info(f"[v2] Removed silence: {silenced_path}")

        # Normalize heartbeat — RMS-based normalization thay vì peak-only
        # Vấn đề: HPSS tạo signal sparse (chỉ có impulse nhịp) → peak normalize không đủ
        # RMS target -12dBFS → heartbeat nghe rõ hơn trong mix
        picked_seg = AudioSegment.from_file(silenced_path, format="wav").normalize()
        # Sau peak normalize, nếu RMS vẫn thấp (signal sparse), boost thêm đến target -12dBFS
        target_rms_dbfs = -12.0
        if picked_seg.dBFS < target_rms_dbfs:
            boost = target_rms_dbfs - picked_seg.dBFS
            picked_seg = picked_seg + min(boost, 18.0)  # cap 18dB tránh noise explosion
            logger.info(f"[v2] RMS boost applied: +{min(boost, 18.0):.1f}dB (dBFS was {picked_seg.dBFS - min(boost, 18.0):.1f})")
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v2] Normalized heartbeat: duration={heart_len_s:.1f}s, dBFS={picked_seg.dBFS:.1f}")

        # Đo volume — numpy direct (0 subprocess)
        vol_picked = fast_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v2] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]adelay=2000|2000,volume={safe_db(max(0, -diff) - 3)}dB[a0];"

        picked_filter = (
            f"[1:a]"
            f"highpass=f=40,lowpass=f=250,"            # Chặt hơn: 40-250Hz, loại bỏ hoàn toàn tần số giọng người
            f"bass=g=5:f=80,"                          # Tăng nhẹ bass 80Hz (tần số tâm thu)
            f"volume={safe_db(max(2, diff+2) + 8)}dB,"   # +8dB (tăng từ +6) → nghe rõ hơn 20% còn thiếu
            f"acompressor=threshold=-18dB:ratio=1.5:attack=10:release=120,"  # Nén rất nhẹ, giữ dynamic tự nhiên
            f"stereowiden=delay=6,"
            f"aloop=loop=-1:size={int(heart_len_s * 44100)}"
            f"[a1];"
        )

        # [ĐẶC TRƯNG V2] weights: nhạc nền 35% / heartbeat 65%
        enc = codec_args(mixed_temp_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            # Giảm dải mid nhạc nền để heartbeat không bị che
            f"[a0]equalizer=f=100:width_type=o:width=2:g=-5[a0clean];"
            f"[a0clean][a1]amix=inputs=2:duration=first:dropout_transition=2"
            f":weights=0.45 0.55,"
            f"alimiter=limit=0.9[a]"
        )
        # FFmpeg call #2 — mix
        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{mixed_temp_path}"')

        # [ĐẶC TRƯNG V2] 432Hz tuning — tần số ấm áp, thư giãn hơn 440Hz
        # FFmpeg call #3 — 432Hz pitch shift
        if not (os.path.exists(mixed_temp_path) and os.path.getsize(mixed_temp_path) > 0):
            logger.error(f"[v2] mixed_temp.mp3 is empty or missing, aborting 432Hz step.")
        else:
            tune_to_432hz(mixed_temp_path, output_path)
        logger.info(f"[v2] Finished successfully -> {output_path}")
    except Exception as e:
        logger.error(f"[v2] Error during mix_audio_v2: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()

def mix_audio_v3(asset_audio, picked_audio, output_path, heart_duration=None, heart_tempo=None, music_tempo=None, shared_data=None):
    """Version 3: Tempo-Synced Music — nhạc nền adapt tempo theo heartbeat.
    - NHẠC NỀN được stretch tempo để match BPM của heartbeat → nhịp đồng bộ
    - BỎ: Cắt 4 nhịp + aloop → nhịp tim máy móc
    - THÊM: duration=shortest, fade-out cuối
    - Nhận shared_data → bỏ preconvert_asset (tiết kiệm tới 7 FFmpeg calls)
    - Dùng shared normalized_asset_path làm input cho atempo stretch
    - Giảm từ ~14 FFmpeg calls → ~4 calls (atempo + loudnorm + mix)
    """
    if heart_tempo is None:
        _, heart_tempo = calculate_duration_from_analysis(picked_audio, num_beats=4)
    if heart_tempo <= 0: heart_tempo = 120.0
    if music_tempo is None:
        music_tempo = detect_tempo(asset_audio)
    if music_tempo <= 0: music_tempo = 120.0

    logger.info(f"[v3] Starting mix_audio_v3 for picked='{picked_audio}', asset='{asset_audio}', output='{output_path}'")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        stretched_asset_path = os.path.join(temp_dir, 'asset_stretched.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

        # === Lấy dữ liệu đã tiền xử lý hoặc fallback ===
        if shared_data and shared_data.get('success'):
            temp_wav_path = shared_data['picked_wav_mono']
            # V3 cần stretch asset → dùng shared normalized asset làm input
            # (đã là PCM WAV chuẩn, bỏ qua preconvert_asset)
            shared_asset_path = shared_data['normalized_asset_path']
            logger.info(f"[v3] Using shared preprocessed data (0 FFmpeg calls for convert/preconvert)")
        else:
            # Fallback: xử lý riêng (backward compatible)
            temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
            run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 1 -ar 44100 "{temp_wav_path}"')
            raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
            if not preconvert_asset(asset_audio, raw_asset_path):
                logger.error(f"[v3] Cannot decode asset audio '{asset_audio}', aborting.")
                return
            shared_asset_path = raw_asset_path

        # HPSS
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        y_denoised = apply_noise_reduction(y, sr)
        logger.info(f"[v3] Finished denoising picked audio: {denoised_path}")
        sf.write(denoised_path, y_denoised, sr)
        logger.info(f"[v3] HPSS denoised: {denoised_path}")

        # Normalize heartbeat — pydub direct WAV load (0 subprocess)
        picked_seg = AudioSegment.from_file(denoised_path, format='wav').normalize() - 14
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v3] Normalized heartbeat (no stretch): duration={heart_len_s:.1f}s")

        # === NHẠC NỀN: Stretch tempo để match heartbeat BPM ===
        # [ĐẶC TRƯNG V3] ĐẢO logic: nhạc adapt theo heartbeat, không phải ngược lại
        # VD: heart=70BPM, music=120BPM → rate=70/120=0.58 → nhạc chậm lại match nhịp tim
        tempo_rate = heart_tempo / music_tempo
        tempo_rate = max(0.75, min(1.25, tempo_rate))  # Clamp tránh FFmpeg atempo quá extreme
        logger.info(f"[v3] Stretching music: rate={tempo_rate:.3f} (heart={heart_tempo:.0f} / music={music_tempo:.0f})")

        # FFmpeg call #1 — atempo stretch asset (dùng shared asset đã convert, bỏ preconvert)
        atempo_str = get_atempo_filter(tempo_rate)
        run_ffmpeg(f'ffmpeg -y -i "{shared_asset_path}" -filter:a "{atempo_str}" "{stretched_asset_path}"')
        # FFmpeg call #2 — loudnorm stretched asset
        run_ffmpeg(f'ffmpeg -y -i "{stretched_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
        logger.info(f"[v3] Normalized stretched asset: {normalized_asset_path}")

        # Đo volume — numpy direct (0 subprocess)
        vol_asset = fast_mean_volume(normalized_asset_path)
        vol_picked = fast_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v3] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]adelay=2000|2000,volume={safe_db(max(0, -diff + 2))}dB[a0];"

        picked_filter = (
            f"[1:a]"
            f"volume={safe_db(max(10, diff + 12))}dB,"
            f"highpass=f=120,lowpass=f=400,"
            f"bass=g=5:f=80,"
            # f"treble=g=4:f=3500,"
            f"acompressor=threshold=-24dB:ratio=2:attack=5:release=80,"
            f"aecho=0.8:0.7:40|80:0.25|0.15,"
            # extrastereo: mở rộng stereo field (m=1.6) → nhịp tim lan toả không gian
            f"extrastereo=m=1.6,"
            f"aloop=loop=-1:size=2e+09"
            f"[a1];"
        )

        # [SỬA LỖI] duration=first → output dài bằng nhạc nền (a0)
        enc = codec_args(output_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            f"[a0][a1]"
            f"amix=inputs=2:duration=first:dropout_transition=2"
            f":weights=0.3 0.7,"
            f"alimiter=limit=0.9"
            f"[a]"
        )
        # FFmpeg call #3 — final mix
        if run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{output_path}"'):
            logger.info(f"[v3] Finished successfully -> {output_path}")
        else:
            logger.error(f"[v3] mix_audio_v3 failed at final stage")
    except Exception as e:
        logger.error(f"[v3] Error during mix_audio_v3: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()

def mix_audio_v4(asset_audio, picked_audio, output_path, heart_duration=None, heart_tempo=None, music_tempo=None, shared_data=None):
    """Version 4: Double-Time + 432Hz — heartbeat tăng gấp đôi, năng lượng cao.

    ĐẶC TRƯNG V4 (phân biệt với v1/v2/v3):
    - Heartbeat stretch lên 2x music tempo → nhịp nhanh, năng lượng cao
      (Đây là version DUY NHẤT được phép thay đổi BPM heartbeat — theo feedback KH)
    - 432Hz tuning → tần số ấm áp kết hợp nhịp nhanh
    - weights 0.7/0.3 → cân bằng nhạc + heartbeat nhanh

    THAY ĐỔI 03/2026 (feedback khách hàng — nhịp tim không tự nhiên):
    - GIỮ: Stretch heartbeat 2x (khách hàng cho phép version này)
    - BỎ: Cắt 4 nhịp + aloop → dùng toàn bộ heartbeat đã stretch
    - THÊM: duration=shortest, fade-out cuối

    TỐI ƯU 03/2026:
    - Nhận shared_data → bỏ preconvert_asset, convert picked, get_mean_volume
    - Giảm từ ~15 FFmpeg calls → ~4 calls (stretch-heartbeat + mix + 432Hz)
    """
    if heart_tempo is None:
        _, heart_tempo = calculate_duration_from_analysis(picked_audio, num_beats=4)
    if heart_tempo <= 0: heart_tempo = 120.0
    if music_tempo is None:
        music_tempo = detect_tempo(asset_audio)
    if music_tempo <= 0: music_tempo = 120.0

    target_heartbeat_tempo = music_tempo * 2
    logger.info(f"[v4] Starting mix_audio_v4: heart={heart_tempo:.0f}BPM → target={target_heartbeat_tempo:.0f}BPM (2x music)")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        stretched_path = os.path.join(temp_dir, 'picked_stretched.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')

        # === Lấy dữ liệu đã tiền xử lý hoặc fallback ===
        if shared_data and shared_data.get('success'):
            temp_wav_path = shared_data['picked_wav_mono']
            normalized_asset_path = shared_data['normalized_asset_path']
            vol_asset = shared_data['asset_volume']
            logger.info(f"[v4] Using shared preprocessed data (0 FFmpeg calls for convert/preconvert)")
        else:
            # Fallback: xử lý riêng (backward compatible)
            temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
            normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
            run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 1 -ar 44100 "{temp_wav_path}"')
            raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
            if not preconvert_asset(asset_audio, raw_asset_path):
                logger.error(f"[v4] Cannot decode asset audio '{asset_audio}', aborting.")
                return
            run_ffmpeg(f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
            vol_asset = get_mean_volume(normalized_asset_path)

        # HPSS
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        logger.info(f"[v4] Finished reading picked audio: {temp_wav_path}")
        y_denoised = apply_noise_reduction(y, sr)
        sf.write(denoised_path, y_denoised, sr)
        logger.info(f"[v4] Finished denoising picked audio: {denoised_path}")

        # [GIỮ NGUYÊN] Stretch heartbeat lên 2x music tempo — khách hàng cho phép v4
        # FFmpeg call #1 — time stretch heartbeat (1-2 calls bên trong)
        time_stretch_heartbeat(denoised_path, stretched_path, target_heartbeat_tempo, heart_tempo)
        logger.info(f"[v4] Stretched heartbeat: {heart_tempo:.0f} → {target_heartbeat_tempo:.0f} BPM")

        # Normalize heartbeat — pydub direct WAV load (0 subprocess)
        # stretched_path is WAV output from time_stretch_heartbeat
        picked_seg = AudioSegment.from_file(stretched_path, format='wav').normalize()
        if picked_seg.dBFS < -25: picked_seg += 3
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v4] Normalized stretched heartbeat: duration={heart_len_s:.1f}s")

        # Đo volume — numpy direct (0 subprocess)
        vol_picked = fast_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v4] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]volume={safe_db(max(0, -diff + 2))}dB[a0];"
        # [SỬA LỖI] Phải dùng aloop để lặp lại heartbeat theo chiều dài bài hát
        picked_filter = f"[1:a]volume={safe_db(max(0, diff))}dB,aloop=loop=-1:size=2e+09[a1];"

        # [SỬA LỖI] duration=first → dài bằng nhạc nền
        # weights=0.75 0.25 → cân bằng nhạc + heartbeat nhanh
        enc = codec_args(mixed_temp_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2"
            f":weights=0.7 0.3[a]"
        )
        # FFmpeg call #2 — mix
        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{mixed_temp_path}"')

        # [ĐẶC TRƯNG V4] 432Hz tuning — ấm áp + nhịp nhanh
        # FFmpeg call #3 — 432Hz pitch shift
        if not (os.path.exists(mixed_temp_path) and os.path.getsize(mixed_temp_path) > 0):
            logger.error(f"[v4] mixed_temp.mp3 is empty or missing, aborting 432Hz step.")
        else:
            tune_to_432hz(mixed_temp_path, output_path)
        logger.info(f"[v4] Finished successfully -> {output_path}")
    except Exception as e:
        logger.error(f"[v4] Error during mix_audio_v4: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()