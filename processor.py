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

logger = logging.getLogger(__name__)

def _librosa_load_safe(audio_path: str, duration: float = 30.0):
    """Load audio via librosa, falling back to a temp WAV conversion if the
    file format is not recognised by libsndfile/audioread."""
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
        # Try with explicit WAV demuxer first, then raw f32le
        converted = False
        for extra in ["-f wav", "", "-f f32le -ar 44100 -ac 2", "-f s16le -ar 44100 -ac 2"]:
            cmd = f'ffmpeg -y {extra} -i "{audio_path}" -ar 44100 -ac 1 -sample_fmt s16 "{tmp_path}"'
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

    Some track files shipped with the app use non-standard WAV variants
    (Wave64 / RF64 / float-32 PCM / ADPCM / etc.) that libsndfile and newer
    FFmpeg refuse to auto-detect.  We attempt several fallback strategies:

    1. Auto-detect (let FFmpeg probe normally)
    2. Force -f wav  (explicit demuxer, handles RF64/W64 mis-labelled as wav)
    3. Force -f f32le / f32be  (float-32 raw PCM, no header)
    4. Force -f s16le / s16be  (16-bit signed raw PCM, no header)
    5. Force -f s24le          (24-bit signed raw PCM)

    Returns True if any strategy succeeded and `output_path` was created.
    """
    strategies = [
        # (label, extra input flags)
        ("auto",   ""),
        ("wav",    "-f wav"),
        ("f32le",  "-f f32le -ar 44100 -ac 2"),
        ("f32be",  "-f f32be -ar 44100 -ac 2"),
        ("s16le",  "-f s16le -ar 44100 -ac 2"),
        ("s16be",  "-f s16be -ar 44100 -ac 2"),
        ("s24le",  "-f s24le -ar 44100 -ac 2"),
    ]
    for label, extra in strategies:
        cmd = f'ffmpeg -y {extra} -i "{asset_audio}" -ar 44100 -ac 2 -sample_fmt s16 "{output_path}"'
        if run_ffmpeg(cmd):
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"[preconvert_asset] Success with strategy '{label}'")
                return True
            # file created but empty – remove and try next strategy
            try:
                os.unlink(output_path)
            except OSError:
                pass
        else:
            # ensure partial file is cleaned up
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
    logger.error(f"[preconvert_asset] All strategies failed for '{asset_audio}'")
    return False

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

def mix_audio_v1(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120, heart_duration=None):
    """Version 1: Pure Natural Mix — Low-pass filter, heartbeat nổi bật nhất.

    ĐẶC TRƯNG V1 (phân biệt với v2/v3/v4):
    - Low-pass filter 500Hz → âm deep/bass đặc trưng nhịp tim (v2-v4 dùng HPSS)
    - weights 0.5/0.5 → heartbeat nổi bật nhất trong 4 version
    - Không 432Hz tuning → giữ nguyên tần số gốc

    THAY ĐỔI 03/2026 (feedback khách hàng — nhịp tim không tự nhiên):
    - BỎ: Cắt 4 nhịp + aloop vô hạn → gây nhịp tim máy móc, lặp lại
    - BỎ: atempo trên nhạc nền → nhạc giữ nguyên tempo gốc
    - THÊM: Dùng toàn bộ heartbeat (≤30s), duration=shortest, fade-out cuối
    """
    logger.info(f"[v1] Starting mix_audio_v1 for picked='{picked_audio}', asset='{asset_audio}', output='{output_path}'")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        filtered_path = os.path.join(temp_dir, 'picked_filtered.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

        # === HEARTBEAT: Giữ nguyên tự nhiên, chỉ lọc noise ===
        # Giới hạn 30s tránh OOM — giữ toàn bộ recording, KHÔNG cắt 4 nhịp
        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 2 -ar 44100 "{temp_wav_path}"')
        logger.info(f"[v1] Converted picked audio: {temp_wav_path}")

        # [ĐẶC TRƯNG V1] Low-pass filter 500Hz (Butterworth bậc 5)
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

        # Loại bỏ silence đầu file
        run_ffmpeg(f'ffmpeg -y -i "{filtered_path}" -af silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:detection=peak "{silenced_path}"')
        logger.info(f"[v1] Removed leading silence: {silenced_path}")

        # [THAY ĐỔI] BỎ trim 4 nhịp — dùng toàn bộ heartbeat tự nhiên
        picked_seg = safe_ffmpeg_load(silenced_path).normalize()
        if picked_seg.dBFS < -50: picked_seg += 10
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v1] Normalized heartbeat: duration={heart_len_s:.1f}s")

        # === NHẠC NỀN: Giữ nguyên tempo, chỉ normalize -> bỏ tempo
        # Pre-convert asset to standard PCM WAV first (handles Wave64/RF64/float variants)
        raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
        if not preconvert_asset(asset_audio, raw_asset_path):
            logger.error(f"[v1] Cannot decode asset audio '{asset_audio}', aborting.")
            return
        run_ffmpeg(f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
        logger.info(f"[v1] Normalized asset: {normalized_asset_path}")

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v1] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]volume={max(0, -diff)}dB[a0];"
        # [SỬA LỖI] Dùng aloop để loop TOÀN BỘ đoạn heartbeat tự nhiên, giúp mix đủ dài bằng nhạc nền
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        # [SỬA LỖI] duration=first → output = độ dài nhạc nền (a0), amix tự ngắt khi nhạc nền hết
        # [ĐẶC TRƯNG V1] weights=0.35 0.65 → heartbeat nổi bật nhất trong 4 version
        enc = codec_args(output_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2"
            f":weights=0.35 0.65[a]"  # Nhịp tim (0.65) to và rõ bám sát tai, nhạc nền nhẹ hơn (0.35)
        )
        if run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{output_path}"'):
            logger.info(f"[v1] Finished successfully -> {output_path}")
        else:
            logger.error(f"[v1] mix_audio_v1 failed at the final step")
    except Exception as e:
        logger.error(f"[v1] Error during mix_audio_v1: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()

def mix_audio_v2(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120, heart_duration=None):
    """Version 2: Clean Heartbeat + 432Hz — HPSS separation, ấm áp hơn.

    ĐẶC TRƯNG V2 (phân biệt với v1/v3/v4):
    - HPSS: Tách percussive → heartbeat sạch, rõ từng nhịp (v1 dùng low-pass)
    - Dynamic silence threshold → loại noise chính xác hơn fixed -40dB của v1
    - 432Hz tuning → tần số ấm áp, thư giãn hơn (v1/v3 không có)
    - weights 0.65/0.35 → heartbeat rõ nhưng nhạc nổi hơn v1

    THAY ĐỔI:
    - BỎ: Cắt 4 nhịp + aloop vô hạn → gây nhịp tim máy móc
    - BỎ: Asset stretch (tempo_factor luôn = 1.0, code thừa)
    - THÊM: Dùng toàn bộ heartbeat (≤30s), duration=shortest, fade-out cuối
    """
    logger.info(f"[v2] Starting mix_audio_v2 for picked='{picked_audio}', asset='{asset_audio}', output='{output_path}'")
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name
    try:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')

        # Limit to 30 seconds to avoid Out-Of-Memory (OOM) on large audio files
        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 2 -ar 44100 "{temp_wav_path}"')
        logger.info(f"[v2] Finished processing picked audio: {temp_wav_path}")

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
        run_ffmpeg(f'ffmpeg -y -i "{denoised_path}" -af silenceremove=start_periods=1:start_duration=0:start_threshold={threshold_db}dB:detection=peak "{silenced_path}"')
        logger.info(f"[v2] Removed silence: {silenced_path}")

        # [THAY ĐỔI] BỎ trim 4 nhịp — giữ toàn bộ heartbeat tự nhiên
        picked_seg = safe_ffmpeg_load(silenced_path).normalize()
        if picked_seg.dBFS < -20: picked_seg += 6
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v2] Normalized heartbeat: duration={heart_len_s:.1f}s")

        # === NHẠC NỀN: Giữ nguyên tempo ===
        # [THAY ĐỔI] BỎ asset stretch — tempo_factor luôn = 1.0 (code thừa)
        # Pre-convert asset to standard PCM WAV first (handles Wave64/RF64/float variants)
        raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
        if not preconvert_asset(asset_audio, raw_asset_path):
            logger.error(f"[v2] Cannot decode asset audio '{asset_audio}', aborting.")
            return
        run_ffmpeg(f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
        logger.info(f"[v2] Normalized asset: {normalized_asset_path}")

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v2] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]volume={max(0, -diff)}dB[a0];"
        # [ĐẶC TRƯNG MỚI] Dùng reverb sâu hơn (500ms delay, decay 0.4) tạo không gian mênh mông như trong hang động
        # Kết hợp highpass(200Hz) nhẹ để gọt bỏ bớt tiếng bụp đục (âm trầm), tiếng đập mảnh hơn, thanh tao hơn
        # [SỬA LỖI] Thêm aloop=loop=-1 ở cuối chuỗi của filter để nhịp tim không chạy hết sớm
        picked_filter = f"[1:a]volume={max(0, diff)}dB,highpass=f=200,aecho=0.8:0.9:500:0.4,aloop=loop=-1:size=2e+09[a1];"

        # [SỬA LỖI] duration=first → output dài bằng nhạc nền (a0)
        # [ĐẶC TRƯNG V2] weights=0.5 0.5 → Nhịp tim vang vọng hòa lẫn sâu vào nhạc, không áp đảo
        enc = codec_args(mixed_temp_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2"
            f":weights=0.5 0.5[a]"
        )
        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{mixed_temp_path}"')

        # [ĐẶC TRƯNG V2] 432Hz tuning — tần số ấm áp, thư giãn hơn 440Hz
        tune_to_432hz(mixed_temp_path, output_path)
        logger.info(f"[v2] Finished successfully -> {output_path}")
    except Exception as e:
        logger.error(f"[v2] Error during mix_audio_v2: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()

def mix_audio_v3(asset_audio, picked_audio, output_path, heart_duration=None, heart_tempo=None, music_tempo=None):
    """Version 3: Tempo-Synced Music — nhạc nền adapt tempo theo heartbeat.

    ĐẶC TRƯNG V3 (phân biệt với v1/v2/v4):
    - NHẠC NỀN được stretch tempo để match BPM của heartbeat → nhịp đồng bộ
    - Heartbeat giữ nguyên BPM gốc → tự nhiên nhất
    - weights 0.8/0.2 → nhạc nổi bật, heartbeat là nhịp nền hài hòa
    - Không 432Hz tuning

    THAY ĐỔI 03/2026 (feedback khách hàng — nhịp tim không tự nhiên):
    - ĐẢO: Stretch NHẠC NỀN thay vì stretch heartbeat
      Lý do: "the music stays the same and the heartbeat gets tuned" — khách hàng
      muốn ngược lại: heartbeat giữ nguyên, nhạc adapt theo
    - BỎ: Cắt 4 nhịp + aloop → nhịp tim máy móc
    - THÊM: duration=shortest, fade-out cuối
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
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        stretched_asset_path = os.path.join(temp_dir, 'asset_stretched.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

        # Limit to 30 seconds to avoid Out-Of-Memory (OOM) on large audio files
        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 1 -ar 44100 "{temp_wav_path}"')
        logger.info(f"[v3] Finished processing picked audio: {temp_wav_path}")

        # HPSS
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        y_denoised = apply_noise_reduction(y, sr)
        logger.info(f"[v3] Finished denoising picked audio: {denoised_path}")
        sf.write(denoised_path, y_denoised, sr)
        logger.info(f"[v3] HPSS denoised: {denoised_path}")

        # [THAY ĐỔI] BỎ time_stretch_heartbeat — giữ heartbeat nguyên BPM
        picked_seg = safe_ffmpeg_load(denoised_path).normalize() - 14
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v3] Normalized heartbeat (no stretch): duration={heart_len_s:.1f}s")

        # === NHẠC NỀN: Stretch tempo để match heartbeat BPM ===
        # [ĐẶC TRƯNG V3] ĐẢO logic: nhạc adapt theo heartbeat, không phải ngược lại
        # VD: heart=70BPM, music=120BPM → rate=70/120=0.58 → nhạc chậm lại match nhịp tim
        tempo_rate = heart_tempo / music_tempo
        tempo_rate = max(0.5, min(2.0, tempo_rate))  # Clamp tránh FFmpeg atempo quá extreme
        logger.info(f"[v3] Stretching music: rate={tempo_rate:.3f} (heart={heart_tempo:.0f} / music={music_tempo:.0f})")

        # Pre-convert asset to standard PCM WAV first (handles Wave64/RF64/float variants)
        raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
        if not preconvert_asset(asset_audio, raw_asset_path):
            logger.error(f"[v3] Cannot decode asset audio '{asset_audio}', aborting.")
            return
        atempo_str = get_atempo_filter(tempo_rate)
        run_ffmpeg(f'ffmpeg -y -i "{raw_asset_path}" -filter:a "{atempo_str}" "{stretched_asset_path}"')
        run_ffmpeg(f'ffmpeg -y -i "{stretched_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
        logger.info(f"[v3] Normalized stretched asset: {normalized_asset_path}")

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v3] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]volume={max(0, -diff + 2)}dB[a0];"
        # [ĐẶC TRƯNG MỚI] Tăng Treble (high EQ) cho nhịp tim để âm thanh sắc nét, dập rõ tiết tấu
        # [SỬA LỖI] Thêm aloop=loop=-1
        picked_filter = f"[1:a]volume={max(0, diff)}dB,treble=g=5:f=1000,aloop=loop=-1:size=2e+09[a1];"

        # [SỬA LỖI] duration=first → output dài bằng nhạc nền (a0)
        # [ĐẶC TRƯNG V3] weights=0.7 0.3 → nhạc nổi bật, heartbeat là nhịp nền đồng bộ
        enc = codec_args(output_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2"
            f":weights=0.7 0.3[a]"
        )
        if run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{output_path}"'):
            logger.info(f"[v3] Finished successfully -> {output_path}")
        else:
            logger.error(f"[v3] mix_audio_v3 failed at final stage")
    except Exception as e:
        logger.error(f"[v3] Error during mix_audio_v3: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()

def mix_audio_v4(asset_audio, picked_audio, output_path, heart_duration=None, heart_tempo=None, music_tempo=None):
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
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        stretched_path = os.path.join(temp_dir, 'picked_stretched.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')

        # Limit to 30 seconds to avoid Out-Of-Memory (OOM) on large audio files
        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -t 30 -ac 1 -ar 44100 "{temp_wav_path}"')
        logger.info(f"[v4] Finished processing picked audio: {temp_wav_path}")

        # HPSS
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        logger.info(f"[v4] Finished reading picked audio: {temp_wav_path}")
        y_denoised = apply_noise_reduction(y, sr)
        sf.write(denoised_path, y_denoised, sr)
        logger.info(f"[v4] Finished denoising picked audio: {denoised_path}")

        # [GIỮ NGUYÊN] Stretch heartbeat lên 2x music tempo — khách hàng cho phép v4
        time_stretch_heartbeat(denoised_path, stretched_path, target_heartbeat_tempo, heart_tempo)
        logger.info(f"[v4] Stretched heartbeat: {heart_tempo:.0f} → {target_heartbeat_tempo:.0f} BPM")

        # [THAY ĐỔI] BỎ trim 4 nhịp — dùng toàn bộ heartbeat đã stretch
        # Lý do: Cắt 4 nhịp rồi loop → nhịp tim máy móc, lặp lại
        picked_seg = safe_ffmpeg_load(stretched_path).normalize()
        if picked_seg.dBFS < -25: picked_seg += 3
        picked_seg.export(normalized_picked_path, format="wav")
        heart_len_s = len(picked_seg) / 1000.0
        logger.info(f"[v4] Normalized stretched heartbeat: duration={heart_len_s:.1f}s")

        # === NHẠC NỀN: Giữ nguyên tempo ===
        # Pre-convert asset to standard PCM WAV first (handles Wave64/RF64/float variants)
        raw_asset_path = os.path.join(temp_dir, 'asset_raw.wav')
        if not preconvert_asset(asset_audio, raw_asset_path):
            logger.error(f"[v4] Cannot decode asset audio '{asset_audio}', aborting.")
            return
        run_ffmpeg(f'ffmpeg -y -i "{raw_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')
        logger.info(f"[v4] Normalized asset: {normalized_asset_path}")

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        logger.info(f"[v4] Volume: asset={vol_asset:.1f}dB, picked={vol_picked:.1f}dB")

        asset_filter = f"[0:a]volume={max(0, -diff + 2)}dB[a0];"
        # [SỬA LỖI] Phải dùng aloop để lặp lại heartbeat theo chiều dài bài hát
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        # [SỬA LỖI] duration=first → dài bằng nhạc nền
        # weights=0.75 0.25 → cân bằng nhạc + heartbeat nhanh
        enc = codec_args(mixed_temp_path)
        mix_filter = (
            f"{asset_filter}{picked_filter}"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2"
            f":weights=0.7 0.3[a]"
        )
        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{mix_filter}" -map "[a]" {enc} "{mixed_temp_path}"')

        # [ĐẶC TRƯNG V4] 432Hz tuning — ấm áp + nhịp nhanh
        tune_to_432hz(mixed_temp_path, output_path)
        logger.info(f"[v4] Finished successfully -> {output_path}")
    except Exception as e:
        logger.error(f"[v4] Error during mix_audio_v4: {e}\n{traceback.format_exc()}")
        raise
    finally:
        temp_dir_obj.cleanup()

