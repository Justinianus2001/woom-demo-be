import subprocess
import os
import tempfile
import librosa
from pydub import AudioSegment
import numpy as np
from scipy import signal
import soundfile as sf

def calculate_duration_from_analysis(picked_audio, num_beats=4):
    """Phân tích file để lấy duration chính xác cho N nhịp tim."""
    try:
        y, sr = librosa.load(picked_audio, sr=None, duration=30.0)
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
        print(f"❌ Phân tích thất bại: {e}")
    return None, 120.0

def detect_tempo(audio_path):
    """Tự detect tempo của file audio dùng Librosa."""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=60.0)
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
        print(f"❌ Detect tempo thất bại: {e}")
        return 120.0

def get_mean_volume(audio_path):
    """Đo mean volume (dBFS) dùng PyDub."""
    try:
        audio = AudioSegment.from_file(audio_path)
        return audio.dBFS
    except Exception as e:
        print(f"❌ Đo volume thất bại: {e}")
        return -16.0

def run_ffmpeg(command):
    """Chạy FFmpeg command và check success."""
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"❌ FFmpeg failed: {process.stderr}")
        return False
    return True


def adjust_bpm(input_path: str, output_path: str, speed_mode: str):
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
    speed = max(0.1, min(10.0, speed))

    print(f"Adjusting BPM: Mode='{speed_mode}', Factor={speed}")
    cmd = f'ffmpeg -y -i "{input_path}" -af "atempo={speed}" "{output_path}"'
    if not run_ffmpeg(cmd):
        # copy through if atempo fails
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')

def apply_noise_reduction(y, sr):
    """Sử dụng HPSS từ Librosa để tách percussive (nhịp tim)."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_percussive

def tune_to_432hz(input_path, output_path):
    """Pitch shift toàn bộ audio xuống 432Hz tuning từ 440Hz dùng FFmpeg."""
    # asetrate changes pitch and speed, atempo corrects the speed back.
    # 432/440 = 0.981818... and 440/432 = 1.018518...
    cmd = f'ffmpeg -y -i "{input_path}" -af "asetrate=44100*432/440,aresample=44100,atempo=1.0185185185185186" "{output_path}"'
    run_ffmpeg(cmd)

def time_stretch_heartbeat(input_path, output_path, target_tempo, original_tempo):
    """Stretch nhịp tim dùng FFmpeg atempo."""
    if original_tempo <= 0 or target_tempo <= 0:
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')
        return

    rate = target_tempo / original_tempo
    if rate <= 0 or np.isinf(rate) or np.isnan(rate):
        rate = 1.0
    
    # Cap rate for stability
    rate = max(0.3, min(3.0, rate))

    stretch_cmd = f'ffmpeg -y -i "{input_path}" -filter:a "atempo={rate}" "{output_path}"'
    if not run_ffmpeg(stretch_cmd):
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')

def mix_audio_v1(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120):
    """Version 1: Basic mixing with volume balancing and trimming."""
    tempo_factor = original_bpm / target_bpm
    duration_seconds, _ = calculate_duration_from_analysis(picked_audio)

    if duration_seconds is None:
        duration_seconds = 4 * (60.0 / original_bpm)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        filtered_path = os.path.join(temp_dir, 'picked_filtered.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -ac 2 -ar 44100 "{temp_wav_path}"')

        # Filter
        y, sr = sf.read(temp_wav_path)
        if y.ndim == 1: y = y[:, np.newaxis]
        nyq = 0.5 * sr
        low = 500 / nyq
        b, a = signal.butter(5, low, btype='low')
        padlen = 3 * (max(len(b), len(a)) - 1)
        if y.shape[0] > padlen:
            y_filtered = signal.filtfilt(b, a, y, axis=0)
        else:
            y_filtered = y
        if len(y_filtered.shape) == 2 and y_filtered.shape[1] == 1:
            y_filtered = y_filtered.squeeze()
        sf.write(filtered_path, y_filtered, sr)

        # Silence remove (using named parameters for compatibility with FFmpeg 7.x)
        run_ffmpeg(f'ffmpeg -y -i "{filtered_path}" -af silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:detection=peak "{silenced_path}"')
        
        # Trim
        run_ffmpeg(f'ffmpeg -y -i "{silenced_path}" -t {duration_seconds} "{normalized_picked_path}"')
        if not os.path.exists(normalized_picked_path) or os.path.getsize(normalized_picked_path) == 0:
            run_ffmpeg(f'ffmpeg -y -i "{filtered_path}" -t {duration_seconds} "{normalized_picked_path}"')
        
        # Normalize Picked
        picked_audio_seg = AudioSegment.from_file(normalized_picked_path).normalize()
        if picked_audio_seg.dBFS < -50: picked_audio_seg += 10
        picked_audio_seg.export(normalized_picked_path, format="wav")

        # Normalize Asset
        run_ffmpeg(f'ffmpeg -y -i "{asset_audio}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        
        if diff > 0:
            asset_filter = f"[0:a]atempo={tempo_factor}[a0];"
            picked_filter = f"[1:a]volume={diff}dB,aloop=loop=-1:size=2e+09[a1];"
        else:
            asset_filter = f"[0:a]atempo={tempo_factor},volume={abs(diff)}dB[a0];"
            picked_filter = f"[1:a]aloop=loop=-1:size=2e+09[a1];"

        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{picked_filter} {asset_filter} [a0][a1]amix=inputs=2:duration=first:dropout_transition=2:weights=0.6 0.4[a]" -map "[a]" -c:a libmp3lame -q:a 2 "{output_path}"')

def mix_audio_v2(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120):
    """Version 2: HPSS, dynamic threshold, tune to 432Hz."""
    tempo_factor = original_bpm / target_bpm
    duration_seconds, _ = calculate_duration_from_analysis(picked_audio)
    if duration_seconds is None:
        duration_seconds = 4 * (60.0 / original_bpm) + 0.5
    else:
        duration_seconds += 0.5

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        trimmed_path = os.path.join(temp_dir, 'picked_trimmed.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        stretched_asset_path = os.path.join(temp_dir, 'asset_stretched.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')

        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -ac 2 -ar 44100 "{temp_wav_path}"')

        # HPSS
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        y_denoised = apply_noise_reduction(y, sr)
        sf.write(denoised_path, y_denoised, sr)

        # Dynamic Threshold (using named parameters for compatibility with FFmpeg 7.x)
        max_val = np.max(np.abs(y_denoised)) if len(y_denoised) > 0 else 0
        if max_val > 0:
            peak_db = librosa.amplitude_to_db(max_val)
            threshold_db = max(-50, peak_db - 30)
        else:
            threshold_db = -50
        run_ffmpeg(f'ffmpeg -y -i "{denoised_path}" -af silenceremove=start_periods=1:start_duration=0:start_threshold={threshold_db}dB:detection=peak "{silenced_path}"')
        
        run_ffmpeg(f'ffmpeg -y -i "{silenced_path}" -t {duration_seconds} "{trimmed_path}"')
        if not os.path.exists(trimmed_path) or os.path.getsize(trimmed_path) == 0:
            run_ffmpeg(f'ffmpeg -y -i "{denoised_path}" -t {duration_seconds} "{trimmed_path}"')
        
        # Normalize Picked
        picked_seg = AudioSegment.from_file(trimmed_path).normalize()
        if picked_seg.dBFS < -20: picked_seg += 6
        picked_seg.export(normalized_picked_path, format="wav")

        # Asset Stretch
        if tempo_factor != 1.0:
            rate = max(0.5, min(2.0, tempo_factor))
            run_ffmpeg(f'ffmpeg -y -i "{asset_audio}" -filter:a "atempo={rate}" "{stretched_asset_path}"')
        else:
            # Just copy if no stretch needed to save processing time
            run_ffmpeg(f'ffmpeg -y -i "{asset_audio}" -c copy "{stretched_asset_path}"')

        run_ffmpeg(f'ffmpeg -y -i "{stretched_asset_path}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        asset_filter = f"[0:a]volume={max(0, -diff)}dB[a0];"
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{asset_filter}{picked_filter}[a0][a1]amix=inputs=2:duration=first:dropout_transition=3:weights=0.8 0.2[a]" -map "[a]" -c:a libmp3lame -q:a 2 "{mixed_temp_path}"')
        tune_to_432hz(mixed_temp_path, output_path)

def mix_audio_v3(asset_audio, picked_audio, output_path):
    """Version 3: Detect tempo, stretch heartbeat to match music tempo."""
    duration_seconds, heart_tempo = calculate_duration_from_analysis(picked_audio, num_beats=4)
    if heart_tempo <= 0: heart_tempo = 120.0
    if duration_seconds is None:
        duration_seconds = 4 * (60.0 / heart_tempo) + 0.5
    music_tempo = detect_tempo(asset_audio)
    if music_tempo <= 0: music_tempo = 120.0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        stretched_path = os.path.join(temp_dir, 'picked_stretched.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -ac 1 -ar 44100 "{temp_wav_path}"')

        # HPSS
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        y_denoised = apply_noise_reduction(y, sr)
        sf.write(denoised_path, y_denoised, sr)

        time_stretch_heartbeat(denoised_path, stretched_path, music_tempo, heart_tempo)

        # Trim & Normalize
        picked_seg = AudioSegment.from_file(stretched_path)
        adjusted_duration = duration_seconds * (heart_tempo / music_tempo)
        picked_seg = picked_seg[:int(adjusted_duration * 1000)].normalize() - 14
        picked_seg.export(normalized_picked_path, format="wav")

        run_ffmpeg(f'ffmpeg -y -i "{asset_audio}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        asset_filter = f"[0:a]volume={max(0, -diff + 2)}dB[a0];"
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{asset_filter}{picked_filter}[a0][a1]amix=inputs=2:duration=first:dropout_transition=3:weights=0.8 0.2[a]" -map "[a]" -c:a libmp3lame -q:a 2 "{output_path}"')

def mix_audio_v4(asset_audio, picked_audio, output_path):
    """Version 4: Stretch heartbeat to 2x music tempo, 432Hz tuning."""
    duration_seconds, heart_tempo = calculate_duration_from_analysis(picked_audio, num_beats=4)
    if heart_tempo <= 0: heart_tempo = 120.0
    if duration_seconds is None:
        duration_seconds = 4 * (60.0 / heart_tempo) + 0.5
    music_tempo = detect_tempo(asset_audio)
    if music_tempo <= 0: music_tempo = 120.0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        stretched_path = os.path.join(temp_dir, 'picked_stretched.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')

        run_ffmpeg(f'ffmpeg -y -i "{picked_audio}" -ac 1 -ar 44100 "{temp_wav_path}"')

        # HPSS
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        y_denoised = apply_noise_reduction(y, sr)
        sf.write(denoised_path, y_denoised, sr)

        target_heartbeat_tempo = music_tempo * 2
        time_stretch_heartbeat(denoised_path, stretched_path, target_heartbeat_tempo, heart_tempo)

        # Trim & Normalize
        picked_seg = AudioSegment.from_file(stretched_path)
        adjusted_duration_ms = (4 * (60.0 / target_heartbeat_tempo)) * 1000
        picked_seg = picked_seg[:int(adjusted_duration_ms)].normalize()
        if picked_seg.dBFS < -25: picked_seg += 3
        picked_seg.export(normalized_picked_path, format="wav")

        run_ffmpeg(f'ffmpeg -y -i "{asset_audio}" -ar 44100 -ac 2 -af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"')

        # Mix
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        diff = vol_asset - vol_picked
        asset_filter = f"[0:a]volume={max(0, -diff + 2)}dB[a0];"
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        run_ffmpeg(f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" -filter_complex "{asset_filter}{picked_filter}[a0][a1]amix=inputs=2:duration=first:dropout_transition=3:weights=0.75 0.25[a]" -map "[a]" -c:a libmp3lame -q:a 2 "{mixed_temp_path}"')
        tune_to_432hz(mixed_temp_path, output_path)
