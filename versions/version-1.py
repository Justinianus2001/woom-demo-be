import subprocess
import os
import tempfile
import librosa
from pydub import AudioSegment
import numpy as np
from scipy import signal
import soundfile as sf

def calculate_duration_from_analysis(picked_audio):
    """Phân tích file để lấy duration chính xác cho 4 nhịp tim (dùng Librosa)."""
    try:
        y, sr = librosa.load(picked_audio, sr=None)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) >= 5:  # Cần ít nhất 5 beats để có 4 intervals
            duration = librosa.frames_to_time(beats[4] - beats[0], sr=sr)
            return duration
    except Exception as e:
        print(f"❌ Phân tích thất bại: {e}")
    return None

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

def mix_audio(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120):
    """Mix audio cải tiến: Giảm threshold silence, tăng cutoff filter, fallback normalize nếu volume thấp."""
    print("🔎 Đang phân tích file để tìm 4 nhịp tim chính xác...")
    tempo_factor = original_bpm / target_bpm
    analyzed_duration = calculate_duration_from_analysis(picked_audio)

    if analyzed_duration is not None:
        duration_seconds = analyzed_duration
        print(f"✅ PHÂN TÍCH THÀNH CÔNG: Cắt chính xác 4 nhịp = {duration_seconds:.3f}s")
    else:
        duration_seconds = 4 * (60.0 / original_bpm)
        print(f"⚠️ Phân tích thất bại. Dùng công thức chuẩn 4 nhịp/BPM: {duration_seconds:.3f}s")

    print(f"📊 Tempo factor: {tempo_factor}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        filtered_path = os.path.join(temp_dir, 'picked_filtered.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

        # Bước 1: Chuyển đổi picked audio sang WAV (stereo, 44.1kHz)
        print("🔄 Bước 1: Chuyển đổi sang WAV...")
        convert_cmd = f'ffmpeg -y -i "{picked_audio}" -ac 2 -ar 44100 "{temp_wav_path}"'
        if not run_ffmpeg(convert_cmd):
            return

        # Bước 2: Lọc tạp âm (tăng cutoff lên 500Hz để giữ tần số tim thai)
        print("🔊 Bước 2.1: Lọc tạp âm low-pass (cutoff 500Hz)...")
        y, sr = sf.read(temp_wav_path)
        if y.ndim == 1:
            y = y[:, np.newaxis]  # Convert to 2D if mono

        nyq = 0.5 * sr
        low = 500 / nyq  # Tăng cutoff
        b, a = signal.butter(5, low, btype='low')

        padlen = 3 * (max(len(b), len(a)) - 1)
        if y.shape[0] > padlen:
            y_filtered = signal.filtfilt(b, a, y, axis=0)
        else:
            print(f"⚠️ Input too short ({y.shape[0]} samples <= {padlen}), skipping filter.")
            y_filtered = y

        if y_filtered.shape[1] == 1:
            y_filtered = y_filtered.squeeze()

        sf.write(filtered_path, y_filtered, sr)

        # Bước 2.2: Loại bỏ khoảng lặng (giảm threshold xuống -40dB để giữ âm yếu)
        print("🔊 Bước 2.2: Loại bỏ khoảng lặng đầu (-40dB)...")
        silence_cmd = (
            f'ffmpeg -y -i "{filtered_path}" '
            f'-af silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:detection=peak '
            f'"{silenced_path}"'
        )
        if not run_ffmpeg(silence_cmd):
            return

        # Bước 2.3: Cắt 4 nhịp
        print("🔊 Bước 2.3: Cắt đúng 4 nhịp...")
        trim_cmd = f'ffmpeg -y -i "{silenced_path}" -t {duration_seconds} "{normalized_picked_path}"'  # Chưa normalize
        if not run_ffmpeg(trim_cmd):
            return

        if not os.path.exists(normalized_picked_path) or os.path.getsize(normalized_picked_path) == 0:
            print("❌ Trimmed file is empty, fallback to no silence remove.")
            # Fallback: Trim from filtered without silence remove
            fallback_trim_cmd = f'ffmpeg -y -i "{filtered_path}" -t {duration_seconds} "{normalized_picked_path}"'
            run_ffmpeg(fallback_trim_cmd)

        # Bước 2.4: Chuẩn hóa picked dùng PyDub (peak normalize, tránh issue loudnorm với file ngắn)
        print("🔊 Bước 2.4: Chuẩn hóa âm lượng picked (PyDub normalize)...")
        picked_audio_seg = AudioSegment.from_file(normalized_picked_path)
        picked_audio_seg = picked_audio_seg.normalize()  # Peak normalize to 0dBFS
        picked_audio_seg.export(normalized_picked_path, format="wav")

        vol_picked_check = picked_audio_seg.dBFS
        print(f"📊 Picked volume after normalize: {vol_picked_check} dB")
        if np.isinf(vol_picked_check) or vol_picked_check < -50:
            print("⚠️ Volume vẫn thấp, boost thêm +10dB.")
            picked_audio_seg = picked_audio_seg + 10
            picked_audio_seg.export(normalized_picked_path, format="wav")

        # Bước 3: Chuẩn hóa asset audio (giữ loudnorm vì file dài)
        print("🔊 Bước 3: Chuẩn hóa âm lượng asset audio...")
        normalize_asset_cmd = (
            f'ffmpeg -y -i "{asset_audio}" -ar 44100 -ac 2 '
            f'-af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"'
        )
        if not run_ffmpeg(normalize_asset_cmd):
            return

        # Bước 4: Mix (Điều chỉnh volume dựa trên diff, loop picked, amix hài hòa)
        print("🎵 Bước 4: Mix audio (Tỉ lệ 0.6:0.4 để tim thai rõ hơn) - Balancing volumes...")
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        print(f"📊 Post-norm Volumes -> Asset: {vol_asset} dB, Picked: {vol_picked} dB")

        diff = vol_asset - vol_picked
        asset_filter = ""
        picked_filter = ""
        boost = 0

        if diff > 0:
            print(f"💡 Asset louder by {diff:.2f}dB -> Boosting Picked")
            boost = diff
            asset_filter = f"[0:a]atempo={tempo_factor}[a0];"
            picked_filter = f"[1:a]volume={boost}dB,aloop=loop=-1:size=2e+09[a1];"
        else:
            boost = abs(diff)
            print(f"💡 Picked louder by {boost:.2f}dB -> Boosting Asset")
            asset_filter = f"[0:a]atempo={tempo_factor},volume={boost}dB[a0];"
            picked_filter = f"[1:a]aloop=loop=-1:size=2e+09[a1];"

        # Mix với weights 0.6:0.4 (tăng phần tim thai để rõ hơn), thêm fade
        mix_cmd = (
            f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" '
            f'-filter_complex "{picked_filter} {asset_filter} '
            f'[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:weights=0.6 0.4[a]" '
            f'-map "[a]" -c:a libmp3lame -q:a 2 "{output_path}"'
        )
        if run_ffmpeg(mix_cmd):
            print(f"✅ Mixing successful! File saved at {output_path}")
        else:
            print("❌ Mixing failed")

# Usage example (thay bằng paths thực tế)
mix_audio("twinkle_star.mp3", "Heartbeat5_bpm140.wav", "demo_version_1.mp3")