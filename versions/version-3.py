import subprocess
import os
import tempfile
import librosa
from pydub import AudioSegment
import numpy as np
import soundfile as sf

def calculate_duration_from_analysis(picked_audio, num_beats=4):
    """Phân tích để lấy duration cho N nhịp tim (giảm density)."""
    try:
        y, sr = librosa.load(picked_audio, sr=None, duration=30.0)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if tempo.size > 0 else 120.0
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
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if tempo.size > 0 else 120.0
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

def apply_noise_reduction(y, sr):
    """Sử dụng HPSS từ Librosa để tách percussive (nhịp tim)."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_percussive

def time_stretch_heartbeat(input_path, output_path, target_tempo, original_tempo):
    """Stretch nhịp tim dùng FFmpeg atempo để tránh bug librosa."""
    if original_tempo <= 0 or target_tempo <= 0:
        print("⚠️ Tempo không hợp lệ, copy nguyên.")
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')
        return

    rate = target_tempo / original_tempo  # <1 để slow down nếu tim nhanh
    if rate <= 0 or np.isinf(rate) or np.isnan(rate):
        rate = 1.0

    stretch_cmd = f'ffmpeg -y -i "{input_path}" -filter:a "atempo={rate}" "{output_path}"'
    if not run_ffmpeg(stretch_cmd):
        print("⚠️ Stretch thất bại, copy nguyên.")
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')

def mix_audio(asset_audio, picked_audio, output_path):
    """Mix cải tiến: Tự detect tempo, stretch tim khớp tempo nhạc, tỉ lệ 0.8:0.2."""
    print("🔎 Phân tích nhịp tim...")
    duration_seconds, heart_tempo = calculate_duration_from_analysis(picked_audio, num_beats=4)
    if heart_tempo <= 0: heart_tempo = 120.0
    if duration_seconds is None:
        duration_seconds = 4 * (60.0 / heart_tempo) + 0.5

    music_tempo = detect_tempo(asset_audio)
    if music_tempo <= 0: music_tempo = 120.0
    print(f"📊 Heart BPM: {heart_tempo}, Music BPM: {music_tempo}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        stretched_path = os.path.join(temp_dir, 'picked_stretched.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')

        # Bước 1: Convert to WAV (mono cho nhịp tim)
        print("🔄 Bước 1: Chuyển đổi nhịp tim sang WAV (mono)...")
        convert_cmd = f'ffmpeg -y -i "{picked_audio}" -ac 1 -ar 44100 "{temp_wav_path}"'
        if not run_ffmpeg(convert_cmd):
            return

        # Bước 2: Khử tạp âm HPSS
        print("🔊 Bước 2: Khử tạp âm HPSS...")
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y_denoised = apply_noise_reduction(y, sr)
        sf.write(denoised_path, y_denoised, sr)

        # Bước 3: Stretch dùng FFmpeg
        print("🔊 Bước 3: Stretch nhịp tim để khớp tempo...")
        time_stretch_heartbeat(denoised_path, stretched_path, music_tempo, heart_tempo)

        # Bước 4: Trim & Normalize picked
        print("🔊 Bước 4: Cắt & chuẩn hóa nhịp tim...")
        picked_seg = AudioSegment.from_file(stretched_path)
        adjusted_duration = duration_seconds * (heart_tempo / music_tempo)  # Adjust sau stretch
        picked_seg = picked_seg[:int(adjusted_duration * 1000)]
        picked_seg = picked_seg.normalize() - 14  # Giảm vừa phải để phù hợp tỉ lệ 0.8:0.2
        picked_seg.export(normalized_picked_path, format="wav")

        # Bước 5: Normalize asset
        print("🔊 Bước 5: Chuẩn hóa âm lượng nhạc...")
        normalize_asset_cmd = (
            f'ffmpeg -y -i "{asset_audio}" -ar 44100 -ac 2 '
            f'-af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"'
        )
        if not run_ffmpeg(normalize_asset_cmd):
            return

        # Bước 6: Mix với tỉ lệ mới 0.8 (nhạc) : 0.2 (tim)
        print("🎵 Bước 6: Mix với tỉ lệ 0.8:0.2 (nhạc : tim)...")
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        print(f"📊 Volumes → Asset: {vol_asset:.2f} dB, Picked: {vol_picked:.2f} dB")

        diff = vol_asset - vol_picked
        asset_filter = f"[0:a]volume={max(0, -diff + 2)}dB[a0];"
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        mix_cmd = (
            f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" '
            f'-filter_complex "{asset_filter}{picked_filter}[a0][a1]amix=inputs=2:duration=first:dropout_transition=3:weights=0.8 0.2[a]" '
            f'-map "[a]" -c:a libmp3lame -q:a 2 "{output_path}"'
        )
        if not run_ffmpeg(mix_cmd):
            return

        print(f"✅ Mixing hoàn tất! File output: {output_path}")

# Sử dụng với file của bạn
mix_audio("twinkle_star.mp3", "Heartbeat5_bpm140.wav", "demo_version_3.mp3")