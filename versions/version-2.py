import subprocess
import os
import tempfile
import librosa
from pydub import AudioSegment
import numpy as np
from scipy import signal
import soundfile as sf
import torch
import torchaudio  # Sử dụng để noise reduction nếu có model, nhưng fallback HPSS

def calculate_duration_from_analysis(picked_audio):
    """Phân tích file để lấy duration chính xác cho 4 nhịp tim (dùng Librosa)."""
    try:
        y, sr = librosa.load(picked_audio, sr=None)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) >= 5:
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

def apply_noise_reduction(y, sr):
    """Cải tiến: Sử dụng HPSS từ Librosa để tách percussive (nhịp tim) từ harmonic (noise nước ối)."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_percussive  # Giữ percussive là nhịp tim đập

def match_tempo(asset_path, tempo_factor, output_path):
    """Cải tiến: Time-stretch asset để khớp tempo chính xác dùng Librosa."""
    y, sr = librosa.load(asset_path, sr=None)
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_factor)
    sf.write(output_path, y_stretched, sr)
    return output_path

def tune_to_432hz(input_path, output_path):
    """Cải tiến: Pitch shift toàn bộ audio xuống 432Hz tuning từ 440Hz."""
    y, sr = librosa.load(input_path, sr=None)
    n_steps = 12 * np.log2(432 / 440)  # ≈ -0.3176 semitones
    y_tuned = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(output_path, y_tuned, sr)

def mix_audio(asset_audio, picked_audio, output_path, original_bpm=120, target_bpm=120):
    """Mix audio cải tiến: HPSS khử tạp âm, time-stretch tempo, dynamic threshold, tune to 432Hz."""
    print("🔎 Đang phân tích file để tìm 4 nhịp tim chính xác...")
    tempo_factor = original_bpm / target_bpm
    analyzed_duration = calculate_duration_from_analysis(picked_audio)

    if analyzed_duration is not None:
        duration_seconds = analyzed_duration + 0.5  # Thêm buffer
        print(f"✅ PHÂN TÍCH THÀNH CÔNG: Cắt chính xác 4 nhịp = {duration_seconds:.3f}s")
    else:
        duration_seconds = 4 * (60.0 / original_bpm) + 0.5
        print(f"⚠️ Phân tích thất bại. Dùng công thức chuẩn 4 nhịp/BPM: {duration_seconds:.3f}s")

    print(f"📊 Tempo factor: {tempo_factor}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        silenced_path = os.path.join(temp_dir, 'picked_silenced.wav')
        trimmed_path = os.path.join(temp_dir, 'picked_trimmed.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        stretched_asset_path = os.path.join(temp_dir, 'asset_stretched.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')
        tuned_output_path = output_path  # Final tuned

        # Bước 1: Chuyển đổi picked sang WAV
        print("🔄 Bước 1: Chuyển đổi sang WAV...")
        convert_cmd = f'ffmpeg -y -i "{picked_audio}" -ac 2 -ar 44100 "{temp_wav_path}"'
        if not run_ffmpeg(convert_cmd):
            return

        # Bước 2.1: Khử tạp âm dùng HPSS
        print("🔊 Bước 2.1: Khử tạp âm (HPSS tách nhịp tim từ noise nước ối)...")
        y, sr = sf.read(temp_wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # Mono for HPSS
        y_denoised = apply_noise_reduction(y, sr)
        sf.write(denoised_path, y_denoised, sr)

        # Bước 2.2: Loại bỏ khoảng lặng (dynamic threshold)
        print("🔊 Bước 2.2: Loại bỏ khoảng lặng đầu (dynamic threshold)...")
        peak_db = librosa.amplitude_to_db(np.max(np.abs(y_denoised)))
        threshold_db = max(-50, peak_db - 30)
        print(f"Dynamic threshold: {threshold_db}dB")
        silence_cmd = (
            f'ffmpeg -y -i "{denoised_path}" '
            f'-af silenceremove=start_periods=1:start_duration=0:start_threshold={threshold_db}dB:detection=peak '
            f'"{silenced_path}"'
        )
        if not run_ffmpeg(silence_cmd):
            return

        # Bước 2.3: Cắt 4 nhịp
        print("🔊 Bước 2.3: Cắt đúng 4+ nhịp để giữ đầy đủ...")
        trim_cmd = f'ffmpeg -y -i "{silenced_path}" -t {duration_seconds} "{trimmed_path}"'
        if not run_ffmpeg(trim_cmd):
            return

        if os.path.getsize(trimmed_path) == 0:
            print("❌ Trimmed file empty, fallback to no silence remove.")
            fallback_trim_cmd = f'ffmpeg -y -i "{denoised_path}" -t {duration_seconds} "{trimmed_path}"'
            run_ffmpeg(fallback_trim_cmd)

        # Bước 2.4: Chuẩn hóa picked
        print("🔊 Bước 2.4: Chuẩn hóa âm lượng picked...")
        picked_seg = AudioSegment.from_file(trimmed_path)
        picked_seg = picked_seg.normalize()
        if picked_seg.dBFS < -20:
            print("⚠️ Volume thấp, boost +6dB.")
            picked_seg += 6
        picked_seg.export(normalized_picked_path, format="wav")

        # Bước 3: Đồng bộ tempo asset và chuẩn hóa
        print("🔊 Bước 3: Đồng bộ tempo asset và chuẩn hóa...")
        match_tempo(asset_audio, tempo_factor, stretched_asset_path)
        normalize_asset_cmd = (
            f'ffmpeg -y -i "{stretched_asset_path}" -ar 44100 -ac 2 '
            f'-af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"'
        )
        if not run_ffmpeg(normalize_asset_cmd):
            return

        # Bước 4: Mix (tỉ lệ 0.8:0.2 để tim thai làm nền, bỏ reverb để tránh lỗi)
        print("🎵 Bước 4: Mix audio (Tỉ lệ 0.8:0.2 để tim thai làm nền)...")
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        print(f"📊 Post-norm Volumes -> Asset: {vol_asset} dB, Picked: {vol_picked} dB")

        diff = vol_asset - vol_picked
        asset_filter = f"[0:a]volume={max(0, -diff)}dB[a0];"
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        mix_cmd = (
            f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" '
            f'-filter_complex "{asset_filter}{picked_filter}[a0][a1]amix=inputs=2:duration=first:dropout_transition=3:weights=0.8 0.2[a]" '
            f'-map "[a]" -c:a libmp3lame -q:a 2 "{mixed_temp_path}"'
        )
        if run_ffmpeg(mix_cmd):
            print(f"✅ Mixing successful! Tuning to 432Hz...")
            tune_to_432hz(mixed_temp_path, tuned_output_path)
            print(f"✅ Tuned output saved at {output_path}")
        else:
            print("❌ Mixing failed")

# Usage example
mix_audio("twinkle_star.mp3", "Heartbeat5_bpm140.wav", "demo_version_2.mp3")