import subprocess
import os
import tempfile
import librosa
from pydub import AudioSegment
import numpy as np
import soundfile as sf

def calculate_duration_from_analysis(picked_audio, num_beats=4):
    """Phân tích để lấy duration cho N nhịp tim."""
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

def time_stretch_heartbeat(input_path, output_path, target_tempo_for_heartbeat, original_heart_tempo):
    """Stretch nhịp tim dùng FFmpeg atempo để tránh bug librosa."""
    if original_heart_tempo <= 0 or target_tempo_for_heartbeat <= 0:
        print("⚠️ Tempo không hợp lệ, copy nguyên.")
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')
        return

    rate = target_tempo_for_heartbeat / original_heart_tempo
    if rate <= 0 or np.isinf(rate) or np.isnan(rate):
        rate = 1.0

    # Giới hạn tỷ lệ kéo dài để tránh tạo ra quá nhiều artifact
    if rate > 3.0:
        print(f"⚠️ Tốc độ kéo dài quá cao ({rate:.2f}), giới hạn ở 3.0.")
        rate = 3.0
    elif rate < 0.3:
        print(f"⚠️ Tốc độ kéo dài quá thấp ({rate:.2f}), giới hạn ở 0.3.")
        rate = 0.3

    stretch_cmd = f'ffmpeg -y -i "{input_path}" -filter:a "atempo={rate}" "{output_path}"'
    if not run_ffmpeg(stretch_cmd):
        print("⚠️ Stretch thất bại, copy nguyên.")
        run_ffmpeg(f'ffmpeg -y -i "{input_path}" "{output_path}"')

def tune_to_432hz(input_path, output_path):
    cmd = f'ffmpeg -y -i "{input_path}" -af "asetrate=44100*432/440,aresample=44100,atempo=1.0185185185185186" "{output_path}"'
    run_ffmpeg(cmd)

def mix_audio_v4(asset_audio, picked_audio, output_path):
    """Mix cải tiến: Tự detect tempo, stretch tim khớp 2x tempo nhạc, tỉ lệ 0.8:0.2, tinh chỉnh norm, 432Hz tuning."""
    print("🔎 Phân tích nhịp tim...")
    duration_seconds, heart_tempo = calculate_duration_from_analysis(picked_audio, num_beats=4)
    if duration_seconds is None:
        duration_seconds = 4 * (60.0 / heart_tempo) + 0.5

    music_tempo = detect_tempo(asset_audio)
    print(f"📊 Heart BPM: {heart_tempo:.2f}, Music BPM: {music_tempo:.2f}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav_path = os.path.join(temp_dir, 'picked_temp.wav')
        denoised_path = os.path.join(temp_dir, 'picked_denoised.wav')
        stretched_path = os.path.join(temp_dir, 'picked_stretched.wav')
        normalized_picked_path = os.path.join(temp_dir, 'picked_normalized.wav')
        normalized_asset_path = os.path.join(temp_dir, 'asset_normalized.wav')
        mixed_temp_path = os.path.join(temp_dir, 'mixed_temp.mp3')
        tuned_output_path = output_path  # Final tuned

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

        # Bước 3: Stretch nhịp tim để khớp 2x tempo nhạc
        print(f"🔊 Bước 3: Stretch nhịp tim từ {heart_tempo:.2f} BPM để khớp {music_tempo * 2:.2f} BPM của nhạc...")
        target_heartbeat_tempo = music_tempo * 2
        time_stretch_heartbeat(denoised_path, stretched_path, target_heartbeat_tempo, heart_tempo)

        # Bước 4: Trim & Normalize picked (loại bỏ giảm 14dB cố định)
        print("🔊 Bước 4: Cắt & chuẩn hóa nhịp tim (tinh chỉnh để giảm noise)...")
        picked_seg = AudioSegment.from_file(stretched_path)

        # Tính lại duration sau stretch cho 4 nhịp tim ở tốc độ mới
        # 4 nhịp tim ở target_heartbeat_tempo
        adjusted_duration_ms = (4 * (60.0 / target_heartbeat_tempo)) * 1000

        picked_seg = picked_seg[:int(adjusted_duration_ms)]
        picked_seg = picked_seg.normalize() # Chỉ normalize, không giảm cố định 14dB

        # Nếu volume vẫn quá thấp sau normalize, có thể boost nhẹ nhàng để tránh noise
        if picked_seg.dBFS < -25:
             print("⚠️ Volume nhịp tim vẫn thấp, boost nhẹ +3dB.")
             picked_seg += 3

        picked_seg.export(normalized_picked_path, format="wav")

        # Bước 5: Normalize asset
        print("🔊 Bước 5: Chuẩn hóa âm lượng nhạc...")
        normalize_asset_cmd = (
            f'ffmpeg -y -i "{asset_audio}" -ar 44100 -ac 2 '
            f'-af loudnorm=I=-16:TP=-1.5:LRA=11 "{normalized_asset_path}"'
        )
        if not run_ffmpeg(normalize_asset_cmd):
            return

        # Bước 6: Mix với tỉ lệ mới 0.75 (nhạc) : 0.25 (tim)
        print("🎵 Bước 6: Mix với tỉ lệ 0.75:0.25 (nhạc : tim) và cân bằng âm lượng...")
        vol_asset = get_mean_volume(normalized_asset_path)
        vol_picked = get_mean_volume(normalized_picked_path)
        print(f"📊 Volumes → Asset: {vol_asset:.2f} dB, Picked: {vol_picked:.2f} dB")

        # Cân bằng động, ưu tiên nhạc nổi bật hơn theo tỉ lệ 0.8
        diff = vol_asset - vol_picked
        # Nếu asset nhỏ hơn picked, tăng asset lên (diff < 0 => -diff > 0)
        # Thêm 2dB cho asset để đảm bảo nó luôn nổi bật hơn
        asset_filter = f"[0:a]volume={max(0, -diff + 2)}dB[a0];"
        # Nếu picked nhỏ hơn asset, tăng picked lên (diff > 0)
        picked_filter = f"[1:a]volume={max(0, diff)}dB,aloop=loop=-1:size=2e+09[a1];"

        mix_cmd = (
            f'ffmpeg -y -i "{normalized_asset_path}" -i "{normalized_picked_path}" '
            f'-filter_complex "{asset_filter}{picked_filter}[a0][a1]amix=inputs=2:duration=first:dropout_transition=3:weights=0.75 0.25[a]" '
            f'-map "[a]" -c:a libmp3lame -q:a 2 "{mixed_temp_path}"'
        )
        if run_ffmpeg(mix_cmd):
            print(f"✅ Mixing successful! Tuning to 432Hz...")
            tune_to_432hz(mixed_temp_path, tuned_output_path)
            print(f"✅ Tuned output saved at {output_path}")
        else:
            print("❌ Mixing failed")

# Sử dụng với file của bạn (Ví dụ)
mix_audio_v4("twinkle_star.mp3", "Heartbeat5_bpm140.wav", "demo_version_4.1.mp3")