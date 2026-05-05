from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import zipfile
import logging
import base64
import json
import mimetypes
import urllib.parse
import urllib.request
import urllib.error
import re
import time
from functools import lru_cache
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
# Only v1 pipeline is exposed via API.
from processor import mix_audio_v1, adjust_bpm, preprocess_shared

# Task status constants
TASK_STATUS_PROCESSING = "PROCESSING"
TASK_STATUS_COMPLETED = "COMPLETED"
TASK_STATUS_FAILED = "FAILED"

try:
    import boto3
    from botocore.client import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # pragma: no cover - keep API running even if boto3 is absent.
    boto3 = None
    BotoConfig = None
    BotoCoreError = Exception
    ClientError = Exception

# Setup logging
import os
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)
logger.info(f"Logging level: {log_level}")

# In-memory task store for background mix tasks (simple dict, NOT for multi-worker)
# Limit: 2 concurrent tasks for Xeon 4 cores / 8GB RAM
import uuid
import threading
from datetime import datetime, timedelta

mixing_tasks: Dict[str, Dict] = {}
MAX_BACKGROUND_TASKS = 2  # Limit concurrent background tasks (2 per 4 cores)


def load_local_env_file() -> None:
    """Nạp các biến môi trường từ file .env khi chạy local hoặc trong Docker.

    Hàm này cho phép chạy ứng dụng với `docker run -p 8000:8000 ...`
    mà vẫn ưu tiên các biến môi trường thật sự (nếu có).
    Chỉ nạp các giá trị CHƯA được đặt (dùng setdefault).
    """
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),  # .env cùng thư mục với script
        os.path.join(os.getcwd(), ".env"),                 # .env ở thư mục làm việc hiện tại
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue

        try:
            loaded = 0
            with open(env_path, "r", encoding="utf-8") as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key:
                        continue

                    # Loại bỏ dấu nháy đơn/ kép bao quanh giá trị
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    os.environ.setdefault(key, value)  # Chỉ đặt nếu chưa có
                    loaded += 1

            logger.info(f"Loaded {loaded} values from local env file: {env_path}")
            return
        except Exception as exc:
            logger.warning(f"Failed to load local env file {env_path}: {exc}")


load_local_env_file()

R2_PUBLIC_BASE_URL = os.getenv(
    "R2_PUBLIC_BASE_URL",
    "https://pub-be426cc47866401c9c6513aa344cb2f0.r2.dev",
).rstrip("/")
R2_S3_BUCKET = os.getenv("R2_S3_BUCKET", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
R2_S3_ENDPOINT = os.getenv("R2_S3_ENDPOINT", "").strip()
ALLOWED_TRACK_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}
ALLOWED_FILE_TYPES = {"heartbeat", "trackbeat"}
MIX_OUTPUT_FORMATS = {
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
}


def resolve_mix_output_format(raw_value: str) -> str:
    """Chuẩn hoá và kiểm tra định dạng output cho mix.

    Chấp nhận: 'flac' hoặc 'mp3'.
    Mặc định trả về 'mp3' nếu giá trị không hợp lệ.
    """
    normalized = str(raw_value or "mp3").strip().lower()
    if normalized in MIX_OUTPUT_FORMATS:
        return normalized
    return "mp3"

app = FastAPI(title="Woom Audio Mixer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_temp(temp_dir: str):
    """Xoá an toàn thư mục tạm sau khi xử lý xong.

    Được gọi bới BackgroundTasks của FastAPI để dọn dẹp tài nguyên.
    Bỏ qua lỗi nếu thư mục không tồn tại hoặc đang được sử dụng.
    """
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def save_uploadfile_to_disk(upload: UploadFile, destination_path: str) -> int:
    """Ghi nội dung UploadFile vào đĩa và trả về số byte đã ghi.

    Hàm này nhận file upload từ request FastAPI và lưu vào đường dẫn chỉ định.
    Xử lý được cả file có thể seek (di chuyển con trỏ) hoặc không.
    """
    if upload is None or upload.file is None:
        return 0

    try:
        upload.file.seek(0)  # Đưa con trỏ về đầu file nếu có thể
    except Exception:
        # Một số file object không hỗ trợ seek; tiếp tục từ vị trí hiện tại
        pass

    written = 0
    with open(destination_path, "wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)  # Đọc từng khối 1MB
            if not chunk:
                break
            buffer.write(chunk)
            written += len(chunk)

    return written


def sanitize_track_name(track_name: str) -> str:
    """Làm sạch tên track để tránh tấn công path traversal.

    Chỉ cho phép tên file đơn (không chứa đường dẫn).
    Trả về tên sạch hoặc ném HTTP 400 nếu không hợp lệ.
    """
    safe_name = os.path.basename((track_name or "").strip())  # Loại bỏ phần đường dẫn
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid track name")
    if safe_name != track_name.strip():  # Kiểm tra nếu có thử chèn đường dẫn
        raise HTTPException(status_code=400, detail="Invalid track name")
    return safe_name


def build_r2_track_url(track_name: str) -> str:
    """Tạo URL công khai cho track trên Cloudflare R2.

    Hàm này mã hoá tên file (URL encoding) và ghép với base URL R2.
    URL này dùng để frontend phát audio trực tiếp từ R2 (không qua backend proxy).
    """
    safe_name = sanitize_track_name(track_name)  # Làm sạch tên file
    encoded_name = urllib.parse.quote(safe_name)  # Mã hoá các ký tự đặc biệt
    return f"{R2_PUBLIC_BASE_URL}/{encoded_name}"


def resolve_r2_s3_endpoint() -> str:
    """Xác định endpoint S3 cho Cloudflare R2.

    Ưu tiên dùng biến môi trường R2_S3_ENDPOINT (nếu có).
    Nếu không, tự động tạo từ R2_ACCOUNT_ID.
    Trả về chuỗi rỗng nếu không đủ thông tin.
    """
    if R2_S3_ENDPOINT:
        return R2_S3_ENDPOINT.rstrip("/")  # Dùng endpoint chỉ định
    if R2_ACCOUNT_ID:
        return f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"  # Tạo tự động
    return ""


def is_r2_s3_ready() -> bool:
    """Kiểm tra cấu hình R2 S3 đã sẵn sàng chưa.

    Hàm kiểm tra các biến môi trường cân thiết:
    - boto3 đã được import thành công
    - Endpoint R2 hợp lệ
    - Tên bucket, Access Key, Secret Key đã đặt
    Trả về True nếu đủ điều kiện để dùng R2.
    """
    endpoint = resolve_r2_s3_endpoint()
    return bool(
        boto3                          # Thư viện boto3 phải có
        and endpoint                      # Có endpoint hợp lệ
        and R2_S3_BUCKET               # Tên bucket đã đặt
        and R2_ACCESS_KEY_ID            # Access key có
        and R2_SECRET_ACCESS_KEY        # Secret key có
    )


@lru_cache(maxsize=1)
def get_r2_s3_client():
    """Tạo và trả về S3 client để làm việc với Cloudflare R2.

    Hàm được cache (tối đa 1 instance) để không phải tạo lại client mỗi lần.
    Trả về None nếu cấu hình R2 chưa sẵn sàng (is_r2_s3_ready() = False).
    """
    if not is_r2_s3_ready():
        return None

    endpoint = resolve_r2_s3_endpoint()
    kwargs = {
        "service_name": "s3",
        "endpoint_url": endpoint,           # Endpoint R2 đã resolve
        "aws_access_key_id": R2_ACCESS_KEY_ID,
        "aws_secret_access_key": R2_SECRET_ACCESS_KEY,
        "region_name": "auto",                 # R2 dùng region "auto"
    }
    if BotoConfig is not None:
        kwargs["config"] = BotoConfig(signature_version="s3v4")  # Chữ ký S3 v4

    return boto3.client(**kwargs)


def guess_track_file_type(track_name: str) -> str:
    """Đoán loại file track dựa trên tên file.

    Quy tắc:
    - Nếu chứa "heartbeat" hoặc "heartbeart" → heartbeat
    - Nếu chứa token "hb" (legacy naming) → heartbeat
    - Mặc định → trackbeat
    """
    lower_name = (track_name or "").lower()
    if "heartbeat" in lower_name or "heartbeart" in lower_name:
        return "heartbeat"

    # Fallback cho các tên file cũ như *_hb.wav
    tokens = [token for token in re.split(r"[^a-z0-9]+", lower_name) if token]
    if "hb" in tokens:
        return "heartbeat"

    return "trackbeat"


def is_generated_mix_track_name(track_name: str) -> bool:
    """Kiểm tra tên file có phải là kết quả mix được tạo ra (mixed track).

    Trả về True nếu tên file khớp các mẫu:
    - mixed-<hex> hoặc mixed_<hex> (có thể có phần mở rộng)
    - v1_mixed, v2_mixed, ... (các version cũ)
    """
    normalized = os.path.basename(str(track_name or "")).strip().lower()
    if not normalized:
        return False

    return bool(
        # Mẫu: mixed-<hex_chars> (8+ ký tự hex)
        re.match(r"^mixed[-_][a-f0-9]{8,}(?:[-_][a-f0-9]{8,})*(?:\.[a-z0-9]+)?$", normalized)
        # Hoặc: v1_mixed, v2_mixed, ... (legacy)
        or re.match(r"^v\d+_mixed(?:\.[a-z0-9]+)?$", normalized)
    )


def is_ghost_heartbeat_track(track_name: str, display_name: str) -> bool:
    """Lọc các track heartbeat giả (placeholder) không nên hiển thị cho người dùng.

    Các track này thường là file tạm thời hoặc legacy như:
    - display_name = "hb.wav"
    - track_name kết thúc bằng "_hb.wav" và display_name là "hb.wav" hoặc "hb"
    Trả về True nếu là ghost track, False nếu là track thật.
    """
    normalized_track = os.path.basename(str(track_name or "")).strip().lower()
    normalized_display = os.path.basename(str(display_name or "")).strip().lower()

    if normalized_display == "hb.wav":       # File chuẩn legacy
        return True

    if normalized_track.endswith("_hb.wav") and normalized_display in {"hb.wav", "hb"}:
        return True

    return False


def normalize_file_type(file_type: str, fallback_track_name: str = "") -> str:
    """Chuẩn hoá giá trị file_type để frontend phân loại heartbeat/trackbeat.

    Quy tắc:
    - "heartbeat", "heartbeart", "heart" → "heartbeat"
    - "trackbeat", "track", "music", "background" → "trackbeat"
    - Nếu không khớp và có fallback_track_name → đoán từ tên file
    - Mặc định → "trackbeat"
    """
    lowered = str(file_type or "").strip().lower()
    compact = lowered.replace("-", "").replace("_", "").replace(" ", "")

    if compact in {"heartbeat", "heartbeart", "heart"}:
        return "heartbeat"
    if compact in {"trackbeat", "track", "music", "background"}:
        return "trackbeat"
    if fallback_track_name:
        return guess_track_file_type(fallback_track_name)  # Đoán từ tên file
    return "trackbeat"


def derive_display_name_from_key(object_key: str, fallback_name: str) -> str:
    """Trích xuất tên hiển thị cho người dùng từ object key R2.

    Xử lý 2 mẫu tên file phổ biến:
    1. Mẫu mơi: upload_heartbeat_<timestamp>_<hex>_<ten_goc> hoặc upload_trackbeat_...
    2. Mẫu cũ: <timestamp>T<time>Z_<hex>_<ten_goc>
    Nếu không khớp mẫu nào → trả về fallback_name.
    """
    key_name = os.path.basename(object_key or "")
    fallback = fallback_name or key_name or "audio.wav"

    # Mẫu mơi: upload_heartbeat_xxx_<ten_goc> hoặc upload_trackbeat_xxx_<ten_goc>
    modern_match = re.match(
        r"^upload_(?:heartbeat|trackbeat)_(?:[0-9]{14}_[0-9a-f]{8}_)?(.+)$",
        key_name,
        flags=re.IGNORECASE,
    )
    if modern_match:
        candidate = modern_match.group(1).strip()
        if candidate:
            return candidate

    # Mẫu cũ: <timestamp>T<time>Z_<hex>_<ten_goc>
    legacy_match = re.match(
        r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8,}_(.+)$",
        key_name,
        flags=re.IGNORECASE,
    )
    if legacy_match:
        candidate = legacy_match.group(1).strip()
        if candidate:
            return candidate

    return fallback


def build_uploaded_track_name(original_name: str, file_type: str, content_type: str = "") -> str:
    """Tạo tên đối tượng R2 ổn định, tránh nhân bản khi upload cùng file nhiều lần.

    Tên được tạo theo mẫu: upload_<file_type>_<safe_stem><ext>
    - Nếu không có tên gốc → dùng chính file_type làm stem
    - Chỉ giữ lại ký tự alphanumeric, dấu chấm, gạch dưới, gạch ngang
    - Giới hạn stem 80 ký tự để tránh tên quá dài
    - Nếu phần mở rộng không hợp lệ → đoán từ content_type hoặc dùng .wav mặc định
    """
    normalized_type = normalize_file_type(file_type)
    safe_name = os.path.basename((original_name or "").strip())
    stem, ext = os.path.splitext(safe_name)
    # Làm sạch stem: chỉ giữ ký tự an toàn
    safe_stem = "".join([c for c in stem if c.isalnum() or c in "._-"]).strip("._-")
    if not safe_stem:
        safe_stem = normalized_type  # Fallback dùng chính file_type

    ext = ext.lower()
    if ext not in ALLOWED_TRACK_EXTENSIONS:
        # Đoán phần mở rộng từ MIME type nếu có
        guessed_ext = (mimetypes.guess_extension(content_type or "") or "").lower()
        ext = guessed_ext if guessed_ext in ALLOWED_TRACK_EXTENSIONS else ".wav"

    return f"upload_{normalized_type}_{safe_stem[:80]}{ext}"


def upload_track_file_to_r2(
    *,
    local_path: str,
    original_name: str,
    file_type: str,
    content_type: str = "",
) -> str:
    """Upload file audio cục bộ lên R2 kèm metadata.

    Hàm kiểm tra:
    - File tồn tại và không rỗng
    - Cấu hình R2 đã sẵn sàng (is_r2_s3_ready())
    - Loại file hợp lệ (heartbeat hoặc trackbeat)

    Sau đó tạo tên object chuẩn hoá, chuẩn bị metadata:
    - file_type (đã chuẩn hoá)
    - source = "uploaded"
    - display_name (tên hiển thị, giới hạn 120 ký tự)
    - original_name (tên gốc, giới hạn 120 ký tự)

    Trả về object_name trên R2.
    """
    if not os.path.exists(local_path) or os.path.getsize(local_path) <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if not is_r2_s3_ready():
        raise HTTPException(status_code=503, detail="R2 upload is not configured")

    s3_client = get_r2_s3_client()
    if s3_client is None:
        raise HTTPException(status_code=503, detail="R2 upload client is unavailable")

    normalized_type = normalize_file_type(file_type)
    if normalized_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file_type")

    # Tạo tên object chuẩn hoá (tránh trùng lặp)
    object_name = build_uploaded_track_name(original_name, normalized_type, content_type)
    safe_original = "".join([c for c in os.path.basename(original_name or "") if c.isalnum() or c in "._-"])
    if not safe_original:
        safe_original = "upload"

    # Xác định Content-Type cho R2
    object_content_type = (content_type or mimetypes.guess_type(object_name)[0] or "application/octet-stream").strip()
    extra_args = {
        "ContentType": object_content_type,
        "Metadata": {
            "file_type": normalized_type,
            "source": "uploaded",
            "display_name": safe_original[:120],   # Giới hạn 120 ký tự
            "original_name": safe_original[:120],
        },
    }

    try:
        with open(local_path, "rb") as file_obj:
            s3_client.upload_fileobj(file_obj, R2_S3_BUCKET, object_name, ExtraArgs=extra_args)
    except ClientError as e:
        logger.error(f"R2 upload failed for '{object_name}': {e}")
        raise HTTPException(status_code=502, detail="Failed to upload file to R2")
    except BotoCoreError as e:
        logger.error(f"R2 upload botocore error for '{object_name}': {e}")
        raise HTTPException(status_code=502, detail="Failed to upload file to R2")
    except Exception as e:
        logger.error(f"Unexpected R2 upload error for '{object_name}': {e}")
        raise HTTPException(status_code=502, detail="Failed to upload file to R2")

    logger.info(
        "Uploaded file to R2: key='%s', file_type='%s', size=%s bytes",
        object_name,
        normalized_type,
        os.path.getsize(local_path),
    )
    return object_name


def resolve_track_meta_from_r2(s3_client, object_key: str, track_name: str) -> Dict[str, str]:
    """Lấy file_type và display_name từ metadata R2 với cơ chế dự phòng mạnh mẽ.

    Quy trình:
    1. Đoán fallback_type từ tên track (guess_track_file_type)
    2. Trích xuất fallback_display từ object key (derive_display_name_from_key)
    3. Thử đọc metadata từ R2 head_object:
       - file_type (chuẩn hoá)
       - display_name, original_name, originalname (ưu tiên theo thứ tự)
    4. Nếu có lỗi (ClientError, BotoCoreError, hoặc khác) → dùng fallback
    Trả về dict {"file_type": ..., "display_name": ...}
    """
    fallback_type = guess_track_file_type(track_name)
    fallback_display = derive_display_name_from_key(object_key, track_name)

    try:
        # Đọc metadata từ R2
        head_data = s3_client.head_object(Bucket=R2_S3_BUCKET, Key=object_key)
        metadata = head_data.get("Metadata") or {}

        # Lấy file_type từ metadata (thử cả "file_type" và "filetype")
        metadata_type = metadata.get("file_type") or metadata.get("filetype") or ""
        resolved_type = normalize_file_type(metadata_type, fallback_track_name=track_name)

        # Lấy display_name từ metadata (thử nhiều key khác nhau)
        metadata_name = (
            metadata.get("display_name")
            or metadata.get("original_name")
            or metadata.get("originalname")
            or ""
        )
        resolved_display = str(metadata_name).strip() or fallback_display

        return {
            "file_type": resolved_type,
            "display_name": resolved_display,
        }
    except ClientError as e:
        logger.warning(f"Cannot read metadata for '{object_key}', fallback to derived values. error={e}")
    except BotoCoreError as e:
        logger.warning(f"Cannot read metadata for '{object_key}' due to botocore error: {e}")
    except Exception as e:
        logger.warning(f"Unexpected metadata read error for '{object_key}': {e}")

    # Trả về giá trị fallback nếu có lỗi
    return {
        "file_type": fallback_type,
        "display_name": fallback_display,
    }


# ---------------------------------------------------------------------------


def resolve_track_meta_from_head(head_data: dict, object_key: str, track_name: str) -> Dict[str, str]:
    """Lấy file_type và display_name từ head_object response đã fetch trước đó.

    Hàm này dùng khi đã gọi head_object một lần và muốn tái sử dụng dữ liệu.
    Nếu head_data rỗng → dùng fallback ngay lập tức.
    Quy trình tương tự resolve_track_meta_from_r2 nhưng không gọi lại R2.
    """
    fallback_type = guess_track_file_type(track_name)
    fallback_display = derive_display_name_from_key(object_key, track_name)

    if not head_data:
        return {"file_type": fallback_type, "display_name": fallback_display}

    try:
        metadata = head_data.get("Metadata") or {}
        # Lấy file_type từ metadata (thử cả "file_type" và "filetype")
        metadata_type = metadata.get("file_type") or metadata.get("filetype") or ""
        resolved_type = normalize_file_type(metadata_type, fallback_track_name=track_name)

        # Lấy display_name từ metadata (thử nhiều key khác nhau)
        metadata_name = (
            metadata.get("display_name")
            or metadata.get("original_name")
            or metadata.get("originalname")
            or ""
        )
        resolved_display = str(metadata_name).strip() or fallback_display

        return {
            "file_type": resolved_type,
            "display_name": resolved_display,
        }
    except Exception as exc:
        logger.warning("Failed to parse head_data for %s, using fallbacks: %s", track_name, exc)
        return {"file_type": fallback_type, "display_name": fallback_display}
# TTL cache for /tracks listing — avoids hammering R2 on every request.
# ---------------------------------------------------------------------------
_tracks_cache: Dict[str, object] = {"data": None, "expires_at": 0.0}
TRACKS_CACHE_TTL_SECONDS = 60   # Refresh at most once per minute.
HEAD_OBJECT_WORKERS = 10        # Parallel head_object threads per listing call.


def _invalidate_tracks_cache() -> None:
    """Vô hiệu hoá cache danh sách tracks, buộc phái fetch lại từ R2.

    Hàm đặt `expires_at` vể 0.0 để lẩn gọi `list_tracks_from_r2()`
    sau đó sẽ coi như cache đã hết hạn và gọi lại R2.
    """
    _tracks_cache["expires_at"] = 0.0


def _fetch_track_candidate(
    s3_client,
    raw_key: str,
    last_modified,
) -> Optional[Dict]:
    """Fetch head_object cho một key và trả về dict thông tin track hoặc None.

    Được thiết kế chạy trong ThreadPoolExecutor worker — boto3 S3 clients
    thread-safe nên không cân khóa (locking).

    Quy trình:
    1. Gọi head_object để lấy metadata
    2. Nếu lỗi 404 → bỏ qua (ghost file)
    3. Dùng resolve_track_meta_from_head() để lấy file_type, display_name
    4. Lọc bỏ các track mix (is_generated_mix_track_name)
    5. Lọc bỏ ghost heartbeat (is_ghost_heartbeat_track)
    6. Trích xuất kích thước file (Content-Length)
    Trả về dict hoặc None nếu bị lọc bỏ.
    """
    key_name = os.path.basename(raw_key)

    try:
        head_data = s3_client.head_object(Bucket=R2_S3_BUCKET, Key=raw_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code") or ""
        http_status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if code == "404" or http_status == 404:
            logger.warning("Skipping ghost file (not found via head): %s", key_name)
            return None
        logger.warning("head_object failed for %s, using fallback metadata: %s", key_name, exc)
        head_data = {}
    except (BotoCoreError, Exception) as exc:
        logger.warning("head_object unexpected error for %s, using fallback metadata: %s", key_name, exc)
        head_data = {}

    # Giải quyết metadata từ head_data
    resolved_meta = resolve_track_meta_from_head(head_data, raw_key, key_name)
    display_name = str(resolved_meta.get("display_name") or key_name).strip() or key_name

    # Lọc bỏ track mix (kết quả từ các lần mix trước)
    if is_generated_mix_track_name(display_name):
        return None
    # Lọc bỏ ghost heartbeat (track giả không nên hiển thị)
    if (
        normalize_file_type(resolved_meta.get("file_type", ""), fallback_track_name=key_name) == "heartbeat"
        and is_ghost_heartbeat_track(key_name, display_name)
    ):
        logger.warning("Skipping ghost heartbeat entry: key='%s' display='%s'", key_name, display_name)
        return None

    # Trích xuất kích thước file từ head_object response
    content_length = 0
    try:
        if head_data and "ContentLength" in head_data:
            content_length = int(head_data["ContentLength"] or 0)
        elif head_data and "ResponseMetadata" in head_data:
            # Fallback sang HTTP header
            http_headers = head_data["ResponseMetadata"].get("HTTPHeaders") or {}
            if "content-length" in http_headers:
                content_length = int(http_headers["content-length"] or 0)
    except (ValueError, TypeError, KeyError) as e:
        logger.warning("Failed to extract Content-Length for %s: %s", key_name, e)
        content_length = 0

    return {
        "track_name": key_name,
        "file_type": resolved_meta["file_type"],
        "display_name": display_name,
        "size_bytes": content_length,  # Thêm cho metadata endpoint
        "_last_modified": last_modified,
        "_raw_key": raw_key,
    }


def list_tracks_from_r2() -> List[Dict[str, str]]:
    """Liệt kê các file audio hỗ trợ từ Cloudflare R2 bucket (S3 API).

    Tối ưu hiệu năng:
    * **TTL cache**: Kết quả được cache 60 giây (TRACKS_CACHE_TTL_SECONDS).
      Các lần gọi lặp lại trong khoảng này không tốn round-trip mạng.
    * **2-phase parallel fetch**:
      - Phase 1: Gọi ``list_objects_v2`` một lần để thu thập tất cả key names
        (tuần tự, thường <300 ms).
      - Phase 2: Gọi ``head_object`` song song cho mọi key qua
        ``ThreadPoolExecutor`` (HEAD_OBJECT_WORKERS threads). Tổng độ trễ là
        độ trễ của head_object chậm nhất, không phải tổng của tất cả.
        Ví dụ: 10 tracks × 700 ms tuần tự ≈ 7 s → ~700 ms song song.
    """
    # --- serve from cache when still fresh ---
    now = time.monotonic()
    if _tracks_cache["data"] is not None and now < _tracks_cache["expires_at"]:
        logger.debug(
            "Returning tracks from in-process cache (TTL %.0fs remaining)",
            _tracks_cache["expires_at"] - now,
        )
        return _tracks_cache["data"]  # type: ignore[return-value]

    if not is_r2_s3_ready():
        logger.warning("R2 S3 listing is not ready. Missing config or boto3 package.")
        return []

    s3_client = get_r2_s3_client()
    if s3_client is None:
        return []

    # ------------------------------------------------------------------
    # Phase 1: collect all eligible (key, last_modified) pairs via
    # list_objects_v2.  This is inherently sequential but cheap — it is
    # a single paginated API call that returns only key names & sizes.
    # ------------------------------------------------------------------
    eligible: List[tuple] = []   # list of (raw_key, last_modified)
    continuation_token = None

    try:
        while True:
            params: Dict = {"Bucket": R2_S3_BUCKET, "MaxKeys": 1000}
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            response = s3_client.list_objects_v2(**params)
            for item in response.get("Contents", []) or []:
                raw_key = str(item.get("Key") or "").strip()
                key_name = os.path.basename(raw_key)
                if not key_name:
                    continue
                ext = os.path.splitext(key_name)[1].lower()
                if ext not in ALLOWED_TRACK_EXTENSIONS:
                    continue
                if is_generated_mix_track_name(key_name):
                    continue
                eligible.append((raw_key, item.get("LastModified")))

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break
    except ClientError as e:
        logger.error("R2 S3 list_objects_v2 failed: %s", e)
        return []
    except BotoCoreError as e:
        logger.error("R2 S3 list failed due to botocore error: %s", e)
        return []
    except Exception as e:
        logger.error("Unexpected error while listing R2 tracks: %s", e)
        return []

    if not eligible:
        return []

    # ------------------------------------------------------------------
    # Phase 2: fire head_object for all eligible keys IN PARALLEL.
    # boto3 S3 clients are documented as thread-safe, so we can share
    # the same client across workers without a lock.
    # ------------------------------------------------------------------
    logger.info(
        "Fetching metadata for %d eligible tracks with %d workers …",
        len(eligible),
        min(HEAD_OBJECT_WORKERS, len(eligible)),
    )
    t0 = time.monotonic()

    candidates: List[Dict] = []
    with ThreadPoolExecutor(max_workers=min(HEAD_OBJECT_WORKERS, len(eligible))) as pool:
        futures = {
            pool.submit(_fetch_track_candidate, s3_client, raw_key, last_modified): raw_key
            for raw_key, last_modified in eligible
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("Unexpected error in head_object worker for %s: %s", futures[future], exc)
                result = None
            if result is not None:
                candidates.append(result)

    logger.info(
        "Parallel head_object phase done: %d/%d tracks OK in %.2fs",
        len(candidates),
        len(eligible),
        time.monotonic() - t0,
    )

    # ------------------------------------------------------------------
    # Deduplicate: same (file_type, display_name) → keep newest upload.
    # ------------------------------------------------------------------
    tracks_by_identity: Dict[str, Dict] = {}
    for candidate in candidates:
        key_name = candidate["track_name"]
        display_name = candidate["display_name"]
        identity_key = (
            f"{normalize_file_type(candidate.get('file_type', ''), fallback_track_name=key_name)}"
            f"::{display_name.lower()}"
        )
        existing = tracks_by_identity.get(identity_key)
        if existing is None:
            tracks_by_identity[identity_key] = candidate
        else:
            existing_last = existing.get("_last_modified")
            candidate_last = candidate.get("_last_modified")
            if candidate_last and (not existing_last or candidate_last > existing_last):
                tracks_by_identity[identity_key] = candidate

    tracks = [
        {
            "track_name": str(item.get("track_name") or ""),
            "file_type": str(item.get("file_type") or "trackbeat"),
            "display_name": str(item.get("display_name") or item.get("track_name") or ""),
            "size_bytes": int(item.get("size_bytes") or 0),
        }
        for item in tracks_by_identity.values()
    ]
    tracks.sort(key=lambda t: t["display_name"].lower())

    # --- store in cache ---
    _tracks_cache["data"] = tracks
    _tracks_cache["expires_at"] = time.monotonic() + TRACKS_CACHE_TTL_SECONDS
    logger.info("Tracks cache refreshed: %d tracks, TTL=%ds", len(tracks), TRACKS_CACHE_TTL_SECONDS)

    return tracks


def download_track_from_r2(track_name: str, temp_dir: str) -> str:
    """Download selected track from public R2 into temp directory for processing."""
    safe_name = sanitize_track_name(track_name)
    local_path = os.path.join(temp_dir, f"r2_{safe_name}")

    # Prefer signed S3 API when credentials are provided. Fallback to public URL.
    if is_r2_s3_ready():
        s3_client = get_r2_s3_client()
        if s3_client is not None:
            logger.info(f"Downloading track from R2 S3 API: {safe_name}")
            try:
                s3_client.download_file(R2_S3_BUCKET, safe_name, local_path)
                if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                    logger.error(f"R2 S3 download produced empty file: {local_path}")
                    raise HTTPException(status_code=502, detail="Downloaded track is empty")
                logger.info(f"Downloaded R2 track to temp path via S3: {local_path} (size={os.path.getsize(local_path)} bytes)")
                return local_path
            except ClientError as e:
                code = str((e.response or {}).get("Error", {}).get("Code") or "")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    logger.error(f"R2 S3: Track '{safe_name}' not found (404)")
                    raise HTTPException(status_code=404, detail=f"Track '{safe_name}' not found")
                logger.warning(f"R2 S3 download failed, fallback to public URL. error={e}")
            except BotoCoreError as e:
                logger.warning(f"R2 S3 download botocore error, fallback to public URL: {e}")
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"R2 S3 download unexpected error, fallback to public URL: {e}")

    track_url = build_r2_track_url(safe_name)
    logger.info(f"Downloading track from R2 public URL: {track_url}")
    req = urllib.request.Request(track_url, headers={"User-Agent": "woom-mixer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp, open(local_path, "wb") as out:
            total = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
        logger.info(f"Downloaded from public URL: {local_path} (size={total} bytes)")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.error(f"Track not found on R2 public URL: {safe_name}")
            raise HTTPException(status_code=404, detail=f"Track '{safe_name}' not found")
        logger.error(f"R2 HTTP error while downloading '{safe_name}': {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch track from R2")
    except urllib.error.URLError as e:
        logger.error(f"R2 URL error while downloading '{safe_name}': {e}")
        raise HTTPException(status_code=502, detail="Cannot connect to R2")

    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        logger.error(f"Download failed: file not found or empty at {local_path}")
        raise HTTPException(status_code=502, detail="Downloaded track is empty")

    logger.info(f"Downloaded R2 track to temp path: {local_path} (size={os.path.getsize(local_path)} bytes)")
    return local_path


def stream_track_from_r2(track_name: str, as_attachment: bool = False):
    """Phát trực tiếp bytes từ R2 URL cho preview/playback.

    Hàm mở URL công khai của R2 và tạo generator để stream dữ liệu.
    Dùng cho frontend phát nhạc trực tiếp mà không tải toàn bộ file về.
    """
    safe_name = sanitize_track_name(track_name)
    track_url = build_r2_track_url(safe_name)
    req = urllib.request.Request(track_url, headers={"User-Agent": "woom-mixer/1.0"})

    try:
        upstream = urllib.request.urlopen(req, timeout=45)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(status_code=404, detail="Track not found")
        logger.error(f"R2 HTTP error while proxying '{safe_name}': {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch track from R2")
    except urllib.error.URLError as e:
        logger.error(f"R2 URL error while proxying '{safe_name}': {e}")
        raise HTTPException(status_code=502, detail="Cannot connect to R2")

    def stream_from_r2():
        try:
            with upstream as resp:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Error while streaming proxied track '{safe_name}': {e}")

    media_type = (
        upstream.headers.get_content_type()
        or mimetypes.guess_type(safe_name)[0]
        or "application/octet-stream"
    )
    headers = {"Content-Disposition": f'inline; filename="{safe_name}"'}
    if as_attachment:
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    return StreamingResponse(stream_from_r2(), media_type=media_type, headers=headers)



@app.get("/tracks/metadata")
def list_tracks_metadata():
    """Trả về metadata nhẹ của các track (không kèm dữ liệu audio) để lazyload.

    Endpoint này tối ưu cho việc tải trang đầu tiên nhanh chóng (≤2s).
    Trả về JSON chứa track_name, file_type, display_name, size (đã định dạng),
    và file_url cho từng track để frontend có thể hiển thị danh sách mà không
    cần tải toàn bộ dữ liệu audio.
    """
    tracks = list_tracks_from_r2()

    def format_size(bytes_val):
        """Convert bytes to human-readable format (MB/KB)."""
        try:
            b = int(bytes_val or 0)
        except (ValueError, TypeError):
            return "0B"
        if b >= 1024 * 1024:
            return f"{b / (1024 * 1024):.1f}MB"
        elif b >= 1024:
            return f"{b / 1024:.1f}KB"
        else:
            return f"{b}B"

    payload = [
        {
            "track_name": item["track_name"],
            "file_type": normalize_file_type(item.get("file_type", ""), fallback_track_name=item["track_name"]),
            "display_name": item.get("display_name") or item["track_name"],
            "size": format_size(item.get("size_bytes", 0)),
            "file_url": build_r2_track_url(item["track_name"]),
        }
        for item in tracks
    ]

    logger.info(f"Returning metadata for {len(payload)} tracks (lightweight)")
    return {"tracks": payload}


@app.get("/tracks")
def list_tracks():
    """Liệt kê các track audio có sẵn từ Cloudflare R2 bucket để hiển thị cho frontend.

    Hàm gọi list_tracks_from_r2() để lấy danh sách track, sau đó chuẩn hoá
    file_type và bổ sung file_url cho mỗi track.

    Trả về JSON chứa danh sách các track với thông tin:
    - track_name: Tên file track
    - file_type: Loại file (heartbeat hoặc trackbeat)
    - display_name: Tên hiển thị cho người dùng
    - file_url: URL công khai trên R2 để phát trực tiếp
    - source: Nguồn dữ liệu (luôn là "r2")
    """
    tracks = list_tracks_from_r2()
    payload = [
        {
            "track_name": item["track_name"],
            "file_type": normalize_file_type(item.get("file_type", ""), fallback_track_name=item["track_name"]),
            "display_name": item.get("display_name") or item["track_name"],
            "file_url": build_r2_track_url(item["track_name"]),
            "source": "r2",
        }
        for item in tracks
    ]

    logger.info(f"Returning {len(payload)} tracks from R2 library")
    return {"tracks": payload}

@app.get("/tracks/{track_name}")
def get_track(track_name: str):
    """Proxy và stream một track cụ thể từ Cloudflare R2 theo tên.

    Tham số:
    - track_name: Tên track cần phát (sẽ được làm sạch để tránh path traversal)

    Hàm sử dụng stream_track_from_r2() để phát trực tiếp dữ liệu audio
    từ R2 về cho frontend mà không lưu đệm toàn bộ file vào bộ nhớ.
    Trả về StreamingResponse chứa dữ liệu audio.
    """
    try:
        return stream_track_from_r2(track_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving track {track_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tracks/audio/{track_name}")
def get_track_audio(track_name: str):
    """Endpoint alias để phát preview track với phân đoạn '/audio' rõ ràng.

    Tham số:
    - track_name: Tên track cần phát preview

    Hàm này là alias của get_track(), dùng cho frontend dễ phân biệt
    giữa các loại request audio khác nhau (metadata vs audio preview).
    Trả về StreamingResponse chứa dữ liệu audio để phát trực tiếp.
    """
    try:
        return stream_track_from_r2(track_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving track audio {track_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/tracks")
# def list_tracks():
#     """Deprecated: frontend now uses fixed track names and fetches files via /tracks/{name}."""
#     return {"tracks": []}



@app.post("/mix")
def create_mix(
    request: Request,
    background_tasks: BackgroundTasks,
    picked: Optional[UploadFile] = File(None),
    track_name: str = Form(...),
    output_format: str = Form("mp3"),
    heartbeat_name: str = Form(""),
):
    """Tạo một nhiệm vụ mix mới và trả về task_id để frontend polling.

    Tham số:
    - request: Request object của FastAPI
    - background_tasks: Đối tượng BackgroundTasks để dọn dẹp tài nguyên
    - picked: File heartbeat do người dùng upload (tuỳ chọn, thay thế bằng heartbeat_name)
    - track_name: Tên track nhạc nền từ R2
    - output_format: Định dạng file đầu ra (mp3 hoặc flac, mặc định: mp3)
    - heartbeat_name: Tên file heartbeat từ thư viện R2 (tuỳ chọn, thay thế bằng picked)

    Các bước xử lý chính:
    1. Dọn dẹp các task cũ nếu số lượng vượt quá giới hạn (2x MAX_BACKGROUND_TASKS)
    2. Kiểm tra số lượng task đang chạy (giới hạn MAX_BACKGROUND_TASKS)
    3. Tạo thư mục tạm và đăng ký dọn dẹp sau khi xong
    4. Tải track từ R2 về thư mục tạm (download_track_from_r2)
    5. Xử lý heartbeat: ưu tiên picked (upload), fallback heartbeat_name (R2)
    6. Upload heartbeat lên R2 ở chế độ nền (không bắt buộc, chỉ cho uploaded files)
    7. Tạo task mix và lưu vào mixing_tasks dict (create_mix_task)
    8. Khởi chạy thread xử lý mix ở background (run_mix_background)

    Trả về JSON chứa:
    - task_id: ID của nhiệm vụ để polling
    - status: Trạng thái ban đầu (PROCESSING)
    - message: Thông báo tạo task thành công
    - poll_url: URL để kiểm tra trạng thái (/mix/status/{task_id})
    """
    hb_source = (picked.filename if picked and getattr(picked, 'filename', None)
                 else heartbeat_name if heartbeat_name
                 else "unknown")
    logger.info(f"==========> [/mix] NEW TASK. Track: '{track_name}', Heartbeat: '{hb_source}'")

    # Cleanup old tasks periodically
    if len(mixing_tasks) > MAX_BACKGROUND_TASKS * 2:
        cleanup_old_tasks()

    # Check if we can accept more tasks
    active_tasks = sum(1 for t in mixing_tasks.values()
                       if t.get("status") == TASK_STATUS_PROCESSING)
    if active_tasks >= MAX_BACKGROUND_TASKS:
        raise HTTPException(status_code=429, detail="Too many active tasks, please retry later")

    temp_dir = tempfile.mkdtemp()

    try:
        # Download track from R2
        asset_path = download_track_from_r2(track_name, temp_dir)

        # Handle heartbeat: either uploaded file or library heartbeat
        picked_path = None
        if picked and getattr(picked, 'filename', None):
            # Save uploaded heartbeat file
            picked_name = picked.filename or "picked_audio.wav"
            picked_filename = "".join([c for c in picked_name if c.isalnum() or c in "._-"])
            if not picked_filename:
                picked_filename = "picked_audio.wav"
            picked_path = os.path.join(temp_dir, f"picked_{picked_filename}")
            save_uploadfile_to_disk(picked, picked_path)
            logger.info(f"[/mix] Using uploaded heartbeat: {picked.filename}")
        elif heartbeat_name:
            # Download heartbeat from R2 (library heartbeat)
            logger.info(f"[/mix] Downloading library heartbeat: {heartbeat_name}")
            picked_path = download_track_from_r2(heartbeat_name, temp_dir)
        else:
            raise HTTPException(status_code=400, detail="Either heartbeat file upload or heartbeat_name is required")

        if not picked_path or os.path.getsize(picked_path) <= 0:
            raise HTTPException(status_code=400, detail="Heartbeat file is empty")

        # Upload heartbeat to R2 (optional, in background) - only for uploaded files
        if is_r2_s3_ready() and picked and getattr(picked, 'filename', None):
            try:
                upload_track_file_to_r2(
                    local_path=picked_path,
                    original_name=picked.filename or picked_filename,
                    file_type="heartbeat",
                    content_type=picked.content_type or "",
                )
            except Exception as upload_err:
                logger.warning(f"[/mix] R2 upload failed (non-critical): {upload_err}")

        # Create task
        task_id = create_mix_task(asset_path, picked_path, temp_dir, output_format)

        # Start background thread
        thread = threading.Thread(target=run_mix_background, args=(task_id,), daemon=True)
        thread.start()

        return {
            "task_id": task_id,
            "status": TASK_STATUS_PROCESSING,
            "message": "Mix task created",
            "poll_url": f"/mix/status/{task_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        cleanup_temp(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mix/status/{task_id}")
def get_mix_status(task_id: str):
    """Lấy trạng thái của một nhiệm vụ mix.

    Tham số:
    - task_id: ID của nhiệm vụ mix cần kiểm tra

    Các bước xử lý chính:
    1. Kiểm tra task_id có tồn tại trong mixing_tasks không
    2. Lấy thông tin trạng thái từ dict
    3. Nếu task đã hoàn thành (COMPLETED) → bổ sung download_url, output_format, mime_type
    4. Nếu task thất bại (FAILED) → bổ sung thông báo lỗi

    Trả về JSON chứa:
    - task_id: ID của task
    - status: Trạng thái (PROCESSING/COMPLETED/FAILED)
    - progress: Tiến trình dạng "x/7"
    - message: Thông báo chi tiết
    - download_url: (nếu completed) URL để tải kết quả
    - output_format: (nếu completed) Định dạng file đầu ra
    - mime_type: (nếu completed) MIME type của file
    - error: (nếu failed) Thông báo lỗi
    """
    if task_id not in mixing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = mixing_tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
    }

    if task["status"] == TASK_STATUS_COMPLETED:
        response["download_url"] = f"/mix/download/{task_id}"
        response["output_format"] = task.get("output_format")
        response["mime_type"] = task.get("mime_type")
    elif task["status"] == TASK_STATUS_FAILED:
        response["error"] = task.get("error")

    return response


@app.get("/mix/download/{task_id}")
def download_mix_result(task_id: str):
    """Tải xuống kết quả mix đã hoàn thành.

    Tham số:
    - task_id: ID của nhiệm vụ mix đã hoàn thành

    Các bước xử lý chính:
    1. Kiểm tra task_id có tồn tại trong mixing_tasks không
    2. Kiểm tra trạng thái task có phải COMPLETED không
    3. Kiểm tra file kết quả có tồn tại trên đĩa không
    4. Trả về FileResponse với đúng MIME type và filename

    Trả về FileResponse chứa file audio đã mix (mp3 hoặc flac).
    Ném HTTP 404 nếu task không tồn tại hoặc file không tìm thấy.
    Ném HTTP 400 nếu task chưa hoàn thành.
    """
    if task_id not in mixing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = mixing_tasks[task_id]
    if task["status"] != TASK_STATUS_COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task not completed (status: {task['status']})")

    output_path = task.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    mime_type = task.get("mime_type", "audio/mpeg")
    filename = f"woom_mix.{task.get('output_format', 'mp3')}"

    return FileResponse(
        output_path,
        media_type=mime_type,
        filename=filename,
    )


@app.post("/adjust-bpm")
def adjust_bpm_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    speeds: List[str] = Form(...)
):
    """Điều chỉnh BPM của file audio theo các mức tốc độ khác nhau.

    Tham số:
    - background_tasks: Đối tượng BackgroundTasks để dọn dẹp tài nguyên
    - file: File audio cần điều chỉnh BPM (UploadFile)
    - speeds: Danh sách các mức tốc độ (ví dụ: "Slow", "Normal", "Fast" hoặc số hệ số)

    Các bước xử lý chính:
    1. Tạo thư mục tạm và đăng ký dọn dẹp sau khi xong
    2. Lưu file upload vào thư mục tạm (save_uploadfile_to_disk)
    3. Kiểm tra file có rỗng không
    4. Tạo file ZIP để chứa kết quả
    5. Với mỗi mức tốc độ trong speeds:
       - Gọi hàm adjust_bpm() để điều chỉnh tempo
       - Thêm file kết quả vào ZIP
    6. Trả về FileResponse chứa file ZIP

    Trả về FileResponse chứa file ZIP với các phiên bản audio đã điều chỉnh BPM.
    """
    logger.info(f"==========> [/adjust-bpm] NEW REQUEST STARTED. UploadFile: '{file.filename}', speeds: {speeds}")
    
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp, temp_dir)

    logger.info(f"[/adjust-bpm] Starting to write user input file to local temp disk...")

    # Save incoming file with best-effort real extension to avoid demux ambiguity.
    incoming_name = os.path.basename(file.filename or "")
    incoming_ext = os.path.splitext(incoming_name)[1].lower()
    if incoming_ext not in ALLOWED_TRACK_EXTENSIONS:
        guessed_ext = (mimetypes.guess_extension(file.content_type or "") or "").lower()
        incoming_ext = guessed_ext if guessed_ext in ALLOWED_TRACK_EXTENSIONS else ".flac"
    input_path = os.path.join(temp_dir, f"input_mix{incoming_ext}")
    reported_upload_size = getattr(file, "size", None)
    written_bytes = save_uploadfile_to_disk(file, input_path)

    file_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
    logger.info(
        "[/adjust-bpm] Finished uploading and saving user file. Saved input file to disk. "
        "Size: %s bytes (written=%s, reported=%s)",
        file_size,
        written_bytes,
        reported_upload_size,
    )
    if file_size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    zip_path = os.path.join(temp_dir, "bpm_adjusted.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for speed in speeds:
            safe = "".join(c for c in speed if c.isalnum() or c in "._-")
            if not safe:
                continue
            out_name = f"{safe}.flac"
            out_path = os.path.join(temp_dir, out_name)
            try:
                adjust_bpm(input_path, out_path, speed)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    zipf.write(out_path, out_name)
                    logger.info(f"Created adjusted file for {speed}")
                else:
                    logger.warning(f"adjust_bpm produced no output for {speed}")
            except Exception as e:
                import traceback
                logger.error(f"Error adjusting bpm ({speed}): {e}\n{traceback.format_exc()}")
    return FileResponse(zip_path, media_type="application/zip", filename="bpm_adjusted.zip")


@app.get("/")
async def health_check():
    """Kiểm tra trạng thái hoạt động của API và trả về thông tin tổng quan.

    Endpoint này dùng để monitoring (ví dụ: Docker healthcheck, load balancer).

    Các bước xử lý chính:
    1. Đếm số lượng task đang xử lý (status = PROCESSING)
    2. Trả về JSON chứa thông tin tổng quan về hệ thống

    Trả về JSON chứa:
    - status: "ok" nếu API đang chạy bình thường
    - active_tasks: Số lượng task mix đang xử lý
    - total_tasks: Tổng số task trong hệ thống (bao gồm cả đã xong và lỗi)
    - message: Thông báo trạng thái hoạt động
    """
    active_count = sum(1 for t in mixing_tasks.values()
                        if t.get("status") == TASK_STATUS_PROCESSING)
    return {
        "status": "ok",
        "active_tasks": active_count,
        "total_tasks": len(mixing_tasks),
        "message": "Woom Audio Mixer API is running."
    }


# ---------------------------------------------------------------------------
# Background Task Helpers
# ---------------------------------------------------------------------------

def cleanup_old_tasks():
    """Xoá các nhiệm vụ cũ hơn 1 giờ để tránh rò rỉ bộ nhớ (memory leak).

    Các bước xử lý chính:
    1. Duyệt qua tất cả task trong mixing_tasks dict
    2. Kiểm tra thời gian tạo (created_at) có cách đây hơn 1 giờ không
    3. Nếu quá 1 giờ → thêm vào danh sách cần xoá
    4. Xoá từng task:
       - Xoá file kết quả trên đĩa (nếu có)
       - Xoá task khỏi mixing_tasks dict

    Hàm này giúp giải phóng bộ nhớ và dung lượng đĩa cho server.
    """
    now = datetime.utcnow()
    to_delete = []
    for task_id, task in mixing_tasks.items():
        created_at = task.get("created_at")
        if created_at and now - created_at > timedelta(hours=1):
            to_delete.append(task_id)
    for task_id in to_delete:
        task = mixing_tasks.get(task_id, {})
        temp_path = task.get("output_path")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        del mixing_tasks[task_id]
    if to_delete:
        logger.info(f"Cleaned up {len(to_delete)} old tasks")


def create_mix_task(asset_path: str, picked_path: str, temp_dir: str,
                    output_format: str) -> str:
    """Tạo một nhiệm vụ mix mới và lưu vào dict mixing_tasks.

    Tham số:
    - asset_path: Đường dẫn đến file track nhạc nền (từ R2 sau khi tải về)
    - picked_path: Đường dẫn đến file heartbeat đã upload và lưu vào thư mục tạm
    - temp_dir: Thư mục tạm chứa các file xử lý
    - output_format: Định dạng file đầu ra (mp3 hoặc flac)

    Các bước xử lý chính:
    1. Tạo task_id duy nhất bằng uuid4
    2. Khởi tạo dict chứa thông tin task:
       - status: PROCESSING (đang xử lý)
       - progress: "0/7" (tiến trình ban đầu)
       - message: Thông báo tạo task thành công
       - created_at, updated_at: Thời gian tạo và cập nhật
       - output_path, output_format, mime_type: Thông tin file kết quả
       - error: Thông báo lỗi (nếu có)
       - temp_dir, asset_path, picked_path: Đường dẫn file

    Trả về: task_id (chuỗi UUID4)
    """
    task_id = str(uuid.uuid4())
    mixing_tasks[task_id] = {
        "status": TASK_STATUS_PROCESSING,
        "progress": "0/7",
        "message": "Task created, waiting to start...",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "output_path": None,
        "output_format": output_format,
        "mime_type": None,
        "error": None,
        "temp_dir": temp_dir,
        "asset_path": asset_path,
        "picked_path": picked_path,
    }
    return task_id


def update_mix_task(task_id: str, **kwargs):
    """Cập nhật trạng thái nhiệm vụ mix một cách an toàn trên đa luồng.

    Tham số:
    - task_id: ID của nhiệm vụ cần cập nhật
    - **kwargs: Các cặp key-value cần cập nhật (vd: status, progress, message, error...)

    Các bước xử lý chính:
    1. Kiểm tra task_id có tồn tại trong mixing_tasks không
    2. Nếu có → cập nhật các trường được chỉ định qua kwargs
    3. Tự động cập nhật thời gian updated_at thành thời gian hiện tại

    Lưu ý: Hàm này được gọi từ thread xử lý background,
    nên việc cập nhật dict Python đơn giản là atomic cho các kiểu dữ liệu cơ bản.
    """
    if task_id not in mixing_tasks:
        return
    task = mixing_tasks[task_id]
    task.update(kwargs)
    task["updated_at"] = datetime.utcnow()


def run_mix_background(task_id: str):
    """Chạy pipeline mix_audio_v1 trong thread nền và cập nhật trạng thái task.

    Tham số:
    - task_id: ID của nhiệm vụ mix cần chạy

    Các bước xử lý chính (7 bước):
    1. Tiền xử lý audio assets (preprocess_shared):
       - Chuẩn hoá stereo/mono
       - Cache kết quả để tái sử dụng
    2. Phân tích tempo của heartbeat (calculate_duration_from_analysis)
    3. Phân tích tempo của track nhạc (detect_tempo)
    4. Thực hiện mix heartbeat với track (mix_audio_v1):
       - Chuyển đổi tần số lên 432Hz (nếu được cấu hình)
       - Áp dụng fade in/out
       - Kết hợp audio theo tỷ lệ phù hợp
    5. Kiểm tra file kết quả có hợp lệ không (tồn tại và kích thước > 0)
    6. Cập nhật trạng thái thành COMPLETED (có output_path, output_format, mime_type)
    7. Xử lý lỗi nếu có → cập nhật trạng thái FAILED kèm thông báo lỗi

    Kết quả được lưu vào output_path và cập nhật vào mixing_tasks dict
    để frontend có thể polling và tải về.
    """
    task = mixing_tasks.get(task_id)
    if not task:
        return

    temp_dir = task["temp_dir"]
    output_format = task["output_format"]
    asset_path = task["asset_path"]
    picked_path = task["picked_path"]

    try:
        version_name = "v1"
        resolved_format = resolve_mix_output_format(output_format)
        output_mime_type = MIX_OUTPUT_FORMATS.get(resolved_format, "audio/mpeg")
        out_path = os.path.join(temp_dir, f"{version_name}_mixed.{resolved_format}")

        update_mix_task(task_id,
                       progress="1/7",
                       message="Preprocessing audio assets...")

        # Debug: check input files
        logger.info(f"[task {task_id}] Checking input files...")
        for label, path in [("asset", asset_path), ("picked", picked_path)]:
            if path and os.path.exists(path):
                size = os.path.getsize(path)
                logger.info(f"[task {task_id}] {label}_path={path}, size={size} bytes")
            else:
                logger.error(f"[task {task_id}] {label}_path={path} NOT FOUND or missing!")

        from processor import preprocess_shared, calculate_duration_from_analysis, detect_tempo

        logger.info(f"[task {task_id}] Calling preprocess_shared(asset='{asset_path}', picked='{picked_path}', dir='{temp_dir}')")
        shared_data = preprocess_shared(asset_path, picked_path, temp_dir)
        logger.info(f"[task {task_id}] preprocess_shared result: {shared_data}")
        if not shared_data.get("success"):
            raise RuntimeError(f"Preprocessing failed: {shared_data.get('error')}")

        update_mix_task(task_id, progress="2/7", message="Analyzing heartbeat tempo...")
        heart_duration, heart_tempo = calculate_duration_from_analysis(
            shared_data.get("picked_wav_mono", picked_path)
        )

        update_mix_task(task_id, progress="3/7", message="Analyzing track tempo...")
        music_tempo = detect_tempo(shared_data.get("normalized_asset_path", asset_path))

        update_mix_task(task_id, progress="4/7", message="Mixing heartbeat with track...")
        from processor import mix_audio_v1
        logger.info(f"[task {task_id}] Calling mix_audio_v1 with out_path={os.path.abspath(out_path)}, exists_before={os.path.exists(out_path)}")
        mix_audio_v1(
            asset_path, picked_path, out_path,
            heart_duration=heart_duration,
            heart_tempo=heart_tempo,
            music_tempo=music_tempo,
            shared_data=shared_data,
        )
        logger.info(f"[task {task_id}] mix_audio_v1 returned. out_path exists={os.path.exists(out_path)}, size={os.path.getsize(out_path) if os.path.exists(out_path) else 'N/A'}")

        update_mix_task(task_id, progress="5/7", message="Validating output...")
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
            raise RuntimeError("Output file not created")

        update_mix_task(
            task_id,
            status=TASK_STATUS_COMPLETED,
            progress="7/7",
            message="Mix completed successfully",
            output_path=out_path,
            output_format=resolved_format,
            mime_type=output_mime_type,
        )
        logger.info(f"[task {task_id}] Mix completed: {out_path}")

    except Exception as e:
        import traceback
        logger.error(f"[task {task_id}] Mix failed: {e}\n{traceback.format_exc()}")
        update_mix_task(
            task_id,
            status=TASK_STATUS_FAILED,
            progress="7/7",
            message="Mix failed",
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
