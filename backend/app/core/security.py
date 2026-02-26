import hashlib
import re

ALLOWED_MIME_TYPES = {"application/pdf"}
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
SAFE_FILENAME_RE = re.compile(r"[^\w\-. ]")


def sanitize_filename(filename: str) -> str:
    """Strip path components and replace unsafe characters."""
    # Take only the final component (no directory traversal)
    filename = filename.replace("\\", "/").split("/")[-1]
    # Replace anything that isn't alphanumeric, dash, dot, underscore, or space
    filename = SAFE_FILENAME_RE.sub("_", filename)
    return filename or "unnamed"


def validate_extension(filename: str) -> bool:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return suffix in ALLOWED_EXTENSIONS


def validate_mime_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    # Strip parameters e.g. "application/pdf; charset=utf-8"
    base_type = content_type.split(";")[0].strip().lower()
    return base_type in ALLOWED_MIME_TYPES


def validate_file_size(size_bytes: int) -> bool:
    return size_bytes <= MAX_FILE_SIZE_BYTES


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()