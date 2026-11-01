import os
import sys
import bz2
import time
import argparse
import urllib.request
import urllib.error

MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
BZ2_FILENAME = "shape_predictor_68_face_landmarks.dat.bz2"
EXPECTED_MIN_SIZE_BYTES = 95_000_000

MODEL_URLS = [
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2",
    "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
]

def format_bytes(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:3.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def render_progress(transferred: int, total: int, prefix: str = "", start_time: float = None):
    bar_width = 35
    if total > 0:
        percent = min(1.0, transferred / total)
        filled = int(round(bar_width * percent))
        bar = '=' * filled + '-' * (bar_width - filled)
        elapsed = time.time() - start_time if start_time else 0
        speed = transferred / elapsed if elapsed > 0 else 0
        eta = (total - transferred) / speed if speed > 0 else 0

        sys.stdout.write(
            f"\r{prefix} [{bar}] {percent*100:5.1f}% "
            f"({format_bytes(transferred)} / {format_bytes(total)}) "
            f"{format_bytes(speed)}/s ETA: {int(eta):02d}s"
        )
    else:
        sys.stdout.write(f"\r{prefix} {format_bytes(transferred)} transferred...")
    sys.stdout.flush()

def download_file(url: str, dest_path: str) -> bool:
    print(f"[DOWNLOAD] Fetching archive from: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DrowsinessDetector/1.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
            total_size = response.getheader('Content-Length')
            total_size = int(total_size) if total_size and total_size.isdigit() else 0

            chunk_size = 1024 * 64
            downloaded = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                render_progress(downloaded, total_size, prefix="[DOWNLOAD]", start_time=start_time)

            sys.stdout.write("\n")
            print(f"[OK] Download completed: {format_bytes(downloaded)} in {time.time() - start_time:.1f}s")
            return True

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        sys.stdout.write("\n")
        print(f"[WARN] Failed to download from {url}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False

def decompress_bz2(bz2_path: str, dest_path: str):
    print(f"[DECOMPRESS] Extracting {os.path.basename(bz2_path)} -> {os.path.basename(dest_path)}...")
    archive_size = os.path.getsize(bz2_path)
    start_time = time.time()

    chunk_size = 1024 * 64
    bytes_read = 0
    decompressor = bz2.BZ2Decompressor()

    with open(bz2_path, 'rb') as source, open(dest_path, 'wb') as dest:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            decompressed_data = decompressor.decompress(chunk)
            if decompressed_data:
                dest.write(decompressed_data)
            render_progress(bytes_read, archive_size, prefix="[DECOMPRESS]", start_time=start_time)

    sys.stdout.write("\n")
    uncompressed_size = os.path.getsize(dest_path)
    print(f"[OK] Extraction finished: {format_bytes(uncompressed_size)} in {time.time() - start_time:.1f}s")

def verify_model(model_path: str) -> bool:
    print(f"[VERIFY] Checking integrity of: {model_path}")
    if not os.path.exists(model_path):
        print(f"[ERROR] Target model does not exist: {model_path}")
        return False

    file_size = os.path.getsize(model_path)
    print(f"[INFO] Model file size: {format_bytes(file_size)} ({file_size:,} bytes)")

    if file_size < EXPECTED_MIN_SIZE_BYTES:
        print(f"[ERROR] Model file size ({file_size:,} bytes) is smaller than expected threshold ({EXPECTED_MIN_SIZE_BYTES:,} bytes). File may be corrupted.")
        return False

    try:
        import dlib
        print("[VERIFY] Testing Dlib C-API model loading...")
        start = time.time()
        _ = dlib.shape_predictor(model_path)
        load_duration = time.time() - start
        print(f"[OK] Dlib successfully loaded shape_predictor_68_face_landmarks in {load_duration:.2f}s!")
    except ImportError:
        print("[WARN] 'dlib' library not found in Python environment. Skipping C++ binary validation.")
    except Exception as e:
        print(f"[ERROR] Dlib failed to deserialize landmark model: {e}")
        return False

    return True

def setup_model(target_dir: str = ".", force: bool = False, verify_only: bool = False, keep_archive: bool = True) -> bool:
    target_model_path = os.path.join(target_dir, MODEL_FILENAME)
    target_bz2_path = os.path.join(target_dir, BZ2_FILENAME)
    fallback_bz2_path = os.path.join(target_dir, "model.bz2")

    if verify_only:
        return verify_model(target_model_path)

    if os.path.exists(target_model_path) and not force:
        print(f"[INFO] Detected existing model: {target_model_path}")
        if verify_model(target_model_path):
            print("[SUCCESS] Landmark model is already present, verified, and operational.")
            return True
        else:
            print("[WARN] Existing model verification failed. Re-acquiring model file...")

    bz2_source = None
    if os.path.exists(target_bz2_path) and os.path.getsize(target_bz2_path) > 30_000_000:
        bz2_source = target_bz2_path
        print(f"[INFO] Using existing local archive: {target_bz2_path}")
    elif os.path.exists(fallback_bz2_path) and os.path.getsize(fallback_bz2_path) > 30_000_000:
        bz2_source = fallback_bz2_path
        print(f"[INFO] Using existing local archive: {fallback_bz2_path}")

    if bz2_source is None or force:
        download_success = False
        for url in MODEL_URLS:
            if download_file(url, target_bz2_path):
                bz2_source = target_bz2_path
                download_success = True
                break

        if not download_success:
            print("[FATAL] All mirror URLs failed. Please check internet connection or manually place")
            print(f"        {MODEL_FILENAME} into: {os.path.abspath(target_dir)}")
            return False

    try:
        decompress_bz2(bz2_source, target_model_path)
    except Exception as e:
        print(f"[FATAL] BZ2 extraction failed: {e}")
        return False

    if not verify_model(target_model_path):
        print("[FATAL] Verification failed after decompression. Decompressed file is invalid.")
        return False

    if not keep_archive and os.path.exists(target_bz2_path):
        try:
            os.remove(target_bz2_path)
            print("[INFO] Cleaned up compressed archive.")
        except OSError:
            pass

    print("[SUCCESS] Landmark predictor model is ready for real-time inference.")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Dlib 68-Point Facial Landmark Model Setup & Verification Tool"
    )
    parser.add_argument(
        "--dir", default=".", help="Target directory for model file (default: current directory)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download and re-extraction even if file exists"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Perform integrity checks without downloading"
    )
    parser.add_argument(
        "--no-keep-archive", action="store_true", help="Delete .bz2 archive after extraction"
    )

    args = parser.parse_args()
    success = setup_model(
        target_dir=args.dir,
        force=args.force,
        verify_only=args.verify_only,
        keep_archive=not args.no_keep_archive
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
