#!/usr/bin/env python3
"""
Face Organizer - Organize photos by faces locally
https://github.com/YOUR_USERNAME/face-organizer

Usage:
    python face_organizer.py /path/to/photos
    python face_organizer.py /path/to/photos --dirs "DCIM,Camera,Pictures"
"""
import os
import sys
import argparse
import pickle
import atexit
import platform
import threading
import numpy as np
from pathlib import Path

IS_WINDOWS = platform.system() == 'Windows'

if not IS_WINDOWS:
    import fcntl
    import signal
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import cv2
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

import insightface
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

# ============ CONFIGURATION ============
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
CLUSTER_EPS = 0.5           # Clustering threshold (lower = stricter)
CLUSTER_MIN_SAMPLES = 2     # Minimum faces to form a cluster
TIMEOUT_SECONDS = 30        # Skip images that take too long
CHECKPOINT_INTERVAL = 100   # Save progress every N images

# ============ LOCK MECHANISM ============
def acquire_lock(lock_file: Path):
    """Ensure only one instance runs at a time (cross-platform)"""
    try:
        lock_fd = open(lock_file, 'w')
        if IS_WINDOWS:
            import msvcrt
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        atexit.register(release_lock, lock_fd, lock_file)
        return lock_fd
    except (BlockingIOError, OSError):
        print("ERROR: Another instance is already running!")
        print(f"If stuck, remove: {lock_file}")
        sys.exit(1)

def release_lock(lock_fd, lock_file: Path):
    """Release the lock"""
    try:
        if IS_WINDOWS:
            import msvcrt
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        lock_file.unlink(missing_ok=True)
    except:
        pass

# ============ TIMEOUT ============
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

class TimeoutContext:
    """Cross-platform timeout context manager"""
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None
        
    def _timeout_func(self):
        pass  # On Windows, we can't interrupt - just let it run
        
    def __enter__(self):
        if not IS_WINDOWS:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.seconds)
        return self
    
    def __exit__(self, *args):
        if not IS_WINDOWS:
            signal.alarm(0)

# ============ IMAGE PROCESSING ============
def load_image(path: Path) -> Optional[np.ndarray]:
    """Load image with EXIF rotation handling"""
    try:
        with Image.open(path) as img:
            try:
                exif = img.getexif()
                if exif:
                    orientation = exif.get(274)
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
            except:
                pass
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except:
        return None

def find_all_images(base_dir: Path, scan_dirs: List[str]) -> List[Path]:
    """Find all images in specified directories"""
    images = []
    for dir_name in scan_dirs:
        scan_dir = base_dir / dir_name
        if not scan_dir.exists():
            print(f"  Skipping (not found): {dir_name}")
            continue
        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.startswith('._'):
                    continue
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(Path(root) / file)
    return images

def process_image(path: Path, app: FaceAnalysis) -> Dict[str, Any]:
    """Detect faces in an image"""
    result = {
        'path': str(path),
        'faces': [],
        'has_faces': False,
        'error': None
    }
    
    img = load_image(path)
    if img is None:
        result['error'] = 'Failed to load'
        return result
    
    try:
        faces = app.get(img)
        if faces:
            for face in faces:
                result['has_faces'] = True
                result['faces'].append({
                    'embedding': face.embedding,
                    'bbox': face.bbox.tolist(),
                    'det_score': float(face.det_score)
                })
    except Exception as e:
        result['error'] = str(e)
    
    return result

# ============ CHECKPOINT ============
def load_checkpoint(checkpoint_file: Path) -> Tuple[List[Dict], List[str], List[str], set]:
    """Load progress from checkpoint"""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
                return (
                    data.get('encodings', []),
                    data.get('no_face_files', []),
                    data.get('error_files', []),
                    set(data.get('processed_files', []))
                )
        except:
            print("Warning: Checkpoint corrupted, starting fresh")
    return [], [], [], set()

def save_checkpoint(checkpoint_file: Path, encodings, no_face_files, error_files, processed_files):
    """Save progress to checkpoint"""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'encodings': encodings,
            'no_face_files': no_face_files,
            'error_files': error_files,
            'processed_files': list(processed_files)
        }, f)

# ============ SCANNING ============
def scan_all_images(images: List[Path], checkpoint_file: Path, app: FaceAnalysis) -> Tuple[List[Dict], List[str], List[str]]:
    """Scan all images with checkpointing"""
    encodings, no_face_files, error_files, processed_files = load_checkpoint(checkpoint_file)
    
    if processed_files:
        print(f"Checkpoint loaded: {len(processed_files)} files already processed")
    
    remaining = [p for p in images if str(p) not in processed_files]
    print(f"To process: {len(remaining)} images\n")
    
    timeout_count = 0
    
    for i, path in enumerate(tqdm(remaining, desc="Scanning")):
        try:
            with TimeoutContext(TIMEOUT_SECONDS):
                result = process_image(path, app)
            
            if result['error']:
                error_files.append(result['path'])
            elif result['has_faces']:
                for j, face in enumerate(result['faces']):
                    encodings.append({
                        'path': result['path'],
                        'face_index': j,
                        'embedding': face['embedding'],
                        'bbox': face['bbox'],
                        'score': face['det_score']
                    })
            else:
                no_face_files.append(result['path'])
            
            processed_files.add(str(path))
            
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(checkpoint_file, encodings, no_face_files, error_files, processed_files)
                
        except TimeoutException:
            timeout_count += 1
            error_files.append(str(path))
            processed_files.add(str(path))
            tqdm.write(f"  [TIMEOUT] {path.name}")
        except Exception as e:
            error_files.append(str(path))
            processed_files.add(str(path))
    
    save_checkpoint(checkpoint_file, encodings, no_face_files, error_files, processed_files)
    
    if timeout_count:
        print(f"\n{timeout_count} files skipped due to timeout")
    
    return encodings, no_face_files, error_files

# ============ CLUSTERING ============
def cluster_faces(encodings: List[Dict]) -> Dict[int, List[Dict]]:
    """Cluster faces using DBSCAN"""
    if not encodings:
        return {}
    
    print(f"\nClustering {len(encodings)} faces...")
    
    embeddings = np.array([e['embedding'] for e in encodings])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    clustering = DBSCAN(
        eps=CLUSTER_EPS,
        min_samples=CLUSTER_MIN_SAMPLES,
        metric='cosine',
        n_jobs=-1
    )
    
    labels = clustering.fit_predict(embeddings)
    
    clusters = defaultdict(list)
    noise_cluster = []
    
    for i, label in enumerate(labels):
        if label == -1:
            noise_cluster.append(encodings[i])
        else:
            clusters[label].append(encodings[i])
    
    max_label = max(clusters.keys()) if clusters else -1
    for i, enc in enumerate(noise_cluster):
        clusters[max_label + 1 + i] = [enc]
    
    print(f"Found {len(clusters)} unique people")
    return dict(clusters)

# ============ ORGANIZATION ============
def get_photo_date(path: Path) -> str:
    """Extract date from photo EXIF or filename"""
    import re
    from datetime import datetime
    
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                date_str = exif.get(36867) or exif.get(306)
                if date_str:
                    dt = datetime.strptime(date_str.split()[0], "%Y:%m:%d")
                    return dt.strftime("%Y-%m-%d")
    except:
        pass
    
    name = path.stem
    patterns = [
        r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})',
        r'IMG_(\d{4})(\d{2})(\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            y, m, d = match.groups()
            if 1990 <= int(y) <= 2030 and 1 <= int(m) <= 12 and 1 <= int(d) <= 31:
                return f"{y}-{m}-{d}"
    
    try:
        mtime = path.stat().st_mtime
        from datetime import datetime
        dt = datetime.fromtimestamp(mtime)
        return dt.strftime("%Y-%m-%d")
    except:
        return "0000-00-00"

def create_symlink(source: Path, target_dir: Path, add_date: bool = True) -> bool:
    """Create symbolic link with date prefix"""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if add_date:
            date_prefix = get_photo_date(source)
            target_name = f"{date_prefix}_{source.name}"
        else:
            target_name = source.name
        
        target = target_dir / target_name
        
        if target.exists() or target.is_symlink():
            base = target.stem
            suffix = target.suffix
            counter = 1
            while target.exists() or target.is_symlink():
                target = target_dir / f"{base}_{counter}{suffix}"
                counter += 1
        
        rel_source = os.path.relpath(source, target.parent)
        target.symlink_to(rel_source)
        return True
    except:
        return False

def create_face_thumbnail(face_data: Dict, person_dir: Path) -> bool:
    """Create face thumbnail for a person folder"""
    try:
        img = load_image(Path(face_data['path']))
        if img is None:
            return False
        
        bbox = face_data['bbox']
        x1, y1, x2, y2 = [int(b) for b in bbox]
        
        h, w = img.shape[:2]
        margin_x = int((x2 - x1) * 0.3)
        margin_y = int((y2 - y1) * 0.3)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        face_crop = img[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_pil.save(person_dir / "_YUZ.jpg", quality=85)
        return True
    except:
        return False

def organize_files(clusters: Dict[int, List[Dict]], no_face_files: List[str], 
                   output_dir: Path, no_face_dir: Path, base_dir: Path):
    """Organize files into folders"""
    import shutil
    
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    if no_face_dir.exists():
        shutil.rmtree(no_face_dir, ignore_errors=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    no_face_dir.mkdir(parents=True, exist_ok=True)
    
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: len(set(f['path'] for f in x[1])),
        reverse=True
    )
    
    print(f"\nCreating person folders ({len(sorted_clusters)} people)...")
    
    total_links = 0
    thumbnails = 0
    
    for new_id, (old_id, faces) in enumerate(tqdm(sorted_clusters, desc="Organize")):
        person_dir = output_dir / f"Kisi_{new_id:03d}"
        person_dir.mkdir(exist_ok=True)
        
        best_face = max(faces, key=lambda f: f.get('score', 0))
        if create_face_thumbnail(best_face, person_dir):
            thumbnails += 1
        
        seen = set()
        for face in faces:
            if face['path'] not in seen:
                seen.add(face['path'])
                if create_symlink(Path(face['path']), person_dir):
                    total_links += 1
    
    print(f"  {total_links} links, {thumbnails} thumbnails created")
    
    print(f"\nOrganizing no-face photos ({len(no_face_files)})...")
    no_face_links = 0
    
    for path_str in tqdm(no_face_files, desc="No-face"):
        source = Path(path_str)
        try:
            rel_path = source.relative_to(base_dir)
            source_folder = rel_path.parts[0] if rel_path.parts else "Other"
        except:
            source_folder = "Other"
        
        target_dir = no_face_dir / source_folder
        if create_symlink(source, target_dir):
            no_face_links += 1
    
    print(f"  {no_face_links} links created")

def print_summary(clusters, no_face_files, error_files):
    """Print final summary"""
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    total_faces = sum(len(faces) for faces in clusters.values())
    total_photos = len(set(f['path'] for faces in clusters.values() for f in faces))
    
    print(f"Total people: {len(clusters)}")
    print(f"Total faces: {total_faces}")
    print(f"Photos with faces: {total_photos}")
    print(f"Photos without faces: {len(no_face_files)}")
    print(f"Errors/Timeouts: {len(error_files)}")
    
    print("\nTop 10 people:")
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(set(f['path'] for f in x[1])), reverse=True)
    for i, (cid, faces) in enumerate(sorted_clusters[:10]):
        photo_count = len(set(f['path'] for f in faces))
        print(f"  Kisi_{i:03d}: {len(faces)} faces, {photo_count} photos")
    
    print("="*60)

# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(
        description='Organize photos by faces - locally and privately',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_organizer.py /path/to/photos
  python face_organizer.py /path/to/photos --dirs "DCIM,Camera,Pictures"
  python face_organizer.py /path/to/photos --eps 0.4  # Stricter clustering
        """
    )
    parser.add_argument('path', help='Path to photos directory')
    parser.add_argument('--dirs', help='Comma-separated list of subdirectories to scan (default: scan all)')
    parser.add_argument('--eps', type=float, default=0.5, help='Clustering threshold (default: 0.5, lower=stricter)')
    parser.add_argument('--output', help='Output directory name (default: _Kisiler)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.path).resolve()
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)
    
    project_dir = base_dir / "_face_organizer"
    project_dir.mkdir(exist_ok=True)
    
    output_dir = base_dir / (args.output or "_Kisiler")
    no_face_dir = base_dir / "_yuz_yok"
    checkpoint_file = project_dir / "checkpoint.pkl"
    lock_file = project_dir / ".lock"
    
    global CLUSTER_EPS
    CLUSTER_EPS = args.eps
    
    if args.dirs:
        scan_dirs = [d.strip() for d in args.dirs.split(',')]
    else:
        scan_dirs = [d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(('_', '.'))]
    
    lock_fd = acquire_lock(lock_file)
    print("Lock acquired, starting...\n")
    
    print("="*60)
    print("FACE ORGANIZER")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Directories to scan: {', '.join(scan_dirs)}")
    print(f"Clustering threshold: {CLUSTER_EPS}")
    
    print("\nLoading InsightFace model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model ready!\n")
    
    print("-"*40)
    print("STEP 1: FINDING IMAGES")
    print("-"*40)
    images = find_all_images(base_dir, scan_dirs)
    print(f"Found: {len(images)} images")
    
    print("\n" + "-"*40)
    print("STEP 2: SCANNING")
    print("-"*40)
    encodings, no_face_files, error_files = scan_all_images(images, checkpoint_file, app)
    
    print(f"\nScan complete:")
    print(f"  Faces: {len(encodings)}")
    print(f"  No faces: {len(no_face_files)}")
    print(f"  Errors: {len(error_files)}")
    
    print("\n" + "-"*40)
    print("STEP 3: CLUSTERING")
    print("-"*40)
    clusters = cluster_faces(encodings)
    
    print("\n" + "-"*40)
    print("STEP 4: ORGANIZING")
    print("-"*40)
    organize_files(clusters, no_face_files, output_dir, no_face_dir, base_dir)
    
    print_summary(clusters, no_face_files, error_files)
    
    print(f"\nOutput: {output_dir}")
    print("\nDONE!")

if __name__ == "__main__":
    main()
