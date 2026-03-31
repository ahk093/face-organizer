# Face Organizer

**Automatically organize your photo library by faces - like iPhone's People album, but local and private.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

## What it does

Face Organizer scans your entire photo library, detects faces using state-of-the-art AI, and automatically groups photos by person. No cloud upload, no privacy concerns - everything runs locally on your machine.

### Features

- **Face Detection & Recognition**: Uses InsightFace (ArcFace) for highly accurate face detection and embedding
- **Smart Clustering**: DBSCAN algorithm groups similar faces automatically
- **Checkpoint System**: Stop anytime and resume later - progress is saved every 100 photos
- **Symbolic Links**: Organizes photos without duplicating files (saves disk space)
- **Date Sorting**: Photos are sorted chronologically within each person's folder
- **Thumbnail Preview**: Each person folder includes a `_YUZ.jpg` face thumbnail
- **No-Face Organization**: Photos without faces are organized by source folder
- **Export Tool**: Export any person's photos to share with them

### Sample Output

```
_Kisiler/
├── Kisi_000/           # Person with most photos
│   ├── _YUZ.jpg        # Face thumbnail
│   ├── 2019-03-15_IMG_1234.jpg → /original/path/IMG_1234.jpg
│   ├── 2020-07-22_photo.jpg → /original/path/photo.jpg
│   └── ...
├── Kisi_001/
│   └── ...
└── ...

_yuz_yok/               # Photos without faces
├── DCIM/
├── WhatsApp Images/
└── ...
```

## Installation


### Setup

```bash
# Clone the repository
git clone https://github.com/ahk093/face-organizer.git
cd face-organizer

# Create virtual environment (recommended)
conda create -n face-organizer python=3.9
conda activate face-organizer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Scan all subdirectories
python face_organizer.py /path/to/photos

# Scan specific directories only
python face_organizer.py /path/to/photos --dirs "DCIM,Camera,Pictures"

# Use stricter clustering (fewer mixed groups)
python face_organizer.py /path/to/photos --eps 0.4
```

The script will:
1. Find all images (JPG, PNG, HEIC)
2. Detect and encode faces
3. Cluster similar faces
4. Create organized folders with symbolic links

### Export Photos

Share someone's photos with them:

```bash
./export_person.sh Kisi_001 John
# Creates: _Export/John/ with actual photo copies (not symlinks)
```


## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLUSTER_EPS` | 0.5 | Clustering strictness (lower = stricter, more clusters) |
| `CLUSTER_MIN_SAMPLES` | 2 | Minimum photos to form a cluster |
| `TIMEOUT_SECONDS` | 30 | Skip photos that take too long |
| `CHECKPOINT_INTERVAL` | 100 | Save progress every N photos |

## How it Works

1. **Face Detection**: InsightFace detects faces and extracts 512-dimensional embeddings
2. **Normalization**: Embeddings are L2-normalized for cosine similarity
3. **Clustering**: DBSCAN groups embeddings with cosine distance < 0.5
4. **Organization**: Symbolic links preserve original files while creating organized structure

## Limitations

- Siblings/family members may be grouped together (genetic similarity)
- Same person at very different ages might be split
- Low quality or side-profile faces may not be detected
- Windows: Symbolic links may require admin privileges or Developer Mode enabled

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the amazing face recognition models
- [scikit-learn](https://scikit-learn.org/) for DBSCAN clustering

---
