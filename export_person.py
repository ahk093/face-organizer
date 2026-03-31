#!/usr/bin/env python3
"""
Export a person's photos to a folder for sharing.
Copies actual files (not symlinks) and removes macOS metadata files.

Usage:
    python export_person.py /path/to/photos Person_001 John
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

def export_person(base_dir: Path, person_folder: str, output_name: str):
    """Export photos from a person folder to a new directory"""
    
    person_dir = base_dir / "_People" / person_folder
    output_dir = base_dir / "_Export" / output_name
    
    if not person_dir.exists():
        print(f"Error: {person_dir} not found")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying: {person_folder} -> {output_name}")
    
    count = 0
    for item in person_dir.iterdir():
        # Skip thumbnails, hidden files, and metadata
        if item.name.startswith('.') or item.name.startswith('._') or item.name == '_FACE.jpg':
            continue
        
        if item.is_symlink() or item.is_file():
            try:
                # Resolve symlink and copy actual file
                source = item.resolve() if item.is_symlink() else item
                dest = output_dir / item.name
                
                # Handle duplicates
                if dest.exists():
                    base = dest.stem
                    suffix = dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = output_dir / f"{base}_{counter}{suffix}"
                        counter += 1
                
                shutil.copy2(source, dest)
                count += 1
            except Exception as e:
                print(f"  Warning: Could not copy {item.name}: {e}")
    
    # Clean up macOS metadata files
    for meta in output_dir.glob('._*'):
        meta.unlink()
    for ds in output_dir.glob('.DS_Store'):
        ds.unlink()
    
    print(f"✓ {count} photos copied")
    print(f"→ {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description='Export a person\'s photos to a folder for sharing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_person.py /path/to/photos Person_001 John
  python export_person.py . Person_042 "Mom"
        """
    )
    parser.add_argument('path', help='Base directory (where _People folder is)')
    parser.add_argument('person', help='Person folder name (e.g., Person_001)')
    parser.add_argument('name', help='Output folder name (e.g., John)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.path).resolve()
    export_person(base_dir, args.person, args.name)

if __name__ == "__main__":
    main()
