#!/usr/bin/env python3
"""
Simple MP4 atom checker.

Usage:
  python tools/check_mp4_atoms.py <path_or_url>

If given a URL (http/https), the script will attempt to request the first and last
1MiB using HTTP Range requests and search for 'moov' and 'mdat' atoms. For a local
file it reads the first and last 1MiB from disk.

This helps determine whether the 'moov' atom (metadata) is located before 'mdat'
(good for progressive streaming) or after it (often causes browsers to not start
playback until the full file is downloaded).
"""
import sys
import os
import argparse

CHUNK = 1024 * 1024


def read_ranges_from_url(url):
    try:
        import requests
    except ImportError:
        print("requests library is required for URL mode. Install via: pip install requests")
        return None
    headers = {}
    out = bytearray()
    # head chunk
    try:
        headers['Range'] = f'bytes=0-{CHUNK-1}'
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code in (200, 206):
            out.extend(r.content)
    except Exception as e:
        print(f"Failed to fetch head chunk: {e}")

    # tail chunk
    try:
        headers['Range'] = f'bytes=-{CHUNK}'
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code in (200, 206):
            tail = r.content
            # ensure we can search both head and tail concatenated safely
            out.extend(tail)
    except Exception as e:
        print(f"Failed to fetch tail chunk: {e}")

    return bytes(out)


def read_ranges_from_file(path):
    size = os.path.getsize(path)
    head = b''
    tail = b''
    with open(path, 'rb') as f:
        head = f.read(CHUNK)
        if size > CHUNK:
            try:
                f.seek(max(0, size - CHUNK))
                tail = f.read(CHUNK)
            except Exception:
                tail = b''
    return head + tail


def find_atoms(buf: bytes):
    # simple search for ascii atom names
    atoms = {}
    for name in (b'moov', b'mdat', b'ftyp'):
        idx = buf.find(name)
        atoms[name.decode()] = idx if idx >= 0 else None
    return atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='Local file path or http/https URL to MP4')
    args = ap.parse_args()

    path = args.path
    buf = None
    if path.startswith('http://') or path.startswith('https://'):
        print(f"Fetching head/tail ranges from URL: {path}")
        buf = read_ranges_from_url(path)
        if buf is None:
            print("Failed to fetch URL ranges. Exiting.")
            sys.exit(2)
    else:
        if not os.path.exists(path):
            print(f"Local file not found: {path}")
            sys.exit(2)
        print(f"Reading head/tail from local file: {path}")
        buf = read_ranges_from_file(path)

    atoms = find_atoms(buf)
    print("Atom search results (offsets in fetched window, -1 indicates not found):")
    for k, v in atoms.items():
        print(f"  {k}: {v}")

    moov = atoms.get('moov')
    mdat = atoms.get('mdat')
    if moov is None and mdat is None:
        print("Could not find 'moov' or 'mdat' in the fetched ranges. The file may be too small or atoms are outside the sampled ranges.")
        sys.exit(0)

    if moov is not None and (mdat is None or moov < mdat):
        print("Likely streaming-friendly: 'moov' appears before 'mdat' in the sampled regions.")
    else:
        print("Likely NOT streaming-friendly: 'moov' appears after 'mdat' (or only found in tail). Consider remuxing with -movflags +faststart.")


if __name__ == '__main__':
    main()
