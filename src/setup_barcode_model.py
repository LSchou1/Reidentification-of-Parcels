"""Download the pinned barcode detector and verify its SHA-256 checksum."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/barcode_model.json"
DEFAULT_WEIGHTS = ROOT / "models/barcode_yolov8s.pt"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weights(destination=DEFAULT_WEIGHTS):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != config["sha256"]:
            raise ValueError(f"Checksum mismatch: {destination}. Move it aside before downloading again.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/{config['repository']}/resolve/"
           f"{config['revision']}/{config['file']}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
        if sha256(temporary) != config["sha256"]:
            raise ValueError("Downloaded model failed checksum verification.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args(argv)
    try:
        path = download_weights(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Barcode model ready; checksum verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
