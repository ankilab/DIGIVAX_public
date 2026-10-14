"""Select the images to process and write <out>/technical/images.tsv (stdlib only).

    python -m digivax_runner.prepare <image_dir> <out_dir> [-n N]

Writes images.tsv (all selected images) and todo.tsv (those not finished yet,
so an interrupted run resumes) and prints "<selected> <remaining>".

Output layout of a run folder:
    fhir/<key>.json           FHIR R4 Bundle per image
    standard/<key>.json       extracted label (country + entries, no bounding boxes)
    digital_card/<key>.png    rendered vaccination card
    technical/                everything else; technical/raw/<key>.json is the
                              full record and marks the image as done
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def tech_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "technical"


def done_path(out_dir: Path, key: str) -> Path:
    """Full per-image record; written last, so its presence means the image is done."""
    return tech_dir(out_dir) / "raw" / f"{key}.json"


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".") and p.is_file())


def image_keys(paths: list[Path]) -> dict[Path, str]:
    """Output file name per image: the stem, or the full file name when two
    images share a stem (scan_01.jpg + scan_01.png)."""
    stems = Counter(p.stem for p in paths)
    return {p: (p.stem if stems[p.stem] == 1 else p.name) for p in paths}


def read_images_tsv(out_dir: Path, name: str = "images.tsv") -> list[tuple[str, str]]:
    rows = []
    for line in (tech_dir(out_dir) / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, path = line.split("\t", 1)
            rows.append((key, path))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("-n", type=int, default=None, help="Only the first N images (sorted by name).")
    ap.add_argument("--allow-local", action="store_true",
                    help="Accept images under /tmp or /dev/shm (only for runs on this node).")
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        sys.exit(f"ERROR: image folder not found: {args.image_dir}")
    images = list_images(args.image_dir.resolve())
    if not images:
        sys.exit(f"ERROR: no images ({', '.join(sorted(IMAGE_EXTS))}) in {args.image_dir}")
    keys = image_keys(images)          # over the whole folder, so keys do not depend on -n
    if args.n is not None:
        images = images[:args.n]
    # The GPU job reads the files on another node: resolve symlinks so a folder
    # of links works, and refuse node-local storage that the job cannot see.
    keys = {p.resolve(): keys[p] for p in images}
    images = list(keys)
    local = [p for p in images if str(p).startswith(("/tmp/", "/dev/shm/"))]
    if local and not args.allow_local:
        sys.exit(f"ERROR: {len(local)} image(s) are on node-local storage ({local[0].parent}), "
                 f"which the GPU node cannot read. Copy them to $WORK or $HOME, "
                 f"or use --direct on a GPU node.")

    todo = [p for p in images if not done_path(args.out_dir, keys[p]).exists()]

    # images.tsv = everything selected (what the summary reports on);
    # todo.tsv   = what still needs a GPU, split evenly across array tasks.
    tech = tech_dir(args.out_dir)
    tech.mkdir(parents=True, exist_ok=True)
    for name, paths in (("images.tsv", images), ("todo.tsv", todo)):
        (tech / name).write_text(
            "".join(f"{keys[p]}\t{p}\n" for p in paths), encoding="utf-8")
    print(len(images), len(todo))


if __name__ == "__main__":
    main()
