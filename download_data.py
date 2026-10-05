"""Download only selected TAO support/query JPEGs and write a FOCUS manifest."""

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from PIL import Image

if __package__:
    from . import prepare_data
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare_data


REPO_ID = "chengyenhsieh/TAO-Amodal"
REVISION = "5604e751e3c1c130e31c81e9e80cee85a5420dfc"
ANNOTATIONS = {
    "train": "annotations/train.json",
    "test": "annotations/tao_test_annotations.json",
}


def output_path(output, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe output path: {relative}")
    path = output / relative
    if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
        raise ValueError(f"Output path escapes the dataset or is a symlink: {path}")
    return path


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            # Publish a complete file atomically, without replacing an existing file.
            os.link(temporary, path)
        finally:
            temporary.unlink()


def download_archive(archive_path, frames, output):
    metadata = get_hf_file_metadata(
        hf_hub_url(REPO_ID, archive_path, repo_type="dataset", revision=REVISION),
        token=True,
    )
    if metadata.size is None:
        raise ValueError(f"Archive size unavailable: {archive_path}")
    prefix = PurePosixPath(archive_path).relative_to("frames").with_suffix("")
    members = {
        frame["id"]: str(PurePosixPath(frame["file_name"]).relative_to(prefix))
        for frame in frames
    }
    downloaded = reused = 0
    fs = HTTPFileSystem()
    # Signed URLs remain in memory. ZIP indexes and JPEGs are read with HTTP ranges.
    with fs.open(
        metadata.location, "rb", size=metadata.size,
        block_size=256 * 1024, cache_type="bytes",
    ) as remote:
        with zipfile.ZipFile(remote) as archive:
            ordered = sorted(
                frames, key=lambda frame: archive.getinfo(members[frame["id"]]).header_offset,
            )
            for index, frame in enumerate(ordered, 1):
                entry = archive.getinfo(members[frame["id"]])
                destination = output_path(output, f"frames/{frame['file_name']}")
                cached = destination.exists()
                if cached:
                    content = destination.read_bytes()
                    if len(content) != entry.file_size or zlib.crc32(content) != entry.CRC:
                        raise ValueError(
                            f"Cached image differs from the pinned source: {destination}. "
                            "Move or remove that file explicitly before retrying."
                        )
                    reused += 1
                else:
                    content = archive.read(entry)
                    downloaded += 1
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    if image.format != "JPEG" or image.size != (frame["width"], frame["height"]):
                        raise ValueError(f"JPEG format or dimensions mismatch: {frame['file_name']}")
                if not cached:
                    write_new(destination, content)
                frame["sha256"] = hashlib.sha256(content).hexdigest()
                frame["jpeg_bytes"] = len(content)
                if index % 25 == 0 or index == len(ordered):
                    print(f"{archive_path}: {index}/{len(ordered)} selected images", flush=True)
        fetched = remote.cache.total_requested_bytes
    return {
        "archive": archive_path, "downloaded": downloaded, "reused": reused,
        "range_bytes_fetched": fetched,
    }


def selection_without_checksums(data):
    return {
        **data,
        "images": [
            {key: value for key, value in frame.items() if key not in ("sha256", "jpeg_bytes")}
            for frame in data["images"]
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(ANNOTATIONS), required=True)
    parser.add_argument("--n_shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--output", type=prepare_data.training.local_path, default=Path("data"))
    parser.add_argument(
        "--video_id", type=int, action="append",
        help="Optionally select one video ID; repeat to select several. Use a separate output directory.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the selection without fetching frame archives or writing outputs; annotations may download.",
    )
    args = parser.parse_args(argv)
    if args.split == "train" and args.n_shot != 4:
        parser.error("The FOCUS workflow trains with four shots; use --n_shot 4 for train")
    manifest_path = output_path(args.output, f"{args.split}_{args.n_shot}shot.json")
    annotation_path = Path(hf_hub_download(
        REPO_ID, ANNOTATIONS[args.split], repo_type="dataset", revision=REVISION, token=True,
    ))
    source_bytes = annotation_path.read_bytes()
    source = json.loads(source_bytes)
    if args.video_id is not None:
        requested = set(args.video_id)
        missing = requested - {video["id"] for video in source["videos"]}
        if missing:
            raise ValueError(f"Video IDs not in the {args.split} split: {sorted(missing)}")
        source["videos"] = [video for video in source["videos"] if video["id"] in requested]
    data = prepare_data.select_manifest(source, args.split, args.n_shot)
    data["info"].update({
        "repo_id": REPO_ID, "revision": REVISION, "annotation_file": ANNOTATIONS[args.split],
        "annotation_sha256": hashlib.sha256(source_bytes).hexdigest(),
    })
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if selection_without_checksums(previous) != data:
            raise ValueError(f"Existing manifest has different selection or provenance: {manifest_path}")
    for frame in data["images"]:
        output_path(args.output, f"frames/{frame['file_name']}")
    print(
        f"Selected {len(data['videos'])} {args.n_shot}-shot {args.split} examples, "
        f"{len(data['images'])} images; skipped {len(data['info']['skipped_videos'])} videos.",
        flush=True,
    )
    if args.dry_run:
        print("No frame archives fetched or outputs written; annotation JSON may be cached.")
        return

    archives = defaultdict(list)
    for frame in data["images"]:
        split, component, _ = frame["file_name"].split("/", 2)
        archives[f"frames/{split}/{component}.zip"].append(frame)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_archive, archive, frames, args.output)
            for archive, frames in sorted(archives.items())
        ]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    if previous is not None:
        if previous != data:
            raise ValueError(f"Existing manifest checksums differ; refusing to overwrite {manifest_path}")
        print(f"Reused unchanged manifest: {manifest_path}", flush=True)
    else:
        write_new(manifest_path, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode())
        print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
