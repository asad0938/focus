"""Prepare a k-shot manifest from local TAO annotations and extracted JPEGs."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

if __package__:
    from . import train as training
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train as training


def select_manifest(data, split, n_shot):
    if n_shot < 1:
        raise ValueError("n_shot must be positive")
    grouped = {key: defaultdict(list) for key in ("images", "annotations", "tracks")}
    for key, groups in grouped.items():
        for record in data[key]:
            groups[record["video_id"]].append(record)
    result = {
        "info": {
            "split": split, "n_shot": n_shot, "bbox_format": "xywh_pixels",
            "bbox_source": "Original TAO bbox, not amodal_bbox", "skipped_videos": [],
        },
        "videos": [], "images": [], "annotations": [], "tracks": [], "categories": [],
    }
    categories = {category["id"]: category for category in data["categories"]}
    used_categories, file_names = set(), set()
    for video in data["videos"]:
        video_name = PurePosixPath(video["name"])
        if not video_name.parts or video_name.parts[0] != split:
            raise ValueError(f"Video {video['id']} is not in the {split} split")
        video_data = {key: groups[video["id"]] for key, groups in grouped.items()}
        track = training.select_track(video_data, video["id"])
        distinct_frames = {
            annotation["image_id"] for annotation in video_data["annotations"]
            if annotation["track_id"] == track["id"]
        }
        if len(distinct_frames) < n_shot + 1:
            reason = f"Only {len(distinct_frames)} distinct annotated frames; need {n_shot + 1}"
            result["info"]["skipped_videos"].append({
                "video_id": video["id"], "track_id": track["id"], "reason": reason,
            })
            print(f"Skipping video {video['id']}, track {track['id']}: {reason}", file=sys.stderr)
            continue
        selected = training.select_frames(video_data, track["id"], n_shot, video["id"])
        category = categories[track["category_id"]]
        if not isinstance(category.get("name"), str) or not category["name"].strip():
            raise ValueError(f"Category {category['id']} has no object name")
        for index, (annotation, frame) in enumerate(selected):
            relative = PurePosixPath(frame["file_name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe image path: {relative}")
            relative.relative_to(video_name)
            if str(relative) in file_names:
                raise ValueError(f"Duplicate image path: {relative}")
            file_names.add(str(relative))
            if annotation["category_id"] != track["category_id"]:
                raise ValueError(f"Annotation {annotation['id']} category differs from track")
            result["images"].append({
                **{key: frame[key] for key in (
                    "id", "video_id", "file_name", "frame_index", "width", "height",
                )},
                "role": "support" if index < n_shot else "query",
            })
            result["annotations"].append({
                **{key: annotation[key] for key in (
                    "id", "video_id", "track_id", "image_id", "category_id", "bbox",
                )},
                "category_name": category["name"],
            })
        result["videos"].append({key: video[key] for key in ("id", "name")})
        result["tracks"].append({key: track[key] for key in ("id", "video_id", "category_id")})
        used_categories.add(category["id"])
    result["categories"] = [
        {"id": category_id, "name": categories[category_id]["name"]}
        for category_id in sorted(used_categories)
    ]
    if not result["videos"]:
        raise ValueError(f"No videos have {n_shot + 1} distinct annotated frames")
    return result


def build_manifest(data, frames_root, split, n_shot):
    result = select_manifest(data, split, n_shot)
    for frame in result["images"]:
        path = frames_root / frame["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=training.local_path, required=True)
    parser.add_argument("--frames", type=training.local_path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--n_shot", type=training.positive_int, required=True)
    parser.add_argument("--output", type=training.local_path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    data = json.loads(args.annotations.read_text())
    result = build_manifest(data, args.frames, args.split, args.n_shot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(
        f"Saved {len(result['videos'])} {args.n_shot}-shot examples "
        f"({len(result['images'])} images) to {args.output}; "
        f"skipped {len(result['info']['skipped_videos'])} videos."
    )


if __name__ == "__main__":
    main()
