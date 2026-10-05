import copy
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import zipfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

import download_data
import prepare_data


def source_data(split):
    data = {
        "videos": [{"id": 16, "name": f"{split}/source/video"}],
        "tracks": [{"id": 101, "video_id": 16, "category_id": 1}],
        "categories": [{"id": 1, "name": "object"}],
        "images": [], "annotations": [],
    }
    for index in range(9):
        data["images"].append({
            "id": index, "video_id": 16, "frame_index": index * 10,
            "file_name": f"{split}/source/video/frame{index:04d}.jpg",
            "width": 100, "height": 200,
        })
        data["annotations"].append({
            "id": index, "video_id": 16, "track_id": 101, "image_id": index,
            "category_id": 1, "bbox": [10 + index, 20, 30, 40],
        })
    return data


@contextmanager
def serve_archive(payload, ignore_ranges=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            requested = self.headers.get("Range")
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested or "")
            start, end = 0, len(payload) - 1
            status = 200
            if match and not ignore_ranges:
                start = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
                status = 206
            body = payload[start:end + 1]
            requests.append({"range": requested, "bytes": len(body), "status": status})
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # A client rejecting unsupported ranges can close before consuming the body.
                return

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            size=len(payload), location=f"http://127.0.0.1:{server.server_port}/frames.zip",
            requests=requests,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "data"
        self.data = source_data("test")
        self.annotations = self.root / "test.json"
        self.annotations.write_text(json.dumps(self.data))
        self.args = ["--split", "test", "--n_shot", "2", "--output", str(self.output)]
        self.jpeg = io.BytesIO()
        Image.new("RGB", (100, 200), "blue").save(self.jpeg, format="JPEG")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
            for frame in self.data["images"]:
                zipped.writestr(
                    str(Path(frame["file_name"]).relative_to("test/source")),
                    self.jpeg.getvalue(),
                )
            zipped.writestr("unselected/padding.bin", b"x" * (5 * 1024 * 1024))
        self.archive = archive.getvalue()

    @contextmanager
    def source(self, ignore_ranges=False):
        with (
            serve_archive(self.archive, ignore_ranges) as metadata,
            patch.object(download_data, "hf_hub_download", return_value=str(self.annotations)),
            patch.object(download_data, "get_hf_file_metadata", return_value=metadata),
            redirect_stdout(io.StringIO()),
        ):
            yield metadata

    def test_selection_does_not_require_downloaded_images_but_local_preparation_does(self):
        for shots, ids in ((1, [0, 8]), (2, [0, 4, 8]), (4, [0, 2, 4, 6, 8])):
            manifest = prepare_data.select_manifest(self.data, "test", shots)
            self.assertEqual([frame["id"] for frame in manifest["images"]], ids)
            self.assertEqual([frame["role"] for frame in manifest["images"]],
                             ["support"] * shots + ["query"])
            with self.assertRaises(FileNotFoundError):
                prepare_data.build_manifest(self.data, self.output / "frames", "test", shots)

    def test_real_http_ranges_fetch_only_selected_files_and_resume_without_overwriting(self):
        with self.source() as metadata:
            download_data.main(self.args)
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in self.output.rglob("*") if path.is_file()}
            first_requests = metadata.requests[:]
            download_data.main(self.args)
        self.assertEqual(
            {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}, before,
        )
        self.assertTrue(first_requests)
        self.assertTrue(all(request["range"] and request["status"] == 206 for request in first_requests))
        self.assertLess(sum(request["bytes"] for request in first_requests), len(self.archive) // 2)
        manifest = json.loads((self.output / "test_2shot.json").read_text())
        self.assertEqual([frame["id"] for frame in manifest["images"]], [0, 4, 8])
        self.assertEqual(len(list(self.output.rglob("*.jpg"))), 3)
        self.assertEqual(len(before), 4)
        self.assertNotIn(metadata.location, json.dumps(manifest))
        for frame in manifest["images"]:
            self.assertEqual((self.output / "frames" / frame["file_name"]).read_bytes(), self.jpeg.getvalue())
            self.assertEqual(frame["sha256"], hashlib.sha256(self.jpeg.getvalue()).hexdigest())
            self.assertEqual(frame["jpeg_bytes"], len(self.jpeg.getvalue()))
        self.assertEqual(manifest["info"]["revision"], download_data.REVISION)
        self.assertEqual(manifest["info"]["annotation_sha256"],
                         hashlib.sha256(self.annotations.read_bytes()).hexdigest())

    def test_evaluation_shot_counts_share_images_without_changing_earlier_manifests(self):
        with self.source():
            previous = {}
            for shots in (1, 2, 4):
                download_data.main([
                    "--split", "test", "--n_shot", str(shots), "--output", str(self.output),
                ])
                for path, content in previous.items():
                    self.assertEqual(path.read_bytes(), content)
                path = self.output / f"test_{shots}shot.json"
                previous[path] = path.read_bytes()
        self.assertEqual(len(list(self.output.rglob("*.jpg"))), 5)

    def test_four_shot_training_manifest_uses_training_paths(self):
        self.annotations.write_text(json.dumps(source_data("train")))
        with self.source():
            download_data.main([
                "--split", "train", "--n_shot", "4", "--output", str(self.output),
            ])
        manifest = json.loads((self.output / "train_4shot.json").read_text())
        self.assertEqual(len(manifest["images"]), 5)
        self.assertTrue(all(frame["file_name"].startswith("train/") for frame in manifest["images"]))

    def test_dry_run_and_video_filter_write_nothing_or_access_frame_archives(self):
        data = copy.deepcopy(self.data)
        data["videos"].append({"id": 17, "name": "test/source/other-video"})
        data["tracks"].append({"id": 202, "video_id": 17, "category_id": 1})
        data["images"].extend([
            {**frame, "id": frame["id"] + 100, "video_id": 17,
             "file_name": frame["file_name"].replace("/video/", "/other-video/")}
            for frame in self.data["images"]
        ])
        data["annotations"].extend([
            {**annotation, "id": annotation["id"] + 100, "video_id": 17,
             "image_id": annotation["image_id"] + 100, "track_id": 202}
            for annotation in self.data["annotations"]
        ])
        self.annotations.write_text(json.dumps(data))
        with (
            patch.object(download_data, "hf_hub_download", return_value=str(self.annotations)),
            patch.object(download_data, "download_archive") as download,
            redirect_stdout(io.StringIO()) as output,
        ):
            download_data.main([*self.args, "--video_id", "16", "--dry_run"])
            self.assertIn("1 2-shot test examples, 3 images", output.getvalue())
            download_data.main([
                *self.args, "--video_id", "16", "--video_id", "17", "--dry_run",
            ])
        download.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertIn("2 2-shot test examples, 6 images", output.getvalue())

    def test_unknown_video_and_non_four_shot_training_fail_explicitly(self):
        with (
            patch.object(download_data, "hf_hub_download", return_value=str(self.annotations)) as source,
            patch.object(download_data, "download_archive") as download,
        ):
            with self.assertRaisesRegex(ValueError, "not in the test split"):
                download_data.main([*self.args, "--video_id", "999"])
            source.reset_mock()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                download_data.main(["--split", "train", "--n_shot", "1"])
            source.assert_not_called()
            download.assert_not_called()

    def test_conflicting_manifest_is_rejected_before_accessing_frame_archives(self):
        with self.source():
            download_data.main(self.args)
        path = self.output / "test_2shot.json"
        manifest = json.loads(path.read_text())
        manifest["info"]["revision"] = "different"
        path.write_text(json.dumps(manifest))
        original = path.read_bytes()
        with (
            patch.object(download_data, "hf_hub_download", return_value=str(self.annotations)),
            patch.object(download_data, "download_archive") as download,
            self.assertRaisesRegex(ValueError, "different selection or provenance"),
        ):
            download_data.main(self.args)
        download.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_corrupt_cached_image_is_not_replaced(self):
        path = self.output / "frames" / self.data["images"][0]["file_name"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"preserve corrupt existing file")
        with self.source(), self.assertRaisesRegex(ValueError, "Cached image differs"):
            download_data.main(self.args)
        self.assertEqual(path.read_bytes(), b"preserve corrupt existing file")
        self.assertFalse((self.output / "test_2shot.json").exists())

    def test_wrong_dimensions_and_missing_archive_member_do_not_publish_a_manifest(self):
        for change, error in (("dimensions", ValueError), ("missing", KeyError)):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                if change == "dimensions":
                    data["images"][0]["width"] = 101
                else:
                    data["images"][0]["file_name"] = "test/source/video/missing.jpg"
                self.annotations.write_text(json.dumps(data))
                with self.source(), self.assertRaises(error):
                    download_data.main(self.args)
                self.assertFalse((self.output / "test_2shot.json").exists())

    def test_range_support_is_required_without_full_archive_fallback(self):
        with self.source(ignore_ranges=True), self.assertRaisesRegex(ValueError, "range"):
            download_data.main(self.args)
        self.assertFalse((self.output / "test_2shot.json").exists())
        self.assertEqual(list(self.output.rglob("*.jpg")), [])

    def test_failed_download_does_not_publish_manifest(self):
        with (
            patch.object(download_data, "hf_hub_download", return_value=str(self.annotations)),
            patch.object(download_data, "download_archive", side_effect=OSError("network failure")),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(OSError, "network failure"),
        ):
            download_data.main(self.args)
        self.assertFalse((self.output / "test_2shot.json").exists())

    def test_unsafe_and_symlinked_destinations_are_rejected(self):
        for relative in ("../outside.jpg", "/absolute.jpg"):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                download_data.output_path(self.output, relative)
        self.output.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.output / "frames").symlink_to(outside, target_is_directory=True)
        with (
            patch.object(download_data, "hf_hub_download", return_value=str(self.annotations)),
            patch.object(download_data, "download_archive") as download,
            self.assertRaisesRegex(ValueError, "escapes"),
        ):
            download_data.main(self.args)
        download.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_atomic_publication_preserves_existing_files_and_cleans_temporary_files(self):
        path = self.root / "existing.json"
        path.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            download_data.write_new(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"existing")
        self.assertEqual(list(self.root.glob(".existing.json.*")), [])


if __name__ == "__main__":
    unittest.main()
