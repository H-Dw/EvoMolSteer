import hashlib
import io
import json
import tarfile

import pytest

from evomolsteer.io import digest
from evomolsteer.storage.generation_archive import verify_generation_archive, MANIFEST


def make_archive(tmp_path, extra=None, corrupt=False):
    payloads = {"results/example/COMPLETE.json": b'{"status":"complete"}',
                "inputs/example.txt": b"molecular coordinates\n"}
    checks = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
              for name, data in payloads.items()}
    manifest = {"format": "evomolsteer.generation_archive.v1", "complete": True,
                "campaign": "example", "code_commit": "abc", "files": checks}
    if corrupt:
        payloads["inputs/example.txt"] = b"Molecular coordinates\n"
    payloads[MANIFEST] = json.dumps(manifest).encode()
    if extra:
        payloads[extra] = b"unmanifested"
    archive = tmp_path / "example.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    meta = {"bytes": archive.stat().st_size, "archive_sha256": digest(archive),
            "files": len(checks), "source_bytes": sum(c["bytes"] for c in checks.values())}
    return archive, meta


def test_generation_archive_roundtrip_refuses_overwrite(tmp_path):
    archive, meta = make_archive(tmp_path)
    dest = tmp_path / "restored"
    result = verify_generation_archive(archive, meta, dest)
    assert result["verified"] and result["files_verified"] == 2
    assert (dest / "inputs/example.txt").read_bytes() == b"molecular coordinates\n"
    with pytest.raises(FileExistsError):
        verify_generation_archive(archive, meta, dest)


@pytest.mark.parametrize("extra", ["../outside.txt", "C:/outside.txt", "inputs/extra.txt"])
def test_generation_archive_rejects_unsafe_or_unmanifested_members(tmp_path, extra):
    archive, meta = make_archive(tmp_path, extra=extra)
    with pytest.raises(ValueError):
        verify_generation_archive(archive, meta, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_generation_archive_rejects_payload_and_transport_corruption(tmp_path):
    archive, meta = make_archive(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="Payload checksum"):
        verify_generation_archive(archive, meta, tmp_path / "restored")
    meta["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="transport"):
        verify_generation_archive(archive, meta, tmp_path / "restored")
