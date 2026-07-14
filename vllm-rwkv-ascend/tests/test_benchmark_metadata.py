import pytest

from perf.benchmark_metadata import (
    checkpoint_metadata,
    collect_cann_metadata,
    collect_npu_metadata,
    infer_huggingface_revision,
    npu_device_id,
)


class _FakeNPU:
    @staticmethod
    def get_device_name(device):
        assert device == "npu:3"
        return "Ascend 910B2C"

    @staticmethod
    def device_count():
        return 4


class _FakeTorch:
    __version__ = "2.9.0"
    npu = _FakeNPU()


class _FakeTorchNPU:
    __version__ = "2.9.0.post1"


def test_collect_npu_metadata_has_pairing_identity():
    metadata = collect_npu_metadata(
        _FakeTorch, _FakeTorchNPU, "npu:3", device_count=2
    )

    assert metadata["device"] == "npu:3"
    assert metadata["device_name"] == "Ascend 910B2C"
    assert metadata["device_count"] == 2
    assert metadata["visible_device_count"] == 4
    assert metadata["torch"] == "2.9.0"
    assert metadata["torch_npu"] == "2.9.0.post1"
    assert metadata["python"]


def test_checkpoint_metadata_uses_absolute_path_and_exact_size(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"rwkv7")

    metadata = checkpoint_metadata(str(checkpoint))

    assert metadata["model"] == str(checkpoint.resolve())
    assert metadata["checkpoint_bytes"] == 5


def test_checkpoint_metadata_sums_model_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}")
    shard = model / "weights"
    shard.mkdir()
    (shard / "model.safetensors").write_bytes(b"1234")

    metadata = checkpoint_metadata(str(model))

    assert metadata["model"] == str(model.resolve())
    assert metadata["checkpoint_bytes"] == 6


def test_npu_device_id_is_strict():
    assert npu_device_id("npu:3") == 3
    with pytest.raises(ValueError, match="expected npu:N"):
        npu_device_id("cuda:0")


def test_huggingface_revision_is_inferred_only_when_unique(tmp_path):
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    revision = "a" * 40
    (metadata / "config.metadata").write_text(
        revision + "\netag\n123\n", encoding="utf-8"
    )

    assert infer_huggingface_revision(tmp_path) == revision

    (metadata / "weights.metadata").write_text(
        "b" * 40 + "\netag\n123\n", encoding="utf-8"
    )
    assert infer_huggingface_revision(tmp_path) is None


def test_collect_cann_metadata_reads_runtime_version(tmp_path, monkeypatch):
    version_file = tmp_path / "share" / "info" / "runtime" / "version.info"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("Version=8.5.1\n", encoding="utf-8")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))

    metadata = collect_cann_metadata()

    assert metadata["cann"] == "8.5.1"
    assert metadata["ascend_home_path"] == str(tmp_path.resolve())
