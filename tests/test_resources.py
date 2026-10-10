import pytest

from tablely import resources
from tablely.resources import build_inventory, detect_gpus, format_cpu_list, parse_cpu_list


def test_cpu_list_round_trip():
    assert parse_cpu_list("0-3, 8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert format_cpu_list([11, 0, 1, 2, 3, 8, 10]) == "0-3,8,10-11"
    assert format_cpu_list([]) == ""
    for bad in ("3-1", "a", "-2"):
        with pytest.raises(ValueError):
            parse_cpu_list(bad)


def test_detect_gpus_respects_cuda_visible_devices():
    assert detect_gpus({"CUDA_VISIBLE_DEVICES": "2,3"}) == ["2", "3"]
    assert detect_gpus({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert detect_gpus({"CUDA_VISIBLE_DEVICES": "-1"}) == []
    assert detect_gpus({"CUDA_VISIBLE_DEVICES": "0,-1,1"}) == ["0"]


@pytest.fixture
def eight_cores_two_gpus(monkeypatch):
    monkeypatch.setattr(resources, "detect_cpus", lambda: list(range(8)))
    monkeypatch.setattr(resources, "detect_gpu_devices", lambda env=None: [("0", 24 * 1024 ** 3), ("1", None)])


def test_build_inventory_defaults_and_reserve(eight_cores_two_gpus):
    inv = build_inventory()
    assert inv.cpus == tuple(range(8)) and inv.gpus == ("0", "1")
    assert build_inventory(reserve_cpus=2).cpus == tuple(range(2, 8))
    with pytest.raises(ValueError):
        build_inventory(reserve_cpus=8)


def test_build_inventory_narrowing(eight_cores_two_gpus):
    assert build_inventory(cpus=4).cpus == (0, 1, 2, 3)
    assert build_inventory(cpus="4-7").cpus == (4, 5, 6, 7)
    assert build_inventory(gpus=1).gpus == ("0",)
    assert build_inventory(gpus=4).gpus == ("0", "1", "2", "3")  # more than detected: assumed ids
    assert build_inventory(gpus="1").gpus == ("1",)
    assert build_inventory(gpus=0).gpus == ()


def test_build_inventory_rejects_missing_cores_unless_simulating(eight_cores_two_gpus):
    with pytest.raises(ValueError):
        build_inventory(cpus=16)
    with pytest.raises(ValueError):
        build_inventory(cpus="6-9")
    assert build_inventory(cpus=16, simulate=True).cpus == tuple(range(16))


def test_gpu_memory_is_detected_or_configured(eight_cores_two_gpus):
    inv = build_inventory()
    assert inv.memory_of("0") == 24 * 1024 ** 3 and inv.memory_of("1") is None
    assert "0 (24GiB)" in inv.describe()
    assert build_inventory(gpu_memory="80GiB").gpu_memory == (80 * 1024 ** 3,) * 2


def test_parse_bytes():
    assert resources.parse_bytes("10GiB") == resources.parse_bytes("10G") == resources.parse_bytes("10gb") == 10 * 1024 ** 3
    assert resources.parse_bytes("512MiB") == 512 * 1024 ** 2 and resources.parse_bytes(4096) == 4096
    for bad in ("lots", "-1G", "0", True):
        with pytest.raises(ValueError):
            resources.parse_bytes(bad)


def test_mig_instances_replace_their_gpu():
    listing = """GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-aaaa)
  MIG 3g.40gb     Device  0: (UUID: MIG-bbbb)
  MIG 1g.10gb     Device  1: (UUID: MIG-cccc)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-dddd)
"""
    devices = resources.parse_gpu_listing(listing, {"0": 80 * 1024 ** 3, "1": 80 * 1024 ** 3})
    assert devices == [("MIG-bbbb", 40 * 1024 ** 3), ("MIG-cccc", 10 * 1024 ** 3), ("1", 80 * 1024 ** 3)]
