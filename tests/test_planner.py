import pytest

from tablely.planner import Allocation, Policy, check_feasible, plan, share_cpus
from tablely.resources import Inventory
from tablely.spec import JobSpec


def inv(cpus=8, gpus=2):
    return Inventory(cpus=tuple(range(cpus)), gpus=tuple(str(i) for i in range(gpus)))


def jobs(*specs):
    return {spec.name: spec for spec in specs}


def job(name, **kw):
    kw.setdefault("command", "true")
    return JobSpec(name=name, **kw)


def first_round(inventory, specs, policy=Policy()):
    return plan(inventory, specs, {}, list(specs), policy)


def test_gpus_go_to_the_highest_priority_jobs_first():
    specs = jobs(
        job("low", priority=1, device="gpu"),
        job("high", priority=9, device="gpu"),
        job("mid", priority=5, device="gpu"),
    )
    p = first_round(inv(gpus=2), specs)
    assert p.started == ["high", "mid"]
    assert p.allocations["high"].gpus == ("0",)
    assert p.allocations["mid"].gpus == ("1",)
    assert "low" in p.waiting


def test_equal_priority_keeps_submission_order():
    specs = jobs(job("a", device="gpu"), job("b", device="gpu"))
    p = first_round(inv(gpus=1), specs)
    assert p.started == ["a"]
    assert list(p.waiting) == ["b"]


def test_any_job_falls_back_to_cpu_when_gpus_are_taken():
    specs = jobs(job("gpu-job", priority=5, device="gpu"), job("flex", priority=1, device="any"))
    p = first_round(inv(gpus=1), specs)
    assert p.allocations["gpu-job"].on_gpu
    assert p.allocations["flex"].device == "cpu"
    assert p.allocations["flex"].gpus == ()


def test_cpu_jobs_never_get_gpus():
    specs = jobs(job("cpu-job", device="cpu"))
    p = first_round(inv(gpus=2), specs)
    assert p.allocations["cpu-job"].gpus == ()


def test_multi_gpu_job_gets_contiguous_set():
    specs = jobs(job("big", device="gpu", gpus=3))
    p = first_round(inv(gpus=4), specs)
    assert p.allocations["big"].gpus == ("0", "1", "2")


def test_strict_priority_holds_free_gpus_for_a_blocked_job():
    # "big" needs both GPUs but one is busy; the free one must not go to "small".
    specs = jobs(
        job("busy", priority=1, device="gpu"),
        job("big", priority=9, device="gpu", gpus=2),
        job("small", priority=5, device="any"),
    )
    running = {"busy": Allocation("gpu", ("0",), (0,))}
    p = plan(inv(gpus=2), specs, running, ["big", "small"])
    assert "big" in p.waiting
    assert p.allocations["small"].device == "cpu"  # not GPU 1


def test_backfill_lets_lower_priority_jobs_use_idle_gpus():
    specs = jobs(
        job("busy", priority=1, device="gpu"),
        job("big", priority=9, device="gpu", gpus=2),
        job("small", priority=5, device="gpu"),
    )
    running = {"busy": Allocation("gpu", ("0",), (0,))}
    p = plan(inv(gpus=2), specs, running, ["big", "small"], Policy(backfill=True))
    assert "big" in p.waiting
    assert p.allocations["small"].gpus == ("1",)


def test_blocked_gpu_job_reserves_its_minimum_cores_in_strict_mode():
    # 4 cores: busy holds 1, big (waiting for GPU) sets 2 aside, so a
    # 2-core CPU job no longer fits but a 1-core one does.
    specs = jobs(
        job("busy", priority=1, device="gpu"),
        job("big", priority=9, device="gpu", gpus=2, cpus=2),
        job("cpu2", priority=5, device="cpu", cpus=2),
    )
    running = {"busy": Allocation("gpu", ("0",), (0,))}
    p = plan(inv(cpus=4, gpus=2), specs, running, ["big", "cpu2"])
    assert set(p.waiting) == {"big", "cpu2"}

    specs = jobs(
        job("busy", priority=1, device="gpu"),
        job("big", priority=9, device="gpu", gpus=2, cpus=2),
        job("cpu1", priority=5, device="cpu", cpus=1),
    )
    p = plan(inv(cpus=4, gpus=2), specs, running, ["big", "cpu1"])
    assert p.started == ["cpu1"]


def test_core_starved_job_blocks_lower_priority_jobs_unless_backfill():
    specs = jobs(
        job("hog", priority=1, device="cpu", cpus=3),
        job("needs3", priority=9, device="cpu", cpus=3),
        job("needs1", priority=5, device="cpu", cpus=1),
    )
    running = {"hog": Allocation("cpu", (), (0, 1, 2))}
    strict = plan(inv(cpus=4, gpus=0), specs, running, ["needs3", "needs1"])
    assert strict.started == []
    assert strict.waiting["needs1"] == "cores held for a higher-priority job"

    backfill = plan(inv(cpus=4, gpus=0), specs, running, ["needs3", "needs1"], Policy(backfill=True))
    assert backfill.started == ["needs1"]


def test_spare_cores_are_split_by_priority():
    specs = jobs(job("a", priority=2, device="cpu"), job("b", priority=1, device="cpu"))
    p = first_round(inv(cpus=12, gpus=0), specs)
    assert len(p.allocations["a"].cpus) == 8
    assert len(p.allocations["b"].cpus) == 4


def test_gpu_jobs_keep_their_requested_cores_and_cpu_jobs_get_the_rest():
    specs = jobs(job("trainer", priority=10, device="gpu", cpus=4), job("sweep", priority=1, device="cpu"))
    p = first_round(inv(cpus=16, gpus=1), specs)
    assert len(p.allocations["trainer"].cpus) == 4
    assert len(p.allocations["sweep"].cpus) == 12


def test_max_cpus_caps_growth():
    specs = jobs(
        job("trainer", priority=10, device="gpu", cpus=2, max_cpus=6),
        job("capped", priority=5, device="cpu", max_cpus=2),
    )
    p = first_round(inv(cpus=16, gpus=1), specs)
    assert len(p.allocations["trainer"].cpus) == 6
    assert len(p.allocations["capped"].cpus) == 2  # remaining 8 cores stay idle


def test_core_sets_are_disjoint_and_cover_the_machine():
    specs = jobs(*(job(f"j{i}", priority=i + 1, device="cpu") for i in range(5)))
    p = first_round(inv(cpus=16, gpus=0), specs)
    seen = [cpu for alloc in p.allocations.values() for cpu in alloc.cpus]
    assert sorted(seen) == list(range(16))


def test_running_jobs_keep_gpus_and_existing_cores_when_rebalanced():
    specs = jobs(job("old", priority=1, device="cpu"), job("new", priority=1, device="cpu"))
    running = {"old": Allocation("cpu", (), tuple(range(8)))}
    p = plan(inv(cpus=8, gpus=0), specs, running, ["new"])
    assert p.allocations["old"].cpus == (0, 1, 2, 3)  # shrunk, but kept its own cores
    assert p.allocations["new"].cpus == (4, 5, 6, 7)

    running = {"old": Allocation("cpu", (), (4, 5, 6, 7))}
    p = plan(inv(cpus=8, gpus=0), jobs(specs["old"]), running, [])
    assert p.allocations["old"].cpus == tuple(range(8))  # grows back when alone


def test_running_gpu_assignment_is_never_moved():
    specs = jobs(job("a", priority=1, device="gpu"), job("b", priority=9, device="gpu"))
    running = {"a": Allocation("gpu", ("1",), (0,))}
    p = plan(inv(gpus=2), specs, running, ["b"])
    assert p.allocations["a"].gpus == ("1",)
    assert p.allocations["b"].gpus == ("0",)


@pytest.mark.parametrize("backfill", [False, True])
def test_top_job_always_starts_on_an_idle_machine(backfill):
    specs = jobs(
        job("cpu-heavy", priority=3, device="cpu", cpus=8),
        job("gpu-heavy", priority=2, device="gpu", gpus=2, cpus=8),
        job("tiny", priority=1, device="any"),
    )
    for name in specs:
        p = plan(inv(cpus=8, gpus=2), specs, {}, [name], Policy(backfill=backfill))
        assert p.started == [name]


def test_share_cpus_weighted_with_minimums_and_caps():
    assert share_cpus(10, [(1, None, 1.0), (1, None, 1.0)]) == [5, 5]
    assert share_cpus(10, [(4, None, 1.0), (1, None, 1.0)]) == [5, 5]
    assert share_cpus(10, [(1, 2, 5.0), (1, None, 1.0)]) == [2, 8]
    assert share_cpus(3, [(1, 1, 1.0), (1, 1, 1.0)]) == [1, 1]
    with pytest.raises(ValueError):
        share_cpus(1, [(1, None, 1.0), (1, None, 1.0)])


def test_check_feasible():
    errors, warnings = check_feasible(
        inv(cpus=4, gpus=1),
        [
            job("too-many-gpus", device="gpu", gpus=2),
            job("too-many-cores", device="cpu", cpus=5),
            job("flex", device="any", gpus=2),
            job("flex", device="cpu"),
        ],
    )
    assert any("too-many-gpus" in e for e in errors)
    assert any("too-many-cores" in e for e in errors)
    assert any("duplicate" in e for e in errors)
    assert len(warnings) == 1 and "flex" in warnings[0]


def test_elastic_job_takes_every_gpu_nobody_else_wants():
    specs = jobs(job("big", device="gpu", gpus=1, max_gpus="all"), job("cpu-side", device="cpu"))
    p = first_round(inv(gpus=4), specs)
    assert p.allocations["big"].gpus == ("0", "1", "2", "3")
    assert p.spare_gpus == []


def test_elastic_gpus_are_split_by_priority_and_capped():
    specs = jobs(
        job("a", priority=3, device="gpu", gpus=1, max_gpus=8),
        job("b", priority=1, device="gpu", gpus=1, max_gpus=2),
    )
    p = first_round(inv(gpus=8), specs)
    assert len(p.allocations["b"].gpus) == 2  # capped
    assert len(p.allocations["a"].gpus) == 6  # the rest
    assert not set(p.allocations["a"].gpus) & set(p.allocations["b"].gpus)


def test_no_elastic_growth_while_someone_waits_for_a_gpu():
    # 5 GPUs, 2 busy: "needs4" cannot start, so the 2 GPUs left after the
    # backfilled elastic job's minimum must stay free for it.
    specs = jobs(
        job("busy", priority=1, device="gpu", gpus=2),
        job("needs4", priority=9, device="gpu", gpus=4),
        job("elastic", priority=5, device="gpu", gpus=1, max_gpus="all"),
    )
    running = {"busy": Allocation("gpu", ("0", "1"), (0,))}
    p = plan(inv(gpus=5), specs, running, ["needs4", "elastic"], Policy(backfill=True))
    assert "needs4" in p.waiting
    assert len(p.allocations["elastic"].gpus) == 1  # takes its minimum only
    assert p.spare_gpus == []


def test_spare_gpus_are_reported_when_nobody_waits():
    specs = jobs(job("one", device="gpu"))
    p = first_round(inv(gpus=3), specs)
    assert p.spare_gpus == ["1", "2"]


def test_max_gpus_validation():
    with pytest.raises(ValueError, match="max_gpus"):
        job("x", device="gpu", gpus=2, max_gpus=1)
    with pytest.raises(ValueError, match="max_gpus"):
        job("x", device="gpu", max_gpus="many")
    assert job("c", device="cpu", max_gpus=4).max_gpus is None


# -- sharing one GPU ----------------------------------------------------------

GiB = 1024 ** 3


def mem_inv(*sizes, cpus=8):
    return Inventory(cpus=tuple(range(cpus)), gpus=tuple(str(i) for i in range(len(sizes))),
                     gpu_memory=tuple(sizes))


def test_small_jobs_share_one_gpu_and_leave_the_other_free():
    specs = jobs(job("a", gpu_share=0.5), job("b", gpu_share=0.25), job("c", gpu_share=0.25))
    p = first_round(inv(gpus=2), specs)
    assert [p.allocations[n].gpus for n in "abc"] == [("0",)] * 3
    assert [p.allocations[n].gpu_share for n in "abc"] == [0.5, 0.25, 0.25]
    assert p.gpu_load == {"0": 1.0, "1": 0.0}
    assert p.spare_gpus == ["1"]


def test_shared_jobs_pack_onto_the_fullest_gpu_they_fit():
    running = {
        "half": Allocation("gpu", ("0",), (0,), gpu_share=0.5),
        "most": Allocation("gpu", ("1",), (1,), gpu_share=0.75),
    }
    specs = jobs(job("half", gpu_share=0.5), job("most", gpu_share=0.75), job("q", gpu_share=0.25),
                 job("h", gpu_share=0.5))
    p = plan(inv(gpus=3), specs, running, ["q", "h"])
    assert p.allocations["q"].gpus == ("1",)  # fills GPU 1 exactly
    assert p.allocations["h"].gpus == ("0",)  # then GPU 0; GPU 2 stays whole
    assert p.spare_gpus == ["2"]


def test_whole_gpu_jobs_skip_partly_used_gpus():
    running = {"small": Allocation("gpu", ("0",), (0,), gpu_share=0.25)}
    specs = jobs(job("small", gpu_share=0.25), job("big", gpus=2))
    p = plan(inv(gpus=2), specs, running, ["big"])
    assert p.waiting["big"] == "needs 2 GPU(s), 1 free"
    p = plan(inv(gpus=3), specs, running, ["big"])
    assert p.allocations["big"].gpus == ("1", "2")


def test_gpu_memory_requests_become_shares_of_the_gpu_they_land_on():
    specs = jobs(job("a", gpu_memory="10GiB"), job("b", gpu_memory="12GiB"), job("c", gpu_memory="4GiB"))
    p = first_round(mem_inv(80 * GiB, 24 * GiB), specs)
    # Best fit: the 24GiB card fills up first, the 80GiB card stays free for big jobs.
    assert p.allocations["a"].gpus == ("1",) and p.allocations["b"].gpus == ("1",)
    assert p.allocations["a"].gpu_share == pytest.approx(10 / 24)
    assert p.allocations["c"].gpus == ("0",) and p.allocations["c"].gpu_share == pytest.approx(4 / 80)


def test_shared_job_waits_when_no_gpu_has_room():
    running = {"x": Allocation("gpu", ("0",), (0,), gpu_share=0.75)}
    specs = jobs(job("x", gpu_share=0.75), job("y", gpu_share=0.5), job("z", gpu_memory="20GiB"))
    p = plan(mem_inv(24 * GiB), specs, running, ["y", "z"], Policy(backfill=True))
    assert p.waiting["y"] == "needs 0.5 of a GPU, at most 0.25 free on one"
    assert p.waiting["z"] == "needs 20GiB of GPU memory, at most 6GiB free on one"


def test_shared_any_job_falls_back_to_cpu_and_strict_priority_holds_gpus():
    running = {"x": Allocation("gpu", ("0",), (0,), gpu_share=0.75)}
    specs = jobs(job("x", gpu_share=0.75), job("top", priority=5, gpu_share=0.5),
                 job("flex", device="any", gpu_share=0.25), job("small", gpu_share=0.25))
    p = plan(inv(gpus=1), specs, running, ["top", "flex", "small"])
    assert "top" in p.waiting
    assert p.allocations["flex"].device == "cpu"  # the 0.25 left is held for "top"
    assert p.waiting["small"] == "GPUs held for a higher-priority job"


def test_shared_jobs_never_grow_onto_spare_gpus():
    specs = jobs(job("s", gpu_share=0.5), job("e", max_gpus="all"))
    p = first_round(inv(gpus=3), specs)
    assert p.allocations["s"].gpus == ("0",) and p.allocations["e"].gpus == ("1", "2")


def test_gpu_sharing_validation():
    for bad in ({"gpu_share": 0}, {"gpu_share": 1}, {"gpu_share": "half"}, {"gpu_memory": "lots"},
                {"gpu_share": 0.5, "gpu_memory": "1GiB"}, {"gpu_share": 0.5, "gpus": 2},
                {"gpu_share": 0.5, "max_gpus": 2}):
        with pytest.raises(ValueError):
            job("x", **bad)
    assert job("x", gpu_memory="1GiB").gpu_memory == GiB
    assert job("c", device="cpu", gpu_share=0.5).shared is False


def test_check_feasible_needs_gpu_memory_sizes_for_memory_requests():
    errors, _ = check_feasible(inv(gpus=1), [job("m", gpu_memory="8GiB")])
    assert "memory size is unknown" in errors[0]
    errors, _ = check_feasible(mem_inv(24 * GiB), [job("m", gpu_memory="40GiB")])
    assert "largest GPU has 24GiB" in errors[0]
    errors, warnings = check_feasible(mem_inv(24 * GiB), [job("m", device="any", gpu_memory="40GiB")])
    assert not errors and "always run on CPU" in warnings[0]
    assert check_feasible(mem_inv(24 * GiB), [job("m", gpu_memory="24GiB")]) == ([], [])
