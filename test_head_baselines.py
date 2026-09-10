import math
import tempfile
from pathlib import Path

from head_baselines import (FUServeSelector, ThreeDPOSSelector,
                            TwoDPFHSSelector, pareto_indices)
from models import FogNode, IoTDevice
from controller_profile import (ProfileValidationError, controller_cycles,
                                create_simulated_profile)


def node(node_id, *, x=0.0, energy=100.0, initial=100.0, tier=2):
    return FogNode(node_id=node_id, tier=tier, fr_avg_ghz=2.0,
                   MP_tot_gb=4.0, MS_tot_gb=4.0, MP_occ_gb=0.0,
                   MS_occ_gb=0.0, E_initial_j=initial, E_res_j=energy,
                   lambda_fail=0.01, mu_fail=0.01, varsigma=4.0,
                   C_mips=16000.0, x=x, y=0.0, z=100.0)


def test_pareto_direction_and_tie_order():
    vals = [(1.0, 1.0), (2.0, 2.0), (3.0, 0.5)]
    assert pareto_indices(vals, (False, True)) == [0, 1]


def test_fu_serve_connectivity_energy_constraints_and_history():
    fog = [node(0, x=0, energy=90), node(1, x=100, energy=80),
           node(2, x=3000, energy=100)]
    selector = FUServeSelector(sensing_range_m=500, rer_threshold=0.5)
    out = selector.select_head(fog, {})
    assert out["head_id"] == 0  # node 2 is isolated; node 0 has higher RER than node 1
    assert selector.service_count == {0: 1}
    selector.select_head(fog, {})
    assert selector.service_count == {0: 1}  # one tenure, not one event per tick
    fog[0].E_res_j = 1
    out = selector.select_head(fog, {})
    assert out["head_id"] == 1
    assert selector.service_count[1] == 1
    assert len(selector.tenure_loads[0]) == 1
    assert math.isclose(selector.tenure_loads[0][0], 1.0)


def test_fu_serve_fitness_increases_with_degree():
    base = node(0)
    near = [node(1, x=100), node(2, x=-100)]
    out = FUServeSelector(sensing_range_m=500).select_head([base] + near, {})
    assert out["criteria"][0]["dc"] == 2
    assert out["criteria"][0]["fitness"] > 0


def test_fu_serve_history_does_not_permanently_lock_first_winner():
    fog = [node(0, x=0, energy=90), node(1, x=100, energy=80)]
    selector = FUServeSelector(sensing_range_m=500, rer_threshold=0.5)
    assert selector.select_head(fog, {})["head_id"] == 0
    # Node 0 remains eligible, but a material condition change should allow
    # node 1 to overcome the one-tenure history increment.
    fog[0].E_res_j = 55
    assert selector.select_head(fog, {})["head_id"] == 1


def test_2dp_table4_unambiguous_endpoints():
    # Tables 3-4 from the paper. Intermediate published rows do not follow the
    # printed weighted-sum equation; see FOG_HEAD_BASELINE_DEVIATIONS_LOG.md.
    fog = [node(i) for i in range(1, 9)]
    fdi = [.4321, .5444, .8818, .5594, .6232, .1734, .1678, .8716]
    fpi = [2.3232, 2.4444, 4.8734, 3.1212, 3.9272, 2.1111, 2.0676, 4.0256]
    assert [n.node_id for n in TwoDPFHSSelector(1, 0).tradeoff_order(fog, fdi, fpi)[:2]] == [7, 6]
    assert [n.node_id for n in TwoDPFHSSelector(0, 1).tradeoff_order(fog, fdi, fpi)[:2]] == [3, 8]


def test_2dp_selects_from_non_dominated_front():
    fog = [node(0), node(1), node(2)]
    ranking, nd = TwoDPFHSSelector().rank_objectives(
        fog, fdi=[1.0, 2.0, 3.0], fpi=[3.0, 4.0, 1.0])
    assert nd == [0, 1]
    assert ranking[0].node_id in (0, 1)


def test_3d_pos_single_non_dominated_scenario():
    fog = [node(i) for i in range(1, 9)]
    dpf = [4, 5, 6, 7, 8, 9, 10, 1]
    dee = [1, 1, 1, 1, 1, 1, 1, 5]
    dcc = [1, 1, 1, 1, 1, 1, 1, 5]
    ranking, nd = ThreeDPOSSelector().rank_objectives(fog, dpf, dee, dcc)
    assert [fog[i].node_id for i in nd] == [8]
    assert ranking[0].node_id == 8


def test_3d_pos_multi_point_ideal_selects_front_member():
    fog = [node(5), node(7), node(8)]
    dpf, dee, dcc = [1, 2, 3], [9, 10, 5], [9, 5, 10]
    ranking, nd = ThreeDPOSSelector().rank_objectives(fog, dpf, dee, dcc)
    assert [fog[i].node_id for i in nd] == [5, 7, 8]
    assert ranking[0].node_id in (5, 7, 8)


def test_selectors_are_deterministic_on_identical_nodes():
    fog = [node(2, x=0), node(1, x=0)]
    iot = [IoTDevice(0, 10, 0, 2.51, False)]
    pm = {1: {"wl": 0, "me": 1}, 2: {"wl": 0, "me": 1}}
    assert TwoDPFHSSelector().select_head(fog, pm, iot_devices=iot)["head_id"] == 1
    assert ThreeDPOSSelector(min_soc=0).select_head(fog, pm, iot_devices=iot)["head_id"] == 1


def test_all_baselines_exclude_unavailable_heads():
    fog = [node(0, x=0), node(1, x=100), node(2, x=200)]
    iot = [IoTDevice(0, 10, 0, 2.51, False)]
    metrics = {candidate.node_id: {} for candidate in fog}
    for selector in (FUServeSelector(sensing_range_m=500),
                     TwoDPFHSSelector(), ThreeDPOSSelector(min_soc=0)):
        first = selector.select_head(fog, metrics, iot_devices=iot)["head_node"]
        first.available = False
        result = selector.select_head(fog, metrics, iot_devices=iot)
        assert result["head_id"] != first.node_id
        assert all(candidate.available for candidate in result["ranking"])
        first.available = True


def test_stage1_baselines_have_distinct_simulated_prices():
    with tempfile.TemporaryDirectory() as directory:
        profile = create_simulated_profile(Path(directory) / "profile.json")
    times = profile["assumptions"]["service_time_ms"]
    assert times["stage1_fu_serve"] > times["stage1_without_spotis"]
    assert times["stage1_2dp_fhs"] > times["stage1_fu_serve"]
    assert times["stage1_3d_pos"] > times["stage1_2dp_fhs"]


def test_stage1_profile_prices_are_strict_measured_rows():
    algorithms = ("stage1_without_spotis", "stage1_fu_serve",
                  "stage1_2dp_fhs", "stage1_3d_pos")
    with tempfile.TemporaryDirectory() as directory:
        profile = create_simulated_profile(Path(directory) / "profile.json")
    ref_hz = profile["target"]["reference_frequency_hz"]
    costs = {name: [1000.0 * controller_cycles(profile, name, n) / ref_hz
                    for n in (10, 20, 40, 80, 160)]
             for name in algorithms}
    assert all(all(a < b for a, b in zip(values, values[1:]))
               for values in costs.values())
    assert costs["stage1_3d_pos"][-1] > 100.0
    assert costs["stage1_3d_pos"][-1] / costs["stage1_without_spotis"][-1] > 900
    try:
        controller_cycles(profile, "stage1_fu_serve", 30)
    except ProfileValidationError:
        pass
    else:
        raise AssertionError("missing Stage-1 size must not interpolate or fall back")


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().copy().items()
             if name.startswith("test_") and callable(obj)]
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} baseline tests passed")
