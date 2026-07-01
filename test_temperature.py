"""Schedule logic only — no eval/net needed (see plan.md §4 temperature schedule)."""
import jax.numpy as jnp

from jax_az.search import SearchConfig, scheduled_temperature


def test_no_schedule_is_constant():
    cfg = SearchConfig(temperature=0.7)  # temp_drop_ply None
    assert scheduled_temperature(cfg, 0) == 0.7
    assert scheduled_temperature(cfg, 99) == 0.7


def test_drop_after_cutoff():
    cfg = SearchConfig(temperature=1.0, temp_drop_ply=4, temp_final=0.0)
    assert float(scheduled_temperature(cfg, 0)) == 1.0
    assert float(scheduled_temperature(cfg, 3)) == 1.0   # last exploring ply
    assert float(scheduled_temperature(cfg, 4)) == 0.0   # drop at cutoff
    assert float(scheduled_temperature(cfg, 10)) == 0.0


def test_traced_ply_vectorizes():
    cfg = SearchConfig(temperature=1.0, temp_drop_ply=4, temp_final=0.0)
    plies = jnp.arange(6)
    got = scheduled_temperature(cfg, plies)
    assert got.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_multi_step_schedule():
    # 2.0 for ply<1, 1.0 for ply<6, then temp_final 0.0
    cfg = SearchConfig(temp_schedule=[[1, 2.0], [6, 1.0]], temp_final=0.0)
    got = scheduled_temperature(cfg, jnp.arange(8))
    assert got.tolist() == [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_schedule_overrides_drop_ply():
    cfg = SearchConfig(temp_drop_ply=4, temp_schedule=[[2, 1.5]], temp_final=0.0)
    assert float(scheduled_temperature(cfg, 1)) == 1.5
    assert float(scheduled_temperature(cfg, 3)) == 0.0  # schedule wins, not drop_ply


if __name__ == "__main__":
    test_no_schedule_is_constant()
    test_drop_after_cutoff()
    test_traced_ply_vectorizes()
    test_multi_step_schedule()
    test_schedule_overrides_drop_ply()
    print("ok")
