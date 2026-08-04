"""Rule and belief tests.

Runs under pytest, or standalone: `python -m tests.test_all`.

The engine-fidelity question is answered by `tools/verify_engine.py`, which
diffs the whole simulator against the official JAX implementation. These tests
cover the things that module cannot: the *derived* rules the policy reasons
with, and the inferences the belief state makes.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.belief import Belief
from bot.board import bfs_field_from, room_field
from sim import engine, mapgen


def _blank(h=7, w=7):
    grid = np.zeros((h, w), dtype=np.int32)
    grid[0, 0] = 1
    grid[h - 1, w - 1] = 2
    return grid


# --- castle pricing -----------------------------------------------------------
def test_build_cost_is_flat_far_from_own_structures():
    structures = np.zeros((15, 15), dtype=bool)
    structures[7, 7] = True
    cost = rules.build_cost_grid(structures)
    assert cost[7, 7] == 35 + 14                       # on top of it: +14
    assert cost[7, 8] == 35 + 12                       # adjacent: +12
    assert cost[7, 13] == 35 + 2                       # manhattan 6: +2
    assert cost[7, 14] == 35                           # manhattan 7: flat
    assert cost[0, 0] == 35


def test_build_cost_stacks_over_structures():
    structures = np.zeros((15, 15), dtype=bool)
    structures[7, 7] = True
    structures[7, 9] = True
    cost = rules.build_cost_grid(structures)
    # distance 1 from one, distance 1 from the other
    assert cost[7, 8] == 35 + 12 + 12


# --- growth phase -------------------------------------------------------------
def test_structures_grow_on_even_ticks_only():
    grid = _blank()
    st = engine.from_grid(grid)
    seen = []
    for _ in range(6):
        engine.step(st, [1, 0, 0, 0, 0], [1, 0, 0, 0, 0])
        seen.append(int(st.armies[0, 0]))
    # spawns with 1; +1 at ticks 2, 4, 6
    assert seen == [1, 2, 2, 3, 3, 4], seen


def test_every_tile_grows_on_the_fiftieth_tick():
    grid = _blank()
    st = engine.from_grid(grid)
    st.own[0][1, 1] = True
    st.neutral[1, 1] = False
    st.time = 48
    engine.step(st, [1, 0, 0, 0, 0], [1, 0, 0, 0, 0])   # -> 49, nothing
    assert int(st.armies[1, 1]) == 0
    engine.step(st, [1, 0, 0, 0, 0], [1, 0, 0, 0, 0])   # -> 50, all tiles +1
    assert int(st.armies[1, 1]) == 1


# --- combat and move order ----------------------------------------------------
def test_ties_favour_the_defender():
    grid = _blank()
    st = engine.from_grid(grid)
    st.armies[0, 0] = 4          # moves 3
    st.own[0][1, 0] = True       # not adjacent to the fight, keeps it simple
    st.neutral[1, 0] = False
    st.armies[0, 1] = 3          # neutral tile holding 3
    engine.step(st, [0, 0, 0, 3, 0], [1, 0, 0, 0, 0])
    assert not st.own[0][0, 1], "3 vs 3 must not capture"
    assert int(st.armies[0, 1]) == 0


def test_bigger_army_resolves_second_and_holds_the_tile():
    """chasing > reinforcing > smaller-army-first: the big stack moves last."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, 0] = 1
    grid[4, 4] = 2
    st = engine.from_grid(grid)
    for cell, player, army in (((2, 1), 0, 10), ((2, 3), 1, 4)):
        st.own[player][cell] = True
        st.neutral[cell] = False
        st.armies[cell] = army
    # both move onto (2, 2)
    engine.step(st, [0, 2, 1, 3, 0], [0, 2, 3, 2, 0])
    assert st.own[0][2, 2], "player 0 had the bigger army and should hold the cell"


# --- deathtouch ---------------------------------------------------------------
def _two_player_board():
    grid = np.zeros((9, 9), dtype=np.int32)
    grid[0, 0] = 1
    grid[8, 8] = 2
    return engine.from_grid(grid)


def test_deathtouch_ignores_army_after_turn_800():
    st = _two_player_board()
    st.time = rules.DEATHTOUCH_TURN
    st.armies[8, 8] = 9999
    st.own[0][8, 7] = True
    st.neutral[8, 7] = False
    st.armies[8, 7] = 2                       # moves 1 unit onto a 9999 general
    done = engine.step(st, [0, 8, 7, 3, 0], [1, 0, 0, 0, 0])
    assert done and st.winner == 0


def test_before_turn_800_the_same_move_just_bounces():
    st = _two_player_board()
    st.time = rules.DEATHTOUCH_TURN - 50
    st.armies[8, 8] = 9999
    st.own[0][8, 7] = True
    st.neutral[8, 7] = False
    st.armies[8, 7] = 2
    done = engine.step(st, [0, 8, 7, 3, 0], [1, 0, 0, 0, 0])
    assert not done and st.winner == -1


def test_chase_defence_beats_a_touch():
    """Taking the attacker's source cell first cancels the touch."""
    st = _two_player_board()
    st.time = rules.DEATHTOUCH_TURN
    for cell, player, army in (((8, 7), 0, 2), ((7, 7), 1, 20)):
        st.own[player][cell] = True
        st.neutral[cell] = False
        st.armies[cell] = army
    # p0 touches (8,7)->(8,8); p1 chases (7,7)->(8,7), which resolves first
    done = engine.step(st, [0, 8, 7, 3, 0], [0, 7, 7, 1, 0])
    assert st.winner != 0, "the chase should have taken the source before the touch"
    assert not done or st.winner == 1


def test_mutual_capture_is_a_draw():
    st = _two_player_board()
    st.time = rules.DEATHTOUCH_TURN
    for cell, player in (((8, 7), 0), ((0, 1), 1)):
        st.own[player][cell] = True
        st.neutral[cell] = False
        st.armies[cell] = 5
    done = engine.step(st, [0, 8, 7, 3, 0], [0, 0, 1, 2, 0])
    assert done and st.winner == -1


# --- fog ----------------------------------------------------------------------
def test_vision_is_the_full_3x3_around_owned_tiles():
    st = _two_player_board()
    obs = engine.observe(st, 0)
    visible = (obs.type_grid >= rules.T_PLAIN) & (obs.type_grid <= rules.T_GENERAL)
    assert visible[:2, :2].all()
    assert not visible[2, 2]
    assert int(visible.sum()) == 4          # corner general sees a 2x2


# --- belief -------------------------------------------------------------------
def test_terrain_is_fully_known_on_the_first_frame():
    for seed in range(12):
        grid = mapgen.generate(seed)
        st = engine.from_grid(grid)
        b = Belief(0, *grid.shape)
        b.update(engine.observe(st, 0))
        assert np.array_equal(b.mountains, grid == -2), f"seed {seed}"


def test_enemy_general_prior_contains_the_truth():
    """The generator's own spawn constraints, run in reverse."""
    hits = 0
    sizes = []
    for seed in range(25):
        grid = mapgen.generate(seed)
        st = engine.from_grid(grid)
        b = Belief(0, *grid.shape)
        b.update(engine.observe(st, 0))
        sizes.append(int(b.candidates.sum()))
        if b.candidates[st.gpos[1]]:
            hits += 1
    assert hits == 25, f"prior missed the general in {25 - hits} boards"
    mean = sum(sizes) / len(sizes)
    passable = 18 * 18 * 0.75
    assert mean < passable / 3, f"prior is not narrowing anything (mean {mean:.0f})"


def test_a_new_structure_in_fog_is_read_as_an_enemy_castle():
    grid = mapgen.generate(3)
    st = engine.from_grid(grid)
    b = Belief(0, *grid.shape)
    b.update(engine.observe(st, 0))
    assert not b.enemy_castles.any()

    # opponent builds far away, out of our sight
    r, c = st.gpos[1]
    st.castles[r, c - 1 if c else c + 1] = True
    b.update(engine.observe(st, 0))
    assert b.enemy_castles.sum() == 1


def test_room_field_matches_a_direct_count():
    grid = mapgen.generate(5)
    passable = grid != -2
    room = room_field(passable, 7)
    for cell in [(0, 0), (5, 5), (10, 9)]:
        if not passable[cell]:
            continue
        d = bfs_field_from(passable, cell)
        assert room[cell] == int(((d <= 7) & passable).sum()) - 1


def test_numpy_conv_matches_the_training_conv():
    """The submission runs numpy; training runs JAX. If the two convolutions
    disagree the network trains fine and plays like noise, and nothing about the
    failure points at the forward pass."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return
    from bot.policy.net import _conv3x3

    rng = np.random.default_rng(0)
    for cin, cout, n in ((6, 5, 7), (12, 32, 21), (32, 8, 21)):
        x = rng.normal(size=(cin, n, n)).astype("f4")
        w = rng.normal(size=(cout, cin, 3, 3)).astype("f4")
        b = rng.normal(size=cout).astype("f4")
        # HIGHEST forces true float32: on a GPU jax defaults to TF32, whose
        # ~10-bit mantissa gives errors around 1e-2 at these magnitudes and would
        # make this test about float formats rather than about weight layout.
        ref = np.asarray(jax.lax.conv_general_dilated(
            jnp.asarray(x)[None], jnp.asarray(w), (1, 1), "SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            precision=jax.lax.Precision.HIGHEST)[0]) + b[:, None, None]
        got = _conv3x3(x, w, b)
        scale = max(float(np.abs(ref).max()), 1.0)
        assert np.abs(got - ref).max() < 1e-4 * scale, f"conv mismatch at {cin}->{cout}"


def test_a_checkpoint_states_its_own_architecture():
    """Load a 6-layer file into a loader that assumes 4 and the extra layers are
    silently dropped: the head still matmuls, the logits still have the right
    shape, and the bot plays confident nonsense. So the loader assumes nothing,
    and anything it cannot read unambiguously is an error."""
    import tempfile
    from pathlib import Path

    from bot import features
    from bot.obs import Obs
    from bot.policy import net as npnet

    rng = np.random.default_rng(0)

    def weights(layers, ch, residual):
        p, prev = {}, features.C
        for n in npnet.trunk_keys(layers, residual):
            p[f"{n}_w"] = (rng.normal(size=(ch, prev, 3, 3)) * 0.2).astype("f4")
            p[f"{n}_b"] = (rng.normal(size=ch) * 0.2).astype("f4")
            prev = ch
        p["head_w"] = (rng.normal(size=(features.PER_CELL, ch, 3, 3)) * 0.2).astype("f4")
        p["head_b"] = np.zeros(features.PER_CELL, "f4")
        p["pass_w"] = np.zeros(ch, "f4")
        p["pass_b"] = np.float32(0.0)
        return p

    ty = np.full((18, 18), rules.T_PLAIN, dtype=np.int8)
    ty[0, 0] = rules.T_GENERAL
    ow = np.zeros((18, 18), dtype=np.int8)
    ow[0, 0] = rules.OWNER_ME
    ar = np.zeros((18, 18), dtype=np.int32)
    ar[0, 0] = 9
    obs = Obs(H=18, W=18, turn=1, my_land=1, my_army=9, opp_land=1, opp_army=1,
              type_grid=ty, owner_grid=ow, army_grid=ar)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for layers, ch, residual in ((4, 32, False), (6, 16, False), (5, 8, True)):
            p = weights(layers, ch, residual)
            np.savez(td / "w.npz", **p, **npnet.arch_record(p))
            net = npnet.Net(str(td / "w.npz"))
            assert net.arch == {"layers": layers, "channels": ch, "residual": residual}
            assert net.logits(obs).shape == (features.N_ACTIONS,)

        # a residual file whose keys were renamed into a plain stack has entirely
        # valid shapes; only the marker says it would compute the wrong thing
        p = weights(5, 8, True)
        renamed = {"conv0_w": p["conv0_w"], "conv0_b": p["conv0_b"], "residual": np.int8(1)}
        for i, n in enumerate(["res0a", "res0b", "res1a", "res1b"], start=1):
            renamed[f"conv{i}_w"], renamed[f"conv{i}_b"] = p[f"{n}_w"], p[f"{n}_b"]
        renamed.update({k: p[k] for k in ("head_w", "head_b", "pass_w", "pass_b")})
        np.savez(td / "renamed.npz", **renamed)

        # and a gap in the numbering is a truncated file, not a shallow network
        p = weights(4, 32, False)
        gap = {k: v for k, v in p.items() if k not in ("conv2_w", "conv2_b")}
        gap["conv4_w"], gap["conv4_b"] = p["conv2_w"], p["conv2_b"]
        np.savez(td / "gap.npz", **gap)

        # Truncated at the TAIL, which no key name can betray: the scan just
        # stops at conv3 and reports a 4-layer net that loads and plays. Only
        # the recorded depth catches it, which is why the depth is recorded.
        p = weights(6, 16, False)
        rec = npnet.arch_record(p)
        cut = {k: v for k, v in p.items()
               if k not in ("conv4_w", "conv4_b", "conv5_w", "conv5_b")}
        np.savez(td / "tail.npz", **cut, **rec)
        # the same file WITHOUT the record is the failure this test exists for:
        # it must read back as something other than the 6 layers it was
        assert npnet.arch_of(cut)["layers"] == 4, "the key scan should not see a tail cut"
        # residual too: a block pair removed is still an odd, loadable count
        p = weights(7, 16, True)
        rec = npnet.arch_record(p)
        cut = {k: v for k, v in p.items() if not k.startswith("res2")}
        np.savez(td / "tailres.npz", **cut, **rec)

        for name in ("renamed.npz", "gap.npz", "tail.npz", "tailres.npz"):
            try:
                npnet.Net(str(td / name))
                raise AssertionError(f"{name} loaded instead of raising")
            except ValueError:
                pass


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
