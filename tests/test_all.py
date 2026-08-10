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


def test_competition_maps_strip_every_neutral_castle():
    """The temporal castle inference is valid only under this ruleset invariant."""
    for seed in range(32):
        grid = mapgen.generate(seed)
        assert not (grid > 2).any(), f"seed {seed} retained a neutral castle"
        assert not engine.from_grid(grid).castles.any(), seed


def test_temporal_enemy_history_survives_recapture():
    from bot.memory import TemporalMemory
    from bot.obs import Obs

    ty = np.full((3, 3), rules.T_PLAIN, np.int8)
    owner = np.zeros((3, 3), np.int8)
    army = np.zeros((3, 3), np.int32)
    owner[1, 1], army[1, 1] = rules.OWNER_OPP, 7
    mem = TemporalMemory(3, 3)
    mem.update(Obs(3, 3, 1, 1, 1, 1, 7, ty, owner, army))
    assert mem.ever_enemy[1, 1]
    owner = owner.copy(); army = army.copy()
    owner[1, 1], army[1, 1] = rules.OWNER_ME, 2
    mem.update(Obs(3, 3, 2, 2, 3, 0, 0, ty, owner, army))
    assert mem.ever_enemy[1, 1], "EVER_ENEMY must encode history, not current owner"


def test_temporal_new_structure_in_fog_does_not_need_stale_owner():
    """A castle built on a never-seen cell is observable only as a new SIF."""
    from bot.memory import TemporalMemory
    from bot.obs import Obs

    h = w = 3
    ty = np.full((h, w), rules.T_FOG, np.int8)
    owner = np.zeros((h, w), np.int8)
    army = np.zeros((h, w), np.int32)
    mem = TemporalMemory(h, w)
    mem.update(Obs(h, w, 1, 1, 1, 1, 1, ty, owner, army))
    assert not mem.ever_seen[1, 1]
    assert mem.mem_owner[1, 1] == rules.OWNER_NEUTRAL

    ty = ty.copy()
    ty[1, 1] = rules.T_STRUCTURE_IN_FOG
    mem.update(Obs(h, w, 2, 1, 1, 1, 1, ty, owner, army))
    assert mem.enemy_castles[1, 1]


def test_temporal_first_frame_structure_is_mountain_and_update_is_idempotent():
    from bot.memory import TemporalMemory
    from bot.obs import Obs

    h = w = 3
    ty = np.full((h, w), rules.T_FOG, np.int8)
    ty[1, 1] = rules.T_STRUCTURE_IN_FOG
    z8 = np.zeros((h, w), np.int8)
    z32 = np.zeros((h, w), np.int32)
    obs = Obs(h, w, 7, 1, 1, 1, 1, ty, z8, z32)
    mem = TemporalMemory(h, w)
    mem.update(obs)
    assert mem.known_mountains[1, 1]
    assert not mem.enemy_castles[1, 1]
    before = {k: v.copy() for k, v in mem.__dict__.items()
              if isinstance(v, np.ndarray)}
    mem.update(obs)
    assert all(np.array_equal(before[k], getattr(mem, k)) for k in before)


def test_jax_temporal_new_structure_in_fog_does_not_need_stale_owner():
    try:
        import jax.numpy as jnp
    except ImportError:
        return
    from types import SimpleNamespace

    from bot import features
    from learn import rlenv

    n = features.PAD
    zero = jnp.zeros((n, n), dtype=jnp.bool_)

    def frame(turn, structures):
        return SimpleNamespace(
            fog_cells=~structures,
            structures_in_fog=structures,
            mountains=zero,
            owned_cells=zero,
            opponent_cells=zero,
            castles=zero,
            generals=zero,
            armies=jnp.zeros((n, n), dtype=jnp.int32),
            timestep=jnp.asarray(turn, dtype=jnp.int32),
            owned_army_count=jnp.asarray(1, dtype=jnp.int32),
            opponent_army_count=jnp.asarray(1, dtype=jnp.int32),
            owned_land_count=jnp.asarray(1, dtype=jnp.int32),
            opponent_land_count=jnp.asarray(1, dtype=jnp.int32),
        )

    mem = {k: v[0] for k, v in rlenv.empty_memory_jax(1).items()}
    mem = rlenv.update_memory_jax(mem, frame(1, zero))
    structures = zero.at[1, 1].set(True)
    mem = rlenv.update_memory_jax(mem, frame(2, structures))
    assert bool(np.asarray(mem["enemy_castles"])[1, 1])


def test_neural_strategic_planes_contain_spawn_distance_and_build_price():
    from bot import features
    from bot.memory import TemporalMemory

    for seed in range(8):
        grid = mapgen.generate(seed)
        st = engine.from_grid(grid)
        obs = engine.observe(st, 0)
        mem = TemporalMemory(*grid.shape)
        mem.update(obs)
        x = features.encode(obs, mem)
        assert mem.general_candidates[st.gpos[1]], f"seed {seed} lost true general"
        assert x[features.ENEMY_GENERAL_PRIOR, st.gpos[1][0], st.gpos[1][1]] == 1
        assert x[features.DIST_HOME, st.gpos[0][0], st.gpos[0][1]] == 0
        mine = obs.owner_grid == rules.OWNER_ME
        structures = mine & ((obs.type_grid == rules.T_GENERAL)
                              | (obs.type_grid == rules.T_CASTLE))
        expected = np.log1p(rules.build_cost_grid(structures)) / 6.0
        assert np.allclose(x[features.BUILD_COST, :obs.H, :obs.W], expected)


def test_legacy_encoded_rows_zero_extend_without_inventing_history():
    from bot import features

    old = np.arange(2 * features.STRATEGIC_OFFSET * 3 * 4, dtype=np.float16).reshape(
        2, features.STRATEGIC_OFFSET, 3, 4)
    grown = features.ensure_channels(old)
    assert grown.shape[-3] == features.C
    assert np.array_equal(grown[:, :features.STRATEGIC_OFFSET], old)
    assert not grown[:, features.STRATEGIC_OFFSET:].any()


def test_temporal_memory_copy_is_independent():
    from bot.memory import TemporalMemory

    mem = TemporalMemory(4, 5)
    mem.mem_army[1, 2] = 17
    mem.ever_seen[1, 2] = True
    copied = mem.copy()
    copied.mem_army[1, 2] = 3
    copied.ever_seen[0, 0] = True
    assert mem.mem_army[1, 2] == 17
    assert not mem.ever_seen[0, 0]
    assert copied.H == mem.H and copied.W == mem.W


def test_paired_rating_uses_board_clusters():
    from arena import rating

    def rows(pair_outcomes):
        out = []
        for seed, (a, b) in enumerate(pair_outcomes):
            out.extend(({"seed": seed, "a_seat": 0, "a_result": a},
                        {"seed": seed, "a_seat": 1, "a_result": b}))
        return out

    # Same raw 50/50 W/L. Correlated boards vary between 1 and 0; anti-correlated
    # boards are exactly 0.5 after the seat swap and therefore have no board
    # difficulty variance.
    positive = rows([("win", "win")] * 50 + [("loss", "loss")] * 50)
    negative = rows([("win", "loss")] * 100)
    p, n = rating.paired_summary(positive), rating.paired_summary(negative)
    assert p["score"] == n["score"] == 0.5
    assert p["boards"] == n["boards"] == 100
    assert p["pair_stderr"] > 0.04
    assert n["pair_stderr"] == 0.0
    assert p["elo_hi"] - p["elo_lo"] > n["elo_hi"] - n["elo_lo"]


def test_paired_rating_rejects_malformed_pairs():
    from arena import rating

    bad = [{"seed": 1, "a_seat": 0, "a_result": "win"},
           {"seed": 2, "a_seat": 1, "a_result": "loss"}]
    try:
        rating.paired_summary(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("different seeds were accepted as a board pair")


def test_army_three_has_a_distinct_legal_split_action():
    from bot import features

    grid = _blank(3, 3)
    st = engine.from_grid(grid)
    st.armies[0, 0] = 3
    obs = engine.observe(st, 0)
    mask = features.legal_mask(obs)
    full = features.action_to_index((rules.MOVE, 0, 0, 1, 0))
    split = features.action_to_index((rules.MOVE, 0, 0, 1, 1))
    assert mask[full] and mask[split]
    valid_full, _, _, amount_full = engine._move_params(
        st, 0, (rules.MOVE, 0, 0, 1, 0))
    valid_split, _, _, amount_split = engine._move_params(
        st, 0, (rules.MOVE, 0, 0, 1, 1))
    assert valid_full and valid_split
    assert (amount_full, amount_split) == (2, 1)


def test_smoke_evaluation_can_never_approve_promotion():
    from tools.evaluate import approval_status

    approved, exit_ok = approval_status(True, True, smoke=True)
    assert not approved and exit_ok
    approved, exit_ok = approval_status(True, True, smoke=False)
    assert approved and exit_ok


def test_critic_replay_is_game_balanced_not_row_balanced():
    from learn.replay import GameBalancedReplay

    replay = GameBalancedReplay(2)
    replay.add(np.zeros((100, 1, 1, 1), np.float16), 1.0, "long")
    replay.add(np.zeros((2, 1, 1, 1), np.float16), -1.0, "short")
    _, z = replay.sample(np.random.default_rng(4), 20_000)
    assert abs(float(z.mean())) < 0.03, z.mean()


def test_restart_curriculum_burns_in_without_changing_full_start_jobs():
    from learn.selfplay import build_jobs

    full = build_jobs(3, 12, 2, restart_frac=0.0)
    focused = build_jobs(3, 12, 2, restart_frac=1.0,
                         restart_min=91, restart_max=91)
    assert all(len(j) == 5 for j in full)
    assert all(len(j) == 6 and j[-1] == 91 for j in focused)
    assert [j[:5] for j in focused] == full


def test_neural_package_identity_is_fail_closed():
    from tools.package import validate_identity

    for args in ((None, None, False, False),
                 ("abc", None, False, False),
                 ("abc", "def", False, False)):
        try:
            validate_identity(*args)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"unsafe package identity was accepted: {args}")
    validate_identity("abc", "abc")
    validate_identity(None, None, allow_heuristic=True)


def test_stdio_agent_honours_deadline_and_reports_fallback():
    import tempfile
    import time
    from pathlib import Path

    from arena.stdio_agent import StdioAgent
    from bot.obs import Obs

    z8 = np.zeros((1, 1), np.int8)
    z32 = np.zeros((1, 1), np.int32)
    obs = Obs(1, 1, 1, 1, 1, 1, 1, z8, z8, z32)
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "run.sh"
        run.write_text("#!/usr/bin/env bash\nsleep 10\n")
        agent = StdioAgent(str(run), 0, 1, 1)
        try:
            agent.act(obs, time.perf_counter() + 0.05)
        except TimeoutError:
            pass
        else:
            raise AssertionError("hung stdio child ignored its deadline")
        finally:
            agent.close()

        run.write_text(
            "#!/usr/bin/env bash\n"
            "read handshake\n"
            "while read frame; do\n"
            "  read type; read owner; read army\n"
            "  echo 'policy: FALLING BACK to heuristic' >&2\n"
            "  echo '1 0 0 0 0'\n"
            "done\n")
        agent = StdioAgent(str(run), 0, 1, 1)
        try:
            agent.act(obs, time.perf_counter() + 1.0)
        except RuntimeError as e:
            assert "FALLING BACK" in str(e)
        else:
            raise AssertionError("stdio fallback was hidden as a valid action")
        finally:
            agent.close()


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

    def weights(layers, ch, residual, cin=None, per_cell=None):
        cin = features.C if cin is None else cin
        per_cell = features.PER_CELL if per_cell is None else per_cell
        p, prev = {}, cin
        for n in npnet.trunk_keys(layers, residual):
            p[f"{n}_w"] = (rng.normal(size=(ch, prev, 3, 3)) * 0.2).astype("f4")
            p[f"{n}_b"] = (rng.normal(size=ch) * 0.2).astype("f4")
            prev = ch
        p["head_w"] = (rng.normal(size=(per_cell, ch, 3, 3)) * 0.2).astype("f4")
        p["head_b"] = np.zeros(per_cell, "f4")
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
            assert net.arch == {"layers": layers, "channels": ch,
                                "residual": residual, "context": False}
            assert net.logits(obs).shape == (features.N_ACTIONS,)

        # Adding the spatial strategy path is an exact incumbent migration.
        from tools.grow import grow as grow_net
        p = weights(4, 8, False)
        source = td / "strategy-source.npz"
        target = td / "strategy-target.npz"
        np.savez(source, **p, **npnet.arch_record(p))
        strategic = grow_net(np.load(source), 4, 8, strategy_hidden=6)
        np.savez(target, **strategic)
        before, after = npnet.Net(str(source)), npnet.Net(str(target))
        assert after.arch["strategy_hidden"] == 6
        assert np.allclose(before.logits(obs), after.logits(obs), atol=1e-6)

        # The same migration is valid for the separately checkpointed critic;
        # tools.grow must not assume every trunk is followed by a policy head.
        critic = {k: v for k, v in p.items()
                  if k.startswith("conv")}
        critic["v_w"] = rng.normal(size=11 * 8).astype("f4")
        critic["v_b"] = np.float32(0.2)
        critic["value_schema"] = np.int32(2)
        critic.update(npnet.arch_record(critic))
        critic_source = td / "critic-source.npz"
        critic_target = td / "critic-target.npz"
        np.savez(critic_source, **critic)
        strategic_critic = grow_net(np.load(critic_source), 4, 8,
                                    strategy_hidden=6)
        np.savez(critic_target, **strategic_critic)
        vb = npnet.ValueNet(str(critic_source)).value(obs)
        va = npnet.ValueNet(str(critic_target)).value(obs)
        assert np.isclose(vb, va, atol=1e-6)

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

        # A checkpoint from before the action space or the observation changed.
        # Both are internally consistent, carry an honest arch_record, and load
        # without complaint if nobody checks the two widths that are not part of
        # "architecture": the head would die later in a broadcast against the
        # legal mask, the stem in a BLAS shape error inside _conv3x3 -- both
        # mid-match, from lines that name neither the file nor the reason.
        for name, kw in (("stale_head.npz", {"per_cell": features.PER_CELL - 1}),
                         ("stale_obs.npz", {"cin": features.C - 1})):
            p = weights(4, 32, False, **kw)
            np.savez(td / name, **p, **npnet.arch_record(p))

        for name in ("renamed.npz", "gap.npz", "tail.npz", "tailres.npz",
                     "stale_head.npz", "stale_obs.npz"):
            try:
                npnet.Net(str(td / name))
                raise AssertionError(f"{name} loaded instead of raising")
            except ValueError:
                pass


def test_legacy_24_channel_policy_migration_preserves_its_function():
    import tempfile
    from pathlib import Path

    from bot import features
    from bot.policy import net as npnet

    rng = np.random.default_rng(91)
    ch = 6
    p = {
        "conv0_w": rng.normal(0, .1, (ch, features.BASE_C, 3, 3)).astype("f4"),
        "conv0_b": rng.normal(0, .1, ch).astype("f4"),
        "head_w": rng.normal(0, .1, (features.PER_CELL, ch, 3, 3)).astype("f4"),
        "head_b": rng.normal(0, .1, features.PER_CELL).astype("f4"),
        "pass_w": rng.normal(0, .1, ch).astype("f4"),
        "pass_b": np.float32(.2),
    }
    x = np.zeros((features.C, features.PAD, features.PAD), np.float32)
    x[:features.BASE_C] = rng.normal(size=x[:features.BASE_C].shape)
    old_h = np.maximum(npnet._conv3x3(x[:features.BASE_C], p["conv0_w"],
                                      p["conv0_b"]), 0)
    old_move = npnet._conv3x3(old_h, p["head_w"], p["head_b"])
    old = np.concatenate([np.transpose(old_move, (1, 2, 0)).reshape(-1),
                          [float(p["pass_w"] @ old_h.mean(axis=(1, 2)) + p["pass_b"])]])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "legacy24.npz"
        np.savez(path, **p, **npnet.arch_record(p))
        got = npnet.Net(str(path))._logits_from(x)
    assert np.allclose(got, old, atol=1e-6), np.max(np.abs(got - old))


def test_configs_are_strict_and_legacy_migration_is_explicit():
    import json
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    from bot.config import Config

    cfg = Config()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.json"
        cfg.save(path)
        assert asdict(Config.load(path)) == asdict(cfg)
        raw = json.loads(path.read_text())
        raw.pop("guard_radius")
        path.write_text(json.dumps(raw))
        try:
            Config.load(path)
            raise AssertionError("strict loader accepted a missing field")
        except ValueError:
            pass
        raw["guard_radius"] = cfg.guard_radius
        raw["typo_radius"] = 3
        try:
            Config.from_dict(raw, strict=True)
            raise AssertionError("strict loader accepted an unknown field")
        except ValueError:
            pass
    legacy = asdict(cfg)
    legacy.pop("guard_radius")
    legacy["general_reserve"] = 2
    assert Config.migrate_legacy(legacy).guard_radius == Config().guard_radius


def test_league_materialises_strict_configs():
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    from bot.config import Config
    from learn.league import materialise

    members = [{"name": "oracle-0", "spec": "", "config": asdict(Config())}]
    with tempfile.TemporaryDirectory() as td:
        materialise(members, Path(td))
        path = Path(members[0]["spec"].removeprefix("ours:"))
        assert Config.load(path).to_dict() == Config().to_dict()


def test_league_forwards_safe_neural_optimizer_settings():
    from pathlib import Path
    from types import SimpleNamespace

    from learn.league import net_oracle_command

    args = SimpleNamespace(
        nn_init="incumbent.npz", nn_init_critic="critic.npz",
        workers=60, nn_iters=200, nn_games=256, max_turns=1200,
        nn_epochs=1, nn_minibatch=2048, nn_lr=7e-5,
        nn_critic_lr=1e-4, nn_value_hidden=64, nn_lam=0.99,
        nn_reward_mode="terminal", nn_tempo_eps=0.03, nn_augment=False,
        nn_warm_evar=0.12,
        nn_sigma_floor=0.25, seed=3)
    cmd = net_oracle_command(args, Path("oracle.npz"), Path("league.json"),
                             "maps.json", 2)

    def value(flag):
        return cmd[cmd.index(flag) + 1]

    assert value("--init") == "incumbent.npz"
    assert value("--init-critic") == "critic.npz"
    assert value("--epochs") == "1"
    assert value("--minibatch") == "2048"
    assert value("--lr") == "7e-05"
    assert value("--critic-lr") == "0.0001"
    assert value("--value-hidden") == "64"
    assert value("--lam") == "0.99"
    assert value("--reward-mode") == "terminal"
    assert value("--tempo-eps") == "0.03"
    assert value("--warm-evar") == "0.12"
    assert value("--sigma-floor") == "0.25"
    assert value("--defense-aux-weight") == "0.0"
    assert value("--defense-tail-turns") == "60"
    assert value("--defense-hidden-ratio") == "2.5"
    assert value("--defense-margin") == "0.05"
    assert value("--defense-opponent") == "snipe"
    assert value("--seed") == "2003"
    assert value("--maps") == "maps.json"
    args.nn_augment = True
    assert "--augment" in net_oracle_command(
        args, Path("oracle.npz"), Path("league.json"), "maps.json", 2)
    args.nn_reward_mode, args.nn_tempo_eps = "tempo", 0.05
    cmd = net_oracle_command(args, Path("oracle.npz"), Path("league.json"),
                             "maps.json", 2)
    assert value("--reward-mode") == "tempo"
    assert value("--tempo-eps") == "0.05"
    args.nn_augment = False
    args.nn_defense_aux_weight = 0.05
    args.nn_defense_tail_turns = 48
    args.nn_defense_hidden_ratio = 3.0
    args.nn_defense_margin = 0.10
    args.nn_defense_opponent = "snipe"
    cmd = net_oracle_command(args, Path("oracle.npz"), Path("league.json"),
                             "maps.json", 2)
    assert value("--defense-aux-weight") == "0.05"
    assert value("--defense-tail-turns") == "48"
    assert value("--defense-hidden-ratio") == "3.0"
    assert value("--defense-margin") == "0.1"


def test_league_default_lambda_resume_migration_is_narrow():
    from learn.league import migrate_resume_params

    old = {"nn_lr": 1e-4}
    default = {"nn_lr": 1e-4, "nn_lam": 0.95, "nn_value_hidden": 0,
               "nn_augment": False, "nn_reward_mode": "terminal",
               "nn_tempo_eps": 0.03}
    experimental = {"nn_lr": 1e-4, "nn_lam": 0.99, "nn_value_hidden": 64,
                    "nn_augment": True, "nn_reward_mode": "tempo",
                    "nn_tempo_eps": 0.05}
    assert migrate_resume_params(old, default) == default
    assert migrate_resume_params(old, experimental) == old
    assert migrate_resume_params(default, default) is default


def test_netoracle_reward_mode_is_opt_in_and_bounded():
    from learn.netoracle import episode_return

    # Terminal mode is exactly the historical objective, independent of time.
    assert episode_return(1, 1, 100) == 1.0
    assert episode_return(-1, 100, 100, "terminal", 0.2) == -1.0
    # Tempo is only a tie-break: faster wins and longer survival on losses.
    assert episode_return(1, 1, 100, "tempo", 0.05) > episode_return(
        1, 100, 100, "tempo", 0.05)
    assert episode_return(-1, 1, 100, "tempo", 0.05) < episode_return(
        -1, 100, 100, "tempo", 0.05)
    assert episode_return(0, 1, 100, "tempo", 0.05) == 0.0
    assert -1.0 <= episode_return(1, 100, 100, "tempo", 0.05) <= 1.0
    assert -1.0 <= episode_return(-1, 1, 100, "tempo", 0.05) <= 1.0


def test_league_runoff_manifest_value_is_json_safe():
    import json

    from learn.league import serialise_runoff

    got = serialise_runoff({4: np.float64(0.55), 1: 0.625})
    assert got == {"1": 0.625, "4": 0.55}
    assert json.loads(json.dumps({"fresh_min": got}))["fresh_min"] == got


def test_top3_promotion_suite_uses_fresh_seeds_and_a_sanity_greedy_floor():
    from tools.evaluate import bucket_alphas, load_suite, planning_score_to_clear

    suite = load_suite("evaluation/top3-v2.json")
    assert suite["seed0"] == 8_000_000
    assert suite["confirmation_seed0"] != suite["seed0"]
    assert suite["max_attempts"] >= 1
    assert all(b["games"] == 2000 for b in suite["buckets"])
    greedy = next(b for b in suite["buckets"] if b["name"] == "greedy")
    assert greedy["min_lower_score"] <= 0.70
    alphas = dict(zip((b["name"] for b in suite["buckets"]),
                      bucket_alphas(suite, suite["buckets"])))
    assert alphas["champion"] == 0.05
    assert alphas["fog"] == 0.05 / 4
    corrected = dict(zip((b["name"] for b in suite["buckets"]),
                         bucket_alphas(suite, suite["buckets"],
                                       repeated_blocks=2, attempt_correct=True)))
    assert corrected["champion"] == 0.05 / 2 / suite["max_attempts"]
    assert corrected["fog"] == 0.05 / 4 / 2 / suite["max_attempts"]
    assert all(b.get("family") is None for b in suite["buckets"]
               if b["name"] in {"hunter", "greedy", "expander"})
    assert {b.get("style_role") for b in suite["buckets"] if b.get("role") == "signal"} \
        == set(suite["required_style_roles"])
    planning = suite["planning_notes"]["fog_detectable_score"]
    assert planning["approx_true_score_to_clear_on_expectation"] == 0.508
    estimate = planning_score_to_clear(
        planning["games"], planning["alpha"], planning["floor"],
        planning["worst_case_pair_variance"])
    assert abs(estimate - planning["approx_true_score_to_clear_on_expectation"]) < 0.001
    alpha4 = planning_score_to_clear(
        planning["games"], planning["alpha"] / 4, planning["floor"],
        planning["worst_case_pair_variance"])
    assert abs(alpha4 - planning["counterfactual_score_if_alpha_were_divided_by_four"]) < 0.001
    champion = next(b for b in suite["buckets"] if b["name"] == "champion")
    assert champion["requires_paired_superiority"] is True
    assert "requires_sprt" not in champion


def test_promotion_registry_consumes_attempts_and_rejects_reuse():
    import tempfile
    from pathlib import Path

    from tools.evaluate import load_suite, reserve_attempt

    suite = load_suite("evaluation/top3-v2.json")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        candidate, reference, registry = d / "candidate.npz", d / "reference.npz", d / "r.json"
        candidate.write_bytes(b"candidate")
        reference.write_bytes(b"reference")
        index, attempt_id = reserve_attempt(registry, suite, candidate, reference, "test")
        assert index == 0 and attempt_id.startswith("001-")
        try:
            reserve_attempt(registry, suite, candidate, reference, "duplicate")
            raise AssertionError("duplicate candidate reused a promotion seed block")
        except SystemExit:
            pass


def test_value_three_way_split_never_leaks_a_game():
    import tempfile
    from pathlib import Path

    from learn import valuetrain

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, games in enumerate(([10, 10, 20], [20, 30, 30])):
            path = Path(td) / f"shard_{i:04d}.npz"
            n = len(games)
            np.savez(path, x=np.zeros((n, 1, 1, 1), np.float16),
                     y=np.asarray([-1, 0, 1][:n], np.float32),
                     game=np.asarray(games, np.int64), seat=np.zeros(n, np.int8),
                     t=np.arange(n))
            paths.append(path)
        # A second generator has equal raw IDs and must contribute its own
        # selection and test game instead of being swallowed by the larger set.
        other = Path(td) / "selfplay"
        other.mkdir()
        other_path = other / "shard_0000.npz"
        np.savez(other_path, x=np.zeros((3, 1, 1, 1), np.float16),
                 y=np.asarray([-1, 0, 1], np.float32),
                 game=np.asarray([10, 40, 50], np.int64), seat=np.zeros(3, np.int8),
                 t=np.arange(3))
        all_paths = paths + [other_path]
        game_ids, refs, _ = valuetrain.canonical_game_ids(all_paths)
        selection, held, select_games, test_games = valuetrain.split_three_way(
            all_paths, .34, .34, np.random.default_rng(2), game_ids, refs)
        _, _, gs = selection
        _, _, gt = valuetrain._materialize(all_paths, game_ids, test_games)
        assert set(np.unique(gs)) == select_games
        assert set(np.unique(gt)) == test_games
        assert not (select_games & test_games)
        assert {refs[g][0] for g in select_games} == {0, 1}
        assert {refs[g][0] for g in test_games} == {0, 1}
        for path in all_paths:
            training_games = set(game_ids[path][~held[path]])
            assert not (training_games & select_games)
            assert not (training_games & test_games)

        # Equal raw ids from independent generators are different games.
        mixed, refs, _ = valuetrain.canonical_game_ids(all_paths)
        official_ten = mixed[paths[0]][0]
        selfplay_ten = mixed[other_path][0]
        assert official_ten != selfplay_ten
        assert refs[int(official_ten)] != refs[int(selfplay_ten)]


def test_residual_value_head_is_an_exact_trainable_migration():
    import jax
    import jax.numpy as jnp

    from bot import features
    from learn import valuetrain

    arch = {"layers": 1, "channels": 4, "residual": False, "context": False}
    key = jax.random.PRNGKey(17)
    linear = valuetrain.init_params(key, arch, value_hidden=0)
    residual = valuetrain.init_params(key, arch, value_hidden=8)
    x = np.random.default_rng(3).normal(
        size=(3, features.C, features.PAD, features.PAD)).astype(np.float32)
    x[:, features.VALID] = 1.0
    before = np.asarray(valuetrain.forward(linear, jnp.asarray(x)))
    after = np.asarray(valuetrain.forward(residual, jnp.asarray(x)))
    assert np.array_equal(before, after), "W2=0 migration moved critic predictions"
    grad = jax.grad(lambda p: valuetrain.forward(p, jnp.asarray(x)).sum())(residual)
    assert np.any(np.asarray(grad["v_res2_w"]) != 0), "new residual branch is dead"


def test_dataset_rebuild_removes_only_stale_artifacts():
    import tempfile
    from pathlib import Path

    from learn import dataset, valuedata

    for prepare in (dataset.prepare_output, valuedata.prepare_output):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            np.savez(out / "shard_0000.npz", x=np.zeros(1))
            np.savez(out / "shard_0042.npz", x=np.zeros(1))
            (out / "meta.json").write_text("stale")
            keep = out / "notes.txt"
            keep.write_text("preserve me")
            assert prepare(out) == 2
            assert not list(out.glob("shard_*.npz"))
            assert not (out / "meta.json").exists()
            assert keep.read_text() == "preserve me"


def test_value_pooling_matches_numpy_and_jax():
    try:
        import jax.numpy as jnp
    except ImportError:
        print("  SKIPPED (no jax): value pooling parity is UNVERIFIED")
        return
    from bot.policy.net import ValueNet
    from learn.valuetrain import pooled

    rng = np.random.default_rng(17)
    h = rng.normal(size=(3, 5, 21, 21)).astype(np.float32)
    valid = np.zeros((3, 21, 21), np.float32)
    valid[0, :21, :21] = 1
    valid[1, :15, :18] = 1
    valid[2, :9, :12] = 1
    got = np.asarray(pooled(jnp.asarray(h), jnp.asarray(valid)))
    want = np.stack([ValueNet._pooled(h[i], valid[i]) for i in range(len(h))])
    assert got.shape == want.shape == (3, 55)
    assert np.max(np.abs(got - want)) < 2e-6


def test_the_training_seam_matches_the_submission():
    """Step both engines in lockstep and compare what a JAX rollout would feed a
    policy: the observation, the encoder, the legal mask and the flat action map.

    This is the check `learn/rlenv.py` claimed in its docstring and never ran.
    Everything downstream — behaviour cloning, PPO, the vectorised rollout —
    computes its gradients against whatever these two agree on, so a single
    channel out of order is a policy trained on one game and played in another,
    with nothing about the failure pointing at the encoder.

    Read the COVERAGE line it prints. The build half of the action space is
    all-False in a random game, and two all-False masks compare equal.
    """
    try:
        import jax  # noqa: F401
    except ImportError:
        print("  SKIPPED (no jax): the encoder equivalence is UNVERIFIED in this run")
        return
    from tools.verify_engine import check_encoders, check_memory_encoders

    check_encoders(boards=12, turns=320, stride=8)
    # Keep this long and close-range enough that every temporal plane becomes
    # active.  The verifier asserts activity, so a broken enemy-memory half can
    # no longer pass merely because both encoders emitted zeros.
    check_memory_encoders(boards=4, turns=240)


def test_the_scanned_rollout_matches_the_python_loop():
    """`--backend scan` must play the SAME games as `--backend gpu`.

    The scan moves the per-turn loop onto the device. It calls the same jitted
    `step`, splits the same keys in the same order and passes the same absolute
    turn index — so everything that DECIDES a game must be exact: the
    observation, the legal mask, the sampled action, the length, the outcome.
    `logp` is compared to `vecroll.LOGP_TOL` instead, because a scan body is a
    different XLA compilation and the conv trunk fuses differently in it
    (measured: logits move 3.6e-07, logp 4.8e-07 = 1 ulp). Do not "fix" a
    failure by widening that bound.

    HORIZON. 480 turns, not 160, and this is the reason: at 160 the two paths
    are bitwise equal and the test looks stronger than it is, so a later change
    that widened the gap would first show up in production. 480 is roughly a
    competition-distance episode and it is where the 1-ulp gap appears.

    Read the COVERAGE line. `turns` is `where(ended >= 0, ended + 1, steps)` and
    `steps` is the only new state the scan carries: if no game ends, both paths
    compare arrays of -1 and the test passes with the turn counter completely
    wrong. So it asserts that BOTH branches are live — some games decided,
    some still running at the horizon.

    On failure it prints the first divergent (turn, column) and the two `logp`
    values there. A single flipped `idx` deep into a game, with everything
    before it identical, is the known near-tie in `categorical` and is a
    property of the drift above; a divergence in `x` or `mask`, one from turn 0,
    or a different length, is a bug.
    """
    try:
        import jax  # noqa: F401
    except ImportError:
        print("  SKIPPED (no jax): --backend scan is UNVERIFIED in this run")
        return
    import jax.random as jr

    from learn import train as bc
    from learn import vecroll
    from tools import pools

    n, max_turns = 8, 480     # see HORIZON above
    key = jr.PRNGKey(0)
    theta = bc.init_params(key, {"layers": 4, "channels": 8, "residual": False})
    step = vecroll.make_step(bc.forward)
    idx = np.arange(n)

    def diverges(a, b):
        """(turn, field) of the first mismatch in one trajectory pair, or None."""
        if a["turns"] != b["turns"]:
            return (min(a["turns"], b["turns"]), "turns")
        for f in ("x", "idx", "mask"):
            bad = np.nonzero(np.any((a[f] != b[f]).reshape(len(a[f]), -1), axis=1))[0]
            if len(bad):
                return (int(bad[0]), f)
        bad = np.nonzero(np.abs(a["logp"] - b["logp"]) > vecroll.LOGP_TOL)[0]
        return (int(bad[0]), "logp") if len(bad) else None

    # Stage 0 ONLY, and that is measured rather than assumed: a random 4x8 net
    # decides nothing at stage 1+ inside 480 turns (stage 3, n=4: 4/4 still
    # running at 480), so a second pool would buy a vacuous COVERAGE line at
    # three times the cost. Stage 0's own boards already vary h (18-19) and
    # w (19-21), which is the only thing a second stage was there for.
    chunks = (480, 40, 16)          # 1 chunk, 12 chunks, 30 chunks
    host = pools.build(0, size=n)
    assert len(set(host["h"])) > 1 and len(set(host["w"])) > 1, "h/w do not vary"
    pool = vecroll.device_pool(host)
    ref = vecroll.rollout(step, pool, idx, theta, key, max_turns)
    decided = sum(r["turns"] < max_turns for r in ref[::2])
    print(f"  COVERAGE: {decided}/{n} games decided, {n - decided} still running "
          f"at turn {max_turns}, chunks {chunks}")
    assert 0 < decided < n, (
        f"{decided}/{n} decided — one of the two `turns` branches is never "
        "taken, so this comparison is vacuous on the only state the scan adds. "
        "Raise max_turns and record the number.")
    for chunk in chunks:
        got = vecroll.rollout_scan(vecroll.make_scan(step, chunk), pool, idx,
                                   theta, key, max_turns, chunk)
        assert len(got) == len(ref), (chunk, len(got), len(ref))
        for j, (a, b) in enumerate(zip(ref, got)):
            for k in ("z", "turns", "dist", "seat", "opp"):
                assert a[k] == b[k], (chunk, j, k, a[k], b[k])
            d = diverges(a, b)
            assert d is None, (
                f"chunk {chunk}: column {j} (game {j // 2}, seat {j % 2}) "
                f"diverges at turn {d[0]} in {d[1]}; "
                f"logp {a['logp'][d[0]]!r} vs {b['logp'][d[0]]!r}, "
                f"idx {a['idx'][d[0]]} vs {b['idx'][d[0]]}")


def test_the_row_index_reproduces_the_packed_columns():
    """The identity a device-resident rollout would stand on, asserted before
    that path exists: gathering `vecroll.row_index` out of the flat
    `(T*2n, ...)` buffer gives EXACTLY what `_pack_columns` slices out today,
    in the same order and with the same total length.

    Ragged on purpose — the five columns end on different turns, which is the
    whole reason the cut cannot be a reshape. The COVERAGE assert below keeps it
    that way: with every column running to the horizon this test would pass
    against a plain reshape and prove nothing.

    Numpy only, no jax, no games. If it ever fails, a resident buffer would feed
    the critic padded rows with a real outcome attached, `evar` would stay
    plausible and nothing downstream would say a word — read `vecroll.row_index`
    before touching the assert.
    """
    from learn import vecroll

    n, T, C = 5, 7, 3
    ended = np.array([2, -1, 6, 0, 4], np.int32)       # -1 = never ended
    winner = np.array([0, -1, 1, 1, -1], np.int32)
    # a distinct value per (turn, column, channel), so a transposed, shifted or
    # column-major gather cannot compare equal by accident
    X = np.arange(T * 2 * n * C, dtype=np.float16).reshape(T, 2 * n, C)
    I = np.arange(T * 2 * n, dtype=np.int32).reshape(T, 2 * n)
    out = vecroll._pack_columns(X, (I % 251).astype(np.uint8), I,
                                I.astype(np.float32), ended, winner, T,
                                {"dist": np.arange(n)}, np.arange(n))

    # `(ended, T)`, the rollout's own pair, NOT a length vector read back out of
    # `out` — that would be per-emitted-column where `row_index` wants per-game,
    # the two differ only when a game is dropped, and both give a gather of the
    # right total length, so the assert below would not see the difference.
    turns = vecroll._turns(ended, T)
    assert 0 < turns.min() < T == turns.max(), (
        f"turns {turns.tolist()} are not ragged against T={T}; this comparison "
        "is vacuous unless both the ended and the still-running branch are live")
    rows = vecroll.row_index(ended, T, n)
    assert len(rows) == sum(len(r["idx"]) for r in out) == 2 * int(turns.sum()), (
        len(rows), sum(len(r["idx"]) for r in out))
    assert np.array_equal(X.reshape(T * 2 * n, C)[rows],
                          np.concatenate([r["x"] for r in out])), "x"
    assert np.array_equal(I.reshape(-1)[rows],
                          np.concatenate([r["idx"] for r in out])), "idx"


def test_the_trainers_check_themselves():
    """The two PPO trainers each carry a numpy-only `--selfcheck`; run them here
    so `make test` covers the shared objective, GAE, the curriculum's advance
    rule and the fact that a requested generals distance actually reaches the
    map generator. No games, no GPU, well under a second.

    They print: the per-stage in-range percentages are the curriculum's own
    proof and are worth seeing on every test run."""
    from learn import netoracle, selfplay
    from tools import pools

    netoracle.selfcheck()
    selfplay.selfcheck()
    pools.selfcheck()
    # `train.selfcheck` needs jax and so does importing `learn.train` at all, so
    # both live under the guard: the whole point of the two skip branches is that
    # `make test` still runs on a machine that only has what the SUBMISSION needs.
    try:
        import jax  # noqa: F401
    except ImportError:
        print("  SKIPPED (no jax): the trainer and the vectorised rollout are "
              "UNCHECKED in this run")
        return
    from learn import train, vecroll
    train.selfcheck()
    vecroll.selfcheck()


def test_dihedral_averaging_is_exact_on_an_equivariant_function():
    """Averaging over the group must return an equivariant function unchanged.

    That is the whole correctness condition for test-time augmentation: if the
    cell gather, the direction permutation or the action relabel is wrong, a
    function that genuinely commutes with the group stops surviving the round
    trip. A net is not equivariant, so it cannot be used as the probe -- this
    builds one that is.
    """
    import numpy as np

    from bot import features, symmetry

    PAD, PER, SPL = features.PAD, features.PER_CELL, features.SPLITS

    def equivariant(x):
        # logit(cell, dir) = v(cell) - v(neighbour in dir); build = 2*v(cell).
        # A rigid motion sends (cell, dir) -> (cell', dperm[dir]) and carries v
        # along, so this commutes with the group exactly.
        v = x[0]
        out = np.zeros(features.N_ACTIONS, np.float64)
        vp = np.pad(v, 1, constant_values=0.0)
        for d, (dr, dc) in enumerate(symmetry.DIRS):
            diff = (v - vp[1 + dr:1 + dr + PAD, 1 + dc:1 + dc + PAD]).reshape(-1)
            for s in range(SPL):
                out[np.arange(PAD * PAD) * PER + d * SPL + s] = diff * (s + 1)
        out[np.arange(PAD * PAD) * PER + features.BUILD_OFFSET] = 2.0 * v.reshape(-1)
        out[-1] = 7.0
        return out

    rng = np.random.default_rng(0)
    for h, w in ((21, 21), (15, 15), (18, 21), (9, 12)):
        x = np.zeros((features.C, PAD, PAD))
        x[0, :h, :w] = rng.normal(size=(h, w))
        base = equivariant(x)
        avg = symmetry.average_logits(x, h, w, equivariant)
        cells = (np.arange(h)[:, None] * PAD + np.arange(w)[None, :]).reshape(-1)
        idx = (cells[:, None] * PER + np.arange(PER)[None, :]).reshape(-1)
        d = float(np.abs(base[idx] - avg[idx]).max())
        assert d < 1e-9, f"{h}x{w}: round trip off by {d:.2e}"

    # the identity element alone must reproduce the plain forward byte for byte,
    # so `tta=False` and a one-element group are the same code path
    x = np.zeros((features.C, PAD, PAD))
    x[0, :21, :21] = rng.normal(size=(21, 21))
    assert np.array_equal(symmetry.average_logits(x, 21, 21, equivariant, elements=(0,)),
                          equivariant(x))


def test_ppo_dihedral_observation_and_action_relabel_commute():
    """Catch a self-consistent but inverse/wrong observation gather.

    Mask/action consistency alone cannot do this: both can share one wrong map.
    An explicitly equivariant function couples board content to action geometry,
    so f(g.x)[g.a] must equal f(x)[a] for every active action and every g.
    """
    from bot import features, symmetry
    from learn import train

    pad, per, splits = features.PAD, features.PER_CELL, features.SPLITS

    def equivariant(x):
        v = x[0]
        out = np.zeros(features.N_ACTIONS, np.float64)
        vp = np.pad(v, 1, constant_values=0.0)
        for direction, (dr, dc) in enumerate(symmetry.DIRS):
            delta = (v - vp[1 + dr:1 + dr + pad,
                            1 + dc:1 + dc + pad]).reshape(-1)
            for split in range(splits):
                out[np.arange(pad * pad) * per + direction * splits + split] = (
                    delta * (split + 1))
        out[np.arange(pad * pad) * per + features.BUILD_OFFSET] = 2.0 * v.reshape(-1)
        out[features.PASS_INDEX] = 7.0
        return out

    rng = np.random.default_rng(20260809)
    for h, w in ((21, 21), (18, 21), (9, 12)):
        x = np.zeros((features.C, pad, pad), np.float32)
        x[0, :h, :w] = rng.normal(size=(h, w))
        x[features.VALID, :h, :w] = 1.0
        cells = (np.arange(h)[:, None] * pad + np.arange(w)[None, :]).reshape(-1)
        active = np.concatenate([
            (cells[:, None] * per + np.arange(per)[None, :]).reshape(-1),
            np.asarray([features.PASS_INDEX])])
        mask = np.zeros(features.N_ACTIONS, bool)
        mask[active] = True
        base = equivariant(x)
        for g in range(8):
            moved_x, moved_mask, moved_idx = train.augment_ppo(
                x[None], mask[None], np.asarray([active[len(active) // 3]]), g)
            _, actmap = train._dihedral_maps(h, w, g)
            mapped = actmap[active]
            assert np.array_equal(equivariant(moved_x[0])[mapped], base[active]), \
                f"observation gather and action relabel disagree at {h}x{w}, g={g}"
            assert moved_mask[0, moved_idx[0]], f"sampled action illegal at g={g}"


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
