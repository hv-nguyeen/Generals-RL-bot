"""Self-contained HTML replay viewer.

Re-simulates a replay, packs every frame into two base64 blobs and writes one
standalone HTML file — no server, no pygame, no dependencies. That matters
because the games worth watching are the ones that ran on the university box,
and this opens over `scp` or a file:// URL.

    python -m analysis.viewer runs/baseline/replays/game_0007.json -o /tmp/g.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from analysis import replay as replay_mod


def pack(rep: dict, every: int = 1) -> dict:
    """Pack one of our own replays (initial grid + action streams)."""
    meta = {k: rep[k] for k in ("seed", "spec0", "spec1", "winner", "reason", "turns")}
    return pack_states(replay_mod.frames(rep), meta, every)


def pack_states(states, meta: dict, every: int = 1) -> dict:
    """Pack any (turn, State) stream. Official replays arrive as per-tick states
    rather than actions, so they enter here instead of through `pack`."""
    codes, armies, series = [], [], []
    h = w = 0
    for turn, st in states:
        if turn % every and turn != meta["turns"]:
            continue
        h, w = st.armies.shape
        code = np.zeros((h, w), dtype=np.uint8)
        code[st.mountains] = 1
        code[st.castles] = 2
        code[st.generals] = 3
        code += (st.own[0].astype(np.uint8) * 4 + st.own[1].astype(np.uint8) * 8)
        codes.append(code.tobytes())
        armies.append(np.clip(st.armies, 0, 65535).astype("<u2").tobytes())
        land, army = st.land(), st.army()
        series.append([turn, land[0], army[0], land[1], army[1]])
    return {
        "h": h, "w": w, "n": len(codes),
        "codes": base64.b64encode(b"".join(codes)).decode(),
        "armies": base64.b64encode(b"".join(armies)).decode(),
        "series": series,
        "meta": meta,
    }


_HTML = """<!doctype html>
<meta charset="utf-8">
<title>generals replay — seed {seed}</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; background:#12141a; color:#e6e6e6; font:13px/1.45 ui-monospace,Menlo,monospace; }}
 header {{ padding:10px 16px; border-bottom:1px solid #262a35; display:flex; gap:18px; flex-wrap:wrap; align-items:baseline }}
 h1 {{ font-size:14px; margin:0; font-weight:600 }}
 .p0 {{ color:#ff6b6b }} .p1 {{ color:#4dabf7 }} .muted {{ color:#8b93a7 }}
 main {{ display:flex; gap:16px; padding:16px; flex-wrap:wrap }}
 canvas {{ background:#0d0f14; border:1px solid #262a35; image-rendering:pixelated }}
 #side {{ min-width:280px; flex:1 }}
 #bar {{ display:flex; gap:10px; align-items:center; padding:0 16px 12px }}
 input[type=range] {{ flex:1 }}
 button, select {{ background:#1b1f2a; color:#e6e6e6; border:1px solid #303646; border-radius:4px; padding:4px 10px; font:inherit; cursor:pointer }}
 table {{ border-collapse:collapse; width:100% }} td {{ padding:2px 6px }} td:first-child {{ color:#8b93a7 }}
 svg {{ width:100%; height:120px; background:#0d0f14; border:1px solid #262a35; margin-top:10px }}
</style>
<header>
  <h1>seed {seed} &middot; <span class=p0>{spec0}</span> vs <span class=p1>{spec1}</span></h1>
  <span class=muted>{reason}, {turns} turns &mdash; winner: <b id=winner></b></span>
</header>
<div id=bar>
  <button id=play>play</button>
  <input type=range id=scrub min=0 value=0>
  <span id=label class=muted></span>
  <select id=speed><option value=1>1x</option><option value=4 selected>4x</option><option value=16>16x</option><option value=64>64x</option></select>
</div>
<main>
  <canvas id=board></canvas>
  <div id=side>
    <table id=stats></table>
    <svg id=chart viewBox="0 0 400 120" preserveAspectRatio=none></svg>
    <div class=muted style="margin-top:6px">land (solid) &middot; army (dashed), scaled independently</div>
  </div>
</main>
<script id=data type=application/json>{data}</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const H = D.h, W = D.w, N = D.n, CELLS = H * W;
const dec = s => {{ const b = atob(s), a = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }};
const CODES = dec(D.codes);
const ARMY_BYTES = dec(D.armies);
const ARMY = new Uint16Array(ARMY_BYTES.buffer, ARMY_BYTES.byteOffset, CELLS * N);

const cv = document.getElementById('board'), ctx = cv.getContext('2d');
const CELL = Math.max(14, Math.min(30, Math.floor(760 / Math.max(H, W))));
cv.width = W * CELL; cv.height = H * CELL;
document.getElementById('winner').textContent =
  D.meta.winner < 0 ? 'draw' : 'player ' + D.meta.winner;

const COL = {{ neutral:'#2a2f3c', mountain:'#080a0e', p0:'#8a2f38', p1:'#2b5c8f',
              p0s:'#ff6b6b', p1s:'#4dabf7' }};

function draw(f) {{
  const off = f * CELLS;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.font = (CELL < 20 ? 9 : 11) + 'px ui-monospace,monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < CELLS; i++) {{
    const c = CODES[off + i], kind = c & 3, owner = (c >> 2) & 3;
    const x = (i % W) * CELL, y = ((i / W) | 0) * CELL;
    let fill = kind === 1 ? COL.mountain
             : owner === 1 ? COL.p0 : owner === 2 ? COL.p1 : COL.neutral;
    ctx.fillStyle = fill; ctx.fillRect(x, y, CELL - 1, CELL - 1);
    if (kind === 3 || kind === 2) {{
      ctx.strokeStyle = kind === 3 ? '#ffd43b' : '#adb5bd';
      ctx.lineWidth = kind === 3 ? 2 : 1;
      ctx.strokeRect(x + 1.5, y + 1.5, CELL - 4, CELL - 4);
    }}
    const a = ARMY[off + i];
    if (a > 0 && kind !== 1) {{
      ctx.fillStyle = owner ? '#fff' : '#98a2b8';
      ctx.fillText(a > 999 ? (a / 1000).toFixed(1) + 'k' : a, x + CELL / 2 - 0.5, y + CELL / 2);
    }}
  }}
  const s = D.series[f];
  document.getElementById('label').textContent = 'turn ' + s[0] + ' / ' + D.meta.turns;
  document.getElementById('stats').innerHTML =
    `<tr><td>land</td><td class=p0>${{s[1]}}</td><td class=p1>${{s[3]}}</td></tr>` +
    `<tr><td>army</td><td class=p0>${{s[2]}}</td><td class=p1>${{s[4]}}</td></tr>`;
}}

function chart() {{
  const S = D.series, n = S.length;
  const maxLand = Math.max(1, ...S.map(s => Math.max(s[1], s[3])));
  const maxArmy = Math.max(1, ...S.map(s => Math.max(s[2], s[4])));
  const path = (idx, scale) => S.map((s, i) =>
    (i ? 'L' : 'M') + (i / (n - 1 || 1) * 400).toFixed(1) + ' ' +
    (118 - s[idx] / scale * 116).toFixed(1)).join(' ');
  document.getElementById('chart').innerHTML =
    `<path d="${{path(1, maxLand)}}" fill="none" stroke="${{COL.p0s}}" stroke-width="1.5"></path>` +
    `<path d="${{path(3, maxLand)}}" fill="none" stroke="${{COL.p1s}}" stroke-width="1.5"></path>` +
    `<path d="${{path(2, maxArmy)}}" fill="none" stroke="${{COL.p0s}}" stroke-width=1 stroke-dasharray="3 3"></path>` +
    `<path d="${{path(4, maxArmy)}}" fill="none" stroke="${{COL.p1s}}" stroke-width=1 stroke-dasharray="3 3"></path>`;
}}

const scrub = document.getElementById('scrub');
scrub.max = N - 1;
scrub.oninput = () => draw(+scrub.value);
let timer = null;
const playBtn = document.getElementById('play');
playBtn.onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; playBtn.textContent = 'play'; return; }}
  playBtn.textContent = 'pause';
  timer = setInterval(() => {{
    if (+scrub.value >= N - 1) {{ clearInterval(timer); timer = null; playBtn.textContent = 'play'; return; }}
    scrub.value = +scrub.value + 1; draw(+scrub.value);
  }}, 1000 / +document.getElementById('speed').value);
}};
document.getElementById('speed').onchange = () => {{ if (timer) {{ playBtn.click(); playBtn.click(); }} }};
addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') {{ scrub.value = Math.min(N - 1, +scrub.value + 1); draw(+scrub.value); }}
  if (e.key === 'ArrowLeft') {{ scrub.value = Math.max(0, +scrub.value - 1); draw(+scrub.value); }}
  if (e.key === ' ') {{ e.preventDefault(); playBtn.click(); }}
}});
chart(); draw(0);
</script>
"""


def render(rep: dict, out_path: str | Path, every: int = 1) -> Path:
    data = pack(rep, every)
    html = _HTML.format(
        seed=data["meta"]["seed"], spec0=data["meta"]["spec0"], spec1=data["meta"]["spec1"],
        reason=data["meta"]["reason"], turns=data["meta"]["turns"],
        data=json.dumps(data),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    args = ap.parse_args()
    rep = replay_mod.load(args.replay)
    out = args.out or str(Path(args.replay).with_suffix(".html"))
    print(render(rep, out, args.every))


if __name__ == "__main__":
    main()
