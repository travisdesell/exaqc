"""Animate a search: the circuit as it is rewritten, its QASM, and infidelity.

Reads a trace written by `evolve_*_pareto.py --trace`. Layout:

  top left     the circuit on a real nanosecond axis, so gate spans are to
               scale and the gaps between them ARE the idle time the device
               card charges for. CNOTs carry an explicit control/target
               marking -- filled dot labelled C on the control, oplus labelled
               T on the target -- and are coloured by which control->target
               pair they act on, with the palette fixed over the whole trace so
               a pair keeps its colour as the circuit is rewritten.
  top right    the OpenQASM for that iteration, each cx line tinted to match
               its gate in the circuit panel, with the ASAP window in a comment
               because QASM cannot express duration.
  bottom       tier-1 state infidelity against step, spanning the full width,
               with iteration 0 (the textbook circuit, exact in the closed
               system) held as a dashed reference, and p_fail alongside it so
               the disagreement between the screening objective and the truth
               stays visible.

The qiskit drawer was tried for the circuit panel and dropped: it gives
standard notation but lays gates out by order, so it cannot show that a wire
waits 450 ns for its only gate, which is the quantity this study is about.

Writes MP4 if an ffmpeg binary is available (imageio-ffmpeg supplies one),
otherwise an animated GIF, plus a static final frame as PDF for the paper.

    .venv_win/Scripts/python scripts/make_circuit_movie.py \
        --trace results/trace_ghz4_mixed.json --out results/ghz4_mixed.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Ellipse, FancyBboxPatch

# matplotlib needs an ffmpeg binary to write MP4 and there is none on PATH here.
# imageio-ffmpeg ships one as a Python package, which keeps the dependency
# contained rather than installing ffmpeg system-wide.
try:
    import imageio_ffmpeg as _iio
    matplotlib.rcParams["animation.ffmpeg_path"] = _iio.get_ffmpeg_exe()
except Exception:
    pass

GATE_COLOR = {"h": "#4C72B0", "x": "#DD8452", "z": "#55A868",
              "s": "#8172B3", "t": "#C44E52", "sdg": "#8172B3", "tdg": "#C44E52"}
CX_COLOR = "#1a1a1a"
QASM_GATE = {"h": "h", "x": "x", "z": "z", "s": "s", "t": "t",
             "sdg": "sdg", "tdg": "tdg", "cx": "cx"}


def schedule(gates: Sequence[Tuple[str, List[int]]], n_wires: int,
             dur: Dict[str, float]) -> Tuple[List[tuple], float]:
    """ASAP start times, so the drawing's x-axis is physical time."""
    free = [0.0] * n_wires
    placed = []
    for name, wires in gates:
        t0 = max(free[w] for w in wires)
        placed.append((t0, name, list(wires)))
        for w in wires:
            free[w] = t0 + dur[name]
    return placed, (max(free) if free else 0.0)


def cnot_palette(traces) -> Dict[Tuple[int, int], str]:
    """One colour per (control, target) pair, assigned over the WHOLE trace.

    Stability matters in a movie: per-frame assignment would make a pair change
    colour as the circuit is rewritten and the eye would read a rewiring where
    there was none. (c,t) and (t,c) are deliberately distinct keys, since
    swapping control and target is a different gate.
    """
    pairs = []
    for r in traces:
        for name, wires in r["gates"]:
            if name == "cx":
                key = (int(wires[0]), int(wires[1]))
                if key not in pairs:
                    pairs.append(key)
    wheel = ["#C44E52", "#4C72B0", "#55A868", "#8172B3", "#CCB974",
             "#64B5CD", "#DD8452", "#937860", "#DA8BC3", "#8C8C8C"]
    return {p: wheel[i % len(wheel)] for i, p in enumerate(sorted(pairs))}


def draw_cnot(ax, t0: float, T: float, ctrl: int, targ: int, xmax: float,
              color: str) -> None:
    """Control and target marked explicitly and coloured by the pair acted on.

    The axes are nanoseconds against wire index, so a circle in data units
    would come out as a slit; the markers are ellipses whose x radius is scaled
    by the time extent.
    """
    xc = t0 + T / 2
    rx_dot, ry_dot = xmax * 0.009, 0.075
    rx_op, ry_op = xmax * 0.018, 0.15

    # the entangler's duration as a faint span, so its cost against a 50 ns
    # single-qubit gate is visible rather than asserted
    ax.add_patch(FancyBboxPatch(
        (t0, min(ctrl, targ) - 0.20), T, abs(ctrl - targ) + 0.40,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=color, alpha=0.13, zorder=2))
    ax.plot([xc, xc], [ctrl, targ], color=color, lw=2.0, zorder=3)

    ax.add_patch(Ellipse((xc, ctrl), 2 * rx_dot, 2 * ry_dot,
                         facecolor=color, edgecolor="none", zorder=5))
    ax.annotate("C", (xc, ctrl), textcoords="offset points", xytext=(7, 5),
                fontsize=7.5, fontweight="bold", color=color, zorder=6)

    ax.add_patch(Ellipse((xc, targ), 2 * rx_op, 2 * ry_op, facecolor="white",
                         edgecolor=color, lw=2.0, zorder=5))
    ax.plot([xc - rx_op, xc + rx_op], [targ, targ], color=color, lw=1.6, zorder=6)
    ax.plot([xc, xc], [targ - ry_op, targ + ry_op], color=color, lw=1.6, zorder=6)
    ax.annotate("T", (xc, targ), textcoords="offset points", xytext=(8, 5),
                fontsize=7.5, fontweight="bold", color=color, zorder=6)


def draw_circuit(ax, gates, n_wires, dur, title: str, xmax: float,
                 cx_colors: Dict[Tuple[int, int], str]) -> None:
    ax.clear()
    placed, _ = schedule(gates, n_wires, dur)

    for w in range(n_wires):
        ax.plot([0, xmax], [w, w], color="#c8c8c8", lw=1.0, zorder=1)
        ax.text(-xmax * 0.03, w, f"$q_{w}$", ha="right", va="center", fontsize=9)

    for t0, name, wires in placed:
        T = dur[name]
        if name == "cx":
            key = (int(wires[0]), int(wires[1]))
            draw_cnot(ax, t0, T, wires[0], wires[1], xmax,
                      cx_colors.get(key, CX_COLOR))
        else:
            w = wires[0]
            ax.add_patch(FancyBboxPatch(
                (t0 + T * 0.08, w - 0.21), T * 0.84, 0.42,
                boxstyle="round,pad=0.01",
                facecolor=GATE_COLOR.get(name, "#777777"),
                edgecolor="none", alpha=0.92, zorder=3))
            ax.text(t0 + T / 2, w, name.upper(), ha="center", va="center",
                    color="white", fontsize=7, zorder=4)

    ax.set_xlim(-xmax * 0.08, xmax * 1.02)
    ax.set_ylim(n_wires - 0.55, -0.75)
    ax.set_xlabel("time (ns) — spans to scale, gaps are idle",
                  fontsize=8.5, labelpad=2)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=9, loc="left", pad=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def qasm_lines(gates, n_wires, dur, cx_colors) -> List[Tuple[str, str]]:
    """(text, colour) per QASM line, cx tinted to match the circuit panel.

    Ordering.  QASM is a sequential instruction list, so the order has to be a
    valid topological order of the circuit.  The genome order already is one --
    it is exactly what the simulator applies -- but it is NOT monotonic in
    start time, because gates on disjoint wires are emitted in genome order
    rather than in schedule order, so a gate at t = 50 ns could be listed after
    one at t = 250 ns.  That reads as a bug even though the circuit is right.

    Sorting by start time fixes the presentation and stays a valid order: two
    gates sharing a wire have strictly different start times, so their relative
    order is preserved, and two gates with the same start time act on disjoint
    wires and therefore commute.  Verified directly -- the prepared state is
    bit-identical under both orderings.

    `barrier` is emitted between ASAP layers so a downstream compiler cannot
    reorder across the schedule we deliberately chose.
    """
    placed, total = schedule(gates, n_wires, dur)
    placed = sorted(placed, key=lambda x: (x[0], min(x[2])))
    out = [("OPENQASM 2.0;", "#444444"),
           ('include "qelib1.inc";', "#444444"),
           (f"qreg q[{n_wires}];", "#444444"),
           ("// comments: ASAP window, ns", "#999999"),
           ("", "#444444")]
    prev_t0 = None
    for t0, name, wires in placed:
        if prev_t0 is not None and t0 != prev_t0:
            out.append(("barrier q;", "#bbbbbb"))
        prev_t0 = t0
        T = dur[name]
        if name == "cx":
            c, t = wires
            col = cx_colors.get((int(c), int(t)), CX_COLOR)
            out.append((f"cx q[{c}],q[{t}];".ljust(15)
                        + f"// C{c}->T{t} {t0:.0f}-{t0 + T:.0f}", col))
        else:
            w = wires[0]
            out.append((f"{QASM_GATE[name]} q[{w}];".ljust(15)
                        + f"//       {t0:.0f}-{t0 + T:.0f}",
                        GATE_COLOR.get(name, "#555555")))
    out.append(("", "#444444"))
    out.append((f"// total {total:.0f} ns", "#444444"))
    return out


def draw_qasm(ax, gates, n_wires, dur, cx_colors, max_lines: int = 26) -> None:
    """Render the QASM, sized so the lines cannot overlap.

    max_lines is the longest listing in the whole trace, so the block keeps a
    fixed line pitch and the text does not jump around as gates are added and
    pruned. The font is then chosen from the panel's actual height rather than
    assumed: at 26 lines a fixed 7.2 pt overlapped badly.  The panel spans
    both grid rows, which is what buys the font its ~10 pt.
    """
    ax.clear()
    ax.set_axis_off()
    ax.set_title("OpenQASM at this iteration", fontsize=10, loc="left", pad=8)
    lines = qasm_lines(gates, n_wires, dur, cx_colors)

    fig = ax.figure
    h_pt = ax.get_window_extent().height / fig.dpi * 72.0
    n_slots = max(max_lines, 1)
    # 1.32 is the usual monospace line-height factor; leave a little slack
    fs = max(5.5, min(11.0, 0.95 * h_pt / (n_slots * 1.32)))
    pitch = 1.0 / n_slots

    # If even the floor font cannot fit, say so rather than overlap.
    n_fit = int(h_pt / (5.5 * 1.32))
    if len(lines) > n_fit:
        keep = max(n_fit - 1, 1)
        lines = lines[:keep] + [(f"... {len(lines) - keep} more lines", "#999999")]

    y = 1.0
    for text, col in lines:
        ax.text(0.0, y, text, transform=ax.transAxes, family="monospace",
                fontsize=fs, color=col, va="top", ha="left")
        y -= pitch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default=None,
                    help="device card for gate durations (default: from trace)")
    ap.add_argument("--fps", type=float, default=1.4)
    ap.add_argument("--frames-dir", default=None,
                    help="also write numbered PNG frames here")
    ap.add_argument("--hold", type=int, default=6,
                    help="extra frames on the final circuit")
    args = ap.parse_args()

    d = json.load(open(args.trace))
    tr: List[dict] = d["trace"]
    if not tr:
        print("empty trace")
        return 1
    n_wires = int(d["n_wires"])

    cardp = args.card or (
        Path("../../Arch_Aware_Circuit_Evolution/device_cards")
        / f"device_card_{d['architecture']}_{d['model']}.json")
    card = json.load(open(cardp))
    dur = {g: float(e["T_ns"]) for g, e in card["gates"].items()}

    xmax = max(schedule([(a, b) for a, b in r["gates"]], n_wires, dur)[1] for r in tr)
    cx_colors = cnot_palette(tr)
    print("CNOT pairs in this trace: "
          + ", ".join(f"q{c}->q{t}" for c, t in sorted(cx_colors)))
    max_lines = max(len(qasm_lines([(a, b) for a, b in r["gates"]], n_wires,
                                   dur, cx_colors)) for r in tr)

    inf = [r["infidelity"] for r in tr]
    seed_inf = inf[0]
    it = [r["iteration"] for r in tr]

    # Explicit gridspec with generous spacing: tight_layout cannot know that the
    # QASM panel is a text block and kept colliding with the circuit title.
    fig = plt.figure(figsize=(12.4, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.38, 1.0],
                          height_ratios=[1.15, 1.0],
                          left=0.060, right=0.985, top=0.880, bottom=0.115,
                          wspace=0.13, hspace=0.70)
    axc = fig.add_subplot(gs[0, 0])
    axq = fig.add_subplot(gs[:, 1])      # full height: the QASM needs the room
    axp = fig.add_subplot(gs[1, 0])

    tag = " (pruned)" if d.get("pruned") else ""
    fig.suptitle(f"{d['task']} on {d['architecture']} / {d['model']}{tag} — "
                 f"textbook circuit refined against the device card",
                 fontsize=11, y=0.965)

    lo, hi = min(inf + [seed_inf]), max(inf + [seed_inf])
    span = max(hi - lo, hi * 1e-3)
    lo, hi = lo - 0.18 * span, hi + 0.18 * span

    def frame(i: int) -> None:
        k = min(i, len(tr) - 1)
        r = tr[k]
        gates = [(a, b) for a, b in r["gates"]]
        gain = seed_inf / r["infidelity"] if r["infidelity"] > 0 else float("inf")
        draw_circuit(
            axc, gates, n_wires, dur,
            # The tag is the SEARCH's own verdict, read off the state-blind
            # p_fail surrogate. Steps 1-8 here are tagged "improvement" while
            # the tier-1 infidelity plotted below gets worse, so name whose
            # verdict it is rather than letting it read as an improvement in
            # the quantity on the y-axis.
            f"iteration {r['iteration']}  (search tag: {r['tag']}, "
            f"p_fail = {r['p_fail']:.5f})\n"
            f"$1-F$ = {r['infidelity']:.4e}   |   "
            f"{r['n_gates']} gates, {r['n_entanglers']} CNOT, {r['time_ns']:.0f} ns"
            + ("" if k == 0 else
               f"   |   {max(gain, 1.0 / gain):.3f}$\\times$ "
               + ("better" if gain >= 1.0 else "worse")
               + " than iteration 0"),
            xmax, cx_colors)
        draw_qasm(axq, gates, n_wires, dur, cx_colors, max_lines)

        axp.clear()
        # Baseline: the initial (textbook) circuit's infidelity, dotted red and
        # held for the whole movie, so every later point is read against it.
        axp.axhline(seed_inf, ls=":", lw=1.8, color="#C44E52",
                    label="baseline — initial (textbook) circuit, "
                          f"$1-F$ = {seed_inf:.3e}")
        axp.plot(it[:k + 1], inf[:k + 1], "o-", ms=5, color="#4C72B0",
                 label="$1-F$ of the refined circuit (density matrix)")
        axp.plot([it[k]], [inf[k]], "o", ms=11, mfc="none", mec="#4C72B0", mew=2)
        axp.set_xlim(-0.4, max(it) + 0.4)
        axp.set_ylim(lo, hi)
        axp.set_xticks(it)
        axp.set_ylabel("state infidelity $1-F$", fontsize=9)
        # The explanatory line lives in the x-label, not in an axes title: the
        # title shared a strip of figure with the external legend and with the
        # circuit panel's x-label, and all three collided.
        axp.set_xlabel("improvement step\n"
                       "tier 1: card channels composed independently, "
                       "no bath memory across gates", fontsize=8.5)
        # Stacked above the axes: the panel is now only the left column wide, so
        # two side-by-side entries no longer fit on one line.
        axp.legend(frameon=False, fontsize=8.5, ncol=1,
                   loc="lower center", bbox_to_anchor=(0.5, 1.01),
                   borderaxespad=0.0, handlelength=2.4, columnspacing=2.5)
        axp.grid(alpha=0.25)
        axp.tick_params(labelsize=8)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)

    n_frames = len(tr) + args.hold
    if args.frames_dir:
        fd = Path(args.frames_dir)
        fd.mkdir(parents=True, exist_ok=True)
        for k in range(n_frames):
            frame(k)
            fig.savefig(fd / f"frame{k:03d}.png", dpi=130)
        print(f"wrote {n_frames} frames to {fd}")

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=1000 / args.fps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out), writer=FFMpegWriter(fps=args.fps, bitrate=3000))
        print("wrote", out)
    except Exception as e:
        gif = out.with_suffix(".gif")
        print(f"ffmpeg unavailable ({type(e).__name__}); writing GIF instead")
        anim.save(str(gif), writer=PillowWriter(fps=args.fps))
        print("wrote", gif)
    frame(len(tr) - 1)
    try:
        fig.savefig(str(out.with_suffix(".pdf")))
        print("wrote", out.with_suffix(".pdf"))
    except PermissionError:
        print(f"could not write {out.with_suffix('.pdf')} (open in a viewer?)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
