"""Plot the recorded Apple M4 Max accumulator measurements."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "docs" / "m4-accumulator-benchmark.md"
OUTPUT = ROOT / "images" / "m4-dot-f32-benchmark.png"

COLORS = ("#64748b", "#f59e0b", "#2563eb")


def read_measurements() -> tuple:
    """Read the marked benchmark table, which is the measurement source."""
    text = BENCHMARK.read_text()
    table = text.split("<!-- benchmark-data:start -->", 1)[1].split(
        "<!-- benchmark-data:end -->", 1
    )[0]
    fixtures = {}
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in {"Fixture", "---"}:
            continue
        fixture, variant, median, _ = cells
        fixtures.setdefault(fixture, []).append((variant.replace(",", ""), float(median)))

    if len(fixtures) != 2 or any(len(values) != 3 for values in fixtures.values()):
        raise ValueError("expected two fixtures with three benchmark variants each")

    result = []
    for fixture, rows in fixtures.items():
        variants, values = zip(*rows)
        ratio = values[1] / values[2]
        result.append((fixture.replace(", ", " · "), variants, values, f"{ratio:.2f}×"))
    return tuple(result)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#0f172a",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    figure.patch.set_facecolor("#f8fafc")

    fixtures = read_measurements()
    for index, (axis, (title, variants, values, ratio)) in enumerate(
        zip(axes, fixtures)
    ):
        axis.set_facecolor("#ffffff")
        bars = axis.barh(variants, values, color=COLORS, height=0.58)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", pad=18)
        axis.set_xlim(0.0, 0.55)
        axis.set_xlabel("Median nanoseconds per element  ·  lower is better")
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        if index:
            axis.tick_params(axis="y", labelleft=False)

        for bar, value in zip(bars, values):
            axis.text(
                value + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                ha="left",
                color="#0f172a",
                fontweight="bold",
            )

        axis.text(
            0.525,
            1.55,
            f"four-register throughput\n{ratio} one-register",
            ha="right",
            va="center",
            color="#1d4ed8",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#eff6ff",
                "edgecolor": "#bfdbfe",
            },
        )

    figure.suptitle(
        "Four accumulators break the NEON dependency chain",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
    )
    figure.text(
        0.06,
        0.925,
        "Apple M4 Max · rustc 1.98.0-nightly · 11-sample medians",
        ha="left",
        fontsize=12,
        color="#475569",
    )
    figure.text(
        0.06,
        0.025,
        "Recorded run: cargo run --release --features bench-api --example bench",
        ha="left",
        fontsize=10,
        color="#64748b",
    )
    figure.subplots_adjust(left=0.19, right=0.98, top=0.84, bottom=0.14, wspace=0.24)
    figure.savefig(OUTPUT, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
