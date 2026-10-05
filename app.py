"""Interactive Streamlit visualizer for the DCP simulation pipeline."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ExperimentConfig
from src.orchestrator import Orchestrator, TraceEvent
from src.utils.math_utils import wilson_score_interval


ROOT = Path(__file__).resolve().parent
CORE_RESULTS = ROOT / "results" / "aggregated" / "dcp_core_summary.csv"
SAMPLE_RESULTS = ROOT / "results" / "aggregated" / "dcp_sample_complexity_summary.csv"


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Read an existing project result table for dashboard display."""
    return pd.read_csv(path)


def phase_status(events: list[TraceEvent], stage: str) -> bool:
    """Return whether a stage has appeared in the visible event prefix."""
    return any(event.stage == stage for event in events)


def render_state(event: TraceEvent) -> None:
    payload = event.payload
    amplitudes = payload["amplitudes"]
    st.caption(
        f"Prepared DCP state: N={payload['N']}, s={payload['s']}, "
        f"x={payload['x']}, (x+s) mod N={payload['target']}"
    )
    st.code(payload["circuit_text"], language="text")
    figure = go.Figure(
        go.Bar(
            x=[item["basis"] for item in amplitudes],
            y=[item["magnitude"] for item in amplitudes],
            marker_color=["#20a36a", "#3478c7"],
            text=[f"P={item['probability']:.3f}" for item in amplitudes],
            textposition="outside",
        )
    )
    figure.update_layout(
        title="Non-zero statevector amplitudes",
        xaxis_title="Computational basis state",
        yaxis_title="Amplitude magnitude",
        yaxis_range=[0, 1],
        height=330,
        margin=dict(t=55, b=35, l=35, r=20),
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(f"Circuit depth: {payload['circuit_depth']}")


def render_qft(event: TraceEvent) -> None:
    payload = event.payload
    distribution = payload["distribution"]
    phases = payload["phases"]
    left, right = st.columns(2)
    with left:
        figure = go.Figure(
            go.Bar(
                x=list(distribution),
                y=list(distribution.values()),
                marker_color="#4c78a8",
            )
        )
        figure.update_layout(
            title="Fourier-label probability",
            xaxis_title="Fourier label y",
            yaxis_title="P(y)",
            height=330,
            margin=dict(t=55, b=35, l=35, r=20),
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        labels = list(phases)
        angles = [float(np.angle(phases[y], deg=True)) for y in labels]
        figure = go.Figure(
            go.Scatterpolar(
                r=[1.0] * len(labels),
                theta=angles,
                mode="markers+text",
                text=[str(y) for y in labels],
                textposition="top center",
                marker=dict(
                    size=10,
                    color=labels,
                    colorscale="Viridis",
                    colorbar=dict(title="y"),
                ),
            )
        )
        figure.update_layout(
            title="Secret-dependent relative phases",
            polar=dict(radialaxis=dict(visible=False, range=[0, 1.2])),
            showlegend=False,
            height=330,
            margin=dict(t=55, b=35, l=35, r=25),
        )
        st.plotly_chart(figure, width="stretch")
    st.info(
        "For DCP, P(y) is flat while the relative phase varies as "
        "exp(2πisy/N): the secret is encoded in phase, not in the label magnitudes."
    )


def render_sample(event: TraceEvent) -> None:
    payload = event.payload
    n = payload["n"]
    k = payload["k"] or n
    noisy_label = payload["Y_noisy"]
    bits = format(noisy_label, f"0{n}b")
    flipped = set(payload["bit_flips"])
    styled_bits = []
    for position, bit in enumerate(bits):
        bit_index = n - position - 1
        if payload["truncation_mode"] == "msb":
            retained = position < k
        else:
            retained = position >= n - k
        color = "#d62728" if bit_index in flipped else "#20a36a" if retained else "#777777"
        decoration = "line-through" if not retained else "none"
        styled_bits.append(
            f"<span style='color:{color};text-decoration:{decoration};"
            f"font-weight:{'700' if retained else '400'}'>{bit}</span>"
        )

    columns = st.columns(4)
    columns[0].metric("Measured Fourier label y", payload["Y_full"])
    columns[1].metric("Noisy label", noisy_label)
    columns[2].metric("Observed truncated label", payload["Y_truncated"])
    columns[3].metric(
        "Flag b: measured → observed",
        f"{payload['b_sampled']} → {payload['b']}",
    )
    st.markdown(
        "Noisy n-bit label (green = retained, gray/struck = discarded, "
        "red = flipped):<br><code>"
        + "".join(styled_bits)
        + "</code>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Independent y-bit flips at positions {payload['bit_flips']}; "
        f"flag-bit flip applied: {payload['flag_bit_flipped']}."
    )


def render_posterior(event: TraceEvent, true_secret: int, modulus: int) -> None:
    posterior = event.payload["posterior"]
    candidates = list(posterior)
    mirror = (-true_secret) % modulus
    colors = [
        "#20a36a" if candidate == true_secret else
        "#f28e2b" if candidate == mirror else
        "#9aa0a6"
        for candidate in candidates
    ]
    figure = go.Figure(
        go.Bar(
            x=candidates,
            y=[posterior[candidate] for candidate in candidates],
            marker_color=colors,
        )
    )
    figure.update_layout(
        title=f"Bayesian posterior after {event.payload['samples_seen']} sample(s)",
        xaxis_title="Secret candidate",
        yaxis_title="Posterior probability",
        height=360,
        margin=dict(t=55, b=35, l=35, r=20),
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"True secret s={true_secret} (green); mirror N−s={mirror} (orange). "
        "The DCP measurement model cannot distinguish this mirror pair."
    )


def render_verdict(event: TraceEvent) -> None:
    payload = event.payload
    columns = st.columns(4)
    columns[0].metric("Recovered secret ŝ", payload["s_hat"])
    columns[1].metric("Exact match", "Yes" if payload["correct"] else "No")
    columns[2].metric("Mirror match", "Yes" if payload["mirror_correct"] else "No")
    columns[3].metric("Posterior confidence", f"{payload['confidence']:.3f}")
    st.write(
        "Per-bit correctness (least-significant bit first):",
        " · ".join(
            f"bit {index}: {'✓' if correct else '✗'}"
            for index, correct in enumerate(payload["bit_correct"])
        ),
    )


def render_trace_player() -> None:
    events: list[TraceEvent] = st.session_state.get("trace_events", [])
    if not events:
        st.info("Set parameters in the sidebar and click **Generate trace**.")
        return

    if "trace_index" not in st.session_state:
        st.session_state.trace_index = 0
    visible_index = min(st.session_state.trace_index, len(events) - 1)

    auto_play = st.toggle(
        "Play / Pause",
        help="Turn on automatic step advancement; turn off to pause.",
        key="trace_auto_play",
    )
    controls = st.columns(3)
    if controls[0].button("Step", disabled=visible_index >= len(events) - 1):
        st.session_state.trace_index = min(visible_index + 1, len(events) - 1)
        st.rerun()
    if controls[1].button("Replay"):
        st.session_state.trace_index = 0
        st.rerun()

    visible_events = events[: visible_index + 1]
    st.progress((visible_index + 1) / len(events))
    current = events[visible_index]
    st.caption(
        f"Step {visible_index + 1}/{len(events)} — {current.stage}"
        + (f", sample {current.sample + 1}" if current.sample is not None else "")
    )

    stage_labels = ["State preparation", "QFT", "Sampling", "Posterior", "Verdict"]
    stage_names = ["state", "qft", "sample", "posterior", "verdict"]
    status_columns = st.columns(len(stage_labels))
    for column, label, stage in zip(status_columns, stage_labels, stage_names):
        complete = phase_status(visible_events, stage)
        column.markdown(f"**{'🟢' if complete else '⚪'} {label}**")

    state_event = next((event for event in visible_events if event.stage == "state"), None)
    qft_event = next((event for event in visible_events if event.stage == "qft"), None)
    if state_event:
        with st.expander("1 · State preparation", expanded=current.stage == "state"):
            render_state(state_event)
    if qft_event:
        with st.expander("2 · Quantum Fourier Transform", expanded=current.stage == "qft"):
            render_qft(qft_event)

    latest_sample = next(
        (event for event in reversed(visible_events) if event.stage == "sample"),
        None,
    )
    if latest_sample:
        with st.expander("3 · Measurement and truncation", expanded=current.stage == "sample"):
            render_sample(latest_sample)

    latest_posterior = next(
        (event for event in reversed(visible_events) if event.stage == "posterior"),
        None,
    )
    if latest_posterior:
        with st.expander("4 · Posterior evolution", expanded=current.stage == "posterior"):
            render_posterior(
                latest_posterior,
                true_secret=int(st.session_state.trace_parameters["s"]),
                modulus=int(st.session_state.trace_parameters["N"]),
            )

    verdict = next(
        (event for event in reversed(visible_events) if event.stage == "verdict"),
        None,
    )
    if verdict:
        with st.expander("5 · Recovery verdict", expanded=current.stage == "verdict"):
            render_verdict(verdict)

    if auto_play and visible_index < len(events) - 1:
        time.sleep(st.session_state.get("trace_delay", 0.5))
        st.session_state.trace_index = visible_index + 1
        st.rerun()


def render_experiment_dashboard() -> None:
    st.subheader("Previously verified experiment results")
    if CORE_RESULTS.exists():
        core = load_csv(str(CORE_RESULTS))
        st.markdown("#### Truncation and information profile")
        n_filter = st.multiselect(
            "Moduli to show",
            sorted(core["N"].unique().tolist()),
            default=sorted(core["N"].unique().tolist()),
            key="dashboard_moduli",
        )
        shown = core[core["N"].isin(n_filter)].sort_values(["N", "k"])
        left, right = st.columns(2)
        with left:
            figure = go.Figure()
            for modulus, group in shown.groupby("N"):
                figure.add_trace(
                    go.Scatter(
                        x=group["k"],
                        y=group["recovery_prob"],
                        mode="lines+markers",
                        name=f"N={modulus}",
                    )
                )
            figure.update_layout(
                title="Exact recovery vs retained bits",
                xaxis_title="Retained bits k",
                yaxis_title="Recovery probability",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(figure, width="stretch")
        with right:
            figure = go.Figure()
            for modulus, group in shown.groupby("N"):
                figure.add_trace(
                    go.Scatter(
                        x=group["k"],
                        y=group["mi_truncated"],
                        mode="lines+markers",
                        name=f"N={modulus}",
                    )
                )
            figure.update_layout(
                title="Mutual information vs retained bits",
                xaxis_title="Retained bits k",
                yaxis_title="Mutual information (bits)",
            )
            st.plotly_chart(figure, width="stretch")
        st.dataframe(shown, width="stretch", hide_index=True)
    else:
        st.warning(f"Core results file not found: {CORE_RESULTS}")

    st.markdown("#### Sample-complexity results")
    if SAMPLE_RESULTS.exists():
        samples = load_csv(str(SAMPLE_RESULTS))
        figure = go.Figure()
        for (modulus, retained), group in samples.groupby(["N", "k"]):
            figure.add_trace(
                go.Scatter(
                    x=group["m"],
                    y=group["recovery_prob"],
                    mode="lines+markers",
                    name=f"N={modulus}, k={retained}",
                )
            )
        figure.update_layout(
            title="Recovery probability vs independent samples",
            xaxis_title="Samples m",
            yaxis_title="Recovery probability",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(figure, width="stretch")
        st.dataframe(samples, width="stretch", hide_index=True)
    else:
        st.info("No archived sample-complexity summary is present in results/aggregated.")


def render_golden_validation() -> None:
    st.subheader("Published result vs. live simulation")
    if not CORE_RESULTS.exists():
        st.info("The published core summary is not available in this checkout.")
        return

    core = load_csv(str(CORE_RESULTS))
    labels = {
        index: (
            f"N={int(row.N)}, k={int(row.k)}, m={int(row.m)}, "
            f"s={int(row.s)}, ε={row.epsilon:g}"
        )
        for index, row in core.iterrows()
    }
    selected_index = st.selectbox(
        "Core configuration",
        list(labels),
        format_func=lambda index: labels[index],
        key="golden_config",
    )
    row = core.loc[selected_index]
    st.caption(
        f"Published P(success)={row.recovery_prob:.3f}; 95% Wilson CI "
        f"[{row.ci_lower:.3f}, {row.ci_upper:.3f}] from {int(row.shots)} trials."
    )
    if st.button("Run this configuration live", key="run_golden"):
        config = ExperimentConfig(
            N=int(row.N),
            s=int(row.s),
            k=int(row.k),
            m=int(row.m),
            epsilon=float(row.epsilon),
            shots=int(row.shots),
            seed=42,
            problem_type="dcp",
            recovery_strategy="bayesian",
            truncation_mode=str(row.truncation_mode),
        )
        with st.spinner("Running the selected experiment through the project engines..."):
            result = Orchestrator().run(config)
        stats = result.statistics
        if stats is None:
            st.error("The live run completed without summary statistics.")
            return
        inside = float(row.ci_lower) <= stats.recovery_prob <= float(row.ci_upper)
        if inside:
            st.success(
                f"Live P(success)={stats.recovery_prob:.3f} is inside the published interval."
            )
        else:
            st.warning(
                f"Live P(success)={stats.recovery_prob:.3f} is outside the published interval."
            )
        st.metric(
            "Live 95% Wilson interval",
            f"{stats.recovery_prob_ci[0]:.3f}–{stats.recovery_prob_ci[1]:.3f}",
        )
        chart = go.Figure()
        chart.add_trace(
            go.Scatter(
                x=["Published", "Live"],
                y=[float(row.recovery_prob), stats.recovery_prob],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        float(row.ci_upper) - float(row.recovery_prob),
                        stats.recovery_prob_ci[1] - stats.recovery_prob,
                    ],
                    arrayminus=[
                        float(row.recovery_prob) - float(row.ci_lower),
                        stats.recovery_prob - stats.recovery_prob_ci[0],
                    ],
                ),
                mode="markers",
                marker=dict(
                    size=12,
                    color=["#4c78a8", "#20a36a" if inside else "#e45756"],
                ),
            )
        )
        chart.update_layout(
            title="Exact recovery probability with 95% Wilson intervals",
            yaxis_title="P(success)",
            yaxis_range=[0, 1],
            showlegend=False,
        )
        st.plotly_chart(chart, width="stretch")


def render_live_noise_sweep() -> None:
    st.subheader("Interactive noise robustness sweep")
    st.caption(
        "Runs the existing Orchestrator on a small selected sweep; this uses the same "
        "state preparation, measurement, truncation, and recovery implementation."
    )
    modulus = st.selectbox("Noise-sweep modulus N", [4, 8, 16, 32], index=2, key="noise_N")
    secret = st.selectbox(
        "Noise-sweep secret s",
        list(range(int(modulus))),
        index=min(5, int(modulus) - 1),
        key="noise_s",
    )
    sample_count = st.slider("Samples m", 1, 16, 8, key="noise_m")
    trial_count = st.slider("Trials per noise level", 20, 300, 100, step=20, key="noise_trials")
    if st.button("Run noise sweep", key="run_noise"):
        n = max(1, (int(modulus) - 1).bit_length())
        base = ExperimentConfig(
            N=int(modulus),
            s=int(secret),
            k=n,
            m=int(sample_count),
            shots=int(trial_count),
            seed=42,
            recovery_strategy="bayesian",
            truncation_mode="msb",
        )
        with st.spinner("Running five noise levels..."):
            trials = Orchestrator().run_sweep(
                base_config=base,
                param_grid={"epsilon": [0.0, 0.05, 0.1, 0.15, 0.2]},
            )
        summary_records: list[dict[str, Any]] = []
        for epsilon, group in trials.groupby("epsilon"):
            successes = int(group["correct"].sum())
            lower, upper = wilson_score_interval(successes, len(group))
            summary_records.append(
                {
                    "epsilon": float(epsilon),
                    "recovery_prob": float(group["correct"].mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
        summary = pd.DataFrame(summary_records)
        figure = go.Figure(
            go.Scatter(
                x=summary["epsilon"],
                y=summary["recovery_prob"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=summary["ci_upper"] - summary["recovery_prob"],
                    arrayminus=summary["recovery_prob"] - summary["ci_lower"],
                ),
                mode="lines+markers",
            )
        )
        figure.update_layout(
            title=f"Noise robustness: N={modulus}, k=n={n}, m={sample_count}",
            xaxis_title="Independent bit-flip probability ε",
            yaxis_title="Exact recovery probability",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(figure, width="stretch")
        st.dataframe(summary, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="DCP Lattice Visualizer",
        page_icon="🔬",
        layout="wide",
    )
    st.title("See the DCP lattice problem being solved")
    st.markdown(
        "An interactive, trace-driven view of the repository's DCP simulator. "
        "The charts below consume events from the existing Qiskit and inference engines."
    )
    st.info(
        "Measurement convention: the sampler jointly observes Fourier label y and "
        "the flag qubit in the Hadamard (X) basis, "
        "P(y,b|s) = [1 + (−1)^b cos(2πsy/N)]/(2N). "
        "Noise is applied to y and b before y is truncated."
    )

    with st.sidebar:
        st.header("Single-trial parameters")
        with st.form("trace_config"):
            modulus = st.selectbox("Modulus N", [4, 8, 16, 32, 64], index=2)
            secret = st.number_input("Secret s", min_value=0, max_value=63, value=5)
            offset = st.number_input("Offset x", min_value=0, max_value=63, value=11)
            n = max(1, (int(modulus) - 1).bit_length())
            retained = st.slider("Retained Fourier bits k", 1, n, n)
            samples = st.slider("Independent samples m", 1, 16, 8)
            epsilon = st.slider("Bit-flip noise ε", 0.0, 0.2, 0.0, step=0.01)
            mode = st.selectbox("Truncation", ["msb", "lsb"])
            strategy = st.selectbox(
                "Recovery strategy",
                ["bayesian", "brute_force", "ml", "bitwise"],
            )
            seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42)
            submitted = st.form_submit_button("Generate trace", type="primary")
        delay = st.slider(
            "Playback delay (seconds per event)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.trace_delay = delay

    if submitted:
        if int(secret) >= int(modulus) or int(offset) >= int(modulus):
            st.error("The secret and offset must both be smaller than N.")
        else:
            config = ExperimentConfig(
                N=int(modulus),
                s=int(secret),
                k=int(retained),
                m=int(samples),
                epsilon=float(epsilon),
                shots=1,
                seed=int(seed),
                recovery_strategy=strategy,
                truncation_mode=mode,
            )
            with st.spinner("Preparing the Qiskit states and inference trace..."):
                trace = list(
                    Orchestrator().run_traced(
                        config,
                        offset_override=int(offset),
                    )
                )
            st.session_state.trace_events = trace
            st.session_state.trace_parameters = {
                "N": int(modulus),
                "s": int(secret),
            }
            st.session_state.trace_index = 0
            st.session_state.trace_auto_play = False

    trace_tab, results_tab, validation_tab, noise_tab = st.tabs(
        ["Single-trial visualizer", "Results dashboard", "Golden validation", "Noise sweep"]
    )
    with trace_tab:
        render_trace_player()
    with results_tab:
        render_experiment_dashboard()
    with validation_tab:
        render_golden_validation()
    with noise_tab:
        render_live_noise_sweep()


if __name__ == "__main__":
    main()
