# fyp2027

## Interactive DCP Visualizer

Install the optional UI dependencies and launch the Streamlit app from the
repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app traces the existing DCP, QFT, measurement, truncation, Bayesian
inference, and recovery code. It includes a step/replay visualizer, charts for
the archived truncation and sample-complexity results, a live comparison
against the published Wilson interval, and an on-demand noise sweep. See
[ui_implementation.md](./ui_implementation.md) for phase status and design
notes.