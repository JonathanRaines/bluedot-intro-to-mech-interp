import plotly.graph_objects as go
import numpy as np


def log_loss(train_losses: list[int], test_losses: list[int]) -> go.Figure:
    """Generates a logarithmic line plot of loss over epochs."""
    x = np.arange(0, len(train_losses), 100)
    y1 = train_losses[::100]
    y2 = test_losses[::100]

    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=y1, name="train"),
            go.Scatter(x=x, y=y2, name="test"),
        ],
        layout={
            "xaxis": {"title": "Epoch"},
            "yaxis": {"title": "Loss", "type": "log"},
            "title": "Training Curve for Modular Addition",
            "template": "simple_white",
        },
    )
    return fig
