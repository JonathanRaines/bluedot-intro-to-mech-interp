import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from transformer_lens import utils

TEMPLATE = "simple_white"


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
            "template": TEMPLATE,
        },
    )
    return fig


def imshow(tensor, xaxis="", yaxis="", **kwargs) -> go.Figure:
    return px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        template=TEMPLATE,
        **kwargs,
    )


def line(tensor, xaxis="", yaxis="", **kwargs) -> go.Figure:
    return px.line(
        utils.to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis},
        template=TEMPLATE,
        **kwargs,
    )


def multiline(tensors, xaxis="", yaxis="", labels=None, **kwargs) -> go.Figure:
    fig = go.Figure()
    for i, tensor in enumerate(tensors):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(tensor)),
                y=utils.to_numpy(tensor),
                mode="lines",
                name=labels[i] if labels else f"Line {i}",
            )
        )
    fig.update_layout(
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        template=TEMPLATE,
        **kwargs,
    )
    return fig
