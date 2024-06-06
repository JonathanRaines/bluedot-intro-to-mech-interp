import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch

from src import const
from src import plot


def plot_training_losses(base: int, save_path: str) -> None:
    # Load the data from the file saved during training
    saved: dict = torch.load(save_path)
    train_losses = saved["train_losses"]
    test_losses = saved["test_losses"]

    # Create the x-axis
    x = np.arange(0, len(train_losses), 100)

    # Down-sample the data for plotting, taking every 100th value.
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
            "title": f"Training Curve for Base {base} Modular Addition",
            "template": "simple_white",
        },
    )

    fig.add_vline(x=1_600, line_width=1, line_dash="solid", line_color="black")
    fig.add_vline(x=9_500, line_width=1, line_dash="solid", line_color="black")

    # Save the plot as a png and csv (for use in LateX or Typst)
    fig.write_image("figures/log_loss.png")
    plot.line_to_csv(fig, "figures/log_loss.csv")


if __name__ == "__main__":
    plot_training_losses(
        base=const.MOD,
        save_path=const.MODEL_SAVE_PATH,
    )
