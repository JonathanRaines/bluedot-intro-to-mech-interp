import pandas as pd
import torch

from src import plot


def make_figures() -> None:
    model = torch.load("persist/grokking_demo.pth")

    pd.DataFrame.from_records([model["train_losses"], model["test_losses"]]).to_csv(
        "figures/log_loss.csv"
    )
    plot.log_loss(model["train_losses"], model["test_losses"]).write_image(
        "figures/log_loss.png"
    )


if __name__ == "__main__":
    make_figures()
