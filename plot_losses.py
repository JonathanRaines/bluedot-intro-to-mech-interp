import torch

from src import plot


def main() -> None:
    model = torch.load("persist/grokking_demo.pth")

    fig = plot.log_loss(model["train_losses"], model["test_losses"])
    fig.write_image("figures/log_loss.png")
    plot.line_to_csv(fig, "figures/log_loss.csv")


if __name__ == "__main__":
    main()
