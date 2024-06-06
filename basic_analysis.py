import einops
import torch
import transformer_lens

from src import const
from src import loss
from src import plot
from src import task
from src import utils


def main():
    # Load the model saved from the training Google Colab
    hooked_model: transformer_lens.HookedTransformer = utils.load_model(
        path="persist/grokking_demo.pth", mod=const.MOD, device="cpu"
    )

    # Make a dataset and labels for the model
    dataset, labels = task.make_dataset_and_labels(
        base=const.MOD,
        device="cpu",
    )

    # Get a prediction for every possible input and cache the activations
    original_logits, cache = hooked_model.run_with_cache(dataset)

    print(f"{original_logits.numel():,}")

    W_E = hooked_model.embed.W_E[:-1]
    print("W_E", W_E.shape)

    # @ is matrix multiplication from PyTorch

    # W_neur = W_E @ W_V @ W_O @ W_in
    # The linear transformation from the input to the ReLU
    W_neur = (
        W_E
        @ hooked_model.blocks[0].attn.W_V
        @ hooked_model.blocks[0].attn.W_O
        @ hooked_model.blocks[0].mlp.W_in
    )
    print("W_neur", W_neur.shape)

    # W_logit = W_out @ W_U
    # The linear transformation from the ReLU to the logits
    W_logit = hooked_model.blocks[0].mlp.W_out @ hooked_model.unembed.W_U
    print("W_logit", W_logit.shape)

    original_loss = loss.mean_log_prob_loss(original_logits, labels).item()
    print("original_loss", original_loss)

    for param_name, param in cache.items():
        print(param_name, param.shape)

    # plot_attention_patterns(cache=cache, P=const.MOD)
    singular_value_decomposition(W_E)


def plot_attention_patterns(cache, P: int):
    # pattern_a = cache["pattern", 0, "attn"][:, :, -1, 0]
    # pattern_b = cache["pattern", 0, "attn"][:, :, -1, 1]
    neuron_acts = cache["post", 0, "mlp"][:, -1, :]
    # neuron_pre_acts = cache["pre", 0, "mlp"][:, -1, :]

    plot.imshow(
        cache["pattern", 0].mean(dim=0)[:, -1, :],
        title="Average Attention Pattern per Head",
        xaxis="Source",
        yaxis="Head",
        x=["a", "b", "="],
        y=[f"Head {i}" for i in range(4)],
        width=800,
        height=800,
    ).write_image("figures/average_attention_pattern_per_head.png")

    plot.imshow(
        cache["pattern", 0][5][:, -1, :],
        title="Attention Pattern for Head",  # TODO: what is this?
        xaxis="Source",
        yaxis="Head",
        x=["a", "b", "="],
        y=[f"Head {i}" for i in range(4)],
        width=800,
        height=800,
    ).write_image("figures/idk.png")

    plot.imshow(
        # Show the attention pattern for the first head (0) from the equals sign (-1) to the first number (0)
        cache["pattern", 0][:, 0, -1, 0].reshape(P, P),
        title="Attention for Head 0 from a -> =",
        xaxis="b",
        yaxis="a",
    ).write_image("figures/attention_for_head_0_from_a_to_equals.png")

    plot.imshow(
        einops.rearrange(
            cache["pattern", 0][:, :, -1, 0],
            "(a b) head -> head a b",
            a=const.MOD,
            b=const.MOD,
        ),
        title="Attention for each head from a -> =",
        xaxis="b",
        yaxis="a",
        facet_col=0,
    ).write_image("figures/attention_for_each_head_from_a_to_equals.png")

    plot.imshow(
        einops.rearrange(
            neuron_acts[:, :5],
            "(a b) neuron -> neuron a b",
            a=const.MOD,
            b=const.MOD,
        ),
        title="First 5 neuron activations",
        xaxis="b",
        yaxis="a",
        facet_col=0,
        facet_col_wrap=3,
    ).write_image("figures/first_5_neuron_activations.png")


def singular_value_decomposition(W_E):
    U, S, _ = torch.linalg.svd(W_E)
    _, S_rand, _ = torch.linalg.svd(torch.randn_like(W_E))

    fig = plot.multiline(
        [S, S_rand],
        title="Singular Values",
        labels=["W_E", "Random Matrix for Comparison"],
        xaxis="Input Vocabulary",
    )
    fig.write_image("figures/singular_values.png")
    plot.line_to_csv(fig, "figures/singular_values.csv")

    plot.imshow(U, title="Principal Components on the Input").write_image(
        "figures/principal_components_on_the_input.png"
    )

    # The singular values plot shows the first 8 elements of the embeddings are the most important
    # The next two are slightly important, the rest are basically unused.
    plot.imshow(
        U[:, :10],
        title="Principal Components on Most Important Inputs",
        aspect="auto",
        xaxis="Input Vocabulary",
    ).write_image("figures/principal_components_on_the_input_most_important.png")


if __name__ == "__main__":
    main()
