import einops
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import fft
import torch
import transformer_lens

from src import const
from src import plot
from src import task
from src import utils


def main(model_path: str, base: int, device: str) -> None:
    hooked_model: transformer_lens.HookedTransformer = utils.load_model(
        path=model_path,
        mod=base,
        device=device,
    )

    # Make a dataset and labels for the model
    dataset, _ = task.make_dataset_and_labels(
        p=const.MOD,
        device=device,
    )

    # Get a prediction for every possible input and cache the activations
    _, cache = hooked_model.run_with_cache(dataset)

    W_E = hooked_model.embed.W_E[:-1]
    U, _, _ = torch.svd(W_E)
    # plot.multiline(U[:, :8].T, xaxis="Input", title="First 8 Singular Values").show()

    # There is a periodicity in the singular values, I explore this using Fourier Analysis
    # Fourier Analysis decomposes a signal into its constituent frequencies so we can add them
    # "across" the singular values to analyse them all at once.
    # Doing so we see 4 key frequencies.
    principal_components = U[:, :8].sum(dim=1).detach().numpy()
    yf = fft.fft(principal_components)
    fig = px.line(
        x=np.arange(base // 2),
        y=2.0 / base * np.abs(yf[0 : base // 2]),
        template=plot.TEMPLATE,
        labels={"x": "Frequency", "y": "Amplitude"},
    )
    fig.write_image("figures/key_frequencies.png")
    plot.line_to_csv(fig, "figures/key_frequencies.csv")

    # Digging a bit deeper, plotting each singular value's Fourier Transform,
    # we see that the first 8 singular values are composed of differing amounts of the same 4 key frequencies.
    fig = go.Figure()
    for i, component in enumerate(U[:, :8].T):
        component = component.detach().numpy()
        yf = fft.fft(component)
        fig.add_trace(
            go.Scatter(
                x=np.arange(base // 2),
                y=2.0 / base * np.abs(yf[0 : base // 2]),
                mode="lines",
                name=f"Singular value {i}",
            )
        )
    fig.update_layout(
        template=plot.TEMPLATE,
    )
    fig.write_image("figures/key_frequencies_by_singular_value.png")
    plot.line_to_csv(fig, "figures/key_frequencies_by_singular_value.csv")

    # The Fourier Basis is a set of orthogonal basis functions that can be used to represent any periodic function.
    fourier_basis, fourier_basis_names = make_fourier_basis(base=base, device=device)
    plot.imshow(
        fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names
    ).write_image("figures/fourier_basis.png")

    # plot.line(
    #     fourier_basis[:8],
    #     xaxis="Input",
    #     # line_labels=fourier_basis_names[:8],
    #     title="First 8 Fourier Components",
    # ).show()
    # plot.line(
    #     fourier_basis[25:29],
    #     xaxis="Input",
    #     # line_labels=fourier_basis_names[25:29],
    #     title="Middle Fourier Components",
    # ).show()

    # Multiplying the Fourier Basis by the singular values we can see how the singular values are composed sin and cos functions.
    # The rows correspond to the key frequencies from the previous plots.
    # The additional thing we learn from this is that it's using both the sin and cos of the 4 key frequencies.
    fourier_embed = fourier_basis @ W_E
    plot.imshow(
        fourier_embed,
        yaxis="Fourier Component",
        xaxis="Residual Stream",
        y=fourier_basis_names,
        title="Embedding in Fourier Basis",
    ).write_image("figures/embedding_in_fourier_basis.png")

    ## 2D Fourier
    # Show that for a given neuron, it's only activated by sin and cos terms of a given freq as well as constant terms.
    neuron_activations = cache["post", 0, "mlp"][:, -1, :]
    plot.imshow(
        fourier_basis @ neuron_activations[:, 0].reshape(base, base) @ fourier_basis.T,
        title="2D Fourier Transform of Neuron 0",
        xaxis="b",
        yaxis="a",
        x=fourier_basis_names,
        y=fourier_basis_names,
    ).write_image("figures/2d_fourier_transform_of_neuron_0.png")

    ## Neuron Clusters
    fourier_neuron_activations = (
        fourier_basis
        @ einops.rearrange(
            neuron_activations, "(a b) neuron -> neuron a b", a=base, b=base
        )
        @ fourier_basis.T
    )
    # Center these by removing the mean - doesn't matter!
    fourier_neuron_activations[:, 0, 0] = 0.0
    print("fourier_neuron_activations", fourier_neuron_activations.shape)

    neuron_freq_norm = get_neuron_frequency_contributions(
        base, device, hooked_model, fourier_neuron_activations
    )
    # We see again the key frequencies as lines across the plot.
    # Also most neurons appear to rely only on one neuron.
    plot.imshow(
        neuron_freq_norm,
        xaxis="Neuron",
        yaxis="Freq",
        y=torch.arange(1, base // 2 + 1),
        title="Neuron Frac Explained by Freq",
        aspect="auto",
    ).show()

    # Doesn't look like the walkthrough, much shallower.
    plot.line(
        neuron_freq_norm.max(dim=0).values.sort().values,
        xaxis="Neuron",
        title="Max Neuron Frac Explained over Freqs",
    ).show()


def get_neuron_frequency_contributions(
    base, device, hooked_model, fourier_neuron_activations
):
    """Determine which key frequencies are most important for each neuron in the model."""

    # Initialise the tensor to store the normalized frequency contributions
    neuron_freq_norm = torch.zeros(base // 2, hooked_model.cfg.d_mlp).to(
        device
    )  # 56 (freq) x 512 (neurons in MLP)

    # Loop through the frequencies
    for freq in range(0, base // 2):
        # Generate the indices for the constant, sin, and cos terms in the fourier basis for this frequency
        const_sin_cos_indices = [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]

        # For each combination of sin and cos terms at this frequency, sum the squared activations
        for x in const_sin_cos_indices:
            for y in const_sin_cos_indices:
                neuron_freq_norm[freq] += fourier_neuron_activations[:, x, y] ** 2

    # Normalize the frequency contributions by the sum of the squared activations for each neuron
    neuron_freq_norm = (
        neuron_freq_norm / fourier_neuron_activations.pow(2).sum(dim=[-1, -2])[None, :]
    )

    return neuron_freq_norm


# W_neur = (
#     W_E
#     @ hooked_model.blocks[0].attn.W_V
#     @ hooked_model.blocks[0].attn.W_O
#     @ hooked_model.blocks[0].mlp.W_in
# )
# W_logit = hooked_model.blocks[0].mlp.W_out @ hooked_model.unembed.W_U


def make_fourier_basis(base: int, device: str) -> torch.Tensor:
    fourier_basis = []
    fourier_basis_names = []
    fourier_basis.append(torch.ones(base))
    fourier_basis_names.append("Constant")
    for freq in range(1, base // 2 + 1):
        fourier_basis.append(torch.sin(torch.arange(base) * 2 * torch.pi * freq / base))
        fourier_basis_names.append(f"Sin {freq}")
        fourier_basis.append(torch.cos(torch.arange(base) * 2 * torch.pi * freq / base))
        fourier_basis_names.append(f"Cos {freq}")
    fourier_basis = torch.stack(fourier_basis, dim=0).to(device)
    fourier_basis = fourier_basis / fourier_basis.norm(dim=-1, keepdim=True)
    return fourier_basis, fourier_basis_names


if __name__ == "__main__":
    main(
        model_path=const.MODEL_SAVE_PATH,
        base=const.MOD,
        device="cpu",
    )
