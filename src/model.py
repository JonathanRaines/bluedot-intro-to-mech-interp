"""
This module contains a function to initialize and return a HookedTransformerModel instance.

The HookedTransformerModel is initialized with the specified configuration parameters.
"""

from typing import Final

from transformer_lens import HookedTransformerConfig, HookedTransformer


def get_hooked_transformer(base: int, device: str):
    """
    Returns a HookedTransformerModel instance with the specified configuration.

    Args:
        base (int): The modular base and number tokens in the vocabulary.
        device (str): The device to use for the model. "cpu" or "cuda".

    Returns:
        HookedTransformerModel: The initialized HookedTransformerModel instance.
    """
    config: Final[HookedTransformerConfig] = HookedTransformerConfig(
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_head=32,
        d_mlp=512,
        act_fn="relu",
        normalization_type=None,
        d_vocab=base + 1,  # +1 for the equals sign
        d_vocab_out=base,
        n_ctx=3,
        init_weights=True,
        device=device,
        seed=999,
    )

    return HookedTransformer(config)
