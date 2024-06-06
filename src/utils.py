import torch
import transformer_lens

from src import model


def load_model(
    path: str, mod: int, device: str = "cpu"
) -> transformer_lens.HookedTransformer:
    saved: dict = torch.load(path)
    hooked_model: transformer_lens.HookedTransformer = model.get_hooked_transformer(
        base=mod, device=device
    )

    hooked_model.load_state_dict(saved["model"])
    return hooked_model
