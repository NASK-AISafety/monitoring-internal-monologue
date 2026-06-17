import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from typing import Dict, List


def generate_hidden_states(
    model,
    dataloader,
    save_folder: Path,
    remove_files_if_exists: bool = False,
    calculate_mean_hidden_states: bool = False,
    layer_names_to_extract: Dict[str, bool] | None = None,
):
    pqwriters = None
    if calculate_mean_hidden_states:
        mean_pqwriters = None
    schema = pa.schema(
        [("labels", pa.float32()), ("hidden_state", pa.list_(pa.float32()))]
    )

    # Dictionary to store captured hidden states from hooks
    captured_hidden_states = {}
    hooks = []

    def get_activation(name, capture_output: bool = True):
        def hook(model, model_input, model_output):
            if capture_output:
                to_capture = model_output
            else:
                to_capture = model_input
            if isinstance(to_capture, tuple):
                captured_hidden_states[name] = to_capture[0].detach()
            else:
                captured_hidden_states[name] = to_capture.detach()

        return hook

    # Register hooks if layer names are provided
    # Create a list of layer names for integer indexing
    layer_names_list = (
        list(layer_names_to_extract.keys()) if layer_names_to_extract else None
    )

    if layer_names_to_extract:
        for name, module in model.named_modules():
            if name in layer_names_to_extract.keys():
                hooks.append(
                    module.register_forward_hook(
                        get_activation(
                            name, capture_output=layer_names_to_extract[name]
                        )
                    )
                )

    try:
        for batch in tqdm(dataloader, desc="Processing", leave=False):
            batch = batch.to(model.device)
            labels = batch.pop("labels", None)

            captured_hidden_states.clear()  # Clear previous batch captures

            with torch.no_grad():
                if layer_names_to_extract:
                    model.forward(
                        **batch, output_hidden_states=False
                    )  # Don't need standard hidden states if using hooks
                    # Collect hidden states from captured dictionary in the order of layer_names_to_extract
                    # Use a list to mimic the structure expected downstream (list of tensors)
                    # Note: Original code expects hidden_states to be indexable by layer_idx.
                    # We will construct a list where the index corresponds to the order in layer_names_to_extract.
                    hidden_states = [
                        captured_hidden_states[name]
                        for name in layer_names_to_extract.keys()
                        if name in captured_hidden_states
                    ]
                else:
                    hidden_states = model.forward(**batch, output_hidden_states=True)[
                        "hidden_states"
                    ]

            if pqwriters is None:
                if remove_files_if_exists:
                    for layer_idx in range(len(hidden_states)):
                        layer_name = (
                            layer_names_list[layer_idx]
                            if layer_names_list
                            else layer_idx
                        )
                        # Only saving token 0 (last token)
                        cur_file = (
                            save_folder
                            / f"layer_{layer_name}_token_0_hidden_states.parquet"
                        )
                        if cur_file.exists():
                            shutil.rmtree(
                                cur_file,
                                ignore_errors=True,
                            )
                        if calculate_mean_hidden_states:
                            cur_file = (
                                save_folder
                                / f"layer_{layer_name}_mean_hidden_states.parquet"
                            )
                            if cur_file.exists():
                                shutil.rmtree(
                                    cur_file,
                                    ignore_errors=True,
                                )
                pqwriters = {
                    (layer_idx, 0): pq.ParquetWriter(
                        save_folder
                        / f"layer_{layer_names_list[layer_idx] if layer_names_list else layer_idx}_token_0_hidden_states.parquet",
                        schema=schema,
                    )
                    for layer_idx in range(len(hidden_states))
                }
                if calculate_mean_hidden_states:
                    mean_pqwriters = {
                        layer_idx: pq.ParquetWriter(
                            save_folder
                            / f"layer_{layer_names_list[layer_idx] if layer_names_list else layer_idx}_mean_hidden_states.parquet",
                            schema=schema,
                        )
                        for layer_idx in range(len(hidden_states))
                    }
            for layer_idx, layer_hidden_states in enumerate(hidden_states):
                # Vectorized processing for the last token only

                # Prepare Labels
                labels_np = labels.detach().to(torch.float32).cpu().numpy().flatten()

                # Prepare Hidden States
                # We access the last token [-1]
                sliced_states = layer_hidden_states[:, -1, :]

                # Convert to numpy:
                sliced_states_np = (
                    sliced_states.detach().to(torch.float32).cpu().numpy()
                )

                # Create PyArrow Arrays
                array_labels = pa.array(labels_np, type=pa.float32())

                hidden_states_list = list(sliced_states_np)
                hidden_states_arrays = pa.array(
                    hidden_states_list, type=pa.list_(pa.float32())
                )

                table = pa.Table.from_arrays(
                    [array_labels, hidden_states_arrays], schema=schema
                )
                pqwriters[(layer_idx, 0)].write_table(table)

                if calculate_mean_hidden_states:
                    mean_layer_hidden_states = layer_hidden_states.clone()
                    attention_mask = batch["attention_mask"]
                    # Calculate mean hidden states only for valid tokens
                    valid_tokens = attention_mask.sum(dim=1)
                    mean_layer_hidden_states = (
                        mean_layer_hidden_states * attention_mask.unsqueeze(-1)
                    ).sum(dim=1) / valid_tokens.unsqueeze(-1)

                    # Vectorized
                    mean_states_np = (
                        mean_layer_hidden_states.detach()
                        .to(torch.float32)
                        .cpu()
                        .numpy()
                    )

                    mean_labels_np = (
                        labels.detach().to(torch.float32).cpu().numpy().flatten()
                    )

                    mean_array_labels = pa.array(mean_labels_np, type=pa.float32())

                    mean_hidden_states_list = list(mean_states_np)
                    mean_hidden_states_arrays = pa.array(
                        mean_hidden_states_list, type=pa.list_(pa.float32())
                    )

                    mean_table = pa.Table.from_arrays(
                        [mean_array_labels, mean_hidden_states_arrays], schema=schema
                    )
                    # Write the mean table to the corresponding layer's file
                    mean_pqwriters[layer_idx].write_table(mean_table)

    finally:
        # Ensure hooks are removed even if an error occurs
        for hook in hooks:
            hook.remove()

    for writer in pqwriters.values():
        writer.close()
    if calculate_mean_hidden_states:
        for writer in mean_pqwriters.values():
            writer.close()


def generate_responses_hidden_states(
    model,
    dataloader,
    save_folder: Path,
    remove_files_if_exists: bool = False,
    calculate_mean_hidden_states: bool = True,
    layer_names_to_extract: list[str] | None = None,
):
    pqwriters = None
    if calculate_mean_hidden_states:
        mean_pqwriters = None
    schema = pa.schema(
        [("labels", pa.float32()), ("hidden_state", pa.list_(pa.float32()))]
    )

    # Dictionary to store captured hidden states from hooks
    captured_hidden_states = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            # output can be a tuple or a tensor. Usually for layers it's a tuple where the first element is the hidden state.
            # However, for some layers (like normalization), it might be just a tensor.
            # We try to inspect the output.
            if isinstance(output, tuple):
                captured_hidden_states[name] = output[0].detach()
            else:
                captured_hidden_states[name] = output.detach()

        return hook

    # Register hooks if layer names are provided
    if layer_names_to_extract:
        for name, module in model.named_modules():
            if name in layer_names_to_extract:
                hooks.append(module.register_forward_hook(get_activation(name)))

        # Verification: check if all requested layers were found
        found_layers = set(
            name for name, _ in model.named_modules() if name in layer_names_to_extract
        )
        missing_layers = set(layer_names_to_extract) - found_layers
        if missing_layers:
            print(
                f"Warning: The following layers were not found in the model: {missing_layers}"
            )

    try:
        for batch in tqdm(dataloader, desc="Processing", leave=False):
            batch = batch.to(model.device)
            labels = batch.pop("labels", None)
            responses_lengths = batch.pop("response_length", None)

            captured_hidden_states.clear()  # Clear previous batch captures

            with torch.no_grad():
                if layer_names_to_extract:
                    model.forward(**batch, output_hidden_states=False)
                    hidden_states = [
                        captured_hidden_states[name]
                        for name in layer_names_to_extract
                        if name in captured_hidden_states
                    ]
                else:
                    hidden_states = model.forward(**batch, output_hidden_states=True)[
                        "hidden_states"
                    ]

            if pqwriters is None:
                if remove_files_if_exists:
                    for layer_idx in range(len(hidden_states)):
                        layer_name = (
                            layer_names_to_extract[layer_idx]
                            if layer_names_to_extract
                            else layer_idx
                        )
                        cur_file = (
                            save_folder / f"layer_{layer_name}_hidden_states.parquet"
                        )
                        if cur_file.exists():
                            shutil.rmtree(
                                cur_file,
                                ignore_errors=True,
                            )
                        if calculate_mean_hidden_states:
                            cur_file = (
                                save_folder
                                / f"layer_{layer_name}_mean_hidden_states.parquet"
                            )
                            if cur_file.exists():
                                shutil.rmtree(
                                    cur_file,
                                    ignore_errors=True,
                                )
                pqwriters = {
                    layer_idx: pq.ParquetWriter(
                        save_folder
                        / f"layer_{layer_names_to_extract[layer_idx] if layer_names_to_extract else layer_idx}_hidden_states.parquet",
                        schema=schema,
                    )
                    for layer_idx in range(len(hidden_states))
                }
                if calculate_mean_hidden_states:
                    mean_pqwriters = {
                        layer_idx: pq.ParquetWriter(
                            save_folder
                            / f"layer_{layer_names_to_extract[layer_idx] if layer_names_to_extract else layer_idx}_mean_hidden_states.parquet",
                            schema=schema,
                        )
                        for layer_idx in range(len(hidden_states))
                    }
            for layer_idx, layer_hidden_states in enumerate(hidden_states):
                # Vectorized - Last Token
                # layer_hidden_states: (batch, seq, hidden)

                # Prepare Labels
                labels_np = labels.detach().to(torch.float32).cpu().numpy().flatten()
                array_labels = pa.array(labels_np, type=pa.float32())

                # Last token extraction
                last_tokens = layer_hidden_states[:, -1, :]
                last_tokens_np = last_tokens.detach().to(torch.float32).cpu().numpy()

                hidden_states_list = list(last_tokens_np)
                hidden_states_arrays = pa.array(
                    hidden_states_list, type=pa.list_(pa.float32())
                )

                table = pa.Table.from_arrays(
                    [array_labels, hidden_states_arrays], schema=schema
                )
                pqwriters[layer_idx].write_table(table)

                if calculate_mean_hidden_states:
                    # Vectorized - Mean over response length
                    batch_size, seq_len, hidden_dim = layer_hidden_states.shape

                    # Create mask for valid response tokens
                    # We want the suffix of length responses_lengths[i]
                    # Indices: 0, 1, ..., seq_len-1
                    indices = torch.arange(
                        seq_len, device=layer_hidden_states.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    start_indices = (seq_len - responses_lengths).unsqueeze(
                        1
                    )  # [batch, 1]

                    # Mask is True where we want to include the token
                    mask = indices >= start_indices  # [batch, seq]

                    # Apply mask (broadcasting over hidden dim)
                    masked_states = layer_hidden_states * mask.unsqueeze(
                        -1
                    )  # [batch, seq, hidden]

                    # Sum and divide
                    sum_states = masked_states.sum(dim=1)  # [batch, hidden]
                    mean_states = sum_states / responses_lengths.unsqueeze(
                        1
                    )  # [batch, hidden]

                    mean_states_np = (
                        mean_states.detach().to(torch.float32).cpu().numpy()
                    )

                    # Reuse labels array as it is the same
                    mean_array_labels = array_labels

                    mean_hidden_states_list = list(mean_states_np)
                    mean_hidden_states_arrays = pa.array(
                        mean_hidden_states_list, type=pa.list_(pa.float32())
                    )

                    mean_table = pa.Table.from_arrays(
                        [mean_array_labels, mean_hidden_states_arrays], schema=schema
                    )
                    # Write the mean table to the corresponding layer's file
                    mean_pqwriters[layer_idx].write_table(mean_table)

    finally:
        # Ensure hooks are removed even if an error occurs
        for hook in hooks:
            hook.remove()

    for writer in pqwriters.values():
        writer.close()
    if calculate_mean_hidden_states:
        for writer in mean_pqwriters.values():
            writer.close()


def generate_vlm_hidden_states(
    model,
    dataloader,
    save_folder: Path,
    remove_files_if_exists: bool = False,
    calculate_mean_hidden_states: bool = False,
    layer_names_to_extract: Dict[str, bool] | None = None,
):
    """Generate hidden states from a Vision Language Model.

    Similar to generate_hidden_states but handles VLM-specific inputs
    (pixel_values, image_grid_thw, etc.) alongside text inputs.
    Uses attention_mask to find the true last token position.

    Args:
        model: The VLM model
        dataloader: DataLoader yielding dicts with input_ids, attention_mask,
                    pixel_values, labels, and optionally image_grid_thw
        save_folder: Directory to save parquet files
        remove_files_if_exists: Whether to remove existing parquet files
        calculate_mean_hidden_states: Whether to also calculate mean hidden states
        layer_names_to_extract: Optional dict of layer names to extract via hooks
    """
    pqwriters = None
    if calculate_mean_hidden_states:
        mean_pqwriters = None
    schema = pa.schema(
        [("labels", pa.float32()), ("hidden_state", pa.list_(pa.float32()))]
    )

    # Dictionary to store captured hidden states from hooks
    captured_hidden_states = {}
    hooks = []

    def get_activation(name, capture_output: bool = True):
        def hook(model, model_input, model_output):
            if capture_output:
                to_capture = model_output
            else:
                to_capture = model_input
            if isinstance(to_capture, tuple):
                captured_hidden_states[name] = to_capture[0].detach()
            else:
                captured_hidden_states[name] = to_capture.detach()

        return hook

    # Register hooks if layer names are provided
    layer_names_list = (
        list(layer_names_to_extract.keys()) if layer_names_to_extract else None
    )

    if layer_names_to_extract:
        for name, module in model.named_modules():
            if name in layer_names_to_extract.keys():
                hooks.append(
                    module.register_forward_hook(
                        get_activation(
                            name, capture_output=layer_names_to_extract[name]
                        )
                    )
                )

    try:
        for batch in tqdm(dataloader, desc="Processing VLM", leave=False):
            labels = batch.pop("labels", None)

            # Move tensors to model device
            model_inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(model.device)
                else:
                    model_inputs[k] = v

            captured_hidden_states.clear()

            with torch.no_grad():
                if layer_names_to_extract:
                    model.forward(**model_inputs, output_hidden_states=False)
                    hidden_states = [
                        captured_hidden_states[name]
                        for name in layer_names_to_extract.keys()
                        if name in captured_hidden_states
                    ]
                else:
                    hidden_states = model.forward(
                        **model_inputs, output_hidden_states=True
                    )["hidden_states"]

            # Find the true last token position using attention_mask
            attention_mask = model_inputs.get("attention_mask", None)

            if pqwriters is None:
                if remove_files_if_exists:
                    for layer_idx in range(len(hidden_states)):
                        layer_name = (
                            layer_names_list[layer_idx]
                            if layer_names_list
                            else layer_idx
                        )
                        cur_file = (
                            save_folder
                            / f"layer_{layer_name}_token_0_hidden_states.parquet"
                        )
                        if cur_file.exists():
                            shutil.rmtree(cur_file, ignore_errors=True)
                        if calculate_mean_hidden_states:
                            cur_file = (
                                save_folder
                                / f"layer_{layer_name}_mean_hidden_states.parquet"
                            )
                            if cur_file.exists():
                                shutil.rmtree(cur_file, ignore_errors=True)
                pqwriters = {
                    (layer_idx, 0): pq.ParquetWriter(
                        save_folder
                        / f"layer_{layer_names_list[layer_idx] if layer_names_list else layer_idx}_token_0_hidden_states.parquet",
                        schema=schema,
                    )
                    for layer_idx in range(len(hidden_states))
                }
                if calculate_mean_hidden_states:
                    mean_pqwriters = {
                        layer_idx: pq.ParquetWriter(
                            save_folder
                            / f"layer_{layer_names_list[layer_idx] if layer_names_list else layer_idx}_mean_hidden_states.parquet",
                            schema=schema,
                        )
                        for layer_idx in range(len(hidden_states))
                    }

            for layer_idx, layer_hidden_states in enumerate(hidden_states):
                # Prepare Labels
                labels_np = labels.detach().to(torch.float32).cpu().numpy().flatten()

                # Extract last token hidden state using attention_mask
                if attention_mask is not None:
                    # Find position of last non-padding token for each sample
                    # With left-padding, last valid token is at the rightmost
                    # position where attention_mask == 1
                    last_token_indices = attention_mask.sum(dim=1) - 1
                    batch_size = layer_hidden_states.shape[0]
                    sliced_states = layer_hidden_states[
                        torch.arange(batch_size, device=layer_hidden_states.device),
                        last_token_indices.to(layer_hidden_states.device),
                        :,
                    ]
                else:
                    # Fallback: use the actual last token
                    sliced_states = layer_hidden_states[:, -1, :]

                sliced_states_np = (
                    sliced_states.detach().to(torch.float32).cpu().numpy()
                )

                array_labels = pa.array(labels_np, type=pa.float32())
                hidden_states_list = list(sliced_states_np)
                hidden_states_arrays = pa.array(
                    hidden_states_list, type=pa.list_(pa.float32())
                )

                table = pa.Table.from_arrays(
                    [array_labels, hidden_states_arrays], schema=schema
                )
                pqwriters[(layer_idx, 0)].write_table(table)

                if calculate_mean_hidden_states:
                    if attention_mask is not None:
                        mean_layer_hidden_states = layer_hidden_states.clone()
                        valid_tokens = attention_mask.sum(dim=1).to(
                            layer_hidden_states.device
                        )
                        mean_layer_hidden_states = (
                            mean_layer_hidden_states
                            * attention_mask.unsqueeze(-1).to(
                                layer_hidden_states.device
                            )
                        ).sum(dim=1) / valid_tokens.unsqueeze(-1)
                    else:
                        mean_layer_hidden_states = layer_hidden_states.mean(dim=1)

                    mean_states_np = (
                        mean_layer_hidden_states.detach()
                        .to(torch.float32)
                        .cpu()
                        .numpy()
                    )

                    mean_array_labels = pa.array(labels_np, type=pa.float32())
                    mean_hidden_states_list = list(mean_states_np)
                    mean_hidden_states_arrays = pa.array(
                        mean_hidden_states_list, type=pa.list_(pa.float32())
                    )

                    mean_table = pa.Table.from_arrays(
                        [mean_array_labels, mean_hidden_states_arrays], schema=schema
                    )
                    mean_pqwriters[layer_idx].write_table(mean_table)

    finally:
        for hook in hooks:
            hook.remove()

    for writer in pqwriters.values():
        writer.close()
    if calculate_mean_hidden_states:
        for writer in mean_pqwriters.values():
            writer.close()
