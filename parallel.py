"""Distributed execution and portable optimizer state for FSDP2."""

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as distributed


def validate_fsdp(device, world_size, dtype, compile_model):
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 14):
        raise ValueError("FSDP2 requires PyTorch 2.14 or newer")
    if world_size < 2:
        raise ValueError("FSDP2 requires at least two workers; launch with torchrun")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("FSDP2 supports CPU and CUDA devices")
    if dtype == "float16":
        raise ValueError("FSDP2 supports float32 or CUDA bfloat16; use --dtype=float32 on older GPUs")
    if compile_model:
        raise ValueError("FSDP2 currently requires --compile=False")


def shard_model(model, device, world_size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    # Every shard must start from the same parameter values.
    for tensor in list(model.parameters()) + list(model.buffers()):
        distributed.broadcast(tensor.detach(), src=0)
    mesh = init_device_mesh(device.type, (world_size,))
    for layer in model.transformer.h:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True)
    # The shared token embedding and output weight belong to the root group.
    fully_shard(model, mesh=mesh, reshard_after_forward=False)


@contextmanager
def gradient_sync(model, method, enabled, distributed_run):
    if method == "fsdp":
        model.set_requires_gradient_sync(enabled)
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True)
    else:
        with model.no_sync() if distributed_run and not enabled else nullcontext():
            yield


def checkpoint_states(model, optimizer, sharded, include_optimizer):
    if not sharded:
        return model.state_dict(), optimizer.state_dict() if include_optimizer else None
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    )

    # All workers participate; only rank zero retains full CPU tensors.
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    weights = get_model_state_dict(model, options=options)
    state = None
    if include_optimizer:
        if optimizer.state:
            state = get_optimizer_state_dict(model, optimizer, options=options)
        else:
            # The state-dict API initializes AdamW buffers with a dummy step.
            # Preserve an untouched optimizer when saving before the first update.
            state = named_optimizer_state(model, optimizer, optimizer.state_dict()) if distributed.get_rank() == 0 else {}
    return weights, state


def named_optimizer_state(model, optimizer, state):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    mapping = {}
    if len(state["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Checkpoint optimizer groups differ from the model")
    for current, saved in zip(optimizer.param_groups, state["param_groups"]):
        if len(current["params"]) != len(saved["params"]):
            raise ValueError("Checkpoint optimizer parameters differ from the model")
        mapping.update((identifier, names[id(parameter)])
                       for parameter, identifier in zip(current["params"], saved["params"]))
    return {
        "state": {mapping[key]: value for key, value in state["state"].items()},
        "param_groups": [group | {"params": [mapping[key] for key in group["params"]]}
                         for group in state["param_groups"]],
    }


def restore_optimizer(model, optimizer, checkpoint, sharded):
    state = checkpoint["optimizer"]
    # AdamW casts scalar step buffers according to the saved execution options.
    fused = not sharded and next(iter(model.parameters())).device.type == "cuda"
    state = state | {"param_groups": [group | {"fused": fused, "foreach": False if sharded else None}
                                      for group in state["param_groups"]]}
    named = checkpoint.get("optimizer_format", "indexed") == "named"
    if "wpe" in model.transformer:
        width = model.config.block_size
        if named:
            position = state["state"].get("transformer.wpe.weight", {})
        else:
            names = {id(parameter): name for name, parameter in model.named_parameters()}
            position = {}
            for current, saved in zip(optimizer.param_groups, state["param_groups"]):
                for parameter, identifier in zip(current["params"], saved["params"]):
                    if names[id(parameter)] == "transformer.wpe.weight":
                        position = state["state"].get(identifier, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in position:
                position[name] = position[name][:width].contiguous()
    if not sharded and not named:
        optimizer.load_state_dict(state)
        return
    if not named:
        state = named_optimizer_state(model, optimizer, state)
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

    if state["state"]:
        set_optimizer_state_dict(model, optimizer, state,
                                 options=StateDictOptions(full_state_dict=True))
    else:
        optimizer.load_state_dict(state)
