import time 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
import torch.nn as nn

from pdb import set_trace as st 
from .quant import *
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

    
            
        
            


def lexsort(keys, dim=-1):
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx


def maximize_total_value(matrix):
    # linear_sum_assignment
    row_indices, col_indices = linear_sum_assignment(matrix, maximize=True) 
    return col_indices


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

#further modifed check sparsity : moved meta tensors to device
#modified for gptneo
def check_sparsity(args, model):
    total_zeros = 0
    total_elements = 0

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        layers = model.encoder.block
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'block'):
        layers = model.model.encoder.block
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'h'):
        print("[INFO] Detected decoder-only model (e.g. GPT-Neo)")
        layers = model.model.transformer.h
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        print("[INFO] Detected decoder-only model (e.g. GPT-Neo)")
        layers = model.transformer.h
    else:
        raise AttributeError("Could not locate layers for sparsity check in this model.")

    for layer in layers:
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            total_zeros += (W == 0).sum().item()
            total_elements += W.numel()

    sparsity_ratio = total_zeros / total_elements
    return sparsity_ratio



def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    input_ids = []
    attention_masks = []

    for batch in dataloader:
        ids, mask = batch
        input_ids.append(ids)
        attention_masks.append(mask)

        if len(input_ids) * ids.size(0) >= args.nsamples:
            break

    input_ids = torch.cat(input_ids, dim=0)[:args.nsamples].to(device)
    attention_mask = torch.cat(attention_masks, dim=0)[:args.nsamples].to(device)

    model.config.use_cache = use_cache
    return input_ids, attention_mask

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # Ensure layers are always assigned a value
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    elif "bert" in args.model:  # Example: adding BERT handling
        layers = model.encoder.layer
    elif "t5" in args.model:  # Example: adding T5 handling
        layers = model.encoder.block
    else:
        raise ValueError(f"Unsupported model type: {args.model}. Please provide a model with 'llama', 'opt', 'bert', or 't5' in its name.")
    
    per_outneuron = False

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            if args.prune_method == "magnitude":
                W_metric = torch.abs(W)
            elif args.prune_method == "ri":
                W_metric = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)  # initialize a mask to be all False
            
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                if per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0] * W.shape[1] * args.sparsity_ratio)].cpu()
                    W_mask = (W_metric <= thresh)

            subset[name].weight.data[W_mask] = 0





# def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
#     if "llama" in args.model:
#         layers = model.model.layers
#     elif "opt" in args.model:
#         layers = model.model.decoder.layers
        
#     per_outneuron = False

#     for i in range(len(layers)):
#         layer = layers[i]
#         subset = find_layers(layer)

#         for name in subset:
#             print(f"pruning layer {i} name {name}")
#             W = subset[name].weight.data.clone()
#             if args.prune_method == "magnitude":
#                 W_metric = torch.abs(W)
#             elif args.prune_method == "ri":
#                 W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
#             W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
#             if prune_n != 0:
#                 for ii in range(W_metric.shape[1]):
#                     if ii % prune_m == 0:
#                         tmp = W_metric[:,ii:(ii+prune_m)].float()
#                         W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
#             else:
#                 if per_outneuron:
#                     sort_res = torch.sort(W_metric, dim=-1, stable=True)
#                     # unstructured pruning
#                     indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
#                     W_mask.scatter_(1, indices, True)
#                 else:
#                     thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
#                     W_mask = (W_metric<=thresh)

#             subset[name].weight.data[W_mask] = 0

def get_encoder_layers(model):
    # Encoder-decoder models
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        print("[INFO] Using model.encoder.block")
        return model.encoder.block
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
        if hasattr(encoder, 'block'):
            print("[INFO] Using model.model.encoder.block")
            return encoder.block
        elif hasattr(encoder, 'layers'):
            print("[INFO] Using model.model.encoder.layers")
            return encoder.layers

    # Decoder-only models (e.g., GPT-Neo, GPT2, OPT)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        print("[INFO] Using model.transformer.h (decoder-only model)")
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        print("[INFO] Using model.model.decoder.layers")
        return model.model.decoder.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print("[INFO] Using model.model.layers")
        return model.model.layers

    # Fallback
    print("[DEBUG] model attributes:", dir(model))
    if hasattr(model, "model"):
        print("[DEBUG] model.model attributes:", dir(model.model))
    raise AttributeError("Could not locate encoder or decoder layers in this model.")


    
def prune_ria(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask = prepare_calibration_input(args, model, dataloader, device)
        else:
            input_ids, attention_mask = prepare_calibration_input(args, model, dataloader, device)

    layers = get_encoder_layers(model)
    is_encoder_decoder = hasattr(model, 'encoder') or (hasattr(model, 'model') and hasattr(model.model, 'encoder'))

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
            if args.gptq:
                wrapped_layers[name].quantizer = Quantizer()
                wrapped_layers[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Warmup pass to collect activations
        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, use_cache=False)[0]
                elif "opt" in args.model:
                    _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, use_cache=False)[0]
                elif is_encoder_decoder:
                    _ = model.model.encoder(
                        input_ids=input_ids[j].unsqueeze(0),
                        attention_mask=attention_mask[j].unsqueeze(0),
                        return_dict=True
                    )
                else:
                    _ = model(
                        input_ids=input_ids[j].unsqueeze(0),
                        attention_mask=attention_mask[j].unsqueeze(0),
                        output_hidden_states=True,
                        return_dict=True
                    )

        for h in handles:
            h.remove()

        # Compute pruning metric
        for name in subset:
            print(f"[Init] Layer {name}: W.shape={subset[name].weight.shape}, scaler_row.shape={wrapped_layers[name].scaler_row.shape}")
            W = subset[name].weight.data.clone()
            if args.prune_method == "wanda":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            elif args.prune_method == "ria":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) +
                            torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * \
                           (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))) ** args.a

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
            W_mask = (W_metric <= thresh)

            if args.reconstruction:
                wrapped_layers[name].fasterprune(args.sparsity_ratio, mask=W_mask)
            else:
                subset[name].weight.data[W_mask] = 0

            wrapped_layers[name].free()

        # Optional second pass to recompute activations (used in reconstruction)
        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, use_cache=False)[0]
                elif "opt" in args.model:
                    _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, use_cache=False)[0]
                elif is_encoder_decoder:
                    _ = model.model.encoder(
                        input_ids=input_ids[j].unsqueeze(0),
                        attention_mask=attention_mask[j].unsqueeze(0),
                        return_dict=True
                    )
                else:
                    outputs = model(
                        input_ids=input_ids[j].unsqueeze(0),
                        attention_mask=attention_mask[j].unsqueeze(0),
                        output_hidden_states=True,
                        return_dict=True
                    )
                    if j == 0:
                        hidden_size = outputs.hidden_states[-1].shape[-1]
                        outs = torch.zeros(args.nsamples, input_ids.shape[1], hidden_size, device=device)
                    outs[j] = outputs.hidden_states[-1].squeeze(0)

        if "llama" in args.model or "opt" in args.model:
            inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                print(f"layer {i} device {dev}")
                inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "llama" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            if "norm" in args.model:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, norm=True)
            else:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            if "llama" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_ria_scheduled(args, model, tokenizer, device, current_step, prune_n=0, prune_m=0):
    # === Zhu & Gupta cubic schedule ===
    si = 0.0
    sf = args.sparsity_ratio
    t0 = 2000
    n = 10
    delta_t = 2000
    t = current_step
    print(f"Current Step , t = {t}")

    if (t - t0) % delta_t != 0 or t < t0:
        return

    progress = min(1.0, (t - t0) / (n * delta_t))
    st = sf + (si - sf) * (1 - progress) ** 3
    print(f"[Scheduled RIA Pruning] Step {t}: Sparsity target = {st:.4f}")

    # === Prepare calibration data ===
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed,
                                 seqlen=args.seqlen, tokenizer=tokenizer)
    input_ids, attention_mask = prepare_calibration_input(args, model, dataloader, device)

    layers = get_encoder_layers(model)

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        wrapped_layers = {name: WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
                          for name in subset}

        # === Hook to collect activations ===
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(
                lambda _, inp, out, name=name: wrapped_layers[name].add_batch(inp[0].data, out.data)
            ))

        # === Run samples through model ===
        for j in range(args.nsamples):
            with torch.no_grad():
                model.encoder(input_ids=input_ids[j].unsqueeze(0),
                      attention_mask=attention_mask[j].unsqueeze(0),
                      return_dict=True)

        for h in handles:
            h.remove()

        # === Compute RIA metrics and prune ===
        for name in subset:
            W = subset[name].weight.data.clone()
            absW = torch.abs(W)
            col_sum = absW.sum(dim=0)
            row_sum = absW.sum(dim=1).reshape(-1, 1)
            act = wrapped_layers[name].scaler_row.reshape((1, -1)).clamp(min=1e-6)

            if act.numel() != W.shape[1]:
                print(f"[SKIP] Shape mismatch in {name}: act={act.shape}, W={W.shape}")
                continue

            W_metric = (absW / col_sum + absW / row_sum) * (torch.sqrt(act)) ** args.a
            threshold = torch.sort(W_metric.flatten())[0][int(W_metric.numel() * st)]
            W_mask = W_metric <= threshold

            if args.reconstruction:
                wrapped_layers[name].fasterprune(st, mask=W_mask)
            else:
                subset[name].weight.data[W_mask] = 0

            wrapped_layers[name].free()

    torch.cuda.empty_cache()


def prune_magnitude_scheduled(args, model, tokenizer, device, current_step, prune_n=0, prune_m=0):
    # === Zhu & Gupta cubic schedule ===
    si = 0.0
    sf = args.sparsity_ratio
    t0 = 2000
    n = 10
    delta_t = 2000
    t = current_step
    print(f"Current Step , t = {t}")

    if (t - t0) % delta_t != 0 or t < t0:
        return

    progress = min(1.0, (t - t0) / (n * delta_t))
    st = sf + (si - sf) * (1 - progress) ** 3
    print(f"[Scheduled Magnitude Pruning] Step {t}: Sparsity target = {st:.4f}")

    # === Prepare calibration data ===
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed,
                                 seqlen=args.seqlen, tokenizer=tokenizer)
    input_ids, attention_mask = prepare_calibration_input(args, model, dataloader, device)

    # Ensure layers are always assigned a value
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    elif "bert" in args.model:  # Example: adding BERT handling
        layers = model.encoder.layer
    elif "t5" in args.model:  # Example: adding T5 handling
        layers = model.encoder.block
    else:
        raise ValueError(f"Unsupported model type: {args.model}. Please provide a model with 'llama', 'opt', 'bert', or 't5' in its name.")

    # For each layer, prune using magnitude-based method
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # === Magnitude pruning process ===
        for name in subset:
            print(f"Pruning layer {i}, name {name}")
            W = subset[name].weight.data.clone()
            W_metric = torch.abs(W)  # Magnitude-based pruning uses the absolute value of weights

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)  # initialize a mask to be all False

            # If prune_n and prune_m are set, prune columns based on magnitude
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # Apply pruning based on the sparsity ratio
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0] * W.shape[1] * st)].cpu()
                W_mask = (W_metric <= thresh)

            # Apply the pruning mask
            subset[name].weight.data[W_mask] = 0

    torch.cuda.empty_cache()







