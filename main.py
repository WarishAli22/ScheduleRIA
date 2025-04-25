import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from lib.eval import eval_ppl, eval_zero_shot
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

print('cuda', torch.version.cuda)
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

# for T5
# def get_llm(model_name, cache_dir="llm_weights", seqlen=2048):
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         model_name, 
#         torch_dtype=torch.float16, 
#         cache_dir=cache_dir, 
#         low_cpu_mem_usage=True, 
#         device_map="auto"
#     )
#     model.seqlen = seqlen
#     return model

def get_llm(model_name, cache_dir="llm_weights", seqlen=2048):
    if "t5" in model_name or "mbart" in model_name or "bart" in model_name or "marian" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    
    model.seqlen = seqlen
    return model

def sanitize_model_name(model_name):
    return model_name.replace(":", "_").replace("/", "_")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--calib_dataset', type=str, default="c4")
    parser.add_argument('--eval_dataset', type=str, default="wikitext2")
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--sparsity_ratio', type=float, default=0.5)
    parser.add_argument("--sparsity_type", type=str, default="unstructured")
    parser.add_argument("--prune_method", type=str, choices=["svd_finetuned", "magnitude", "ri", "wanda", "svd_ri", "svd", "sparsegpt", "ria", "scheduled", "mag_sched"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--save_model', type=str, default=None)
    parser.add_argument('--semi_sparse_acc', action="store_true")
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--reconstruction", action="store_true")
    parser.add_argument("--reallocation", action="store_true")
    parser.add_argument("--lsa", action="store_true")
    parser.add_argument("--importance_score", type=str, default="sum")
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--per_outneuron", action="store_true")
    parser.add_argument("--test_bs", type=int, default=1)
    parser.add_argument("--use_cusparselt", action="store_true")
    parser.add_argument("--layer_wise", action="store_true")
    parser.add_argument("--svd_threshold", type=float, default=1e-3)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    if args.use_cusparselt:
        SparseSemiStructuredTensor._FORCE_CUTLASS = False

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    sanitized_model_name = sanitize_model_name(model_name)
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, args.seqlen)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False) if "opt" in args.model else AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]

    print("use device ", device)
    print(model)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        from lib.prune import prune_magnitude, prune_sparsegpt, prune_ria, check_sparsity, prune_ria_scheduled, prune_magnitude_scheduled

        if args.prune_method == "scheduled":
            total_steps = 10000
            for current_step in range(total_steps):
                args.current_step = current_step
                prune_ria_scheduled(args, model, tokenizer, device, current_step)
        elif args.prune_method == "mag_sched":
            total_steps = 10000
            for current_step in range(total_steps):
                args.current_step = current_step
                prune_magnitude_scheduled(args, model, tokenizer, device, current_step)
        else:
            if args.prune_method == "wanda":
                prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "magnitude":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "ria":
                prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method in ["ri", "svd", "svd_ri", "svd_finetuned"]:
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        print("*"*30)
        sparsity_ratio = check_sparsity(args, model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)

    ppl_test = eval_ppl(model, tokenizer, args.eval_dataset, args.test_bs, device)
    print(f"wikitext perplexity {ppl_test}")

    if args.save:
        dirname = f"results/{sanitized_model_name}"
        os.makedirs(dirname, exist_ok=True)
        filename = f"log_{args.prune_method}_layer.txt" if args.layer_wise else f"log_{args.prune_method}.txt"
        with open(os.path.join(dirname, filename), "a") as f:
            print("method\tactual_sparsity\tsparsity_pattern\treallocation\timportance_score\tlsa\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.sparsity_type}\t{args.reallocation}\t{args.importance_score}\t{args.lsa}\t{ppl_test:.4f}", file=f, flush=True)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if args.eval_zero_shot:
        accelerate = True
        task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "mnli"]
        num_shot = 0
        results = eval_zero_shot(args.save_model if args.save_model else args.model, task_list, num_shot, accelerate)

if __name__ == '__main__':
    main()
