# Import necessary modules
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
# def eval_ppl(model, tokenizer, dataset, bs, device=torch.device("cuda:0")):
#     # Set dataset
#     # dataset = "wikitext2"
#     # dataset = "wikitext2"
#     # Print status
#     print(f"evaluating on {dataset}")

#     # Get the test loader
#     _, testloader = get_loaders(
#         dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
#     )

#     # Evaluate ppl in no grad context to avoid updating the model
#     with torch.no_grad():
#         ppl_test = eval_ppl_wikitext(model, testloader, bs, device)
#     return ppl_test 

#modified
def eval_ppl(model, tokenizer, dataset, bs, device=torch.device("cuda:0")):
    # Set dataset
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, tokenizer, testloader, bs, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    
    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        
        lm_logits = model(inputs).logits
        
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
    
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
# def eval_ppl_wikitext(model, testenc, bs=8, device=None):
#     # Get input IDs
#     testenc = testenc.input_ids

#     # Calculate number of samples
#     nsamples = testenc.numel() // model.seqlen

#     # List to store negative log likelihoods
#     nlls = []
#     print(f"nsamples {nsamples}")

#     # Loop through each batch
#     for i in range(0,nsamples,bs):
#         if i % 50 == 0:
#             print(f"sample {i}")

#         # Calculate end index
#         j = min(i+bs, nsamples)

#         # Prepare inputs and move to device
#         inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
#         inputs = inputs.reshape(j-i, model.seqlen)

#         # Forward pass through the model
#         # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#         #     with record_function("model_inference"):
#         #         lm_logits = model(inputs).logits
#         # # exit()
#         # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
#         lm_logits = model(inputs).logits
        

#         # Shift logits and labels for next token prediction
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_labels = inputs[:, 1:]

#         # Compute loss
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

#         # Calculate negative log likelihood
#         neg_log_likelihood = loss.float() * model.seqlen * (j-i)

#         # Append to list of negative log likelihoods
#         nlls.append(neg_log_likelihood)
#     # Compute perplexity
#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

#     return ppl.item()

#modified
# def eval_ppl_wikitext(model, testloader, bs, device):
#     model.eval()
#     test_loss = 0
#     for i, batch in enumerate(testloader):
#         # inputs = batch[0].to(device)
#         inputs = batch[0].to(device) #modified
#         with torch.no_grad():
#             # For T5 models, provide decoder_input_ids
#             if "t5" in model.config.model_type:
#                 # Use the same input as decoder_input_ids for simplicity
#                 decoder_input_ids = inputs
#                 outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids)
#             else:
#                 # For other models (e.g., GPT, OPT), call the model directly
#                 outputs = model(inputs)
            
#             # Calculate the loss
#             lm_logits = outputs.logits
#             shift_logits = lm_logits[:, :-1, :].contiguous()
#             shift_labels = inputs[:, 1:].contiguous()
#             loss_fct = torch.nn.CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             test_loss += loss.item()

#         print(f"batch {i} loss {loss.item()}")

#     # Calculate perplexity
#     ppl = torch.exp(torch.tensor(test_loss / len(testloader)))
#     return ppl

#modified-2
# def eval_ppl_wikitext(model, tokenizer, testloader, bs, device):
#     model.eval()
#     test_loss = 0
#     for i, batch in enumerate(testloader):
#         # Tokenize the input if it's raw text
#         if isinstance(batch[0], str):
#             inputs = tokenizer(batch[0], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
#         else:
#             # Assume batch[0] is already a tensor
#             inputs = batch[0].to(device)

#         with torch.no_grad():
#             # For T5 models, provide decoder_input_ids
#             if "t5" in model.config.model_type:
#                 # Use the same input as decoder_input_ids for simplicity
#                 decoder_input_ids = inputs
#                 outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids)
#             else:
#                 # For other models (e.g., GPT, OPT), call the model directly
#                 outputs = model(inputs)
            
#             # Calculate the loss
#             lm_logits = outputs.logits
#             shift_logits = lm_logits[:, :-1, :].contiguous()
#             shift_labels = inputs[:, 1:].contiguous()
#             loss_fct = torch.nn.CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             test_loss += loss.item()

#         print(f"batch {i} loss {loss.item()}")

#     # Calculate perplexity
#     ppl = torch.exp(torch.tensor(test_loss / len(testloader)))
#     return ppl

#modified 3 : added padding handling
def eval_ppl_wikitext(model, tokenizer, testloader, bs, device):
    model.eval()
    test_loss = 0
    
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding if not set

    for i, batch in enumerate(testloader):
        # Tokenize the input with explicit max_length to avoid truncation issues
        if isinstance(batch[0], str):
            inputs = tokenizer(
                batch[0], return_tensors="pt", padding=True, truncation=True, max_length=512
            ).input_ids.to(device)
        else:
            # Assume batch[0] is already a tensor
            inputs = batch[0].to(device)

        with torch.no_grad():
            # For T5 models, provide decoder_input_ids
            if "t5" in model.config.model_type:
                decoder_input_ids = inputs
                outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids)
            else:
                outputs = model(inputs)
            
            # Calculate the loss
            lm_logits = outputs.logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            test_loss += loss.item()

        print(f"batch {i} loss {loss.item()}")

    # Calculate perplexity
    ppl = torch.exp(torch.tensor(test_loss / len(testloader)))
    return ppl

def eval_zero_shot(model_name, task_list=["qqp","rte","mnli","mrpc","sst2","cola", "qnli", "stsb"], 
        num_fewshot=0, use_accelerate=True, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},use_accelerate=True,device_map_option=\"auto\""
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        max_batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        output_base_path=None
    )
    print("********************************")
    print("zero_shot evaluation results")
    print(evaluator.make_table(results))
    return results 