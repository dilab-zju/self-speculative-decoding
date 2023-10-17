
import torch
import torch.nn as nn
import time
from transformers import top_k_top_p_filtering


def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
            
        return output_ids

def base_generate(model, tokenizer, input_ids, max_new_tokens=10, 
                  do_sample=False, top_k=0, top_p=0.85, temperature=0.2,
                  early_stop=False):

    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens], dtype=torch.long, device=model.device)
    past_key_values = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            output = model(input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True)
            logits = output['logits'][:,-1:]
            output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
            generate_ids[:, step] = output_ids
            current_input_ids = output_ids
            past_key_values = output['past_key_values']

            if early_stop and current_input_ids.item() == tokenizer.eos_token_id:
                break

    step = min(step+1, max_new_tokens)
    generate_ids = generate_ids[:, :step]
                
    return {
        'generate_ids': generate_ids,
    }

def exact_self_speculative_generate(model, tokenizer, input_ids, max_new_tokens=10, early_stop=False,
                 max_step_draft=8, th_stop_draft=0.8, auto_th_stop_draft=True, auto_parameters=[1,0.5,0.9,1e-2,0.9],
                 do_sample=False, do_sample_draft=False, top_k=0, top_p=0.85, temperature=0.2):
    
    step = 0
    step_draft = 0
    step_verify = 0
    
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=model.device)
    draft_generate_ids = torch.empty([input_ids.size(0), max_step_draft+1], dtype=torch.long, device=model.device)
    past_key_values = None

    n_matched = 0
    n_drafted = 0
    tmp_n_matched = 0
    tmp_n_drafted = 0
    tmp_matchness = 0
    with torch.no_grad():

        while True:

            if step >= max_new_tokens:
                break

            if step == 0:
                # first token use full model
                output = model(input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True)
                logits = output['logits']
                logits = logits[:,-1:]
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
                generate_ids[:, step] = output_ids
                current_input_ids = output_ids
                past_key_values = output['past_key_values']

                step += 1

            else:
                
                draft_current_input_ids = current_input_ids
                draft_past_key_values = past_key_values
                draft_generate_ids[:, 0] = current_input_ids
                for step_draft in range(max_step_draft):
                    with model.self_draft():
                        draft_output = model(input_ids=draft_current_input_ids,
                            past_key_values=draft_past_key_values,
                            return_dict=True,
                            use_cache=True)
                    draft_probs = draft_output['logits'].softmax(-1)
                    draft_output_ids, draft_output_probs = sample(
                        draft_output['logits'], return_probs=True, do_sample=do_sample_draft, top_k=top_k, top_p=top_p, temperature=temperature)
                    draft_generate_ids[:, step_draft+1] = draft_output_ids
                    draft_current_input_ids = draft_output_ids
                    draft_past_key_values = draft_output['past_key_values']

                    if draft_output_probs.item() < th_stop_draft or step + step_draft + 2 >= max_new_tokens:
                        break
                
                drafted_n_tokens = step_draft + 1
                drafted_input_ids = draft_generate_ids[:, :drafted_n_tokens+1] # raft input + raft completion

                output = model(input_ids=drafted_input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True)
                logits = output['logits']
                # output_ids = torch.argmax(logits, dim=-1)
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)

                past_key_values = output['past_key_values']

                # including the one generated by the base model
                max_matched = ((output_ids[:, :-1] != drafted_input_ids[:, 1:]).cumsum(-1) == 0).sum(-1).item() + 1
                max_of_max_matched = output_ids.size(1)

                if max_of_max_matched != max_matched:
                    output_ids = output_ids[:, :max_matched]
                    
                    past_key_values = [
                        (k[:, :, :-(max_of_max_matched - max_matched)], v[:, :, :-(max_of_max_matched - max_matched)]) for k, v in past_key_values
                    ]

                generate_ids[:, step:step+output_ids.size(1)] = output_ids
                current_input_ids = output_ids[:, -1:]

                step += output_ids.size(1)

                # remove one generated by the base model
                n_matched += max_matched - 1
                n_drafted += drafted_n_tokens
                tmp_n_matched += max_matched - 1
                tmp_n_drafted += drafted_n_tokens
                step_verify += 1

                if auto_th_stop_draft and step_verify % auto_parameters[0] == 0:
                    tmp_matchness = auto_parameters[1]*(tmp_matchness) + (1-auto_parameters[1])*((max_matched - 1)/drafted_n_tokens)
                    if tmp_matchness<auto_parameters[2]:
                        new_th_stop_draft = th_stop_draft+auto_parameters[3]
                    else:
                        if drafted_n_tokens==max_step_draft:
                            new_th_stop_draft = th_stop_draft
                        else:
                            new_th_stop_draft = th_stop_draft-auto_parameters[3]
                    th_stop_draft = auto_parameters[4] * th_stop_draft + (1-auto_parameters[4]) * new_th_stop_draft
                    # print('draft_output_probs: {:.4f}, th_stop_draft: {:.4f}, tmp_matchness: {:.2f}, drafted_n_tokens: {:d}'.format(
                    #     draft_output_probs.item(), th_stop_draft, tmp_matchness, drafted_n_tokens))

            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                break

    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]
            
    return {
        'generate_ids': generate_ids,
        'matchness': n_matched/n_drafted,
        'num_drafted_tokens': n_drafted,
        'th_stop_draft': th_stop_draft,
    }


def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

def self_speculative_sample(model, tokenizer, input_ids, max_new_tokens=10, early_stop=False,
                 max_step_draft=8, th_stop_draft=0.5, th_random_draft=1.0, auto_th_stop_draft=True, auto_parameters=[1,0.5,0.9,1e-2,0.9],
                 do_sample=False, do_sample_draft=False, 
                 top_k=0, top_p=0.85, temperature=0.2):
    
    step = 0
    step_draft = 0
    step_verify = 0
    
    current_input_ids = input_ids
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens+max_step_draft], dtype=torch.long, device=model.device)
    draft_generate_ids = torch.empty([input_ids.size(0), max_step_draft+2], dtype=torch.long, device=model.device)
    draft_generate_probs = torch.empty([input_ids.size(0), max_step_draft, model.config.vocab_size], dtype=torch.float, device=model.device)
    past_key_values = None

    n_matched = 0
    n_drafted = 0
    tmp_matchness = 0
    with torch.no_grad():

        while True:

            if step >= max_new_tokens:
                break

            if step == 0:
                # first token use full model
                output = model(input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True)
                logits = output['logits']
                logits = logits[:,-1:]
                output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
                generate_ids[:, step] = output_ids
                current_input_ids = output_ids
                past_key_values = output['past_key_values']

                step += 1

            else:
                
                draft_current_input_ids = current_input_ids
                draft_past_key_values = past_key_values
                draft_generate_ids[:, 0] = current_input_ids
                random_list = torch.rand(max_step_draft)
                for step_draft in range(max_step_draft):
                    with model.self_draft():
                        draft_output = model(input_ids=draft_current_input_ids,
                            past_key_values=draft_past_key_values,
                            return_dict=True,
                            use_cache=True)
            
                    if do_sample_draft and top_k != 1 and top_p != 0.0 and temperature != 0.0:
                        logits = draft_output['logits']
                        _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
                        draft_probs = _logits.unsqueeze(1).softmax(-1)
                        draft_output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
                    else:
                        draft_probs = draft_output['logits'].softmax(-1)
                        draft_output_ids, _ = sample(draft_output['logits'], return_probs=True, do_sample=do_sample_draft)
                    draft_generate_ids[:, step_draft+1] = draft_output_ids
                    draft_generate_probs[:, step_draft] = draft_probs
                    draft_current_input_ids = draft_output_ids
                    draft_past_key_values = draft_output['past_key_values']
                    origin_output_probs = torch.gather(draft_output['logits'].softmax(-1), -1, draft_output_ids.unsqueeze(-1)).squeeze(-1)
                    if (origin_output_probs.item() < th_stop_draft and (1-random_list[step_draft]) <= th_random_draft) or step + step_draft + 2 >= max_new_tokens:
                        break
                
                drafted_n_tokens = step_draft + 1
                drafted_input_ids = draft_generate_ids[:, :drafted_n_tokens+1] # raft input + raft completion
                output = model(input_ids=drafted_input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True)
                if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
                    logits = output['logits']
                    _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
                    probs = _logits.unsqueeze(0).softmax(-1)
                else:
                    probs = output['logits'].softmax(-1)

                output_ids = draft_generate_ids[:, 1:drafted_n_tokens+2]
                observed_r_list = (probs[0, :drafted_n_tokens] / draft_generate_probs[0, :drafted_n_tokens]).cpu()
                for i in range(drafted_n_tokens):
                    j = output_ids[0, i]
                    r = observed_r_list[i, j]
                    if random_list[i] < min(1, r):
                        pass
                    else:
                        output_ids[0, i] = torch.multinomial(max_fn((probs[0, i] - draft_generate_probs[0, i])), num_samples=1)
                        break
                else:
                    i += 1
                    output_ids[0, i] = sample(output['logits'][0, i], do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)

                past_key_values = output['past_key_values']

                # # including the one generated by the base model
                max_matched = i + 1
                max_of_max_matched = drafted_input_ids.size(1)

                if max_of_max_matched != max_matched:
                    output_ids = output_ids[:, :max_matched]
                    
                    past_key_values = [
                        (k[:, :, :-(max_of_max_matched - max_matched)], v[:, :, :-(max_of_max_matched - max_matched)]) for k, v in past_key_values
                    ]

                generate_ids[:, step:step+output_ids.size(1)] = output_ids
                current_input_ids = output_ids[:, -1:]

                step += output_ids.size(1)

                # remove one generated by the base model
                n_matched += max_matched - 1
                n_drafted += drafted_n_tokens
                step_verify += 1
                
                if auto_th_stop_draft and step_verify % auto_parameters[0] == 0:
                    tmp_matchness = auto_parameters[1]*(tmp_matchness) + (1-auto_parameters[1])*((max_matched - 1)/drafted_n_tokens)
                    if tmp_matchness<auto_parameters[2]:
                        new_th_stop_draft = th_stop_draft+auto_parameters[3]
                    else:
                        if drafted_n_tokens==max_step_draft:
                            new_th_stop_draft = th_stop_draft
                        else:
                            new_th_stop_draft = th_stop_draft-auto_parameters[3]
                    th_stop_draft = auto_parameters[4] * th_stop_draft + (1-auto_parameters[4]) * new_th_stop_draft
                    # print('draft_output_probs: {:.4f}, th_stop_draft: {:.4f}, tmp_matchness: {:.2f}, drafted_n_tokens: {:d}'.format(
                    #     draft_output_probs.item(), th_stop_draft, tmp_matchness, drafted_n_tokens))


            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                break

    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]
            
    return {
        'generate_ids': generate_ids,
        'matchness': n_matched/n_drafted,
        'num_drafted_tokens': n_drafted,
        'th_stop_draft': th_stop_draft,
    }



generate_fn_mapping = {
    'base': base_generate,
    'exact_self_speculative_generate': exact_self_speculative_generate,
    'essg': exact_self_speculative_generate,
    'self_speculative_sample': self_speculative_sample,
    'sss': self_speculative_sample,
}

def infer(model, tokenizer, prompt, generate_fn='base', 
          decode_timing=True, seed=42, *args, **kargs):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)
              
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(model, tokenizer, input_ids, *args, **kargs)
    generate_ids = generate_dict['generate_ids']
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    completion = tokenizer.decode(generate_ids[0])
    generate_dict['completion'] = completion
    generate_dict['time'] = decode_time
    return generate_dict

def clip_input(tokenizer, prompt, task_name, max_new_tokens=512, prompt_shots=''):
    if task_name == 'xsum':
        input_ids = tokenizer(
            prompt_shots +'Article: ' + prompt['document'] + '\nSummary:',
            return_tensors='pt').input_ids
    elif task_name == 'cnndm':
        input_ids = tokenizer(
            prompt_shots +'Article: ' + prompt['article'] + '\nSummary:',
            return_tensors='pt').input_ids
    elif task_name == 'humaneval':
        format_tabs=True
        if format_tabs:
            prompt = prompt['prompt'].replace("    ", "\t")
        else:
            prompt = prompt['prompt']
        input_ids = tokenizer(prompt,return_tensors='pt').input_ids
    if len(input_ids[0])+max_new_tokens>=4096:
        print('(input ids+max token)>4096')
        sample_num = (len(input_ids[0])+max_new_tokens-4096) 
        input_ids = torch.cat((input_ids[0][:2],input_ids[0][2:-3][:-sample_num],input_ids[0][-3:]),dim=0).unsqueeze(0)
    return  input_ids
    
def infer_input_ids(model, tokenizer, input_ids, generate_fn='base', 
          decode_timing=True, seed=42, *args, **kargs):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)
        
    input_ids = input_ids.to(model.device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(model, tokenizer, input_ids, *args, **kargs)
    generate_ids = generate_dict['generate_ids']
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    completion = tokenizer.decode(generate_ids[0, input_ids.size(0):],skip_special_tokens=True)
    generate_dict['completion'] = completion
    generate_dict['time'] = decode_time
    return generate_dict
