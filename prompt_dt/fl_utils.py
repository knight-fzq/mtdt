import numpy as np
import random
import copy
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters

def get_harmo_gradient(gradient_set):
    #harmo_gradient=0
    idx=0
    for env_name in gradient_set:
        if idx==0:
            harmo_gradient=gradient_set[env_name]
        else:
            harmo_gradient+=gradient_set[env_name]
        idx+=1
    harmo_gradient/=idx
    return harmo_gradient
def mask_dead_harmo(harmo_gradient,gradient_set,mask_vectors,ratio=0.0001):
    num_dead=int(len(harmo_gradient)*ratio)
    print(num_dead)
    #index=torch.zeros(harmo_gradient.size())
    num_task=len(mask_vectors)
    #mask_vectors=[dict_to_vector(masks[name]) for name in masks]
    new_vectors={name:torch.zeros(mask_vectors[name].size()).cuda() for name in mask_vectors}
    #new_masks=copy(masks)
    simi_vec=None
    for name in mask_vectors:
        #new_mask=torch.ones(masks_vector[idx].size())
        #mask_vectors[name]=mask_vectors[name]
        #masks_vector[name]=[]
        if simi_vec==None:
            simi_vec=harmo_gradient*gradient_set[name]*(1-mask_vectors[name]*1000)
        else:
            simi_vec+=harmo_gradient*gradient_set[name]*(1-mask_vectors[name]*1000)
    #if simi_vec>
    value, index = torch.topk(simi_vec,k=num_dead,largest=True)
    for name in new_vectors:
        #mask_vectors[i]=0
        new_vectors[name][index]=1
        
    #new_masks[i]=vector_to_dict(new_masks,new_vectors[i])
    return new_vectors
    #new_mask_vector
    #new_mask=copy(mask)
    #gradient_set[i]
    #num_dead=len(harmo_gradient)*ratio
    #pass

def mask_generate_harmo(harmo_gradient,gradient_set,mask_vectors,model_vec,thresh=1,ratio=0.0001):
    num_new=int(len(harmo_gradient)*ratio)
    new_vectors={name:torch.zeros(mask_vectors[name].size()).cuda() for name in mask_vectors}
    eps=1e-6
    for name in mask_vectors:
        #mask_vec=dict_to_vector(env_masks[name])
        simi_vec=harmo_gradient*gradient_set[name]*mask_vectors[name]
        num_conflict=torch.sum(simi_vec<0)
        #print(num_conflict)
        #sum(i < thresh for i in simi_vec)
        #new_mask=copy(mask)
        
        # value, index = torch.topk(simi_vec,k=num_grow,largest=False)
        # new_mask[index]=0
        if num_conflict>=num_new:
            
            value, index = torch.topk(simi_vec,k=num_new,largest=False)
            new_vectors[name][index]=1
        else:
            value, index = torch.topk(simi_vec,k=num_conflict,largest=False)
            new_vectors[name][index]=1
            value, index = torch.topk(model_vec,k=num_new-num_conflict,largest=False)
            new_vectors[name][index]=1
    return new_vectors

def get_new_masks(dead_masks,new_masks,env_masks):
    for name in env_masks:
        env_masks[name]=env_masks[name]+dead_masks[name]-new_masks[name]
    pass



def parameters_to_gradvector(net):
    vec = []
    for param in net.parameters():
        if param.grad !=None:
            
            vec.append(param.grad.view(-1))
        else:
            
            vec.append(torch.zeros(param.size()).view(-1).cuda())
    return torch.cat(vec)

def vector_to_dict(maskdict,dict_vector):
    pointer = 0
    for n, p in maskdict.items():
        # The length of the parameter
        num_param = p.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        maskdict[n] = dict_vector[pointer:pointer + num_param].view_as(p)

        # Increment the pointer
        pointer += num_param

    return maskdict

def dict_to_vector(masks):
    vec = []
    for name in masks:
        #if param.grad !=None:
            
        vec.append(masks[name].view(-1))
        #else:
            
        #vec.append(torch.zeros(param.size()).view(-1).cuda())
    return torch.cat(vec)

def ERK_maskinit(model, density=0.5, erk_power_scale=1.0, seed=0):
    
    masks={}
    for name, tensor in model.named_parameters():
        #avoid bias to be 0
        #if len(tensor.size()) == 4 or len(tensor.size()) == 2:
            #self.names.append(name)
        masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False)
    print('initialize by fixed_ERK')
    total_params = 0
    for name, weight in masks.items():
        total_params += weight.numel()
    is_epsilon_valid = False
    dense_layers = set()
    while not is_epsilon_valid:

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in masks.items():
            n_param = np.prod(mask.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if name in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                raw_probabilities[name] = (
                                                    np.sum(mask.shape) / np.prod(mask.shape)
                                            ) ** erk_power_scale
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    print(f"Sparsity of var:{mask_name} had to be set to 0.")
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    density_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name, mask in masks.items():
        n_param = np.prod(mask.shape)
        if name in dense_layers:
            density_dict[name] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            density_dict[name] = probability_one
        # print(
        #     f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
        # )
        masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data

        total_nonzero += density_dict[name] * mask.numel()
    print(f"Overall sparsity {total_nonzero / total_params}")
    return masks

@torch.no_grad()
def apply_mask_model(model,masks):
    weights_after_mask=copy.deepcopy(model.state_dict())
    for key, mask in masks.items():
        if 'embed' or 'predict' in key:
            weights_after_mask[key]=weights_after_mask[key]
        else:
            weights_after_mask[key]=weights_after_mask[key]*mask
    return weights_after_mask

# @torch.no_grad()
# def get_harmo_gradient(gradient_vecs,masks,weights=None,mode="avg"):
#     if weights==None:
#         weights=1/len(gradient_vecs)
#     if mode =="avg":
#         return torch.sum(gradient_vecs*masks*weight)

# def mask_dead():
#     simi_vectos=gradient_vecs*mask*harmo_gradient:
#     mask=mask+topk(simi_vectors)

# def mask_generate():
#     simi_vectos=gradient_vecs*(1-mask)*harmo_gradient:
#     mask=mask+masktopk(simi_vectors)
#def if_conflicts(gradient_vecs,harmo_gradient,thresh):
#    if gradient_vecs.dot(harmo_gradient)>thresh:



@torch.no_grad()
def apply_final_mask(model, env_masks, threshold):
    weights_after_mask=copy.deepcopy(model.state_dict())
    for name, param in model.state_dict().items():
        if 'embed' or 'predict' in name:
            weights_after_mask[name] = weights_after_mask[name]
        else:
            mask = 0
            for k in env_masks.keys():
                if name in env_masks[k]:
                    mask += env_masks[k][name]
            mask = (mask >= threshold)
            if type(mask) == bool:
                mask = float(mask)
            else:
                mask = mask.float().data
            weights_after_mask[name] = weights_after_mask[name] * mask
    return weights_after_mask

@torch.no_grad()
def apply_mask_grad(model, masks):
    for n, p in model.named_parameters():
        if p.grad is None:  
            continue
            
        if n in masks:
            if 'embed' or 'predict' in n:
                p.grad=p.grad
            else:
                p.grad=p.grad*masks[n]

@torch.no_grad()
def load_original_parameters(model, original_state_dict, masks):
    current_state_dict = model.state_dict()
    for key, value in original_state_dict.items():
        if key in masks:
            m = 1 - masks[key]
            current_state_dict[key] = current_state_dict[key] * masks[key] + value * m
        
    model.load_state_dict(current_state_dict)

@torch.no_grad()
def cosine_annealing(alpha_t,eta_max=50,eta_min=10):
    
    return int(eta_min+0.5*(eta_max-eta_min)*(1+np.cos(np.pi*alpha_t)))

@torch.no_grad()
def mask_grow(num_grow, model, masks, mode="magnitude"):
    
    masks_new={}
    # be careful here, i donot know whether here is a value or zhizhen
    params = {}
    for n, p in model.named_parameters():
        params[n] = torch.abs(p)
    param_vec = torch.cat([p.view(-1) for n, p in params.items()])

    # model_param=model.parameters()
    # model_vec=torch.abs(torch.clone(parameters_to_vector(model_param)))

    # for idx,item in enumerate(model_vec):
    #     if item==0:
    #         model_vec[idx]=1000
    index = param_vec == 0
    param_vec[index] = 1000

    value, index = torch.topk(param_vec,k=num_grow,largest=False)
    param_vec_ = torch.zeros(param_vec.shape).to(param_vec.device)
    param_vec_[index] = 1

    # for idx,item in enumerate(model_vec):
    #     if idx in index:
    #         model_vec[idx]==int(1)
    #     else:
    #         model_vec[idx]==int(0)

    # vector_to_parameters(model_vec_, model_param)
    pointer = 0
    for n, p in params.items():
        # The length of the parameter
        num_param = p.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        params[n] = param_vec_[pointer:pointer + num_param].view_as(p)

        # Increment the pointer
        pointer += num_param

    for name, tensor in params.items():
        if name in masks:
            masks_new[name]=masks[name]+tensor

    return masks_new

@torch.no_grad()
def mask_death(num_death, model, masks, mode="grad_magnitude"):
    masks_new={}

    params = {}
    for n, p in model.named_parameters():
        if p.grad is None:  
            continue
        if n in masks:
            params[n] = torch.abs(p.grad)
        else:
            params[n] = torch.zeros(p.data.shape).to(p.device)

    param_vec = torch.cat([p.view(-1) for n, p in params.items()])
    
    model_vec=torch.clone(param_vec)
    value,index=torch.topk(model_vec,k=num_death)

    model_vec_ = torch.zeros(model_vec.shape).to(model_vec.device)
    model_vec_[index] = 1
    # for idx,item in enumerate(model_vec):
    #     if idx in index:
    #         model_vec[idx]==int(1)
    #     else:
    #         model_vec[idx]==int(0)
    
    pointer = 0
    for n, p in params.items():
        # The length of the parameter
        num_param = p.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        params[n] = model_vec_[pointer:pointer + num_param].view_as(p)

        # Increment the pointer
        pointer += num_param

    for name, tensor in params.items():
        if name in masks:
            masks_new[name]=masks[name]-tensor

    return masks_new


# module=SimpleCNN_header()
# masks=ERK_maskinit(module)
# param_after_mask=apply_mask_model(module,masks)
# module.load_state_dict(param_after_mask)
# apply_mask_grad(module,masks)
# t=1
# total_round=100
# alpha_t=t/total_round
# eta_min=10
# eta_max=1000
# num_grow=cosine_annealing(alpha_t=alpha_t,eta_min=eta_min,eta_max=eta_max)
# num_death=cosine_annealing(alpha_t=1-alpha_t,eta_min=eta_min,eta_max=eta_max)
# masks=mask_grow(num_grow,module,masks,mode="magnitude")
# masks=mask_death(num_death,module,masks,mode="grad_magnitude")
# param_after_mask=apply_mask_model(module,masks)
# module.load_state_dict(param_after_mask)

