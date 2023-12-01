"""
Date: 2023-11-30
Authors: Juan Pablo Triana Martinez, Kuniaki Iwanami
Project: CS236 Final Project, GMVAE for X-rays images.

# We did use some come for reference of HW2 to do the primary setup from 2021 Rui Shu
"""

import numpy as np
import os
import shutil
import sys
import torch

# import tensorflow as tf
from torch.nn import functional as F
from torchvision import datasets, transforms
from utils.models.gmvae import GMVAE

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, dim): Samples
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Sample z = mu + sigma * e where e is from a gaussian distribution. normalized between 0 and Identity matrix
    ################################################################################
    # Determine epsilon distribution
    epsilon = torch.randn_like(v)
    
    #Calculate Z
    z = m + torch.sqrt(v) * epsilon 
    ################################################################################
    # End of code modification
    ################################################################################
    return z

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute element-wise log probability of normal and remember to sum over
    # the last dimension
    ################################################################################
    # ASSUME that ln is log for approx to decompose the normal distribution
    log_first_term = -(torch.pow((x-m), 2)/(2*v))
    log_sec_term = -np.log(np.sqrt(2*np.pi))
    log_third_term = -torch.log(torch.sqrt(v))

    #Add them all
    log_probs = log_first_term + log_sec_term + log_third_term
    log_prob = log_probs.sum(-1)

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob

def log_normal_mixture(z, m, v):
    """
    Computes log probability of Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Compute the uniformly-weighted mixture of Gaussians density for each sample
    # in the batch
    ################################################################################

    #Determine the zs used for multi variate prior distribution
    multi_zs_prior = z.unsqueeze(1).expand_as(m)

    # ASSUME that ln is log for approx to decompose the normal distribution
    log_first_term = -(torch.pow((multi_zs_prior-m), 2)/(2*v))
    log_sec_term = -np.log(np.sqrt(2*np.pi))
    log_third_term = -torch.log(torch.sqrt(v))

    #Add them all
    log_probs = log_first_term + log_sec_term + log_third_term
    prob_sums = log_probs.sum(-1)
    log_prob = log_mean_exp(prob_sums, -1)   

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def duplicate(x, rep):
    """
    Duplicates x along dim=0

    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x

    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])

def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))

def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()

def load_model_by_name(model, global_step, device=None):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    #Had to modify to add a path
    file_path = os.path.join(os.getcwd(), 'checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))

def evaluate_lower_bound(model, labeled_test_subset, run_iwae=True):
    "We will come back to change to our model class GMVAE"
    check_model = isinstance(model, GMVAE)
    assert check_model, "This function is only intended for VAE and GMVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    xl, _ = labeled_test_subset
    xl = torch.bernoulli(xl)

    def detach_torch_tuple(args):
        return (v.detach() for v in args)

    def compute_metrics(fn, repeat):
        metrics = [0, 0, 0]
        for _ in range(repeat):
            niwae, kl, rec = detach_torch_tuple(fn(xl))
            metrics[0] += niwae / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat
        return metrics

    # Run multiple times to get low-var estimate
    nelbo, kl, rec = compute_metrics(model.negative_elbo_bound, 100)
    print("NELBO: {}. KL: {}. Rec: {}".format(nelbo, kl, rec))

    if run_iwae:
        for iw in [1, 10, 100, 1000]:
            repeat = max(100 // iw, 1) # Do at least 100 iterations
            fn = lambda x: model.negative_iwae_bound(x, iw)
            niwae, kl, rec = compute_metrics(fn, repeat)
            print("Negative IWAE-{}: {}".format(iw, niwae))

def save_loss_kl_rec_across_training(model_name, global_step, loss_array, kl_array, rec_array, overwrite_existing=False):
    """
    Additional function :) to save in .npy format inside the checkpoints folder for later to use
    """
    #Set paths for checkpoints saving
    save_dir = os.path.join('checkpoints', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Create file paths for loss, kl, and rec
    loss_path = os.path.join(save_dir, 'loss-{:05d}.pt'.format(global_step)+ ".npy")
    kl_path = os.path.join(save_dir, 'kl-{:05d}.pt'.format(global_step) + ".npy")
    rec_path = os.path.join(save_dir, 'rec-{:05d}.pt'.format(global_step)+ ".npy")

    #np.save
    np.save(loss_path, loss_array)
    np.save(kl_path, kl_array)
    np.save(rec_path, rec_array)


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    maybe_delete_existing(log_dir, overwrite_existing)
    maybe_delete_existing(save_dir, overwrite_existing)
    # Sadly, I've been told *not* to use tensorflow :<
    # writer = tf.summary.FileWriter(log_dir)
    writer = None
    return writer

def maybe_delete_existing(path, overwrite_existing):
    if not os.path.exists(path):
        return

    if overwrite_existing:
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)
    else:
        raise FileExistsError(
            """
    Unpermitted attempt to delete {}.
    1. To overwrite checkpoints and logs when re-running a model, remember to pass --overwrite 1 as argument.
    2. To run a replicate model, pass --run NEW_ID where NEW_ID is incremented from 0.""".format(path))

def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass