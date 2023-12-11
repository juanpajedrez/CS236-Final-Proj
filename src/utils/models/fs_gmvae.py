"""
Date: 2023-11-30
Authors: Kuniaki Iwanami, Juan Pablo Triana Martinez
Project: CS236 Hw2 GMVAE answers

# We did use some come for reference of HW2 to do the primary setup from 2021 Rui Shu
"""


# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
from utils import tools as t
from utils.models import nns
from torch import nn
from torch.nn import functional as F

class FS_GMVAE(nn.Module):
    def __init__(self, nn='FSVAE_CXR14_V1_Kuni', z_dim=2, k=500, name='gmvae', loss_type = "bce"):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)

        #Assign the loss type
        self.loss_type = loss_type
        
        # *** ADDED CODE: set y_dim to 20 ***
        self.enc = nn.Encoder(self.z_dim, y_dim=20)
        self.dec = nn.Decoder(self.z_dim, y_dim=20)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)


    # *** Modified code: Added y ***
    def negative_elbo_bound(self, x, y=None):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = t.gaussian_parameters(self.z_pre, dim=1)
        prior_mean, prior_var = prior

        #Obtain the batch and dimension
        batch = x.shape[0]
        dim = x.shape[1]

        #First, find the q_phi_enc(z|x) by passing the x observations through the encoder
        
        # *** ADDED CODE: Applying y (labels) ***
        if y is not None:
            q_phi_enc = self.enc(x, y)
        else:
            q_phi_enc = self.enc(x)

        q_phi_enc_mean, q_phi_enc_var = q_phi_enc

        #Second, we sample all zs per row in the tensor x batch.
        zs = t.sample_gaussian(q_phi_enc_mean, q_phi_enc_var)

        #Third, calculate the log for posterior distribution q_phi_zx
        log_post_q_phi_zx = t.log_normal(zs, q_phi_enc_mean, q_phi_enc_var)

        #Fifth, determine a multi means and multi variances to find log.
        prior_multi_means = prior_mean.expand(batch, prior_mean.shape[1], prior_mean.shape[2])
        prior_multi_vars = prior_var.expand(batch, prior_var.shape[1],  prior_var.shape[2])

        #Sixth determine the kl using the function
        log_prior_q_z = t.log_normal_mixture(zs, prior_multi_means, prior_multi_vars)
        kls = log_post_q_phi_zx - log_prior_q_z
        kl = torch.mean(kls)
    
        #Seventh determine the rec error with zs and calcualte everything
        
        # *** ADDED CODE: Applying y (labels) ***
        if y is not None:        
            p_x_z_decoder = self.dec(zs, y)
        else:
            p_x_z_decoder = self.dec(zs)        

        #Reshape to batch x 3 x 224 x 224 (original image)
        p_x_z_decoder = p_x_z_decoder.view(p_x_z_decoder.shape[0],3,224,224)

        # *** Added Code: Change the loss from BCE to BCE+MSE ***
        #If condition for the type of loss
        if self.loss_type == "bce":
            recs = -1 *t.log_bernoulli_with_logits(x, p_x_z_decoder)
            rec = torch.mean(recs)
        elif self.loss_type == "mse":
            recs = -1*t.log_pixel_with_logits(x, p_x_z_decoder)
            rec = torch.mean(recs)
        elif self.loss_type == "bce_and_mse":
            #Obtain the mse loss
            mse_1 = -1*t.log_pixel_with_logits(x, p_x_z_decoder)
            #Obtain the log with bernoulli with logits
            bern_1 = -t.log_bernoulli_with_logits(x, p_x_z_decoder)
            #Obtain rec
            rec = torch.mean(mse_1) + torch.mean(bern_1)

        #Calculate nelbo
        nelbo = kl + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = t.gaussian_parameters(self.z_pre, dim=1)
        prior_means , prior_vars = prior

        #Obtain the batch and dimension
        batch = x.shape[0]
        dim = x.shape[1]
        multi_weight_x = t.duplicate(x, iw)

        #Obtain the q_phi_mean and q_phi_variance with multi weighted samples
        q_phi_mean, q_phi_variance = self.enc(x)
        multi_q_phi_mean = t.duplicate(q_phi_mean, iw)
        multi_q_phi_var = t.duplicate(q_phi_variance, iw)

        #Obain the zs using this multi means and variances
        multi_zs = t.sample_gaussian(multi_q_phi_mean, multi_q_phi_var)

        #Obtain the probabibilites from the decoder feeding the zs
        probs_x_z_decoder = self.dec(multi_zs)
        probs_x_z_decoder = probs_x_z_decoder.view(probs_x_z_decoder.shape[0],3,224,224)
        recs = t.log_bernoulli_with_logits(multi_weight_x, probs_x_z_decoder)

        #Addittion to match sizes
        recs = recs.mean(dim=1)
        recs = recs.mean(dim=1)
        rec = -1.0 * torch.mean(recs)

        #Find the log sums now using multi prior_zs means and vars
        p_zs_means = prior_means.expand(batch*iw, *prior_means.shape[1:])
        p_zs_vars = prior_vars.expand(batch*iw, *prior_vars.shape[1:])
        
        #Find the logs of p_zs and multi_q_phis
        log_p_zs = t.log_normal_mixture(multi_zs, p_zs_means, p_zs_vars)
        log_multi_q_phis = t.log_normal(multi_zs, multi_q_phi_mean, multi_q_phi_var)

        #Find kl divergence
        kls = log_multi_q_phis - log_p_zs
        kl = torch.mean(kls)

        #Log sum of the likelihood of all
        log_sum = recs - log_multi_q_phis + log_p_zs 

        # Find the log mean exp.
        niwaes = t.log_mean_exp(log_sum.reshape(iw, batch), 0)
        niwae = -1 * torch.mean(niwaes)
        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    # *** Modified code: Added y ***
    def loss(self, x, y):
        nelbo, kl, rec = self.negative_elbo_bound(x, y)
        loss = nelbo

        return loss, kl, rec

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    # *** Modified code: Added y ***
    def compute_sigmoid_given(self, z, y):
        logits = self.dec(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = t.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return t.sample_gaussian(m, v)

    # *** Modified code: Added y ***
    def sample_x(self, batch, y):
        z = self.sample_z(batch)
        return self.sample_x_given(z, y)

    # *** Modified code: Added y ***
    def sample_x_given(self, z, y):
        return self.compute_sigmoid_given(z, y)
