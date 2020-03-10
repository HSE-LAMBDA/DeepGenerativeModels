from itertools import chain, repeat, islice
from collections import defaultdict

import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

def in_jupyter():
    try:
        from IPython import get_ipython
        from ipykernel.zmqshell import ZMQInteractiveShell
        assert isinstance(get_ipython(), ZMQInteractiveShell)
    except Exception:
        return False
    return True

try:
    if in_jupyter():
        # from tqdm import tqdm as pbar
        from tqdm import tqdm_notebook as pbar
    else:
        from tqdm import tqdm as pbar
except ImportError:
    def pbar(it, *a, **kw):
        return it

# avoid log(0)
_eps = 1e-15

def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False
def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def _as_tuple(x,n):
    if not isinstance(x, tuple):
        return (x,)*n
    assert len(x)==n, 'input is a tuple of incorrect size'
    return x
            
def _take_epochs(X, n_epochs):
    """Get a fractional number of epochs from X, rounded to the batch
    X: torch.utils.DataLoader (has len(), iterates over batches)
    n_epochs: number of iterations through the data.
    """
    n_batches = int(np.ceil(len(X)*n_epochs))
    _take_iters(X, n_batches)

def _take_batches(X, n_batches):
    """Get a integer number of batches from X, reshuffling as necessary
    X: torch.utils.DataLoader (has len(), iterates over batches)
    n_iters: number of batches
    """
    n_shuffles = int(np.ceil(len(X)/n_batches))
    return islice(chain.from_iterable(repeat(X,n_shuffles)),n_batches)

class AlphaGAN(nn.Module):
    def __init__(self, E, G, D, C, latent_dim,
                 lambd=1, z_lambd=0, code_weight=1, adversarial_weight=1):
        """α-GAN as described in Rosca, Mihaela, et al.
            "Variational Approaches for Auto-Encoding Generative Adversarial Networks."
            arXiv preprint arXiv:1706.04987 (2017).
        E: nn.Module mapping X to Z
        G: nn.Module mapping Z to X
        D: nn.module discriminating real from generated/reconstructed X
        C: nn.module discriminating prior from posterior Z
        latent_dim: dimensionality of Z
        lambd: scale parameter for the G distribution
            a.k.a weight for the reconstruction loss
        z_lambd: if nonzero, weight for code reconstruction loss
        code_weight: weight for code loss. if zero, C won't be trained
        adversarial_weight: weight for adversarial loss. if zero, D won't be trained
        """
        super().__init__()
        self.E = E
        self.G = G
        self.D = D
        self.C = C
        self.latent_dim = latent_dim
        self.lambd = lambd
        self.z_lambd = z_lambd
        self.code_weight = code_weight
        self.adversarial_weight = adversarial_weight

    def sample_prior(self, n):
        """Sample self.latent_dim-dimensional unit normal.
        n: batch size
        """
        return self._wrap(torch.randn(n, self.latent_dim))#, requires_grad=False)

    def rec_loss(self, x_rec, x):
        """L1 reconstruction error or Laplace log likelihood"""
        return (x_rec-x).abs().mean()
    
    def autoencoder_loss(self, x):
        """Return reconstruction loss, adversarial loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.E, self.G)

        z_prior = self.sample_prior(len(x))
        if self.use_E(): # encoder case
            z = self.E(x)
            zs = torch.cat((z_prior, z), 0)
            x_fake = self.G(zs)
            x_gen, x_rec = x_fake.chunk(2)
        else: # no encoder (pure GAN) case
            zs = z_prior
            x_fake = x_gen = self.G(zs)
        
        ret = {}
        if self.use_C():
            ret['code_adversarial_loss'] = -(self.C(z) + _eps).log().mean()
        if self.adversarial_weight != 0:
            ret['adversarial_loss'] = -(self.D(x_fake) + _eps).log().mean()
        if self.use_E() and self.lambd != 0:
            ret['reconstruction_loss'] = self.lambd*self.rec_loss(x_rec, x)
        if self.use_E() and self.z_lambd != 0:
            z_rec = self.E(x_fake)
            ret['code_reconstruction_loss'] = self.z_lambd*self.rec_loss(z_rec, zs)
        return ret

    def discriminator_loss(self, x):
        """Return discriminator (D) loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.D)
        z_prior = self.sample_prior(len(x))
        if self.use_E(): # encoder case
            z = self.E(x)
            zs = torch.cat((z_prior, z), 0)
        else: # no encoder (pure GAN) case
            zs = z_prior
        x_fake = self.G(zs)
        return {
            'discriminator_loss':
                - (self.D(x) + _eps).log().mean()
                - (1 - self.D(x_fake) + _eps).log().mean()
        }

    def code_discriminator_loss(self, x):
        """Return code discriminator (C) loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.C)
        z_prior = self.sample_prior(len(x))
        z = self.E(x)
        return {
            'code_discriminator_loss':
                - (self.C(z_prior) + _eps).log().mean()
                - (1 - self.C(z) + _eps).log().mean()
        }

    def _epoch(self, X, loss_fns,
               optimizers=None, n_iter=(1,1,1), n_batches=None):
        """Evaluate/optimize for one epoch.
        X: torch.nn.DataLoader
        loss_fns: each takes an input batch and returns dict of loss component Variables
        optimizers: sequence of torch.nn.Optimizer for each loss or None if not training
        n_iter: sequence of optimization steps per batch for each loss
        n_batches: number of batches to draw or None for all data
        """
        optimizers = optimizers or [None]*3
        iter_losses = defaultdict(list)
        
        it = _take_batches(X, n_batches) if n_batches else X
        desc = 'training batch' if optimizers[0] else 'validating batch'
        for x in pbar(it, desc=desc, leave=False):
            x = self._wrap(x)
            for opt, iters, loss_fn in zip(optimizers, n_iter, loss_fns):
                for _ in range(iters):
                    loss_components = loss_fn(x)
                    if opt:
                        loss = sum(loss_components.values())
                        self.zero_grad()
                        loss.backward()
                        opt.step()
                        del loss
                    for k,v in loss_components.items():
                        iter_losses[k].append(v.data.cpu().numpy())
                    del loss_components
        return {k:np.mean(v) for k,v in iter_losses.items()}

    def fit(self,
            X_train, X_valid=None,
            opt_fn=torch.optim.Adam, opt_params={'lr':8e-4, 'betas':(.5,.9)},
            n_iter=(2,1,1), n_batches=None, n_epochs=10,
            log_fn=None, log_every=1,
            checkpoint_fn=None, checkpoint_every=2):
        """
        X_train: torch.utils.data.DataLoader
        X_valid: torch.utils.data.DataLoader or None
        opt_fn: nn.Optimizer constructor or triple for E/G, D, C
        opt_params: dict of keyword args for optimizer or triple for E/G, D, C
        n_iter: int or triple # of E/G, D, C optimizer steps/batch
        n_batches: int or pair # of train, valid batches per epoch (None for all data)
        n_epochs: number of discriminator, autoencoder training iterations
        log_fn: takes diagnostic dict, called after every nth epoch
        log_every: call log function every nth epoch
        checkpoint_fn: takes model, epoch. called after nth every epoch
        checkpoint_every: call checkpoint function every nth epoch
        """
        _unfreeze(self)
        
        n_iter = _as_tuple(n_iter, 3)
        train_batches, valid_batches = _as_tuple(n_batches, 2)
        EG_opt_fn, D_opt_fn, C_opt_fn = (
            lambda p: fn(p, **hyperparams) for fn, hyperparams in zip(
                _as_tuple(opt_fn, 3),  _as_tuple(opt_params, 3)))
        
        # define optimization order/separation of networks
        optimizers, loss_fns = [], []
         #if neither autoencoder loss is present, skip encoder and just train GAN
        if self.use_E():
            optimizers.append(EG_opt_fn(chain(
                self.E.parameters(), self.G.parameters()
            )))
        else:
            optimizers.append(EG_opt_fn(self.G.parameters()))
        loss_fns.append(self.autoencoder_loss)
        # discriminator
        if self.use_D():
            optimizers.append(D_opt_fn(self.D.parameters()))
            loss_fns.append(self.discriminator_loss)
        # code discriminator
        if self.use_C():
            optimizers.append(C_opt_fn(self.C.parameters()))
            loss_fns.append(self.code_discriminator_loss)
# #         discriminators together
#         optimizers.append(D_opt_fn(chain(
#             self.D.parameters(), self.C.parameters()
#         )))
#         loss_fns.append(lambda x: self.code_discriminator_loss(x)+self.discriminator_loss(x))

        for i in pbar(range(n_epochs), desc='epoch'):
            diagnostic = defaultdict(dict)
            report = log_fn and (i%log_every==0 or i==n_epochs-1)
            checkpoint = checkpoint_every and checkpoint_fn and (
                (i+1)%checkpoint_every==0 or i==n_epochs-1 )
            # train for one epoch
            self.train()
            diagnostic['train'].update( self._epoch(
                X_train, loss_fns, optimizers, n_iter, train_batches ))
            # validate for one epoch
            self.eval()
            diagnostic['valid'].update(self._epoch(
                X_valid, loss_fns, n_batches=valid_batches ))
            # log the dict of loss components
            if report:
                log_fn(diagnostic)
            if checkpoint:
                checkpoint_fn(self, i+1)

    def forward(self, *args, mode=None):
        """
        mode:
            None: return z ~ Q(z|x), x_rec ~ P(x|z); args[0] is x.
            sample: return z ~ P(z), x ~ P(x|z); args[0] is number of samples.
            generate: return x ~ P(x|z); args[0] is z.
            encode: return z ~ Q(z|x); args[0] is x.
            reconstruct: like None, but only return x_rec.
        """
        # get code from prior, args, or by encoding.
        if mode=='sample':
            n = args[0]
            z = self.sample_prior(n)
        elif mode=='generate':
            z = self._wrap(args[0])
        else:
            x = self._wrap(args[0])
            z = self.E(x)
        # step there if reconstruction not desired
        if mode=='encode':
            return z
        # run code through G
        x_rec = self.G(z)
        if mode=='reconstruct' or mode=='generate':
            return x_rec
        # default, 'sample': return code and reconstruction
        return z, x_rec

    def is_cuda(self):
        return any(p.is_cuda for p in self.parameters())

    def use_E(self):
        return bool(
            self.E and (self.code_weight or self.lambd or self.z_lambd))
    
    def use_C(self):
        return bool(self.E and self.C and self.code_weight)
    
    def use_D(self):
        return bool(self.D and self.adversarial_weight)
    
    def _wrap(self, x, **kwargs):
        """ensure x is a Variable on the correct device"""
        if not isinstance(x, Variable):
            # if x isn't a Tensor, attempt to construct one from it
            if not isinstance(x, torch._TensorBase):
                x = torch.Tensor(x)
            x = Variable(x, **kwargs)
        if self.is_cuda():
            x = x.cuda()
        return x

# experiment with using WGAN-GP for the GAN losses
# requires pytorch >= 0.2.0
class AlphaWGAN(AlphaGAN):
    """α-GAN with alternative WGAN-GP based losses"""
    
    def gradient_penalty(self, model, x, x_gen, w=10):
        """WGAN-GP gradient penalty"""
        assert x.size()==x_gen.size(), "real and sampled sizes do not match"
        alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
        alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
        alpha = alpha_t(*alpha_size).uniform_()
        x_hat = x.data*alpha + x_gen.data*(1-alpha)
        x_hat = Variable(x_hat, requires_grad=True)

        def eps_norm(x):
            x = x.view(len(x), -1)
            return (x*x+_eps).sum(-1).sqrt()
        def bi_penalty(x):
            return (x-1)**2

        grad_xhat = torch.autograd.grad(
            model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True
        )[0]
    
        penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
        return penalty
    
    def autoencoder_loss(self, x):
        """Return reconstruction loss, adversarial loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.E, self.G)

        z_prior = self.sample_prior(len(x))
        if self.use_E(): # encoder case
            z = self.E(x)
            zs = torch.cat((z_prior, z), 0)
            x_fake = self.G(zs)
            x_gen, x_rec = x_fake.chunk(2)
        else: # no encoder (pure GAN) case
            zs = z_prior
            x_fake = x_gen = self.G(zs)
        
        ret = {}
        if self.use_C() != 0:
            ret['code_adversarial_loss'] = -self.C(z).mean()
        if self.adversarial_weight != 0:
            ret['adversarial_loss'] = -self.D(x_fake).mean()
        if self.use_E() and self.lambd != 0:
            ret['reconstruction_loss'] = self.lambd*self.rec_loss(x_rec, x)
        if self.use_E() and self.z_lambd != 0:
            z_rec = self.E(x_fake)
            ret['code_reconstruction_loss'] = self.z_lambd*self.rec_loss(z_rec, zs)
        return ret

    def discriminator_loss(self, x):
        """Return discriminator (D) loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.D)
        z_prior = self.sample_prior(len(x))
        if self.use_E(): # encoder case
            z = self.E(x)
            zs = torch.cat((z_prior, z), 0)
            x_real = torch.cat((x,x), 0)
        else: # no encoder (pure GAN) case
            zs = z_prior
            x_real = x
        x_fake = self.G(zs)
        return {
            'D_critic_loss': 
                self.D(x_fake).mean() - self.D(x).mean(),
            'D_gradient_penalty':
                self.gradient_penalty(self.D, x_real, x_fake)
        }

    def code_discriminator_loss(self, x):
        """Return code discriminator (C) loss"""
        _freeze(self) # save memory by excluding parameters from autograd
        _unfreeze(self.C)
        z_prior = self.sample_prior(len(x))
        z = self.E(x)
        return {
            'C_critic_loss': 
                self.C(z).mean() - self.C(z_prior).mean(),
            'C_gradient_penalty':
                self.gradient_penalty(self.C, z, z_prior)
        }