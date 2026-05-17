import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        # PGD: random eps-ball init -> sign(grad CE) step of size alpha -> project
        # back into eps-ball n [0,1]; freeze samples once they succeed (early-stop).
        was_training = self.model.training
        self.model.eval()

        x = x.detach()
        x_adv = x.clone().detach()
        if self.rand_init:
            noise = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)
        x_adv = x_adv.detach()

        active = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        for _ in range(self.n):
            if self.early_stop and not active.any():
                break

            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss = self.loss_func(logits, y)
            grad = torch.autograd.grad(loss.sum(), x_adv)[0]

            with torch.no_grad():
                step = self.alpha * grad.sign()
                if targeted:
                    step = -step
                update_mask = active.view(-1, *([1] * (x_adv.dim() - 1))).float()
                x_adv = x_adv.detach() + step * update_mask
                x_adv = torch.max(torch.min(x_adv, x + self.eps), x - self.eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

                if self.early_stop:
                    preds = self.model(x_adv).argmax(dim=1)
                    if targeted:
                        succeeded = (preds == y)
                    else:
                        succeeded = (preds != y)
                    active = active & ~succeeded

        x_adv = x_adv.detach()

        assert torch.all(x_adv >= 0.0 - 1e-6) and torch.all(x_adv <= 1.0 + 1e-6), \
            'adversarial samples must lie in [0, 1]'
        assert torch.all((x_adv - x).abs() <= self.eps + 1e-6), \
            'adversarial samples must lie within the eps-ball around x'

        if was_training:
            self.model.train()
        return x_adv


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255., momentum=0.,
                 k=200, sigma=1 / 255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        # Antithetic NES gradient estimate: 2k queries/iter at x +- sigma*u_i, then
        # PGD with momentum (g_t = mu*g_{t-1} + grad_est) and per-sample query count.
        was_training = self.model.training
        self.model.eval()

        x = x.detach()
        B = x.size(0)
        device = x.device

        x_adv = x.clone()
        if self.rand_init:
            noise = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        g_prev = torch.zeros_like(x_adv)
        active = torch.ones(B, dtype=torch.bool, device=device)
        n_queries = torch.zeros(B, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(self.n):
                if not active.any():
                    break

                idx = torch.where(active)[0]
                x_a = x_adv[idx]
                y_a = y[idx]
                Ba = x_a.size(0)

                u = torch.randn(Ba, self.k, *x_a.shape[1:], device=device)
                x_pos = x_a.unsqueeze(1) + self.sigma * u
                x_neg = x_a.unsqueeze(1) - self.sigma * u

                flat_pos = x_pos.reshape(-1, *x_a.shape[1:])
                flat_neg = x_neg.reshape(-1, *x_a.shape[1:])

                logits_pos = self.model(flat_pos)
                logits_neg = self.model(flat_neg)

                y_rep = y_a.unsqueeze(1).expand(-1, self.k).reshape(-1)
                loss_pos = self.loss_func(logits_pos, y_rep).reshape(Ba, self.k)
                loss_neg = self.loss_func(logits_neg, y_rep).reshape(Ba, self.k)

                diff = (loss_pos - loss_neg).view(Ba, self.k, *([1] * (x_a.dim() - 1)))
                grad_est = (diff * u).sum(dim=1) / (2.0 * self.k * self.sigma)

                n_queries[idx] += 2 * self.k

                g_a = self.momentum * g_prev[idx] + grad_est
                g_prev[idx] = g_a

                step = self.alpha * g_a.sign()
                if targeted:
                    step = -step

                x_a_new = x_a + step
                x_a_new = torch.max(torch.min(x_a_new, x[idx] + self.eps),
                                    x[idx] - self.eps)
                x_a_new = torch.clamp(x_a_new, 0.0, 1.0)
                x_adv[idx] = x_a_new

                if self.early_stop:
                    preds = self.model(x_a_new).argmax(dim=1)
                    if targeted:
                        succeeded = (preds == y_a)
                    else:
                        succeeded = (preds != y_a)
                    new_active = active.clone()
                    new_active[idx] = ~succeeded
                    active = new_active

        x_adv = x_adv.detach()

        assert torch.all(x_adv >= 0.0 - 1e-6) and torch.all(x_adv <= 1.0 + 1e-6), \
            'adversarial samples must lie in [0, 1]'
        assert torch.all((x_adv - x).abs() <= self.eps + 1e-6), \
            'adversarial samples must lie within the eps-ball around x'

        if was_training:
            self.model.train()
        return x_adv, n_queries


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        # Ensemble PGD (Liu et al., ICLR'17): at each step descend the *average*
        # CE loss across the ensemble; freeze samples only once *all* models fool.
        was_training = [m.training for m in self.models]
        for m in self.models:
            m.eval()

        x = x.detach()
        x_adv = x.clone().detach()
        if self.rand_init:
            noise = torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)
        x_adv = x_adv.detach()

        active = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        n_models = len(self.models)

        for _ in range(self.n):
            if self.early_stop and not active.any():
                break

            x_adv.requires_grad_(True)
            total_loss = 0.0
            for m in self.models:
                logits = m(x_adv)
                total_loss = total_loss + self.loss_func(logits, y)
            total_loss = total_loss / n_models
            grad = torch.autograd.grad(total_loss.sum(), x_adv)[0]

            with torch.no_grad():
                step = self.alpha * grad.sign()
                if targeted:
                    step = -step
                update_mask = active.view(-1, *([1] * (x_adv.dim() - 1))).float()
                x_adv = x_adv.detach() + step * update_mask
                x_adv = torch.max(torch.min(x_adv, x + self.eps), x - self.eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

                if self.early_stop:
                    succeeded_all = torch.ones(x.size(0), dtype=torch.bool,
                                               device=x.device)
                    for m in self.models:
                        preds = m(x_adv).argmax(dim=1)
                        if targeted:
                            succeeded_m = (preds == y)
                        else:
                            succeeded_m = (preds != y)
                        succeeded_all = succeeded_all & succeeded_m
                    active = active & ~succeeded_all

        x_adv = x_adv.detach()

        assert torch.all(x_adv >= 0.0 - 1e-6) and torch.all(x_adv <= 1.0 + 1e-6), \
            'adversarial samples must lie in [0, 1]'
        assert torch.all((x_adv - x).abs() <= self.eps + 1e-6), \
            'adversarial samples must lie within the eps-ball around x'

        for m, was_t in zip(self.models, was_training):
            if was_t:
                m.train()
        return x_adv
