import torch
from functools import reduce


class HessianFree(torch.optim.Optimizer):
    """
    Implements the Hessian-free algorithm presented in `Training Deep and
    Recurrent Networks with Hessian-Free Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1)
        delta_decay (float, optional): Decay of the previous result of computing
            delta with conjugate gradient method for the initialization of the 
            next conjugate gradient iteration
        damping (float, optional): Initial value of the Tikhonov damping
            coefficient. (default: 0.5)
        max_iter (int, optional): Maximum number of Conjugate-Gradient
            iterations (default: 50)
        use_gnm (bool, optional): Use the generalized Gauss-Newton matrix:
            probably solves the indefiniteness of the Hessian (Section 20.6)
        verbose (bool, optional): Print statements (debugging)

    .. _Training Deep and Recurrent Networks with Hessian-Free Optimization:
        https://doi.org/10.1007/978-3-642-35289-8_27
    """

    def __init__(self, params,
                 lr=1,
                 damping=0.5,
                 delta_decay=0.95,
                 cg_max_iter=50,
                 verbose=False,
                 use_gnm=True):

        if not (0.0 < lr <= 1):
            raise ValueError("Invalid lr: {}".format(lr))

        if not (0.0 < damping <= 1):
            raise ValueError("Invalid damping: {}".format(damping))

        if not cg_max_iter > 0:
            raise ValueError("Invalid cg_max_iter: {}".format(cg_max_iter))

        defaults = dict(alpha=lr,
                        damping=damping,
                        delta_decay=delta_decay,
                        cg_max_iter=cg_max_iter,
                        use_gnm=use_gnm,
                        verbose=verbose)
        super(HessianFree, self).__init__(params, defaults)

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _modify_params(self, new_params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = new_params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel()

    def _gather_flat_params(self):
        views = list()
        for p in self._params:
            view = p.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_grad(self):
        views = list()
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def step(self, closure, b=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable): A closure that re-evaluates the model (yielding 
                its output) and returns a tuple of the loss and the output
            b (callable, optional): A closure that calculates the vector b in
                the minimization problem x^T . A . x + x^T b.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        alpha = group['alpha']
        delta_decay = group['delta_decay']
        cg_max_iter = group['cg_max_iter']
        damping = group['damping']
        use_gnm = group['use_gnm']
        verbose = group['verbose']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        loss_before, output = closure()
        current_evals = 1
        state['func_evals'] += 1

        # Gather current parameters and respective gradients
        flat_params = self._gather_flat_params()
        flat_grad = self._gather_flat_grad()

        # Define linear operator
        if use_gnm:
            # Generalized Gauss-Newton vector product
            def A(x):
                return self._Gv(loss_before, output, x, damping)
        else:
            # Hessian-vector product
            def A(x):
                return self._Hv(flat_grad, x, damping)

        b = flat_grad.detach() if b is None else b().detach()

        # Initializing Conjugate-Gradient (Section 20.10)
        # NOTE: This seems to be quite terrible (at least without a PC)
        # if state.get('init_delta') is not None:
        #     init_delta = delta_decay * state.get('init_delta')
        # else:
        init_delta = torch.zeros_like(flat_params)

        # Conjugate-Gradient
        deltas, Ms = self._CG(A=A, b=b.neg(), x0=init_delta,
                              max_iter=cg_max_iter, marteens=True)

        # Update parameters
        delta = state['init_delta'] = deltas[-1]
        M = Ms[-1]

        self._modify_params(flat_params + delta)
        loss_now = closure()[0]
        current_evals += 1
        state['func_evals'] += 1

        # Conjugate-Gradient backtracking (Section 20.8.7)
        if verbose:
            print("Original loss: \t{}".format(float(loss_before)))
            print("Loss before bt: {}".format(float(loss_now)))

        for (d, m) in zip(reversed(deltas[:-1]), reversed(Ms[:-1])):
            self._modify_params(flat_params + d)
            loss_prev = closure()[0]
            if float(loss_prev) > float(loss_now):
                break
            delta = d
            M = m
            loss_now = loss_prev

        if verbose:
            print("Loss after bt: \t{}".format(float(loss_now)))

        # The Levenberg-Marquardt Heuristic (Section 20.8.5)
        reduction_ratio = (float(loss_now) -
                           float(loss_before)) / M if M != 0 else 1

        if reduction_ratio < 0.25:
            group['damping'] *= 3 / 2
        elif reduction_ratio > 0.75:
            group['damping'] *= 2 / 3

        if verbose:
            print("Reduction_ratio: {}".format(reduction_ratio))
            print("Damping: {}".format(group['damping']))

        # Line Searching (Section 20.8.8)
        beta = 0.8
        c = 1e-2
        min_improv = min(c * torch.dot(b, delta), 0)

        for _ in range(60):
            if float(loss_now) <= float(loss_before) + alpha * min_improv:
                break

            alpha *= beta
            self._modify_params(flat_params + alpha * delta)
            loss_now = closure()[0]
        else:  # No good update found
            alpha = 0.0
            loss_now = loss_before

        # Update the parameters (this time fo real)
        self._modify_params(flat_params + alpha * delta)

        if verbose:
            print("Final loss: {}".format(float(loss_now)))
            print("Lr: {}".format(alpha), end='\n\n')

        return loss_now

    def _CG(self, A, b, x0, max_iter=50, tol=1e-12, eps=1e-12, marteens=False):
        """
        Minimizes the linear system x^T.A.x - x^T b using the conjugate gradient
        method

        Arguments:
            A (callable): An an abstract linear operator implementing the
                product A.x. A must represent a hermitian, positive definite
                matrix.
            b (torch.Tensor): The vector b.
            x0 (torch.Tensor): An initial guess for x.
            tol (float, optional): Tolerance for convergence.
            marteens (bool, optional): Flag for Marteens' convergence criterion.
        """

        x = [x0]
        r = A(x[0]) - b
        p = -r
        res_i_norm = r @ r

        if marteens:
            m = [0.5 * (r - b) @ x0]

        for i in range(max_iter):
            Ap = A(p)

            alpha = res_i_norm / ((p @ Ap) + eps)

            x.append(x[i] + alpha * p)
            r = r + alpha * Ap

            res_ip1_norm = r @ r

            beta = res_ip1_norm / (res_i_norm + eps)
            res_i_norm = res_ip1_norm

            # Marteens' Relative Progress stopping condition (Section 20.4)
            if marteens:
                m.append(0.5 * A(x[i + 1]) @ x[i + 1] - b @ x[i + 1])

                k = max(10, int(i / 10))
                if i > k:
                    stop = (m[i] - m[i - k]) / (m[i] + eps)
                    if stop < 1e-4:
                        break

            if res_i_norm < tol:
                break

            p = beta * p - r

        return (x, m) if marteens else (x, None)

    def _Hv(self, gradient, vec, damping):
        """
        Computes the Hessian vector product.
        """
        # gg = torch.autograd.grad(gradient, self._params,
        #                          grad_outputs=vec, retain_graph=True)
        # Hv = torch.cat([g.contiguous().view(-1) for g in gg])

        Hv = self._Rop(gradient, self._params, vec)

        return Hv + damping * vec  # Tikhonov damping (Section 20.8.1)

    def _Gv(self, loss, output, vec, damping):
        """
        Computes the generalized Gauss-Newton vector product.
        """
        Jv = self._Rop(output, self._params, vec)

        [gradient] = torch.autograd.grad(loss, output, create_graph=True)
        HJv = self._Rop(gradient, output, Jv)

        JHJv = torch.autograd.grad(
            output, self._params, grad_outputs=HJv, retain_graph=True)

        Gv = torch.cat([j.contiguous().view(-1) for j in JHJv])
        return Gv + damping * vec  # Tikhonov damping (Section 20.8.1)

    def _Rop(self, y, x, vec):
        """
        Computes the product between a Jacobian (dy/dx) and a vector: R-operator
        """
        ws = torch.zeros_like(y).requires_grad_(True)

        gs = torch.autograd.grad(
            y, x, grad_outputs=ws, create_graph=True, retain_graph=True)
        gs = torch.cat([g.flatten() for g in gs], 0)

        [re] = torch.autograd.grad(
            gs, ws, grad_outputs=vec, create_graph=True, retain_graph=True)

        return re
