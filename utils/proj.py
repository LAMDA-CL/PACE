import torch
import torch.nn as nn


class CrossAttentionProjector:

    def __init__(self, sel_attn, alpha=0.9, threshold=0.95, device="cuda"):
        self.sel_attn = sel_attn
        self.alpha = alpha
        self.threshold = threshold
        self.device = device

        self.params = {
            "w_qs": sel_attn.w_qs.weight,
            "w_ks": sel_attn.w_ks.weight,
            "w_vs": sel_attn.w_vs.weight,
            # "fc_w": sel_attn.fc.weight,
            # "fc_b": sel_attn.fc.bias,
            # "ln_w": sel_attn.layer_norm.weight,
            # "ln_b": sel_attn.layer_norm.bias,
        }

        # 缓存旧任务权重
        self.W_old = {}
        self.P_old = {}

        for name, p in self.params.items():
            if p.dim() == 2:
                self.W_old[name] = p.detach().clone().T.to(device)
            elif p.dim() == 1:
                self.W_old[name] = p.detach().clone().to(device)

    @torch.no_grad()
    def store_old_subspace(self):
        for name, W in self.W_old.items():

            if W.dim() == 2:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                energy = S ** 2
                cum_energy = torch.cumsum(energy, dim=0)
                total_energy = cum_energy[-1]

                rank = torch.searchsorted(cum_energy / total_energy,
                                          self.threshold).item() + 1

                rank = max(1, min(rank, U.shape[1]))
                U_old = U[:, :rank]

                self.P_old[name] = U_old @ U_old.T

            elif W.dim() == 1:
                w = W.unsqueeze(1)  # [d,1]
                norm = torch.norm(w)
                if norm > 0:
                    u = w / norm
                    self.P_old[name] = u @ u.T
                else:
                    self.P_old[name] = torch.zeros((W.numel(), W.numel()), device=self.device)

    @torch.no_grad()
    def project_new_weights(self):

        for name, p in self.params.items():

            P = self.P_old[name]


            if p.dim() == 2:
                W_new = p.detach().clone().T  # [d, d]

                I = torch.eye(W_new.shape[0], device=self.device)

                W_proj = self.alpha * (P @ W_new) + (1 - self.alpha) * ((I - P) @ W_new)

                p.copy_(W_proj.T)
                self.W_old[name] = W_proj.clone()
            elif p.dim() == 1:
                w_new = p.detach().clone()
                dim = w_new.shape[0]
                I = torch.eye(dim, device=self.device)

                w_proj = self.alpha * (P @ w_new) + (1 - self.alpha) * ((I - P) @ w_new)

                p.copy_(w_proj)
                self.W_old[name] = w_proj.clone()
