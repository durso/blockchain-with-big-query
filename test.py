# --- SAS (Sinh–Arcsinh) — NLL block ---

if len(self.schema["sas_idx"]) > 0:
    xi = x[:, self.schema["sas_idx"]]

    sigma = F.softplus(outputs["sas_sigma"]) + eps      # >0
    tail  = F.softplus(outputs["sas_tail"])  + eps      # >0 (tail-weight)
    skew  = outputs["sas_skew"]                       
    mu    = outputs["sas_mu"]                           # location; keep unconstrained

    two_pi = torch.tensor(2.0 * torch.pi, dtype=xi.dtype, device=xi.device)

    w = (xi - mu) / (sigma + eps)
    y = tail * torch.asinh(w) - skew
    z = torch.sinh(y)

    log_pz = -0.5 * (z.pow(2) + torch.log(two_pi))
    log_abs_det = (
        torch.log(torch.cosh(y) + eps)
        + torch.log(tail + eps)
        - 0.5 * torch.log1p(w.pow(2))
        - torch.log(sigma + eps)
    )
    log_sas = log_pz + log_abs_det
    nll_sas = -log_sas

    m = masks[:, self.schema["sas_idx"]] if masks is not None else None
    per_dim, total = self.masked_mean_per_dim(nll_sas, m)
    loss += total
    breakdown["sas"] = {
        "total": float(total.item()),
        "per_dim_mean": per_dim.detach().cpu(),
        "col_idx": self.schema["sas_idx"],
    }

# --- LogNormal — NLL block ---

if len(self.schema["lognorm_idx"]) > 0:
    xi = x[:, self.schema["lognorm_idx"]]

    # Parameters of log X ~ Normal(mu, sigma^2)
    mu     = outputs["lognorm_mu"]                    
    sigma  = F.softplus(outputs["lognorm_sigma"]) + eps  # >0

    # log p(x) = -log x - log sigma - 0.5*log(2π) - ((log x - mu)^2) / (2*sigma^2)
    two_pi = torch.tensor(2.0 * torch.pi, dtype=xi.dtype, device=xi.device)
    log_x  = torch.log(xi + eps)
    quad   = (log_x - mu).pow(2) / (2.0 * sigma.pow(2))

    log_lognorm = -torch.log(xi + eps) - torch.log(sigma + eps) - 0.5 * torch.log(two_pi) - quad
    nll_lognorm = -log_lognorm

    m = masks[:, self.schema["lognorm_idx"]] if masks is not None else None
    per_dim, total = self.masked_mean_per_dim(nll_lognorm, m)
    loss += total
    breakdown["lognorm"] = {
        "total": float(total.item()),
        "per_dim_mean": per_dim.detach().cpu(),
        "col_idx": self.schema["lognorm_idx"],
    }
# --- Laplace (Double Exponential) — NLL block ---

if len(self.schema["laplace_idx"]) > 0:
    xi = x[:, self.schema["laplace_idx"]]

    mu    = outputs["laplace_mu"]                    
    b     = F.softplus(outputs["laplace_b"]) + eps        # scale > 0  (a.k.a. beta)

    # log p(x) = -log(2b) - |x - mu| / b
    log_laplace = -torch.log(2.0 * b) - torch.abs(xi - mu) / b
    nll_laplace = -log_laplace

    m = masks[:, self.schema["laplace_idx"]] if masks is not None else None
    per_dim, total = self.masked_mean_per_dim(nll_laplace, m)
    loss += total
    breakdown["laplace"] = {
        "total": float(total.item()),
        "per_dim_mean": per_dim.detach().cpu(),
        "col_idx": self.schema["laplace_idx"],
    }






# --- SAS (Sinh–Arcsinh) — sampling block (synthetic generation) ---

# Params (same transforms as in NLL)
sigma = F.softplus(outputs["sas_sigma"]) + eps
tail  = F.softplus(outputs["sas_tail"])  + eps
skew  = outputs["sas_skew"]
mu    = outputs["sas_mu"]                 # or F.softplus(...) + eps 

# Reparameterized sample:
# z ~ N(0,1); y = asinh(z); u = (y + skew)/tail; w = sinh(u); x = mu + sigma*w
z = torch.randn_like(mu)
y = torch.asinh(z)
u = (y + skew) / (tail + eps)
w = torch.sinh(u)
x_sas = mu + sigma * w


x_syn[:, self.schema["sas_idx"]] = x_sas






# --- LogNormal — sampling block (synthetic generation) ---

mu    = outputs["lognorm_mu"]
sigma = F.softplus(outputs["lognorm_sigma"]) + eps

# Sample: y ~ Normal(mu, sigma^2), x = exp(y)
z = torch.randn_like(mu)
y = mu + sigma * z
x_lognorm = torch.exp(y)

x_syn[:, self.schema["lognorm_idx"]] = x_lognorm





# --- Laplace (Double Exponential) — sampling block (synthetic generation) ---

mu = outputs["laplace_mu"]
b  = F.softplus(outputs["laplace_b"]) + eps

u = torch.rand_like(mu) - 0.5
x_laplace = mu - b * torch.sign(u) * torch.log1p(-2.0 * torch.abs(u) + eps)

x_syn[:, self.schema["laplace_idx"]] = x_laplace
