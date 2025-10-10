def scale_numeric_features(self, X, fit=False):
    """
    Scales only the numeric columns defined in self.schema["num_idx"].
    - If fit=True, fits the scaler and then transforms.
    - If fit=False, only transforms using the existing fitted scaler.
    
    Stores the scaler inside self.num_scaler.
    
    Parameters
    ----------
    X : np.ndarray
        The input data matrix (already including one-hot/binary columns).
    fit : bool
        Whether to fit the scaler (typically True for training, False for inference).
    
    Returns
    -------
    X_scaled : np.ndarray
        Copy of X with numeric columns scaled, others untouched.
    """
    if not hasattr(self, "num_scaler"):
        self.num_scaler = StandardScaler()

    X = np.asarray(X, dtype=np.float32, copy=True)

    num_idx = self.schema.get("num_idx", [])
    if len(num_idx) == 0:
        return X  # nothing to scale

    if fit:
        self.num_scaler.fit(X[:, num_idx])

    X[:, num_idx] = self.num_scaler.transform(X[:, num_idx]).astype(np.float32)
    return X

def _build_eval_masks(self, X_tensor):
    """Create masks for a full (N,D) tensor using self.schema."""
    schema = self.schema
    masks = {"num": None, "bin": None, "cat": []}
    if schema.get("num_idx"):
        masks["num"] = (~torch.isnan(X_tensor[:, schema["num_idx"]])).float()
    if schema.get("bin_idx"):
        masks["bin"] = (~torch.isnan(X_tensor[:, schema["bin_idx"]])).float()
    for col in schema.get("cat", []):
        idx = col["idx"]
        oh = X_tensor[:, idx]
        observed = (~torch.isnan(oh)).any(dim=1) & (oh.sum(dim=1) != 0)
        masks["cat"].append(observed.float().unsqueeze(1))
    return masks

def calculate_combined_score(self, X, labels, autoencoder=None):
    model = autoencoder if autoencoder is not None else self.autoencoder
    model.eval()

    with torch.no_grad():
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        outputs, mu, logvar, z = model(X_tensor)

        # ---- reconstruction loss using tabular NLL ----
        if isinstance(outputs, dict) and hasattr(self, "_tabular_nll_dict"):
            masks = self._build_eval_masks(X_tensor)
            recon_loss = self._tabular_nll_dict(outputs, X_tensor, self.schema, masks=masks)
        elif hasattr(self, "feat_dists") and hasattr(self, "_tabular_nll_flat"):
            # flat-parameter decoder path (no per-feature mask here; add if you maintain one)
            recon_loss = self._tabular_nll_flat(outputs, X_tensor, self.feat_dists, imp_mask=None)
        else:
            # fallback (not ideal for tabular, but keeps backward compat)
            recon_loss = torch.nn.functional.mse_loss(outputs, X_tensor, reduction="mean")

        # Bound into [0, 1) like your old 1/(1 + MSE)
        reconstruction_score = 1.0 / (1.0 + recon_loss.item())

        gt_metrics = self.evaluate_against_ground_truth(X, labels)

    if gt_metrics and self.labeled_indices is not None:
        combined_score = (
            0.05 * reconstruction_score
            + 0.5 * gt_metrics.get("adjusted_rand_score", 0)
            + 0.45 * gt_metrics.get("normalized_mutual_info_score", 0)
        )
    else:
        combined_score = reconstruction_score

    return combined_score
