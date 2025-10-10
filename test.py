def create_data_loader(X, schema, batch_size=32, shuffle=True):
    """
    X: numpy array or tensor of shape (N, D), already preprocessed & concatenated
    schema: {
        "num_idx": [int, ...],
        "bin_idx": [int, ...],
        "cat": [ {"idx": [int, ...], "K": int}, ... ]
    }
    Returns batches: (x, masks)
      - x: (B, D) float32
      - masks: {"num": (B, D_num) or None,
                "bin": (B, D_bin) or None,
                "cat": [ (B,1), ... ]}  # one per categorical column
    """
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)

    # ---- Build full-dataset masks (NaN = missing; for one-hot, missing if all-NaN OR all-zero) ----
    def build_masks(X, schema):
        masks = {"num": None, "bin": None, "cat": []}

        if schema.get("num_idx"):
            m = ~torch.isnan(X[:, schema["num_idx"]])
            masks["num"] = m.float()

        if schema.get("bin_idx"):
            m = ~torch.isnan(X[:, schema["bin_idx"]])
            masks["bin"] = m.float()

        for col in schema.get("cat", []):
            idx = col["idx"]
            oh = X[:, idx]
            # observed if any non-NaN AND not all zeros
            observed = (~torch.isnan(oh)).any(dim=1) & (oh.sum(dim=1) != 0)
            masks["cat"].append(observed.float().unsqueeze(1))  # (N,1)

        return masks

    full_masks = build_masks(X, schema)

    class TabularVAEDataset(Dataset):
        def __init__(self, X, masks):
            self.X = X
            self.masks = masks
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, i):
            m = {"num": None, "bin": None, "cat": []}
            if self.masks["num"] is not None: m["num"] = self.masks["num"][i]
            if self.masks["bin"] is not None: m["bin"] = self.masks["bin"][i]
            for j in range(len(self.masks["cat"])):
                m["cat"].append(self.masks["cat"][j][i])   # (1,)
            return self.X[i], m

    # Collate masks into properly stacked tensors
    def collate_fn(batch):
        xs, ms = zip(*batch)
        x = torch.stack(xs, dim=0)

        out = {"num": None, "bin": None, "cat": []}
        if ms[0]["num"] is not None:
            out["num"] = torch.stack([m["num"] for m in ms], dim=0)  # (B, D_num)
        if ms[0]["bin"] is not None:
            out["bin"] = torch.stack([m["bin"] for m in ms], dim=0)  # (B, D_bin)
        if len(ms[0]["cat"]) > 0:
            for j in range(len(ms[0]["cat"])):
                out["cat"].append(torch.stack([m["cat"][j] for m in ms], dim=0))  # (B,1)
        return x, out

    ds = TabularVAEDataset(X, full_masks)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)





for batch_x, batch_masks in train_loader:
    batch_x = batch_x.to(self.device)
    # forward
    outputs, mu, logvar, z = autoencoder(batch_x)
    # mixed-type NLL (the version I sent earlier)
    recon_loss = tabular_nll(outputs, batch_x, schema, masks=batch_masks)
    kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / batch_x.size(0)
    loss = recon_loss + beta * kl_loss
