import torch
import torch.nn as nn

class ShapeBiasAttention(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        # Copy parameters from original attention module
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale
        self.qkv = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop

        # Shape bias parameters
        self.alpha = 1.0
        self.dist_scale = 1.0
        self.limit_penalty_radius_to = None

        # Flag to enable/disable shape bias
        self.apply_shape_bias = False
        self.patch_positions = None
        self.batch_patch_embeddings = None

    def vectorized_shape_bias_penalty(self, patch_positions, batch_patch_embeddings):
        """Vectorized version that computes penalties for all images in batch at once"""
        B = batch_patch_embeddings.shape[0]

        # Reshape to [B*N, D] where N is num_patches, D is embedding_dim
        N = patch_positions.shape[0]
        D = batch_patch_embeddings.shape[2]
        reshaped_embeddings = batch_patch_embeddings.reshape(B * N, D)

        # Compute norms and normalize
        norms = reshaped_embeddings.norm(dim=1, keepdim=True) + 1e-6
        normalized = reshaped_embeddings / norms

        # Reshape back to [B, N, D]
        normalized = normalized.reshape(B, N, D)

        # Distance calculation (only needs to be done once)
        row_diff = patch_positions[:, 0].unsqueeze(1) - patch_positions[:, 0].unsqueeze(0)
        col_diff = patch_positions[:, 1].unsqueeze(1) - patch_positions[:, 1].unsqueeze(0)
        dist_matrix = torch.sqrt(row_diff ** 2 + col_diff ** 2)
        dist_weight = 1.0 / (1.0 + dist_matrix * self.dist_scale)

        # Create storage for all penalties
        all_penalties = torch.zeros(B, N, N, device=patch_positions.device)

        # Compute correlation matrices for each image
        for b in range(B):
            corr_matrix = normalized[b] @ normalized[b].T
            all_penalties[b] = self.alpha * corr_matrix * dist_weight

            # Zero diagonal
            all_penalties[b].diagonal().fill_(0)

        # Apply local window mask if specified
        if self.limit_penalty_radius_to is not None:
            local_window_mask = (dist_matrix <= self.limit_penalty_radius_to)
            # Expand mask for batch dimension and apply
            expanded_mask = local_window_mask.unsqueeze(0).expand(B, N, N)
            all_penalties = all_penalties * (~expanded_mask).float()

        return all_penalties

    def compute_shape_bias_penalty(self, patch_positions, patch_embeddings):
        """Compute shape bias penalty for a single image"""
        alpha_tensor = torch.tensor(self.alpha, device=patch_positions.device, dtype=torch.float32)
        dist_scale_tensor = torch.tensor(self.dist_scale, device=patch_positions.device, dtype=torch.float32)

        # Correlation computation
        norms = patch_embeddings.norm(dim=1, keepdim=True) + 1e-6
        normalized = patch_embeddings / norms
        corr_matrix = normalized @ normalized.T

        # Distance calculation
        row_diff = patch_positions[:, 0].unsqueeze(1) - patch_positions[:, 0].unsqueeze(0)
        col_diff = patch_positions[:, 1].unsqueeze(1) - patch_positions[:, 1].unsqueeze(0)
        dist_matrix = torch.sqrt(row_diff ** 2 + col_diff ** 2)
        dist_weight = 1.0 / (1.0 + dist_matrix * dist_scale_tensor)

        # Combine
        penalty = alpha_tensor * corr_matrix * dist_weight

        # Diagonal = 0
        N = patch_positions.shape[0]
        diag_idx = torch.arange(N, device=patch_positions.device)
        penalty[diag_idx, diag_idx] = 0.0

        # Apply local window mask if specified
        if self.limit_penalty_radius_to is not None:
            local_window_mask = (dist_matrix <= self.limit_penalty_radius_to)
            # Zero out penalties within the local window
            penalty = penalty * (~local_window_mask).float()

        return penalty

    def forward(self, x):
        # print(f"ShapeBiasAttention: x.shape = {x.shape}")
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # # Apply shape bias penalty when specified and patch data is available. UNBATCHED. SLOW.
        # if self.apply_shape_bias and self.patch_positions is not None and self.batch_patch_embeddings is not None:
        #     # print(f"ShapeBiasAttention: Applying shape bias penalty")
        #     print(self.alpha, self.dist_scale, self.limit_penalty_radius_to)
        #
        #     # Number of patches (excluding CLS token)
        #     num_patches = self.patch_positions.shape[0]
        #
        #     # Process each image in the batch
        #     for b in range(B):
        #         # Get patch embeddings for this image
        #         patch_embeddings = self.batch_patch_embeddings[b]
        #
        #         # Compute shape bias penalty for this image
        #         penalty = self.compute_shape_bias_penalty(self.patch_positions, patch_embeddings)
        #
        #         # Create a num_heads x (N-1) x (N-1) penalty matrix for this image
        #         expanded_penalty = penalty.unsqueeze(0).expand(self.num_heads, num_patches, num_patches)
        #
        #         # Fill in the non-CLS token part with the penalty
        #         # Skip CLS token (at position 0) in the attention matrix
        #         for h in range(self.num_heads):
        #             attn[b, h, 1:, 1:] = attn[b, h, 1:, 1:] - expanded_penalty[h]

        # BATCHED.FAST.
        if self.apply_shape_bias and self.patch_positions is not None and self.batch_patch_embeddings is not None:
            # print(self.alpha, self.dist_scale, self.limit_penalty_radius_to)
            # Compute penalties for all images
            all_penalties = self.vectorized_shape_bias_penalty(self.patch_positions, self.batch_patch_embeddings)

            # Expand penalties for all attention heads
            expanded_penalties = all_penalties.unsqueeze(1).expand(B, self.num_heads, -1, -1)

            # Create full penalty matrix including CLS token
            full_penalty = torch.zeros_like(attn)
            full_penalty[:, :, 1:, 1:] = expanded_penalties

            # Apply penalty
            attn = attn - full_penalty

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x