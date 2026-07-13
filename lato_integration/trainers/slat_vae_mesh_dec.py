"""
Enhanced SLat VAE Mesh Decoder Trainer with pruning loss and edge prediction.

Inherits from TRELLIS's SLatVaeMeshDecoderTrainer and adds:
1. Occupancy pruning loss from decoder's intermediate predictions
2. Edge prediction loss from ConnectionHead
3. Latent context passing for cross-attention
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from trellis.trainers.vae.structured_latent_vae_mesh_dec import (
    SLatVaeMeshDecoderTrainer as _SLatVaeMeshDecoderTrainer,
)
from trellis.modules.sparse import SparseTensor
from trellis.representations import MeshExtractResult


class EnhancedSLatVaeMeshDecoderTrainer(_SLatVaeMeshDecoderTrainer):
    """
    Enhanced SLat VAE Mesh Decoder Trainer with pruning + edge losses.

    Key enhancements over the original trainer:
    1. Passes latent context to decoder for cross-attention
    2. Adds occupancy pruning loss from intermediate decoder layers
    3. Adds edge prediction loss for mesh topology

    Args:
        (same as SLatVaeMeshDecoderTrainer)
        lambda_pruning: Weight for occupancy pruning BCE loss.
        lambda_edge: Weight for edge prediction BCE loss.
    """

    def __init__(
        self,
        *args,
        lambda_pruning: float = 0.01,
        lambda_edge: float = 0.01,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_pruning = lambda_pruning
        self.lambda_edge = lambda_edge

    def training_losses(
        self,
        latents: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        mesh: List[Dict],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        normal_map: torch.Tensor = None,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses with pruning and edge supervision.

        Args:
            latents: The [N x * x C] sparse latents.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            mesh: The list of dictionaries of GT meshes.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            normal_map: Optional normal map for color training.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
        """
        terms = edict(loss=0.0, rec=0.0)

        # Forward through enhanced decoder with pruning data
        decoder = self.training_models['decoder']
        has_cross_attn = hasattr(decoder, 'use_cross_attn') and decoder.use_cross_attn
        has_pruning = hasattr(decoder, 'use_pruning') and decoder.use_pruning

        if has_pruning:
            reps, pruning_data = decoder(
                latents,
                original_latent=latents if has_cross_attn else None,
                return_pruning=True,
            )
            # Compute pruning loss against ground-truth occupancy at each level
            # pruning_data contains occupancy predictions at each subdivision level
            for i, occ_prob in enumerate(pruning_data):
                if occ_prob is not None:
                    # For now, apply a sparsity regularization on occupancy
                    # In full training, this would compare against GT vertex voxels
                    pruning_loss = torch.sigmoid(occ_prob.feats).mean()
                    terms[f"pruning_{i}"] = pruning_loss
                    terms["loss"] = terms["loss"] + self.lambda_pruning * pruning_loss
        elif has_cross_attn:
            reps = decoder(latents, original_latent=latents)
        else:
            reps = decoder(latents)

        self.renderer.rendering_options.resolution = image.shape[-1]

        # Regularization loss from mesh extraction
        terms['reg_loss'] = sum([rep.reg_loss for rep in reps]) / len(reps)
        terms['loss'] = terms['loss'] + terms['reg_loss']

        # Geometry losses (mask, depth, normal)
        geo_terms = self.geometry_losses(reps, mesh, normal_map, extrinsics, intrinsics)
        terms.update(geo_terms)
        terms['loss'] = terms['loss'] + terms['geo_loss']

        # Color losses
        if self.use_color:
            color_terms = self.color_losses(reps, image, alpha, extrinsics, intrinsics)
            terms.update(color_terms)
            terms['loss'] = terms['loss'] + terms['color_loss']

        # Edge prediction loss using ConnectionHead
        if (
            hasattr(decoder, 'use_edge_pred')
            and decoder.use_edge_pred
            and self.lambda_edge > 0
        ):
            edge_loss = self._compute_edge_loss(reps, mesh)
            if edge_loss is not None:
                terms["edge"] = edge_loss
                terms["loss"] = terms["loss"] + self.lambda_edge * edge_loss

        return terms, {}

    def _compute_edge_loss(
        self,
        reps: List[MeshExtractResult],
        gt_meshes: List[Dict],
    ) -> torch.Tensor:
        """
        Compute edge prediction loss between predicted and GT mesh edges.

        Uses the decoder's ConnectionHead to predict edge probabilities
        and compares against GT edges extracted from mesh faces.

        Args:
            reps: List of predicted mesh extraction results.
            gt_meshes: List of GT mesh dicts with 'vertices' and 'faces'.

        Returns:
            Edge prediction BCE loss, or None if not computable.
        """
        decoder = self.training_models['decoder']
        losses = []

        for i, rep in enumerate(reps):
            if not rep.success or 'vertices' not in gt_meshes[i]:
                continue

            gt_verts = gt_meshes[i]['vertices'].to(self.device)
            gt_faces = gt_meshes[i]['faces'].to(self.device)

            # Build GT edge set from faces
            edges_set = set()
            for face in gt_faces:
                for j in range(3):
                    u, v = face[j].item(), face[(j + 1) % 3].item()
                    edges_set.add((min(u, v), max(u, v)))

            if len(edges_set) == 0:
                continue

            # Sample candidate edges (for efficiency, use k-NN neighbors)
            # This is a simplified version; full implementation would use
            # the same candidate generation as LATO's predict_edges()
            if hasattr(rep, 'vertices') and rep.vertices is not None:
                pred_verts = rep.vertices
                if pred_verts.shape[0] < 2:
                    continue

                # Sample a subset of vertex pairs as candidates
                num_verts = min(pred_verts.shape[0], 1000)
                indices = torch.randperm(pred_verts.shape[0])[:num_verts]
                candidates = torch.combinations(indices, r=2)

                if candidates.shape[0] == 0:
                    continue

                # Get vertex features for edge prediction
                if hasattr(rep, 'vertex_features') and rep.vertex_features is not None:
                    vertex_feats = rep.vertex_features
                else:
                    # Fall back to positional features
                    vertex_feats = pred_verts

                edge_probs = decoder.predict_edges(vertex_feats, candidates)
                edge_probs = torch.clamp(edge_probs, 1e-6, 1 - 1e-6)

                # Build GT labels for candidate edges
                gt_labels = torch.zeros(candidates.shape[0], device=self.device)
                for j, (u, v) in enumerate(candidates):
                    orig_u = indices[u].item()
                    orig_v = indices[v].item()
                    key = (min(orig_u, orig_v), max(orig_u, orig_v))
                    if key in edges_set:
                        gt_labels[j] = 1.0

                # BCE loss
                edge_loss = F.binary_cross_entropy(edge_probs, gt_labels)
                losses.append(edge_loss)

        if losses:
            return torch.stack(losses).mean()
        return None
