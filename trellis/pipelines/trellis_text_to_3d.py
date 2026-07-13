from typing import *
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.
    Supports both standard TRELLIS and LATO-enhanced decoders.
    """

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """Load a pretrained model."""
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline

    def _init_text_cond_model(self, name: str):
        """Initialize the text conditioning model."""
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode the text."""
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        return embeddings

    def get_cond(self, prompt: List[str]) -> dict:
        """Get the conditioning information for the model."""
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """Sample sparse structures with the given conditioning."""
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        return coords

    def decode_slat_lato(
        self,
        slat: sp.SparseTensor,
    ) -> dict:
        """
        Decode structured latent using LATO VoxelVAE.
        This is the LATO-enhanced decode path.

        Converts TRELLIS SparseTensor → LATO SparseTensor before calling
        LATO's VoxelVAE.decode(), since the two libraries use different
        sparse tensor implementations.
        """
        ret = {}
        if 'lato_vae' in self.models:
            # Convert TRELLIS SparseTensor → LATO SparseTensor
            # (they are different classes from different packages)
            try:
                from lato.modules.sparse import SparseTensor as LATOSparseTensor
            except ImportError:
                # LATO not installed; try direct pass (may fail)
                LATOSparseTensor = None

            if LATOSparseTensor is not None:
                lato_slat = LATOSparseTensor(
                    feats=slat.feats.contiguous(),
                    coords=slat.coords.contiguous(),
                )
            else:
                lato_slat = slat

            # LATO decode: training=False for inference branch
            decoded = self.models['lato_vae'].decode(
                lato_slat,
                training=False,
                inference_threshold=0.2
            )
            ret['lato_decoded'] = decoded
        else:
            # Fallback to mesh decoder if lato_vae not available
            if 'slat_decoder_mesh' in self.models:
                ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        return ret

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh'],
    ) -> dict:
        """
        Decode the structured latent.
        
        Modified for LATO integration: defaults to only 'mesh' format.
        If 'lato_vae' is available in models, uses LATO decode path.
        """
        ret = {}
        
        # Check if LATO VAE is available
        if 'lato_vae' in self.models and 'mesh' in formats:
            return self.decode_slat_lato(slat)
        
        # Original TRELLIS decode paths
        if 'mesh' in formats and 'slat_decoder_mesh' in self.models:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats and 'slat_decoder_gs' in self.models:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats and 'slat_decoder_rf' in self.models:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        
        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Note: coords should already be at the correct resolution.
        For LATO, this is 128 (after upsampling from 64).
        """
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
    ) -> dict:
        """
        Run the pipeline.
        
        LATO-enhanced: coords are upsampled from 64 to 128 for LATO SLat Flow.
        """
        cond = self.get_cond([prompt])
        torch.manual_seed(seed)
        
        # 1. Generate sparse structure (coords at resolution 64)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        
        # 2. Upsample coords from 64 to 128 for LATO compatibility
        #    This is the key addition for LATO integration
        coords = coords * 2
        
        # 3. Generate SLAT using LATO-compatible flow model
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        
        # 4. Decode to final format
        return self.decode_slat(slat, formats)

    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """Voxelize a mesh."""
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1/64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5)
        )
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
