import torch
import json
from math import log
from typing import Dict, List, Tuple, Optional
from .objectives import *
from fem_placer.config import PlaceType, GridType
from fem_placer.logger import INFO


def entropy_q(p):
    """
    Calculate entropy for q-dimensional probability distributions.

    Args:
        p: Probabilities [batch, N, q]

    Returns:
        Entropy values [batch]
    """
    return -(p * torch.log(p)).sum(2).sum(1)


def _build_region_distances(
    region_site_coords: Dict[str, torch.Tensor],
    distance_metric: str = 'manhattan',
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Build distance matrices for all region pairs."""
    dists: Dict[str, Dict[str, torch.Tensor]] = {}
    for rA, coordsA in region_site_coords.items():
        dists[rA] = {}
        for rB, coordsB in region_site_coords.items():
            xi = coordsA[:, 0].unsqueeze(1)
            xj = coordsB[:, 0].unsqueeze(0)
            yi = coordsA[:, 1].unsqueeze(1)
            yj = coordsB[:, 1].unsqueeze(0)
            d = torch.abs(xi - xj) + torch.abs(yi - yj)
            if distance_metric == 'sqrt_manhattan':
                d = torch.sqrt(d)
            dists[rA][rB] = d
    return dists


class FPGAPlacementOptimizer:
    """
    FPGA Placement optimizer using QUBO formulation with site coordinates.

    Supports N regions (logic, io, dsp, clock, ...) via dict-based
    configuration.  Coefficients and noise factors are given as lists
    that map positionally to ``regions``.
    """

    def __init__(
            self,
            regions: List[str],
            region_sizes: Dict[str, Tuple[int, int]],
            region_coupling: Dict[str, Dict[str, torch.Tensor]],
            region_site_coords: Dict[str, torch.Tensor],
            constraint_coeffs: Optional[List[float]] = None,
            h_factors: Optional[List[float]] = None,
            # --- Shared params ---
            num_trials: int = 10,
            num_steps: int = 1000,
            dev: str = 'cpu',
            betamin: float = 0.01,
            betamax: float = 0.5,
            anneal: str = 'inverse',
            optimizer: str = 'adam',
            learning_rate: float = 0.1,
            seed: int = 1,
            dtype: torch.dtype = torch.float32,
            manual_grad: bool = False,
            distance_metric: str = 'manhattan',
        ):
        self.regions = list(regions)
        self.region_sizes = region_sizes
        self.region_coupling = region_coupling
        self.region_site_coords = region_site_coords

        # Convert positional lists to region-keyed dicts
        if constraint_coeffs is not None:
            self.constraint_coeffs = dict(zip(self.regions, constraint_coeffs))
        else:
            self.constraint_coeffs = {r: 1.0 for r in self.regions}

        if h_factors is not None:
            self.h_factors = dict(zip(self.regions, h_factors))
        else:
            self.h_factors = {r: 0.01 for r in self.regions}

        self.num_trials = num_trials
        self.num_steps = num_steps
        self.dev = dev
        self.dtype = dtype
        self.seed = seed
        self.manual_grad = manual_grad
        self.distance_metric = distance_metric.lower()
        if self.distance_metric not in ('manhattan', 'sqrt_manhattan'):
            raise ValueError("distance_metric must be 'manhattan' or 'sqrt_manhattan'")

        # --- Annealing schedule ---
        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax), num_steps))
        elif anneal == 'inverse':
            betas = 1 / torch.linspace(betamax, betamin, num_steps)
        self.betas = betas.to(self.dtype).to(self.dev)

        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # --- Distance matrices (built on first use) ---
        self._region_distances: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    @classmethod
    def from_saved_params(cls, params_file: str, **kwargs):
        """
        Create an FPGAPlacementOptimizer from saved parameters in a JSON file.

        Args:
            params_file: Path to the JSON file containing saved parameters
            **kwargs: Additional arguments to override saved parameters

        Returns:
            FPGAPlacementOptimizer instance
        """
        with open(params_file, 'r') as f:
            data = json.load(f)

        params = data['params']
        tensors = data['tensors']

        target_dev = kwargs.get('dev', params.get('device', 'cpu'))

        # Rebuild N-region dicts from the saved legacy scalars
        coupling_matrix = torch.tensor(tensors['coupling_matrix'], dtype=torch.float32, device=target_dev) if 'coupling_matrix' in tensors else None
        site_coords_matrix = torch.tensor(tensors['site_coords_matrix'], dtype=torch.float32, device=target_dev) if 'site_coords_matrix' in tensors else None
        io_site_connect_matrix = torch.tensor(tensors['io_site_connect_matrix'], dtype=torch.float32, device=target_dev) if 'io_site_connect_matrix' in tensors else None
        io_site_coords = torch.tensor(tensors['io_site_coords'], dtype=torch.float32, device=target_dev) if 'io_site_coords' in tensors else None

        regions = ['logic']
        region_sizes = {'logic': (params['num_inst'], params['num_site'])}
        region_coupling = {'logic': {'logic': coupling_matrix}}
        region_site_coords = {'logic': site_coords_matrix}
        constraint_coeffs = [params.get('constraint_alpha', 1.0)]
        h_factors = [params.get('h_factor', 0.01)]

        if params.get('with_io', False) and io_site_coords is not None and io_site_connect_matrix is not None:
            regions.append('io')
            region_sizes['io'] = (params.get('num_fixed_inst', 0), params.get('num_fixed_site', 0))
            region_coupling['logic']['io'] = io_site_connect_matrix
            region_coupling['io'] = {'logic': io_site_connect_matrix.T.clone()}
            region_coupling['io']['io'] = None
            region_site_coords['io'] = io_site_coords
            constraint_coeffs.append(params.get('constraint_beta', 0.0))
            h_factors.append(params.get('io_factor', 1.0) * params.get('h_factor', 0.01))

        default_kwargs = dict(
            regions=regions,
            region_sizes=region_sizes,
            region_coupling=region_coupling,
            region_site_coords=region_site_coords,
            constraint_coeffs=constraint_coeffs,
            h_factors=h_factors,
            dev=target_dev,
            dtype=torch.float32,
            distance_metric=params.get('distance_metric', 'manhattan'),
        )
        default_kwargs.update(kwargs)
        return cls(**default_kwargs)

    def _apply_distance_metric(self, distance_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the configured distance metric transformation."""
        if self.distance_metric == 'sqrt_manhattan':
            return torch.sqrt(distance_tensor)
        return distance_tensor

    def _ensure_distances(self):
        """Build distance matrices if not already built."""
        if self._region_distances is None:
            self._region_distances = _build_region_distances(
                self.region_site_coords, self.distance_metric,
            )

    def _initialize(self):
        """
        Initialise random latent variables h_r for every region,
        and pre-compute all distance matrices.

        Returns:
            Dict[r] → h tensor [num_trials, N_r, M_r]
        """
        torch.manual_seed(self.seed)
        self._ensure_distances()

        hs = {}
        for r in self.regions:
            n_inst, n_site = self.region_sizes[r]
            h = self.h_factors[r] * torch.randn(
                [self.num_trials, n_inst, n_site],
                device=self.dev, dtype=self.dtype,
            )
            if not self.manual_grad:
                h.requires_grad = True
            hs[r] = h

        return hs

    def _setup_optimizer(self, params):
        """Set up the torch optimizer."""
        if self.optimizer == 'adam':
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(
                params, lr=self.learning_rate, alpha=0.98, eps=1e-08,
                weight_decay=0.01, momentum=0.91, centered=False
            )
        else:
            raise ValueError("Unknown optimizer, valid choices are ['adam', 'rmsprop'].")

    def iterate_placement(self):
        """
        Run the full optimisation loop over N regions.

        Returns:
            Dict[r] → softmax probability tensor [num_trials, N_r, M_r]
        """
        import time as _time
        _t_prep = _time.time()
        hs = self._initialize()
        opt = self._setup_optimizer(list(hs.values()))
        _prep_time = _time.time() - _t_prep
        INFO(f"Data preparation (distances + init) took {_prep_time:.2f}s")

        # CUDA memory tracking
        cuda_available = self.dev != 'cpu' and torch.cuda.is_available()
        mem_log = []
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
            mem_initial = torch.cuda.memory_allocated()
            INFO(f"CUDA memory before optimization: {mem_initial / 1024**2:.2f} MB")

        _t_loop = _time.time()
        for step in range(self.num_steps):
            ps = {r: torch.softmax(hs[r], dim=2) for r in self.regions}
            opt.zero_grad()

            loss = expected_fpga_placement_with_regions(
                region_ps=ps,
                region_coupling=self.region_coupling,
                region_distances=self._region_distances,
                alphas=self.constraint_coeffs,
            )

            total_entropy = sum(entropy_q(ps[r]) for r in self.regions)
            free_energy = loss - total_entropy / self.betas[step]
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

            if cuda_available:
                mem_log.append(torch.cuda.memory_allocated())

        _loop_time = _time.time() - _t_loop
        INFO(f"Iteration loop ({self.num_steps} steps) took {_loop_time:.2f}s")

        if cuda_available and mem_log:
            mem_tensor = torch.tensor(mem_log, device='cpu', dtype=torch.float64)
            mem_avg = mem_tensor.mean().item()
            mem_max = torch.cuda.max_memory_allocated()
            INFO(f"CUDA memory report — max: {mem_max / 1024**2:.2f} MB, "
                 f"avg: {mem_avg / 1024**2:.2f} MB, "
                 f"peak over initial: {(mem_max - mem_initial) / 1024**2:.2f} MB")

        return ps

    def optimize(self):
        """
        Run optimisation and infer final placements.

        Returns:
            (config, result) where
              config is a Dict[r] → instance coords [num_trials, N_r, 2]
              result is HPWL values [num_trials]
        """
        import time as _time
        ps = self.iterate_placement()
        _t_infer = _time.time()
        config, result = infer_placements_with_regions(
            region_ps=ps,
            region_site_coords=self.region_site_coords,
            region_coupling=self.region_coupling,
            region_distances=self._region_distances,
        )
        INFO(f"Placement inference took {_time.time() - _t_infer:.2f}s")
        return config, result
