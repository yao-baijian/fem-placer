import torch
import json
from math import log
from typing import Tuple
from .objectives import *
from fem_placer.config import PlaceType, GridType
from fem_placer.logger import INFO

def entropy_q(p):
    """
    Calculate entropy for q-dimensional probability distributions.
    NOTE: Matches master branch exactly - no epsilon added.

    Args:
        p: Probabilities [batch, N, q]

    Returns:
        Entropy values [batch]
    """
    return -(p * torch.log(p)).sum(2).sum(1)


class FPGAPlacementOptimizer:
    """
    FPGA Placement optimizer using QUBO formulation with site coordinates.

    This matches the algorithm from the master branch, using pre-computed
    site coordinate matrices for HPWL calculation.

    """

    def __init__(
            self,
            num_inst: int,
            num_fixed_inst: int,
            num_site: int,
            num_fixed_site: int,
            coupling_matrix: torch.Tensor,
            site_coords_matrix: torch.Tensor,
            io_site_connect_matrix: torch.Tensor = None,
            io_site_coords: torch.Tensor = None,
            constraint_alpha: float = 1.0,
            constraint_beta: float = 1.0,
            num_trials: int = 10,
            num_steps: int = 1000,
            dev: str = 'cpu',
            betamin: float = 0.01,
            betamax: float = 0.5,
            anneal: str = 'inverse',
            optimizer: str = 'adam',
            learning_rate: float = 0.1,
            h_factor: float = 0.01,
            io_factor: float = 1.0,
            seed: int = 1,
            dtype: torch.dtype = torch.float32,
            with_io: bool = False,
            manual_grad: bool = False,
            distance_metric: str = 'manhattan',
        ):
        """
        Initialize the FPGA placement optimizer with QUBO formulation.

        Args:
            num_inst: Number of instances to place
            num_site: Number of available placement sites
            coupling_matrix: Instance connectivity matrix [num_inst, num_inst]
            site_coords_matrix: Site coordinates [num_site, 2]
            drawer: Optional PlacementDrawer for visualization
            visualization_steps: Steps at which to visualize
            constraint_alpha: Weight for constraint loss (alpha parameter)
        """
        self.num_inst = num_inst
        self.fixed_insts_num = num_fixed_inst
        self.num_site = num_site
        self.fixed_site_num = num_fixed_site
        self.coupling_matrix = coupling_matrix
        self.logic_site_coords = site_coords_matrix
        self.io_site_connect_matrix = io_site_connect_matrix
        self.io_site_coords = io_site_coords
        self.constraint_alpha = constraint_alpha
        self.constraint_beta = constraint_beta
        
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.dev = dev
        self.dtype = dtype

        if anneal == 'lin':
            betas = torch.linspace(betamin, betamax, num_steps)
        elif anneal == 'exp':
            betas = torch.exp(torch.linspace(log(betamin), log(betamax),num_steps))
        elif anneal == 'inverse':
            betas = 1 / torch.linspace(betamax, betamin, num_steps)
        self.betas = betas.to(self.dtype).to(self.dev) 
        
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.h_factor = h_factor
        self.io_factor = io_factor
        self.seed = seed
        self.with_io = with_io
        self.manual_grad = manual_grad
        metric = distance_metric.lower()
        if metric not in ('manhattan', 'sqrt_manhattan'):
            raise ValueError("distance_metric must be either 'manhattan' or 'sqrt_manhattan'")
        self.distance_metric = metric

        self.D = None

        self.D_LL = None
        self.D_LI = None

    @classmethod
    def from_saved_params(cls, params_file: str, **kwargs):
        """
        Create an FPGAPlacementOptimizer from saved parameters in a JSON file.
        
        Args:
            params_file: Path to the JSON file containing saved parameters
            **kwargs: Additional arguments to override saved parameters
                      (e.g., num_trials, num_steps, dev, etc.)
        
        Returns:
            FPGAPlacementOptimizer instance
        """
        with open(params_file, 'r') as f:
            data = json.load(f)
        
        params = data['params']
        tensors = data['tensors']
        
        target_dev = kwargs.get('dev', params.get('device', 'cpu'))
        coupling_matrix = torch.tensor(tensors['coupling_matrix'], dtype=torch.float32, device=target_dev) if 'coupling_matrix' in tensors else None
        site_coords_matrix = torch.tensor(tensors['site_coords_matrix'], dtype=torch.float32, device=target_dev) if 'site_coords_matrix' in tensors else None
        io_site_connect_matrix = torch.tensor(tensors['io_site_connect_matrix'], dtype=torch.float32, device=target_dev) if 'io_site_connect_matrix' in tensors else None
        io_site_coords = torch.tensor(tensors['io_site_coords'], dtype=torch.float32, device=target_dev) if 'io_site_coords' in tensors else None
        
        place_orientation = PlaceType[params['place_orientation']] if params['place_orientation'] in PlaceType.__members__ else PlaceType.CENTERED
        grid_type = GridType[params['grid_type']] if params['grid_type'] in GridType.__members__ else GridType.SQUARE
        
        default_kwargs = {
            'num_inst': params['num_inst'],
            'num_fixed_inst': params['num_fixed_inst'],
            'num_site': params['num_site'],
            'num_fixed_site': params['num_fixed_site'],
            'coupling_matrix': coupling_matrix,
            'site_coords_matrix': site_coords_matrix,
            'io_site_connect_matrix': io_site_connect_matrix,
            'io_site_coords': io_site_coords,
            'constraint_alpha': params['constraint_alpha'],
            'constraint_beta': params['constraint_beta'],
            'dev': target_dev,
            'dtype': torch.float32,
            'with_io': params['with_io'],
            'distance_metric': params.get('distance_metric', 'manhattan'),
        }
        
        default_kwargs.update(kwargs)
        
        return cls(**default_kwargs)

    def _apply_distance_metric(self, distance_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the configured distance metric transformation."""
        if self.distance_metric == 'sqrt_manhattan':
            return torch.sqrt(distance_tensor)
        return distance_tensor

    def _initialize(self):

        torch.manual_seed(self.seed)
        
        if self.with_io:
            h_logic = self.h_factor * torch.randn(
                [self.num_trials, self.num_inst, self.num_site], 
                device=self.dev, dtype=self.dtype
            )
            
            h_io = self.io_factor * self.h_factor * torch.randn(
                [self.num_trials, self.fixed_insts_num, self.fixed_site_num], 
                device=self.dev, dtype=self.dtype
            )

            if not self.manual_grad:
                h_logic.requires_grad=True
                h_io.requires_grad=True

            x_logic = self.logic_site_coords[:, 0]
            y_logic = self.logic_site_coords[:, 1]

            Dx_LL = torch.abs(x_logic.unsqueeze(1) - x_logic.unsqueeze(0))
            Dy_LL = torch.abs(y_logic.unsqueeze(1) - y_logic.unsqueeze(0))
            self.D_LL = self._apply_distance_metric(Dx_LL + Dy_LL)

            x_io = self.io_site_coords[:, 0]
            y_io = self.io_site_coords[:, 1]

            Dx_LI = torch.abs(x_logic.unsqueeze(1) - x_io.unsqueeze(0))
            Dy_LI = torch.abs(y_logic.unsqueeze(1) - y_io.unsqueeze(0))
            self.D_LI = self._apply_distance_metric(Dx_LI + Dy_LI)

            return h_logic, h_io

        h = self.h_factor * torch.randn(
            [self.num_trials, self.num_inst, self.num_site],
            device=self.dev, dtype=self.dtype
        )

        if not self.manual_grad:
            h.requires_grad = True

        coords_i = self.logic_site_coords.unsqueeze(1)
        coords_j = self.logic_site_coords.unsqueeze(0)
        base_distance = torch.sum(torch.abs(coords_i - coords_j), dim=2)
        self.D = self._apply_distance_metric(base_distance)

        return h

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
        h = self._initialize()
        opt = self._setup_optimizer([h])

        # CUDA memory tracking
        cuda_available = self.dev != 'cpu' and torch.cuda.is_available()
        mem_log = []
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
            mem_initial = torch.cuda.memory_allocated()
            INFO(f"CUDA memory before optimization: {mem_initial / 1024**2:.2f} MB")

        for step in range(self.num_steps):
            p = torch.softmax(h, dim=2)
            opt.zero_grad()

            loss = expected_fpga_placement(
                J=self.coupling_matrix, 
                p=p, 
                D=self.D, 
                step=step, 
                site_coords_matrix=self.logic_site_coords, 
                alpha=self.constraint_alpha
            )

            free_energy = loss - entropy_q(p) / self.betas[step]
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

            if cuda_available:
                mem_log.append(torch.cuda.memory_allocated())

        if cuda_available and mem_log:
            mem_tensor = torch.tensor(mem_log, device='cpu', dtype=torch.float64)
            mem_avg = mem_tensor.mean().item()
            mem_max = torch.cuda.max_memory_allocated()
            INFO(f"CUDA memory report — max: {mem_max / 1024**2:.2f} MB, "
                 f"avg: {mem_avg / 1024**2:.2f} MB, "
                 f"peak over initial: {(mem_max - mem_initial) / 1024**2:.2f} MB")

        return p
    
    def iterate_placement_with_io(self):
        h_logic, h_io = self._initialize()
        opt = self._setup_optimizer([h_logic, h_io])

        # CUDA memory tracking
        cuda_available = self.dev != 'cpu' and torch.cuda.is_available()
        mem_log = []
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
            mem_initial = torch.cuda.memory_allocated()
            INFO(f"CUDA memory before optimization: {mem_initial / 1024**2:.2f} MB")

        for step in range(self.num_steps):
            p_logic = torch.softmax(h_logic, dim=2)
            p_io = torch.softmax(h_io, dim=2)
            opt.zero_grad()

            loss = expected_fpga_placement_with_io(
                J_LL=self.coupling_matrix, 
                J_LI=self.io_site_connect_matrix, 
                p_logic=p_logic, 
                p_io=p_io, 
                D_LL=self.D_LL, 
                D_LI=self.D_LI, 
                alpha=self.constraint_alpha, 
                beta=self.constraint_beta
            )

            free_energy = loss - ((entropy_q(p_logic) + entropy_q(p_io)) / self.betas[step])
            free_energy.backward(gradient=torch.ones_like(free_energy))
            opt.step()

            if cuda_available:
                mem_log.append(torch.cuda.memory_allocated())

        if cuda_available and mem_log:
            mem_tensor = torch.tensor(mem_log, device='cpu', dtype=torch.float64)
            mem_avg = mem_tensor.mean().item()
            mem_max = torch.cuda.max_memory_allocated()
            INFO(f"CUDA memory report (with IO) — max: {mem_max / 1024**2:.2f} MB, "
                 f"avg: {mem_avg / 1024**2:.2f} MB, "
                 f"peak over initial: {(mem_max - mem_initial) / 1024**2:.2f} MB")

        return p_logic, p_io

    def optimize(self) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.with_io:
            p = self.iterate_placement_with_io()
            config, result = infer_placements_with_io(self.coupling_matrix, 
                                                      self.io_site_connect_matrix,
                                                      p[0], p[1], 
                                                      self.logic_site_coords,
                                                      D_LL=self.D_LL, D_LI=self.D_LI, 
                                                      io_site_coords=self.io_site_coords)
            return config, result
        else:
            p = self.iterate_placement()
            config, result = infer_placements(self.coupling_matrix, 
                                              p, 
                                              self.logic_site_coords, 
                                              D=self.D)
            return config, result
