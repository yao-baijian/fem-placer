import torch
from typing import Iterable, Sequence, Tuple

from .config import *
from .logger import ERROR, WARNING

try:
    from .native_hpwl import HAS_NATIVE_HPWL, hpwl_stats as native_hpwl_stats
except Exception:  # pragma: no cover
    HAS_NATIVE_HPWL = False
    native_hpwl_stats = None

class HPWLCalculator:

    def __init__(self, device, debug=False):
        self.net_hpwl = {}
        self.net_bbox = {}

        self.total_hpwl = 0.0

        self.nets = []
        self.net_names = []
        self.device = device
        self.debug = debug
        pass

    def get_hpwl(self):
        return self.total_hpwl

    def clear(self):
        self.total_hpwl = 0.0
        self.net_hpwl.clear()
        self.net_bbox.clear()

    def compute_net_hpwl_rapidwright(self, net, net_name, logic_instances=None, io_instances=None):
        if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
            return 0.0, {}

        pins = net.getPins()
        if len(pins) < 2:
            return 0.0, {}

        coordinates_set = set()
        for pin in pins:
            site_inst = pin.getSiteInst()
            if not site_inst:
                continue
                
            site_name = site_inst.getName()
            
            # Use pre-classified instances to filter if provided
            is_logic = logic_instances.has_name(site_name)
            is_io = io_instances.has_name(site_name)
            
            if not is_logic and not is_io:
                # Ignore sites not part of our optimizable/fixed problem entirely
                continue
                
            coord = (site_inst.getInstanceX(), site_inst.getInstanceY())
            coordinates_set.add(coord)

        coordinates = list(coordinates_set)

        if len(coordinates) < 2:
            return 0.0, {}

        hpwl, bbox = self._compute_hpwl_from_coordinates(coordinates)

        self.net_hpwl[net_name] = hpwl
        self.net_bbox[net_name] = bbox
        self.total_hpwl += hpwl

    def compute_net_hpwl(self, net_name, connected_sites, instance_coords):
        coordinates = []

        for site_name in connected_sites:
            if site_name in instance_coords:
                coordinates.append(instance_coords[site_name])

        if len(coordinates) < 2:
            return 0.0, {}

        hpwl, bbox = self._compute_hpwl_from_coordinates(coordinates)

        self.net_hpwl[net_name] = hpwl
        self.net_bbox[net_name] = bbox
        self.total_hpwl += hpwl

    def compute_single_instance_hpwl(self, connected_sites, instance_coords, include_io=True):
        coordinates = []

        for site_name in connected_sites:
            if site_name in instance_coords:
                coordinates.append(instance_coords[site_name])

        if len(coordinates) < 2:
            return 0.0, {}

        return self._compute_hpwl_from_coordinates(coordinates)

    def _normalize_coordinates(self, coordinates) -> Sequence[Tuple[float, float]]:
        normalized = []
        for coord in coordinates:
            if isinstance(coord, torch.Tensor):
                x = float(coord[0].item())
                y = float(coord[1].item())
            elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                x = float(coord[0])
                y = float(coord[1])
            else:
                raise ValueError(f"Unsupported coordinate format: {coord}")
            normalized.append((x, y))
        return normalized

    def _compute_hpwl_from_coordinates(self, coordinates):
        if len(coordinates) < 2:
            return 0.0, {}

        coords = self._normalize_coordinates(coordinates)

        if HAS_NATIVE_HPWL and native_hpwl_stats is not None:
            hpwl, min_x, max_x, min_y, max_y = native_hpwl_stats(coords)
        else:
            x_coords = [pt[0] for pt in coords]
            y_coords = [pt[1] for pt in coords]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            hpwl = (max_x - min_x) + (max_y - min_y)

        bbox = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'num_pins': len(coords)
        }

        return hpwl, bbox

    def compute_wa_hpwl(self,
                       p: torch.Tensor,
                       site_coords_matrix: torch.Tensor,
                       net_tensor: torch.Tensor,
                       gamma: float = 0.03) -> torch.Tensor:
        batch_size, num_instances, num_sites = p.shape
        num_nets = net_tensor.shape[0]

        expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
        expected_x = expected_coords[..., 0]  # [batch_size, num_instances]
        expected_y = expected_coords[..., 1]  # [batch_size, num_instances]

        x_min = expected_x.min(dim=1, keepdim=True)[0]
        x_max = expected_x.max(dim=1, keepdim=True)[0]
        y_min = expected_y.min(dim=1, keepdim=True)[0]
        y_max = expected_y.max(dim=1, keepdim=True)[0]

        x_norm = (expected_x - x_min) / (x_max - x_min + 1e-10)
        y_norm = (expected_y - y_min) / (y_max - y_min + 1e-10)

        net_tensor_expanded = net_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        if net_tensor_expanded.dtype == torch.bool:
            net_tensor_expanded = net_tensor_expanded.float()

        total_hpwl = torch.zeros(batch_size, device=p.device)

        for coord_norm in [x_norm, y_norm]:
            coord_expanded = coord_norm.unsqueeze(1).expand(-1, num_nets, -1)  # [batch_size, num_nets, num_instances]
            coord_masked = coord_expanded * net_tensor_expanded

            large_neg = -1e10
            large_pos = 1e10

            zero_mask_max = (coord_masked == 0) & (net_tensor_expanded == 0)
            coord_for_max = torch.where(zero_mask_max,
                                       torch.tensor(large_neg, device=p.device),
                                       coord_masked)
            max_vals = coord_for_max.max(dim=2, keepdim=True)[0]

            zero_mask_min = (coord_masked == 0) & (net_tensor_expanded == 0)
            coord_for_min = torch.where(zero_mask_min,
                                       torch.tensor(large_pos, device=p.device),
                                       coord_masked)
            min_vals = coord_for_min.min(dim=2, keepdim=True)[0]

            weight_pos = torch.exp((coord_masked - max_vals) / gamma)
            numerator_pos = torch.sum(coord_masked * weight_pos * net_tensor_expanded, dim=2)
            denominator_pos = torch.sum(weight_pos * net_tensor_expanded, dim=2)

            weight_neg = torch.exp(-(coord_masked - min_vals) / gamma)
            numerator_neg = torch.sum(coord_masked * weight_neg * net_tensor_expanded, dim=2)
            denominator_neg = torch.sum(weight_neg * net_tensor_expanded, dim=2)

            denominator_pos = torch.where(denominator_pos == 0,
                                         torch.tensor(1e-10, device=p.device),
                                         denominator_pos)
            denominator_neg = torch.where(denominator_neg == 0,
                                         torch.tensor(1e-10, device=p.device),
                                         denominator_neg)

            wa_max = numerator_pos / denominator_pos
            wa_min = numerator_neg / denominator_neg

            coord_hpwl = wa_max - wa_min
            total_hpwl += torch.sum(coord_hpwl, dim=1)

        return total_hpwl

    def compute_jmatrix_hpwl(self,
                            p: torch.Tensor,
                            site_coords_matrix: torch.Tensor,
                            J: torch.Tensor) -> torch.Tensor:
        expected_coords = torch.matmul(p, site_coords_matrix)  # [batch_size, num_instances, 2]
        coords_i = expected_coords.unsqueeze(2)  # [batch_size, num_instances, 1, 2]
        coords_j = expected_coords.unsqueeze(1)  # [batch_size, 1, num_instances, 2]

        manhattan_dist = torch.sum(torch.abs(coords_i - coords_j), dim=-1)
        weighted_dist = manhattan_dist * J.unsqueeze(0)  # [batch_size, num_instances, num_instances]

        triu_mask = torch.triu(torch.ones_like(J), diagonal=1).bool()
        weighted_dist_triu = weighted_dist[:, triu_mask]  # [batch_size, num_pairs]
        total_wirelength = torch.sum(weighted_dist_triu, dim=1)  # [batch_size]

        return total_wirelength