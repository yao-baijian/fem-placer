import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from .grid import Grid
from .placer import FpgaPlacer
from .logger import INFO, WARNING, ERROR

class Legalizer:

    def __init__(self,
                 placer,
                 device,
                 overlap_solver: str = 'greedy',
                 hungarian_distance_weight: float = 0.1,
                 hungarian_max_empty_sites: Optional[int] = None,
                 enable_importance_based_swapping: bool = False,
                 fast_first_improvement: bool = True):
        self.placer: FpgaPlacer = placer
        self.device = device

        # Build grids from whatever regions the placer has
        self.grids: Dict[str, Grid] = {}
        for r in placer.regions:
            try:
                self.grids[r] = placer.get_grid(r)
            except Exception:
                WARNING(f"Legalizer: no grid for region '{r}', skipping")

        self.overlap_solver = overlap_solver
        self.hungarian_distance_weight = hungarian_distance_weight
        self.hungarian_max_empty_sites = hungarian_max_empty_sites
        self.enable_importance_based_swapping = enable_importance_based_swapping
        self.fast_first_improvement = fast_first_improvement

    def legalize_placement(self, region_coords: Dict[str, torch.Tensor],
                           region_ids: Dict[str, torch.Tensor]):
        """
        Legalize placements for all regions.

        Args:
            region_coords: ``{region_name: coords_tensor [N, 2]}``
            region_ids: ``{region_name: ids_tensor [N]}``

        Returns:
            ``(legalized_coords, total_overlap, hpwl_before, hpwl_after)``
            where ``legalized_coords`` is a dict ``{region_name: tensor [N, 2]}``.
        """
        regions = [r for r in region_coords if r in self.grids and r in region_ids]

        INFO(f"Stage 1: solve overlap")
        total_moved = 0
        for r in regions:
            self._load_coords_to_grids(self.grids[r], region_coords[r], region_ids[r])
            moved = self._resolve_grid_overlaps(self.grids[r], region_coords)
            total_moved += moved

        # Build legalized coords dict
        legalized = {}
        for r in regions:
            n = region_coords[r].shape[0]
            legalized[r] = self.grids[r].to_coords_tensor(n)

        # HPWL before / after
        all_coords_before = [region_coords[r] for r in regions]
        all_coords_after = [legalized[r] for r in regions]
        hpwl_before = self.placer.net_manager.analyze_solver_hpwl(*all_coords_before)
        hpwl_after  = self.placer.net_manager.analyze_solver_hpwl(*all_coords_after)
        moved_str = ' + '.join(f"{r}: {m}" for r, m in zip(regions, [0]*len(regions)))
        INFO(f"Hpwl {hpwl_before.get('hpwl', hpwl_before.get('hpwl_no_io', 0)):.2f} -> "
             f"{hpwl_after.get('hpwl', hpwl_after.get('hpwl_no_io', 0)):.2f}, "
             f"moved {total_moved} instances")

        INFO(f"Stage 2: global optimization")
        optimized = self._global_optimization(legalized, region_ids, iteration=3)
        all_coords_opt = [optimized[r] for r in regions]
        hpwl_opt = self.placer.net_manager.analyze_solver_hpwl(*all_coords_opt)
        hpwl_after_val = hpwl_after.get('hpwl', hpwl_after.get('hpwl_no_io', 0))
        hpwl_opt_val = hpwl_opt.get('hpwl', hpwl_opt.get('hpwl_no_io', 0))
        INFO(f"Optimized Hpwl {hpwl_opt_val:.2f}, improve {hpwl_opt_val - hpwl_after_val:.2f}")

        return optimized, total_moved, hpwl_before, hpwl_opt

    # ------------------------------------------------------------------
    # Internal helpers — accept region_coords dict, extract logic/io for
    # the net manager calls (which still use the legacy signature).
    # ------------------------------------------------------------------

    def _logic_io_from(self, region_coords):
        return (region_coords.get('logic'),
                region_coords.get('io'),
                'io' in region_coords)

    def _load_coords_to_grids(self, grid: Grid, coords: torch.Tensor, ids: torch.Tensor):
        grid.clear_all()
        grid.from_coords_tensor(coords, ids)
        INFO(f"Loaded {len(ids)} instance to grid")

    def _resolve_grid_overlaps(self, grid: Grid, region_coords: Dict[str, torch.Tensor]) -> int:
        lc, ioc, inc_io = self._logic_io_from(region_coords)

        if self.overlap_solver == 'hungarian':
            return self._resolve_grid_overlaps_hungarian(grid, lc, ioc, inc_io)

        moved_count = 0
        conflict_groups = self._collect_conflict_groups(grid)
        sorted_conflicts = sorted(conflict_groups.items(), key=lambda x: len(x[1]), reverse=True)

        for conflict_pos, conflict_instances in sorted_conflicts:
            if len(conflict_instances) <= 1:
                continue
            success, num_moved = self._resolve_conflict_in_grid(
                grid, conflict_pos, conflict_instances, lc, ioc, inc_io
            )
            if success:
                moved_count += num_moved

        remaining_conflicts = self._check_remaining_overlaps(grid)
        if remaining_conflicts > 0:
            WARNING(f'{remaining_conflicts} conflicts are not resolved')
        return moved_count

    def _collect_conflict_groups(self, grid: Grid) -> Dict[Tuple[int, int], List[int]]:
        conflict_groups: Dict[Tuple[int, int], List[int]] = {}
        for instance_id, poz in grid.instance_positions.items():
            pos_tuple = tuple(poz)
            if pos_tuple in conflict_groups:
                conflict_groups[pos_tuple].append(instance_id)
            else:
                conflict_groups[pos_tuple] = [instance_id]
        return {pos: insts for pos, insts in conflict_groups.items() if len(insts) > 1}

    def _resolve_grid_overlaps_hungarian(self, grid, logic_coords, io_coords, include_io):
        conflict_groups = self._collect_conflict_groups(grid)
        if not conflict_groups:
            return 0
        conflict_instances = []
        for instances in conflict_groups.values():
            if len(instances) > 1:
                conflict_instances.extend(instances[1:])
        if not conflict_instances:
            return 0
        empty_positions = list(grid._empty_positions)
        if not empty_positions:
            ERROR(f"Grid '{grid.name}' has no empty place")
            return 0
        if self.hungarian_max_empty_sites is not None and len(empty_positions) > self.hungarian_max_empty_sites:
            cx = sum(pos[0] for pos in conflict_groups) / max(1, len(conflict_groups))
            cy = sum(pos[1] for pos in conflict_groups) / max(1, len(conflict_groups))
            empty_positions = sorted(empty_positions, key=lambda p: abs(p[0]-cx)+abs(p[1]-cy))[:self.hungarian_max_empty_sites]
        if len(empty_positions) < len(conflict_instances):
            WARNING(f"Grid '{grid.name}' empty sites ({len(empty_positions)}) < conflict instances ({len(conflict_instances)})")
        candidate_xy = [(x, y) for x, y in empty_positions]
        n_inst, n_cand = len(conflict_instances), len(candidate_xy)
        if n_inst == 0 or n_cand == 0:
            return 0
        cost_matrix = np.zeros((n_inst, n_cand), dtype=np.float32)
        for i, inst_id in enumerate(conflict_instances):
            cp = grid.get_instance_position(inst_id)
            if cp is None:
                cost_matrix[i, :] = 1e6
                continue
            cur = self.placer.net_manager.compute_instance_move_hpwl(inst_id, logic_coords, io_coords, include_io)
            cand = self.placer.net_manager.compute_instance_move_hpwl_batch(inst_id, logic_coords, io_coords, include_io, candidate_xy)
            pen = np.array([abs(cx-cp[0])+abs(cy-cp[1]) for cx,cy in candidate_xy], dtype=np.float32) * self.hungarian_distance_weight
            cost_matrix[i, :] = (np.asarray(cand, dtype=np.float32) - float(cur)) + pen
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        moved = 0
        for ri, ci in zip(row_ind, col_ind):
            inst_id = conflict_instances[ri]
            tx, ty = candidate_xy[ci]
            cp = grid.get_instance_position(inst_id)
            if cp is None or (cp[0]==tx and cp[1]==ty):
                continue
            ok, _, _ = grid.move_instance(inst_id, tx, ty, swap_allowed=False)
            if ok:
                moved += 1
        rem = self._check_remaining_overlaps(grid)
        if rem > 0:
            WARNING(f'{rem} conflicts not resolved after Hungarian stage')
        return moved

    def _check_remaining_overlaps(self, grid: Grid) -> int:
        pos_cnt = {}
        for _, pos in grid.instance_positions.items():
            t = tuple(pos)
            pos_cnt[t] = pos_cnt.get(t, 0) + 1
        rem = [(p, c) for p, c in pos_cnt.items() if c > 1]
        if rem:
            INFO(f" remain overlapped: {rem}")
        return len(rem)

    def _resolve_conflict_in_grid(self, grid, conflict_pos, conflict_instances,
                                  logic_coords, io_coords, include_io):
        cx, cy = conflict_pos
        needed = len(conflict_instances) + 1
        empty = grid.find_empty_positions_nearby(cx, cy, needed)
        if len(empty) < needed - 1:
            ERROR(f"Grid '{grid.name}' has no empty place")
            return False, 0
        empty.insert(0, (cx, cy, 0))
        m, n_max = len(conflict_instances), min(len(empty), len(conflict_instances) + 3)
        cand_pos = empty[:n_max]
        cand_xy = [(x, y) for x, y, _ in cand_pos]
        cost = torch.zeros((m, n_max), device=self.device)
        for i, inst_id in enumerate(conflict_instances):
            cur = self.placer.net_manager.compute_instance_move_hpwl(inst_id, logic_coords, io_coords, include_io)
            hpwl_c = self.placer.net_manager.compute_instance_move_hpwl_batch(inst_id, logic_coords, io_coords, include_io, cand_xy)
            for j in range(n_max):
                _, _, dist = cand_pos[j]
                cost[i, j] = (hpwl_c[j] - cur) + dist * 0.1
        asgn = self._greedy_assignment(cost)
        moved = 0
        for i, j in enumerate(asgn):
            if j < 0:
                continue
            inst_id = conflict_instances[i]
            tx, ty, _ = empty[j]
            cp = grid.get_instance_position(inst_id)
            if cp and (cp[0]!=tx or cp[1]!=ty):
                ok, _, _ = grid.move_instance(inst_id, tx, ty, swap_allowed=True)
                if ok:
                    moved += 1
        return True, moved

    def _greedy_assignment(self, cost_matrix):
        m, n = cost_matrix.shape
        assigned_positions = set()
        assignment = [-1] * m
        for i in range(m):
            best = float('inf')
            best_j = -1
            for j in range(n):
                if j not in assigned_positions and cost_matrix[i, j] < best:
                    best = cost_matrix[i, j]
                    best_j = j
            if best_j != -1:
                assignment[i] = best_j
                assigned_positions.add(best_j)
        return assignment

    def _global_optimization(self, legalized: Dict[str, torch.Tensor],
                             region_ids: Dict[str, torch.Tensor],
                             iteration: int = 3) -> Dict[str, torch.Tensor]:
        lc, ioc, inc_io = self._logic_io_from(legalized)
        for _ in range(iteration):
            improved = False
            for r in legalized:
                if r in self.grids:
                    ok = self._optimize_grid_instances(
                        self.grids[r], legalized[r].shape[0], lc, ioc, inc_io
                    )
                    if ok:
                        improved = True
            if not improved:
                break
        optimized = {}
        for r in legalized:
            if r in self.grids:
                optimized[r] = self.grids[r].to_coords_tensor(legalized[r].shape[0])
        return optimized

    def _optimize_grid_instances(self, grid: Grid, num_instances: int,
                                logic_coords, io_coords, include_io) -> bool:
        improved = False
        if self.enable_importance_based_swapping:
            critical_instances = self._select_critical_instances_for_grid(
                grid, num_instances, logic_coords, io_coords, include_io
            )
            for instance_id in critical_instances:
                success, improvement = self._optimize_instance_in_grid_importance_aware(
                    grid, instance_id, logic_coords, io_coords, include_io
                )
                if success and improvement > 0:
                    improved = True
        else:
            # 快速贪心优化（速度快）
            for instance_id in list(grid.instance_positions.keys())[:num_instances]:
                success, improvement = self._optimize_instance_in_grid_fast(
                    grid, instance_id, logic_coords, io_coords, include_io
                )
                if success and improvement > 0:
                    improved = True
                    if self.fast_first_improvement:
                        break

        return improved

    def _compute_instance_connectivity(self, instance_id: int) -> float:
        """计算实例的连接度得分"""
        net_tensor = self.placer.net_manager.net_tensor
        if net_tensor is None or instance_id >= net_tensor.shape[1]:
            return 0.0
        return net_tensor[:, instance_id].sum().item()

    def _select_critical_instances_for_grid(self, grid: Grid, num_instances: int,
                                           logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                           include_io: bool) -> List[int]:
        """为指定网格选择关键实例 (top 20% by connectivity)"""
        # 如果网格中没有实例，返回空列表
        if not grid.instance_positions:
            return []

        # 根据连接度选择关键实例
        instances_in_grid = list(grid.instance_positions.keys())
        connectivity_scores = []

        for inst_id in instances_in_grid:
            connectivity = self._compute_instance_connectivity(inst_id)
            connectivity_scores.append((connectivity, inst_id))

        if not connectivity_scores:
            return []

        # 按连接度降序排序
        connectivity_scores.sort(reverse=True)
        
        # 选择top 20%作为关键实例
        top_k = max(1, len(connectivity_scores) // 5)
        top_k = min(top_k, len(connectivity_scores))
        
        return [inst_id for _, inst_id in connectivity_scores[:top_k]]

    def _optimize_instance_in_grid_fast(self, grid: Grid, instance_id: int,
                                       logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                       include_io: bool) -> Tuple[bool, float]:
        """快速贪心优化: 仅考虑邻域空位, 不做交换(批量HPWL评估)"""
        current_pos = grid.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0

        # 获取当前位置的HPWL
        current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
            instance_id, logic_coords, io_coords, include_io
        )

        best_pos = current_pos
        best_hpwl = current_hpwl

        # 根据网格类型设置搜索半径
        search_radius = 2 if grid.name == 'logic' else 1

        # 搜索邻域并收集空位候选
        candidate_xy: List[Tuple[int, int]] = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                new_x, new_y = current_pos[0] + dx, current_pos[1] + dy

                if not grid.is_within_bounds(new_x, new_y):
                    continue

                if grid.is_position_empty(new_x, new_y):
                    candidate_xy.append((new_x, new_y))

        if not candidate_xy:
            return False, 0.0

        hpwl_candidates = self.placer.net_manager.compute_instance_move_hpwl_batch(
            instance_id,
            logic_coords,
            io_coords,
            include_io,
            candidate_xy,
        )

        if not hpwl_candidates:
            return False, 0.0

        hpwl_candidates_np = np.asarray(hpwl_candidates, dtype=np.float32)
        best_idx = int(np.argmin(hpwl_candidates_np))
        best_hpwl = float(hpwl_candidates_np[best_idx])
        best_pos = candidate_xy[best_idx]

        # 如果找到更好的位置
        if best_hpwl < current_hpwl and best_pos != current_pos:
            improvement = current_hpwl - best_hpwl
            success, _, _ = grid.move_instance(
                instance_id, best_pos[0], best_pos[1], swap_allowed=False
            )
            return success, improvement

        return False, 0.0

    def _optimize_instance_in_grid_importance_aware(self, grid: Grid, instance_id: int,
                                  logic_coords: torch.Tensor, io_coords: Optional[torch.Tensor],
                                  include_io: bool) -> Tuple[bool, float]:
        """重要性感知优化：支持与非关键实例交换
        
        策略：
        1. 优先查找并移到空位
        2. 其次考虑与低重要性实例交换
        3. 只在总HPWL改进时执行交换
        """
        current_pos = grid.get_instance_position(instance_id)
        if not current_pos:
            return False, 0.0

        # 获取当前位置的HPWL
        current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
            instance_id, logic_coords, io_coords, include_io
        )

        best_pos = current_pos
        best_hpwl = current_hpwl
        best_swap_candidate = None
        best_improvement = 0.0

        # 获取该实例的连接度
        instance_connectivity = self._compute_instance_connectivity(instance_id)

        # 根据网格类型设置搜索半径 (Manhattan distance)
        search_radius = 3 if grid.name == 'logic' else 1

        # 收集邻域内的所有位置（空位或其他实例）
        neighbor_positions = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                # 使用Manhattan距离判断
                if abs(dx) + abs(dy) > search_radius:
                    continue

                new_x, new_y = current_pos[0] + dx, current_pos[1] + dy

                if not grid.is_within_bounds(new_x, new_y):
                    continue

                neighbor_positions.append((new_x, new_y))

        # 首先评估所有邻域位置中的空位（优先级高）
        empty_positions = []
        occupied_positions = []
        
        for new_x, new_y in neighbor_positions:
            occupants = grid.get_position_occupants(new_x, new_y)
            if not occupants:
                empty_positions.append((new_x, new_y))
            elif occupants[0] != instance_id:
                occupied_positions.append((new_x, new_y, occupants[0]))

        # 评估空位（优先级最高）
        for new_x, new_y in empty_positions:
            new_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                instance_id,
                logic_coords,
                io_coords,
                include_io,
                candidate_pos=(new_x, new_y)
            )

            improvement = current_hpwl - new_hpwl
            if improvement > best_improvement:
                best_improvement = improvement
                best_hpwl = new_hpwl
                best_pos = (new_x, new_y)
                best_swap_candidate = None

        # 如果没有找到改进的空位，才考虑交换（只与非关键实例）
        if best_swap_candidate is None and best_improvement <= 0:
            for new_x, new_y, swap_instance_id in occupied_positions:
                # 只考虑与低重要性实例的交换
                swap_connectivity = self._compute_instance_connectivity(swap_instance_id)
                
                # 交换的实例不能是关键实例（连接度不能更高）
                if swap_connectivity >= instance_connectivity:
                    continue

                # 计算交换后的HPWL变化
                swap_current_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                    swap_instance_id, logic_coords, io_coords, include_io
                )
                
                # 当前实例移到新位置
                instance_new_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                    instance_id,
                    logic_coords,
                    io_coords,
                    include_io,
                    candidate_pos=(new_x, new_y)
                )
                
                # 交换的实例移到当前位置
                swap_new_hpwl = self.placer.net_manager.compute_instance_move_hpwl(
                    swap_instance_id,
                    logic_coords,
                    io_coords,
                    include_io,
                    candidate_pos=current_pos
                )
                
                # 总的HPWL变化
                total_hpwl_after_swap = instance_new_hpwl + swap_new_hpwl
                total_hpwl_before_swap = current_hpwl + swap_current_hpwl
                improvement = total_hpwl_before_swap - total_hpwl_after_swap
                
                # 如果交换能改进且改进最好，记录
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_hpwl = instance_new_hpwl
                    best_pos = (new_x, new_y)
                    best_swap_candidate = swap_instance_id

        # 如果找到更好的位置或交换
        if best_improvement > 0 and best_pos != current_pos:
            if best_swap_candidate is not None:
                # 执行交换
                success, _, _ = grid.move_instance(
                    instance_id, best_pos[0], best_pos[1], swap_allowed=True
                )
                if success:
                    return success, best_improvement
            else:
                # 仅移动到空位
                success, _, _ = grid.move_instance(
                    instance_id, best_pos[0], best_pos[1], swap_allowed=False
                )
                if success:
                    return success, best_improvement

        return False, 0.0