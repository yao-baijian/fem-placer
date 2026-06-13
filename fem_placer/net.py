import sys
import os
import torch
import rapidwright
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set, Tuple, List, Optional
from com.xilinx.rapidwright.design import Design, Net
from .hpwl import HPWLCalculator
from .config import *
from .logger import INFO, WARNING, ERROR

class NetManager:

    def __init__(self,
                 get_site_inst_id_by_name_func=None,
                 get_site_inst_name_by_id_func=None,
                 map_coords_to_instance_func=None,
                 debug=False,
                 device='cpu',
                 record_mode='simple',
                 map_mode='simple',
                 offset_coeff: float = 1.0,
                 hpwl_workers=None,
                 hpwl_parallel_threshold=4):

        self.debug = debug
        self.device = device
        self.site_to_site_connectivity = {}
        self.io_to_site_connectivity = {}
        self.net_to_sites = {}
        self.site_to_nets = {}
        self.nets = []
        self.net_names = []
        self.net_tensor = None
        self.insts_matrix = None
        self.io_insts_matrix = None
        self.logic_depth = 1.0  # Store estimated logic depth
        self.max_degree = 0 
        self.avg_degree = 0.0

        self.get_site_inst_id_by_name_func = get_site_inst_id_by_name_func
        self.get_site_inst_name_by_id_func = get_site_inst_name_by_id_func
        self.map_coords_to_instance_func = map_coords_to_instance_func

        self.debug_src_root = "result"
        self.hpwl_calculator = HPWLCalculator(device, debug=debug)
        self.record_mode = record_mode
        self.map_mode = map_mode
        self.offset_coeff = float(offset_coeff)
        cpu_count = os.cpu_count() or 4
        default_workers = max(1, cpu_count // 2)
        self.hpwl_workers = hpwl_workers if hpwl_workers is not None else default_workers
        self.hpwl_parallel_threshold = max(1, hpwl_parallel_threshold)
        self._hpwl_executor: Optional[ThreadPoolExecutor] = None

    def has_net(self, site_name):
        return site_name in self.site_to_nets
    
    def set_debug_path(self, result_dir='result', instance_name=None):
        self.debug_src_root = os.path.join(result_dir, instance_name) if instance_name else result_dir

    def analyze_design_hpwl(self, design, logic_instances=None, io_instances=None):
        self.hpwl_calculator.clear()
        self.nets = design.getNets()
        self.net_names = [net.getName() for net in self.nets]

        total_nets = len(self.nets)
        hpwl_io_count = 0
        hpwl_no_io_count = 0

        # ---- Single-pass HPWL + logic depth + degree estimation ----
        # Previously: 2 passes for HPWL + 1 for logic_depth + 1 for degrees = 4 passes.
        # Now: 1 pass does all.

        # For logic depth estimation
        site_connectivity = {}
        # For degree estimation
        degrees = []

        for idx, net in enumerate(self.nets):
            net_name = net.getName()

            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                # Still need to try both include_io variants to keep dicts consistent
                self.hpwl_calculator.compute_net_hpwl_rapidwright(
                    net, net_name, include_io=True,
                    logic_instances=logic_instances, io_instances=io_instances
                )
                self.hpwl_calculator.compute_net_hpwl_rapidwright(
                    net, net_name, include_io=False,
                    logic_instances=logic_instances, io_instances=io_instances
                )
                continue

            pins = net.getPins()
            if len(pins) < 2:
                continue

            # --- Single pin iteration for ALL metrics ---
            logic_coords_set = set()
            all_coords_set = set()
            sites_in_net = set()
            logic_sites_in_net = set()

            for pin in pins:
                site_inst = pin.getSiteInst()
                if not site_inst:
                    continue

                site_name = site_inst.getName()
                sites_in_net.add(site_name)

                is_logic = logic_instances.has_name(site_name) if logic_instances else False
                is_io = io_instances.has_name(site_name) if io_instances else False

                if not is_logic and not is_io:
                    continue

                coord = (site_inst.getInstanceX(), site_inst.getInstanceY())
                all_coords_set.add(coord)

                if is_logic:
                    logic_coords_set.add(coord)
                    logic_sites_in_net.add(site_name)

            # --- HPWL with IO ---
            if len(all_coords_set) >= 2:
                hpwl, bbox = self.hpwl_calculator._compute_hpwl_from_coordinates(
                    list(all_coords_set)
                )
                self.hpwl_calculator.net_hpwl[net_name] = hpwl
                self.hpwl_calculator.net_bbox[net_name] = bbox
                self.hpwl_calculator.total_hpwl += hpwl
                hpwl_io_count += 1

            # --- HPWL without IO ---
            if len(logic_coords_set) >= 2:
                hpwl_no_io, bbox_no_io = self.hpwl_calculator._compute_hpwl_from_coordinates(
                    list(logic_coords_set)
                )
                self.hpwl_calculator.net_hpwl_no_io[net_name] = hpwl_no_io
                self.hpwl_calculator.net_bbox_no_io[net_name] = bbox_no_io
                self.hpwl_calculator.total_hpwl_no_io += hpwl_no_io
                hpwl_no_io_count += 1

            # --- Logic depth: update site connectivity (from all sites in net) ---
            for site in sites_in_net:
                site_connectivity[site] = site_connectivity.get(site, 0) + len(sites_in_net)

            # --- Net degree: count logic sites in this net ---
            if len(logic_sites_in_net) >= 2:
                degrees.append(len(logic_sites_in_net))

            # Progress indicator for large designs
            if total_nets > 100000 and (idx + 1) % 50000 == 0:
                INFO(f"  HPWL progress: {idx+1}/{total_nets} nets processed ({100*(idx+1)//total_nets}%)")

        hpwl = self.hpwl_calculator.get_hpwl()
        INFO(f"Nets num: {total_nets}, total hpwl: {hpwl['hpwl']:.2f}, without io: {hpwl['hpwl_no_io']:.2f} "
             f"(hpwl_io_nets={hpwl_io_count}, hpwl_no_io_nets={hpwl_no_io_count})")

        # --- Estimate logic depth from collected data ---
        self._estimate_logic_depth_from_data(site_connectivity)

        # --- Calculate net degrees from collected data ---
        self._calculate_net_degrees_from_data(degrees)

        if self.debug:
            self.save_net_debug_info()

        return hpwl

    def _estimate_logic_depth_from_data(self, site_connectivity):
        """Estimate logic depth from pre-collected connectivity data (single-pass)."""
        try:
            if not site_connectivity:
                self.logic_depth = 1.0
                return

            avg_connectivity = sum(site_connectivity.values()) / len(site_connectivity)
            total_sites = len(site_connectivity)

            depth_factor = avg_connectivity / max(1.0, np.sqrt(total_sites))
            self.logic_depth = min(2.0, max(0.5, depth_factor / 10.0))

            INFO(f"Estimated logic depth factor: {self.logic_depth:.3f} "
                 f"(avg_connectivity: {avg_connectivity:.2f}, sites: {total_sites})")
        except Exception as e:
            WARNING(f"Failed to estimate logic depth: {e}, using default value 1.0")
            self.logic_depth = 1.0

    def _calculate_net_degrees_from_data(self, degrees):
        """Calculate degree stats from pre-collected data (single-pass)."""
        if not degrees:
            self.max_degree = 0
            self.avg_degree = 0.0
        else:
            self.max_degree = max(degrees)
            self.avg_degree = sum(degrees) / len(degrees)
    
    def get_net_degrees(self) -> tuple[int, float]:
        return self.max_degree, self.avg_degree

    # Calculate network degree statistics for placer ML
    def _calculate_net_degrees(self):
        """
        Calculate network degree statistics.
        Returns:
            (max_degree, avg_degree)
        """
        if not self.nets or len(self.nets) == 0:
            return 0, 0.0
        degrees = []
        for net in self.nets:
            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                continue
            pins = net.getPins()
            # Count unique sites that this net connects to
            sites_in_net = set()
            for pin in pins:
                site_inst = pin.getSiteInst()
                if site_inst:
                    sites_in_net.add(site_inst.getName())
            degree = len(sites_in_net)
            if degree >= 2:  # Only count nets that connect at least 2 sites
                degrees.append(degree)

        self.max_degree = max(degrees)
        self.avg_degree = sum(degrees) / len(degrees)

    def _estimate_logic_depth(self):
        """
        Estimate the logic depth of the design by analyzing the netlist.
        Uses a heuristic based on net connectivity to estimate critical path depth.
        
        Stores result in self.logic_depth: ratio > 1.0 means deep logic
        """
        try:
            if not self.nets or len(self.nets) == 0:
                self.logic_depth = 1.0
                return
            
            # Count connectivity degree per site
            site_connectivity = {}
            for net in self.nets:
                if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                    continue
                
                pins = net.getPins()
                sites_in_net = set()
                
                for pin in pins:
                    site_inst = pin.getSiteInst()
                    if site_inst:
                        site_name = site_inst.getName()
                        sites_in_net.add(site_name)
                
                # Update connectivity degree
                for site in sites_in_net:
                    site_connectivity[site] = site_connectivity.get(site, 0) + len(sites_in_net)
            
            # Calculate average connectivity degree
            if not site_connectivity:
                self.logic_depth = 1.0
                return
            
            avg_connectivity = sum(site_connectivity.values()) / len(site_connectivity)
            total_sites = len(site_connectivity)
            
            # Heuristic: deep logic has higher connectivity per site and lower site count
            # depth_factor correlates with (avg_connectivity / sqrt(total_sites))
            depth_factor = avg_connectivity / max(1.0, np.sqrt(total_sites))
            
            # Normalize to a reasonable range [0.5, 2.0]
            self.logic_depth = min(2.0, max(0.5, depth_factor / 10.0))
            
            INFO(f"Estimated logic depth factor: {self.logic_depth:.3f} (avg_connectivity: {avg_connectivity:.2f}, sites: {total_sites})")
            
        except Exception as e:
            WARNING(f"Failed to estimate logic depth: {e}, using default value 1.0")
            self.logic_depth = 1.0

    def analyze_solver_hpwl(self, *region_coords):
        """Compute HPWL from N region coordinate tensors.
        Accepts variadic tensors (one per region) and always computes
        full HPWL (including IO nets when io coords are present).
        """
        self.hpwl_calculator.clear()
        instance_coords = self.map_coords_to_instance_func(*region_coords)
        include_io = len(region_coords) > 1
        for net_name, connected_sites in self.net_to_sites.items():
            self.hpwl_calculator.compute_net_hpwl(net_name, connected_sites, instance_coords, include_io=include_io)
        return self.hpwl_calculator.get_hpwl()

    def save_net_debug_info(self, output_path=None):
        if output_path is None:
            output_path = os.path.join(self.debug_src_root, 'net_debug_info.txt')
        with open(output_path, 'w') as f:
            f.write("Net_IDX\tNet_Name\tHPWL\tSite_Count\tSites_Info\n")
            for idx, net in enumerate(self.nets):
                net_name = net.getName()
                hpwl = self.hpwl_calculator.net_hpwl_no_io.get(net_name, 0.0)

                sites_set = set()
                if not (net.isClockNet() or net.isVCCNet() or net.isGNDNet()):
                    pins = net.getPins()

                    for pin in pins:
                        site_inst = pin.getSiteInst()
                        if site_inst:
                            site_name = site_inst.getName()
                            site_x = site_inst.getInstanceX()
                            site_y = site_inst.getInstanceY()
                            site_key = f"{site_name}({site_x},{site_y})"
                            sites_set.add(site_key)

                sites_list = sorted(list(sites_set))
                site_count = len(sites_list)
                sites_str = " | ".join(sites_list)

                if hpwl > 0.0:
                    f.write(f"{idx}\t{net_name}\t{hpwl:.2f}\t{site_count}\t{sites_str}\n")

    def save_solver_hpwl_debug(self, instance_coords, net_to_sites, output_path=None):
        if output_path is None:
            output_path = os.path.join(self.debug_src_root, 'solver_hpwl_debug.txt')
        with open(output_path, 'w') as f:
            f.write("Net_IDX\tNet_Name\tHPWL\tInstance_Count\tInstances_Info\n")

            for idx, (net_name, connected_sites) in enumerate(net_to_sites.items()):
                hpwl, _ = self.hpwl_calculator.compute_net_hpwl(net_name, connected_sites, instance_coords)

                if hpwl == 0.0:
                    continue

                instances_info = []
                for site_name in connected_sites:
                    if site_name in instance_coords:
                        coord = instance_coords[site_name]
                        instance_info = f"{site_name}[({coord[0]:.2f},{coord[1]:.2f})]"
                        instances_info.append(instance_info)

                instance_count = len(instances_info)
                instances_str = " | ".join(instances_info)

                f.write(f"{idx}\t{net_name}\t{hpwl:.2f}\t{instance_count}\t{instances_str}\n")

    def analyze_nets(self, logic_instances, io_instances):

        logic_insts_num = logic_instances.num
        io_insts_num = io_instances.num if io_instances is not None else 0

        self.net_names = [net.getName() for net in self.nets]
        sites_net_list = []
        valid_net_num = 0
        connectivity_groups: List[Tuple[Set[str], Set[str]]] = []

        # Pre-build lookup sets for O(1) site classification
        logic_name_set = set(logic_instances.name_to_id.keys())
        io_name_set = set(io_instances.name_to_id.keys()) if io_instances is not None else set()
        # Use defaultdict for site_to_nets to avoid repeated membership tests
        self.site_to_nets = defaultdict(list)

        total_nets = len(self.nets)

        for idx, net in enumerate(self.nets):
            net_name = net.getName()
            if net.isClockNet() or net.isVCCNet() or net.isGNDNet():
                continue

            sites_in_net = set()
            logic_sites = set()
            io_sites = set()

            for pin in net.getPins():
                site_inst = pin.getSiteInst()
                if not site_inst:
                    continue
                site_name = site_inst.getName()

                sites_in_net.add(site_name)
                if site_name in logic_name_set:
                    logic_sites.add(site_name)
                elif site_name in io_name_set:
                    io_sites.add(site_name)

            if len(logic_sites) + len(io_sites) >= 2:
                self.net_to_sites[net_name] = list(logic_sites) + list(io_sites)

                # Record site-to-net mapping
                for site_name in sites_in_net:
                    self.site_to_nets[site_name].append(net_name)

                connectivity_groups.append((logic_sites, io_sites))

            if len(logic_sites) >= 2:
                valid_net_num += 1
                sites_net_list.append(logic_sites)

            # Progress indicator for large designs
            if total_nets > 100000 and (idx + 1) % 50000 == 0:
                INFO(f"  Net analysis progress: {idx+1}/{total_nets} ({100*(idx+1)//total_nets}%)")

        for logic_sites, io_sites in connectivity_groups:
            self._record_connectivity(logic_sites, io_sites)

        self._create_net_tensor(valid_net_num, sites_net_list, logic_insts_num)
        self._create_net_matrix(logic_insts_num, io_insts_num)
        if self.debug:
            self.save_tensor_debug_info(instance_count=logic_insts_num)
            self.save_matrix_debug_info(logic_insts_num, io_insts_num)
        INFO(f"Processed {valid_net_num} nets, total {len(self.nets)} nets",
              f" {len(self.site_to_site_connectivity)} site-to-site routes",
              f" {len(self.io_to_site_connectivity)} io-to-site routes",
              f" {len(self.net_to_sites)} inter-tile routes")

        return {
            'logic_net_num': len(self.site_to_site_connectivity),
            'total_net_num': len(self.nets)
            }

    def _record_connectivity(self, logic_sites: Set[str], io_sites: Set[str]):
        logic_inst_list = list(logic_sites)
        io_inst_list = list(io_sites)

        weight = 1
        offset_coeff = self.offset_coeff

        if self.record_mode == 'simple':
            io_increment = 1.0
            logic_increment = 1.0
        elif self.record_mode == 'inverse':
            denom_io = float(len(logic_inst_list) + len(io_inst_list) + offset_coeff)
            denom_logic = float(len(logic_inst_list) + offset_coeff)
            io_increment = weight / denom_io if denom_io > 0 else 0.0
            logic_increment = weight / denom_logic if denom_logic > 0 else 0.0
        elif self.record_mode == 'inverse_sqr':
            denom_io = float(len(logic_inst_list) + len(io_inst_list) + offset_coeff) ** 2
            denom_logic = float(len(logic_inst_list) + offset_coeff) ** 2
            io_increment = weight / denom_io if denom_io > 0 else 0.0
            logic_increment = weight / denom_logic if denom_logic > 0 else 0.0
        elif self.record_mode == 'inverse_log':
            denom_io = float(len(logic_inst_list) + len(io_inst_list) + offset_coeff)
            denom_logic = float(len(logic_inst_list) + offset_coeff)
            io_increment = weight / np.log2(denom_io) if denom_io > 1.0 else 0.0
            logic_increment = weight / np.log2(denom_logic) if denom_logic > 1.0 else 0.0
        elif self.record_mode == 'degree_inverse':
            denom_io = float(len(logic_inst_list) + len(io_inst_list) + offset_coeff)
            denom_logic = float(len(logic_inst_list) + offset_coeff)

            def inv_degree_sum(inst_names: List[str]) -> float:
                total = 0.0
                for inst_name in inst_names:
                    degree_weight = len(self.site_to_nets.get(inst_name, []))
                    if degree_weight > 0:
                        total += 1.0 / np.sqrt(float(degree_weight))
                return total

            io_sites_all = logic_inst_list + io_inst_list
            # io_increment = inv_degree_sum(io_sites_all) / denom_io
            # logic_increment = inv_degree_sum(logic_inst_list) / denom_logic

            io_increment = inv_degree_sum(io_sites_all)
            logic_increment = inv_degree_sum(logic_inst_list)
        else:
            WARNING(f"Unknown record_mode '{self.record_mode}', fallback to 'simple'.")
            io_increment = 1.0
            logic_increment = 1.0

        # 1. IO to logic — preserves original nested-loop structure exactly
        io_conn = self.io_to_site_connectivity
        for i in range(len(io_inst_list)):
            for j in range(len(logic_inst_list)):
                io_inst1, inst2 = io_inst_list[i], logic_inst_list[j]

                if io_inst1 not in io_conn:
                    io_conn[io_inst1] = {}
                if inst2 not in io_conn[io_inst1]:
                    io_conn[io_inst1][inst2] = 0
                io_conn[io_inst1][inst2] += io_increment

                if inst2 not in io_conn:
                    io_conn[inst2] = {}
                if io_inst1 not in io_conn[inst2]:
                    io_conn[inst2][io_inst1] = 0
                io_conn[inst2][io_inst1] += io_increment

        # 2. Logic to logic — preserves original nested-loop structure exactly
        site_conn = self.site_to_site_connectivity
        for i in range(len(logic_inst_list)):
            for j in range(i + 1, len(logic_inst_list)):
                inst1, inst2 = logic_inst_list[i], logic_inst_list[j]

                if inst1 not in site_conn:
                    site_conn[inst1] = {}
                if inst2 not in site_conn[inst1]:
                    site_conn[inst1][inst2] = 0
                site_conn[inst1][inst2] += logic_increment

                if inst2 not in site_conn:
                    site_conn[inst2] = {}
                if inst1 not in site_conn[inst2]:
                    site_conn[inst2][inst1] = 0
                site_conn[inst2][inst1] += logic_increment

    def _create_net_tensor(self, valid_net_num, sites_net_list, logic_insts_num):
        self.net_tensor = torch.zeros(valid_net_num, logic_insts_num, dtype=torch.bool)

        for net_idx, sites in enumerate(sites_net_list):
            for site_name in sites:
                instance_idx = self.get_site_inst_id_by_name_func(site_name)
                self.net_tensor[net_idx, instance_idx] = True

        INFO(f"Net tensor shape {self.net_tensor.shape[0]} x {self.net_tensor.shape[1]}")

    def _create_net_matrix(self, logic_insts_num, io_insts_num):
        n = logic_insts_num
        k = io_insts_num

        # ---- Pre-build name→id lookup dict (avoids repeated method calls) ----
        name_to_id = {}
        get_id = self.get_site_inst_id_by_name_func  # local alias

        # Collect all unique site names from both connectivity dicts
        all_sites = set(self.site_to_site_connectivity.keys())
        all_sites.update(self.io_to_site_connectivity.keys())
        for conn_dict in self.site_to_site_connectivity.values():
            all_sites.update(conn_dict.keys())
        for conn_dict in self.io_to_site_connectivity.values():
            all_sites.update(conn_dict.keys())

        for site_name in all_sites:
            sid = get_id(site_name)
            if sid is not None:
                name_to_id[site_name] = sid

        # ---- Logic matrix (site_to_site_connectivity) ----
        self.insts_matrix = torch.zeros((n, n), device=self.device)

        for source_site, connections in self.site_to_site_connectivity.items():
            source_id = name_to_id.get(source_site)
            if source_id is None:
                continue
            for target_site, connection_count in connections.items():
                target_id = name_to_id.get(target_site)
                if target_id is not None:
                    self.insts_matrix[source_id, target_id] += connection_count

        if self.map_mode == 'simple':
            max_val = torch.max(self.insts_matrix)
            self.insts_matrix = self.insts_matrix / max_val
        elif self.map_mode == 'log':
            self.insts_matrix = torch.log1p(self.insts_matrix)
        elif self.map_mode == 'avg':
            denom = float(self.insts_matrix.shape[1]) ** 0.5
            self.insts_matrix = self.insts_matrix / denom

        # ---- IO matrix (io_to_site_connectivity) ----
        self.io_insts_matrix_all = torch.zeros((n + k, n + k), device=self.device)

        for source_site, connections in self.io_to_site_connectivity.items():
            source_id = name_to_id.get(source_site)
            if source_id is None:
                continue
            for target_site, connection_count in connections.items():
                target_id = name_to_id.get(target_site)
                if target_id is not None:
                    self.io_insts_matrix_all[source_id, target_id] += connection_count

        self.io_insts_matrix = self.io_insts_matrix_all[0:n, n:n+k]

        if self.map_mode == 'simple':
            max_val_io = torch.max(self.io_insts_matrix)
            self.io_insts_matrix = self.io_insts_matrix / max_val_io
        elif self.map_mode == 'log':
            self.io_insts_matrix = torch.log1p(self.io_insts_matrix)
        elif self.map_mode == 'avg':
            denom = float(self.io_insts_matrix.shape[1]) ** 0.5
            self.io_insts_matrix = self.io_insts_matrix / denom
        
        # max_val_io = torch.max(self.io_insts_matrix)
        # self.io_insts_matrix = self.io_insts_matrix / (max_val_io)

        def _matrix_stats(tensor):
            if tensor is None or tensor.numel() == 0:
                return 0.0, 0.0, 0.0, 0.0
            non_zero = tensor[tensor != 0]
            if non_zero.numel() == 0:
                return 0.0, 0.0, 0.0, 0.0
            return (
                non_zero.min().item(),
                non_zero.mean().item(),
                non_zero.max().item(),
                non_zero.var(unbiased=False).item() if non_zero.numel() > 1 else 0.0,
            )

        inst_min, inst_mean, inst_max, inst_var = _matrix_stats(self.insts_matrix)
        io_slice_min, io_slice_mean, io_slice_max, io_slice_var = _matrix_stats(self.io_insts_matrix)

        INFO(f"Site matrix {n} x {n} | min={inst_min:.4f}, mean={inst_mean:.4f}, var={inst_var:.4f}, max={inst_max:.4f}")
        INFO(f"IO slice matrix {n} x {k} | min={io_slice_min:.4f}, mean={io_slice_mean:.4f}, var={io_slice_var:.4f}, max={io_slice_max:.4f}")

        # print(f"Site matrix {n} x {n} | min={inst_min:.4f}, mean={inst_mean:.4f}, var={inst_var:.4f}, max={inst_max:.4f}")
        # print(f"IO slice matrix {n} x {k} | min={io_slice_min:.4f}, mean={io_slice_mean:.4f}, var={io_slice_var:.4f}, max={io_slice_max:.4f}")

    def save_matrix_debug_info(self, n, k):
        original_options = np.get_printoptions()
        # Set print options to show full matrix without truncation
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        matrix_debug_path = os.path.join(self.debug_src_root, 'matrix_debug.txt')
        with open(matrix_debug_path, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            
            print("=" * 50)
            print("insts_matrix ({} x {}):".format(n, n))
            print("=" * 50)
            print(self.insts_matrix.cpu().numpy())
            print("\n")
            
            print("=" * 50)
            print("io_insts_matrix_all ({} x {}):".format(n + k, n + k))
            print("=" * 50)
            print(self.io_insts_matrix_all.cpu().numpy())
            print("\n")
            
            print("=" * 50)
            print("io_insts_matrix ({} x {}):".format(n, k))
            print("=" * 50)
            print(self.io_insts_matrix.cpu().numpy())
            print("\n")
            
            # Restore stdout
            sys.stdout = original_stdout
        
        # Restore original numpy print options
        np.set_printoptions(**original_options)

    def save_tensor_debug_info(self, output_path=None, instance_count=None):
        if output_path is None:
            output_path = os.path.join(self.debug_src_root, 'net_to_slice_sites_tensor_debug.txt')
        num_nets = self.net_tensor.shape[0]
        if instance_count is None:
            instance_count = self.net_tensor.shape[1]

        with open(output_path, 'w') as f:
            f.write("Net_IDX\tNet_Name")
            for instance_idx in range(instance_count):
                f.write(f"\t{instance_idx}")
            f.write("\n")
            for net_idx in range(num_nets):
                net_name = self.net_to_sites.get(net_idx, {}).get('name', f'Net_{net_idx}')
                f.write(f"{net_idx}\t{net_name}")

                for instance_idx in range(instance_count):
                    if instance_idx < self.net_tensor.shape[1]:
                        value = 1 if self.net_tensor[net_idx, instance_idx] else 0
                    else:
                        value = 0
                    f.write(f"\t{value}")
                f.write("\n")

            f.write(f"\n=== Summary ===\n")
            f.write(f"Total Nets: {num_nets}\n")
            f.write(f"Total Instances: {instance_count}\n")
            f.write(f"Tensor Shape: {self.net_tensor.shape}\n")

            # 统计每个网络的连接数
            f.write(f"\n=== Connections per Net ===\n")
            f.write("Net_IDX\tNet_Name\tConnections\tConnected_Instances\n")
            for net_idx in range(num_nets):
                if instance_idx < self.net_tensor.shape[1]:
                    connections = self.net_tensor[net_idx].sum().item()
                else:
                    connections = 0

                net_name = self.net_to_sites.get(net_idx, {}).get('name', f'Net_{net_idx}')
                connected_instances = []

                for instance_idx in range(instance_count):
                    if instance_idx < self.net_tensor.shape[1] and self.net_tensor[net_idx, instance_idx]:
                        connected_instances.append(str(instance_idx))

                instances_str = ", ".join(connected_instances) if connected_instances else "None"
                f.write(f"{net_idx}\t{net_name}\t{int(connections)}\t{instances_str}\n")

            # 统计每个instance被多少个网络连接
            f.write(f"\n=== Connections per Instance ===\n")
            f.write("Instance_ID\tConnections\tConnected_Nets\n")
            for instance_idx in range(instance_count):
                if instance_idx < self.net_tensor.shape[1]:
                    connections = self.net_tensor[:, instance_idx].sum().item()
                else:
                    connections = 0

                connected_nets = []
                for net_idx in range(num_nets):
                    if instance_idx < self.net_tensor.shape[1] and self.net_tensor[net_idx, instance_idx]:
                        connected_nets.append(str(net_idx))

                nets_str = ", ".join(connected_nets) if connected_nets else "None"
                f.write(f"{instance_idx}\t{int(connections)}\t{nets_str}\n")

    def get_single_instance_net_hpwl(self, instance_id: int,
                                     coords: torch.Tensor,
                                     io_coords: Optional[torch.Tensor],
                                     include_io: bool = True) -> float:
        return self.compute_instance_move_hpwl(instance_id,
                                               coords,
                                               io_coords,
                                               include_io)

    def _candidate_pos_to_tensor(self,
                                 instance_id: int,
                                 candidate_pos: Tuple[float, float],
                                 logic_coords: torch.Tensor,
                                 io_coords: Optional[torch.Tensor]) -> torch.Tensor:
        if candidate_pos is None:
            return None

        logic_len = logic_coords.shape[0] if logic_coords is not None else 0
        if instance_id < logic_len and logic_coords is not None:
            base_tensor = logic_coords
        else:
            base_tensor = io_coords

        device = base_tensor.device if base_tensor is not None else torch.device(self.device)
        dtype = base_tensor.dtype if base_tensor is not None else torch.float32
        return torch.tensor([float(candidate_pos[0]), float(candidate_pos[1])],
                            dtype=dtype,
                            device=device)

    def _get_coord_for_instance(self,
                                target_instance_id: Optional[int],
                                logic_coords: torch.Tensor,
                                io_coords: Optional[torch.Tensor],
                                include_io: bool,
                                moving_instance_id: int,
                                candidate_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if target_instance_id is None:
            return None

        logic_len = logic_coords.shape[0] if logic_coords is not None else 0

        if target_instance_id < logic_len:
            if logic_coords is None:
                return None
            if target_instance_id == moving_instance_id and candidate_tensor is not None:
                return candidate_tensor
            if 0 <= target_instance_id < logic_coords.shape[0]:
                return logic_coords[target_instance_id]
            return None

        if not include_io or io_coords is None:
            return None

        io_idx = target_instance_id - logic_len
        if io_idx < 0 or io_idx >= io_coords.shape[0]:
            return None

        if target_instance_id == moving_instance_id and candidate_tensor is not None:
            return candidate_tensor
        return io_coords[io_idx]

    def compute_instance_move_hpwl(self,
                                   instance_id: int,
                                   logic_coords: torch.Tensor,
                                   io_coords: Optional[torch.Tensor],
                                   include_io: bool,
                                   candidate_pos: Optional[Tuple[float, float]] = None,
                                   candidate_tensor: Optional[torch.Tensor] = None) -> float:
        site_name = self.get_site_inst_name_by_id_func(instance_id)
        if site_name is None or not self.has_net(site_name):
            return 0.0

        if candidate_tensor is None and candidate_pos is not None:
            candidate_tensor = self._candidate_pos_to_tensor(instance_id,
                                                             candidate_pos,
                                                             logic_coords,
                                                             io_coords)

        total_hpwl = 0.0
        connected_nets = self.site_to_nets.get(site_name, [])

        for net_name in connected_nets:
            connected_sites = self.net_to_sites[net_name]
            coords_list: List[torch.Tensor] = []
            for connected_site in connected_sites:
                target_id = self.get_site_inst_id_by_name_func(connected_site)
                coord = self._get_coord_for_instance(target_id,
                                                     logic_coords,
                                                     io_coords,
                                                     include_io,
                                                     instance_id,
                                                     candidate_tensor)
                if coord is not None:
                    coords_list.append(coord)

            if len(coords_list) < 2:
                continue

            hpwl, _ = self.hpwl_calculator._compute_hpwl_from_coordinates(coords_list)
            total_hpwl += hpwl

        return total_hpwl

    def compute_instance_move_hpwl_batch(self,
                                         instance_id: int,
                                         logic_coords: torch.Tensor,
                                         io_coords: Optional[torch.Tensor],
                                         include_io: bool,
                                         candidate_positions: List[Tuple[float, float]]) -> List[float]:
        if not candidate_positions:
            return []

        candidate_tensors = [
            self._candidate_pos_to_tensor(instance_id, pos, logic_coords, io_coords)
            for pos in candidate_positions
        ]

        if len(candidate_tensors) < self.hpwl_parallel_threshold or self.hpwl_workers == 1:
            return [
                self.compute_instance_move_hpwl(instance_id,
                                                logic_coords,
                                                io_coords,
                                                include_io,
                                                candidate_tensor=tensor)
                for tensor in candidate_tensors
            ]

        executor = self._ensure_hpwl_executor()
        futures = [
            executor.submit(self.compute_instance_move_hpwl,
                             instance_id,
                             logic_coords,
                             io_coords,
                             include_io,
                             None,
                             tensor)
            for tensor in candidate_tensors
        ]
        return [future.result() for future in futures]

    def _ensure_hpwl_executor(self) -> ThreadPoolExecutor:
        if self._hpwl_executor is None:
            workers = max(1, min(self.hpwl_workers, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else self.hpwl_workers))
            self._hpwl_executor = ThreadPoolExecutor(max_workers=workers)
        return self._hpwl_executor

    def shutdown_hpwl_executor(self):
        if self._hpwl_executor is not None:
            self._hpwl_executor.shutdown(wait=False)
            self._hpwl_executor = None