import sys
sys.path.append('.')
import os
import json
import numpy as np
import random
import io
import sys
import torch
import rapidwright
from contextlib import contextmanager
from fem_placer.net import NetManager
from fem_placer.grid import Grid
from fem_placer.hollow_grid import HollowGrid
from fem_placer.instance import InstanceGroup
from fem_placer.config import *
from fem_placer.logger import INFO, WARNING, ERROR

from com.xilinx.rapidwright.design import Design
from com.xilinx.rapidwright.device import Device, Site, SiteTypeEnum
from com.xilinx.rapidwright.rwroute import RWRoute



class FpgaPlacer:

    def __init__(self, 
                 config=None,
                 place_orientation = PlaceType.CENTERED, 
                 grid_type = GridType.SQUARE,
                 place_mode = IoMode.NORMAL,
                 utilization_factor = 0.3,
                 debug = True,
                 device = 'cpu',
                 record_mode = 'inverse_sqr',
                 map_mode = 'no', 
                 net_offset_coeff: float = 1.0,
                 hpwl_workers = None,
                 hpwl_parallel_threshold = 4):
        
        # If a config object is given, pull parameters from it
        if config is not None:
            regions = getattr(config, 'regions', ['logic', 'io'])
            raw_gt = getattr(config, 'grid_type', None)
            if raw_gt is not None:
                grid_type = raw_gt if isinstance(raw_gt, GridType) else GridType[raw_gt]
            raw_pm = getattr(config, 'place_mode', None)
            if raw_pm is not None:
                place_mode = raw_pm if isinstance(raw_pm, IoMode) else IoMode[raw_pm]
            utilization_factor = getattr(config, 'utilization_factor', utilization_factor)
            verbose = getattr(config, 'verbose', debug)
            device = getattr(config, 'dev', device)
            record_mode = getattr(config, 'record_mode', record_mode)
            map_mode = getattr(config, 'map_mode', map_mode)
            net_offset_coeff = getattr(config, 'net_offset_coeff', net_offset_coeff)
            hpwl_workers = getattr(config, 'hpwl_workers', hpwl_workers)
            hpwl_parallel_threshold = getattr(config, 'hpwl_parallel_threshold', hpwl_parallel_threshold)
        else:
            regions = ['logic', 'io']

        self.regions = regions
        self.total_insts_num = 0
        self.other_insts_num = 0

        # Auto-create instance groups and grids for declared regions
        self.instances = {}
        self.grids = {}
        for r in regions:
            self.instances[r] = InstanceGroup(r, device)
            if r == 'io' and place_mode == IoMode.VIRTUAL_NODE:
                self.grids[r] = HollowGrid(name=r, device=device)
            else:
                self.grids[r] = Grid(name=r, device=device)
        # Always create clock and sites
        self.instances['sites'] = InstanceGroup('sites', device)

        self.unfixed_placements = {}
        self.fixed_placements = {}
        self.place_orientation = place_orientation
        self.grid_type = grid_type
        self.place_mode = place_mode
        self.utilization_factor = utilization_factor
        self.debug = debug
        self.net_manager = NetManager(self.get_site_inst_id_by_name, 
                          self.get_inst_name_by_id,
                          self.map_coords_to_instance,
                          debug=self.debug,
                          device=device,
                          record_mode=record_mode,
                          map_mode=map_mode,
                          offset_coeff=net_offset_coeff,
                          hpwl_workers=hpwl_workers,
                          hpwl_parallel_threshold=hpwl_parallel_threshold)
        self.record_mode = record_mode
        self.map_mode = map_mode
        self.net_offset_coeff = net_offset_coeff
        self.hpwl_workers = hpwl_workers
        self.hpwl_parallel_threshold = hpwl_parallel_threshold
        
        self.logic_site_coords = None
        self.site_coords_all = None
        self.device = device
        self.constraint_alpha = 0
        self.constraint_beta = 0
        self.instance_name = None
        self.result_dir = 'result'
        pass
    
    def close(self):
        if hasattr(self, 'net_manager') and self.net_manager is not None:
            self.net_manager.shutdown_hpwl_executor()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_instance_name(self, instance_name, result_dir='result'):
        self.instance_name = instance_name
        self.result_dir = result_dir
        os.makedirs(os.path.join(result_dir, instance_name), exist_ok=True)
        self.net_manager.set_debug_path(result_dir, instance_name)
    
    def get_debug_output_path(self, filename):
        if self.instance_name:
            return os.path.join(self.result_dir, self.instance_name, filename)
        return os.path.join(self.result_dir, filename)
    
    def with_io(self):
        if self.place_orientation == PlaceType.IO:
            return True
        return False
    
    def set_alpha(self, alpha):
        self.constraint_alpha = alpha
        
    def set_beta(self, beta):
        self.constraint_beta = beta
    
    def get_grid(self, grid_name) -> Grid:
        return self.grids[grid_name]
    
    @contextmanager
    def suppress_rapidwright_output(self):
        original_py_out = sys.stdout
        original_py_err = sys.stderr
        
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            from java.lang import System
            from java.io import PrintStream
            from java.io import ByteArrayOutputStream
            
            original_java_out = System.out
            original_java_err = System.err
            
            baos = ByteArrayOutputStream()
            null_stream = PrintStream(baos)
            
            System.setOut(null_stream)
            System.setErr(null_stream)
            
            yield
            
        finally:
            sys.stdout = original_py_out
            sys.stderr = original_py_err
            
            try:
                System.setOut(original_java_out)
                System.setErr(original_java_err)
            except NameError:
                pass
        
    def get_site_inst_id_by_name(self, site_name):
        # Dynamic region-aware lookup — iterate declared regions in order
        # Note: create_mappings already assigns globally-unique IDs with offsets,
        # so get_id() already returns the final global ID — do NOT add offset again.
        for r in self.regions:
            if self.instances[r].has_name(site_name):
                return self.instances[r].get_id(site_name)
        WARNING(f"Cannot find site_inst id for site_name: {site_name}")
        return None
        
    def get_inst_name_by_id(self, id):
        # Dynamic region-aware lookup — iterate declared regions in order
        offset = 0
        for r in self.regions:
            if self.instances[r].has_id(id - offset):
                return self.instances[r].get_name(id - offset)
            offset += self.instances[r].num
        WARNING(f"Cannot find site_inst name for id: {id}")
        return None

    def get_site_id_by_name(self, site_name):
        return self.instances['sites'].name_to_id.get(site_name)
    
    def get_site_name_by_id(self, id):
        return self.instances['sites'].id_to_name.get(id)

    def classify_instances(self, design):
        # 1. Collect instances include IO (IOB), not used in module run
        if self.place_mode == IoMode.NORMAL:
            for site_inst in design.getSiteInsts():
                site_type = site_inst.getSiteTypeEnum()
                if site_type in SLICE_SITE_ENUM:
                    self.instances['logic'].add(site_inst)
                elif site_type in DSP_SITE_ENUM:
                    if 'dsp' in self.instances:
                        self.instances['dsp'].add(site_inst)
                    else:
                        continue
                elif site_type in IO_SITE_ENUM:
                    if 'io' in self.instances:
                        self.instances['io'].add(site_inst)
                elif site_type in OTHER_SITE_ENUM:
                    continue
                elif site_inst not in self.instances['logic'].insts and \
                     ('dsp' not in self.instances or site_inst not in self.instances['dsp'].insts) and \
                     ('io' not in self.instances or site_inst not in self.instances['io'].insts):
                    WARNING(f"Site {site_inst.getName()} with type {site_type} is not classified as optimizable or fixed.")
        # 2. Collect instances boundary node as virtual (IO), used in module run
        elif self.place_mode == IoMode.VIRTUAL_NODE:
            io_file = os.path.join(self.result_dir, self.instance_name, 'io_locations.txt')
            io_site_names = set()
            if os.path.exists(io_file):
                with open(io_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            io_site_names.add(parts[1]) # record site name, e.g. SLICE_X15Y148
            else:
                WARNING(f"IO locations file not found: {io_file}")

            for site_inst in design.getSiteInsts():
                site_type = site_inst.getSiteTypeEnum()
                
                # Check for regular IOs, and other sites
                if site_type in IO_SITE_ENUM or site_type in OTHER_SITE_ENUM:
                     continue

                # RapidWright provides site name (which inherently maps to the fixed location in this case)
                is_boundary = site_inst.getSiteName() in io_site_names
                
                if is_boundary:
                    if 'io' in self.instances:
                        self.instances['io'].add(site_inst)
                elif site_type in SLICE_SITE_ENUM:
                    self.instances['logic'].add(site_inst)
                else:
                    WARNING(f'Cannot orient site type, site: {site_inst.getName()}, type: {site_type}')
    
    def get_available_target_sites(self, device):
        logic_grid = self.grids['logic']
        for site in device.getAllSites():
            site_x = site.getInstanceX()
            site_y = site.getInstanceY()
            
            if (logic_grid.start_x <= site_x <= logic_grid.end_x and
                logic_grid.start_y <= site_y <= logic_grid.end_y):

                site_type = site.getSiteTypeEnum()

                if site_type in SLICE_SITE_ENUM:
                    self.instances['sites'].add(site)
                elif site_type in DSP_SITE_ENUM:
                    # Only track DSP sites if the 'dsp' region is active
                    if 'dsp_sites' not in self.instances:
                        self.instances['dsp_sites'] = InstanceGroup('dsp_sites', self.device)
                    self.instances['dsp_sites'].add(site)

        if self.instances['sites'].num < self.instances['logic'].num:
            WARNING(f"Available sites({self.instances['sites'].num}) less than logic sites({self.instances['logic'].num})")
        
    def get_ids(self):
        return tuple(self.instances[r].ids for r in self.regions)

    def _map_site_to_id(self):
        offset = 0
        for r in self.regions:
            offset = self.instances[r].create_mappings(offset)
        # Target sites — always create for 'sites' and each region's _sites
        self.instances['sites'].create_mappings(0)
        for r in self.regions:
            sname = r + '_sites'
            if sname in self.instances:
                self.instances[sname].create_mappings(0)
        
        if self.debug:
        # Write debug files
            with open(self.get_debug_output_path('logic_inst_mapping_debug.tsv'), 'w') as f:
                f.write("Type\tSiteInst_Name\tID\n")
                for idx, inst in enumerate(self.instances['logic'].insts):
                    site_name = inst.getName()
                    f.write(f"Optimizable\t{site_name}\t{idx}\n")
                f.write(f"TOTAL\t{len(self.instances['logic'])}\n\n")
                
            if 'io' in self.instances:
                with open(self.get_debug_output_path('io_inst_mapping_debug.tsv'), 'w') as f:
                    f.write("Type\tSiteInst_Name\tID\n")
                    for idx, inst in enumerate(self.instances['io'].insts):
                        site_name = inst.getName()
                        f.write(f"Fixed\t{site_name}\t{idx + offset}\n")
                    f.write(f"TOTAL\t{len(self.instances['io'])}\n\n")

    def _init_place_areas(self, design):
        self.total_insts_num = len(design.getSiteInsts())
        classified = sum(self.instances[r].num for r in self.regions if r in self.instances)
        self.other_insts_num = self.total_insts_num - classified
        self.constraint_alpha = self.instances['logic'].num / 2

        stat_parts = ' + '.join(f"{self.instances[r].num} {r}" for r in self.regions if r in self.instances)
        INFO(f"Sites stat: {stat_parts}, {self.other_insts_num} other, "
             f"total {self.total_insts_num} sites.")
        
        if self.place_mode == IoMode.VIRTUAL_NODE and 'io' in self.grids:
            dim_file = os.path.join(self.result_dir, self.instance_name, 'io_dimensions.txt')
            if os.path.exists(dim_file):
                with open(dim_file, 'r') as f:
                    parts = f.read().strip().split()
                    if len(parts) >= 2:
                        io_length = int(parts[0])
                        io_height = int(parts[1])
                        thickness = self.grids['io'].thick
                        if len(parts) >= 3:
                            thickness = int(parts[2])
                            self.grids['io'].thick = thickness
                        
                        # Logic area is the bounding box minus the boundary thickness on both sides
                        # Note: If the dimensions provided in tx are just the overall size of the slice grid used
                        area_length = max(1, io_length - 2 * thickness)
                        area_height = max(1, io_height - 2 * thickness)
                        
                        INFO(f"Virtual IO: Overriding logic area shape to ({area_length}x{area_height}) based on {dim_file}")
        elif self.grid_type == GridType.SQUARE:
            area_length = int(np.ceil(np.sqrt(self.instances['logic'].num / self.utilization_factor)))
            area_height = area_length
        elif self.grid_type == GridType.RECTAN:
            base_area = self.instances['logic'].num / self.utilization_factor
            logic_depth = self.net_manager.logic_depth
            
            aspect_ratio = np.sqrt(logic_depth)
            area_length = int(np.ceil(np.sqrt(base_area * aspect_ratio)))
            area_height = int(np.ceil(np.sqrt(base_area / aspect_ratio)))
            INFO(f"Using RECT grid: logic_depth_factor={logic_depth:.3f}, aspect_ratio={aspect_ratio:.3f}")
        else:
            # Default fallback
            area_length = int(np.ceil(np.sqrt(self.instances['logic'].num / self.utilization_factor)))
            area_height = 0
            
        # Optional override for Virtual Node mode to align logic grid to physical bounded area
        if self.place_mode == IoMode.VIRTUAL_NODE and 'io' in self.grids:
            thickness = self.grids['io'].thick
            start_x = thickness
            end_x = start_x + area_length
            start_y = thickness
            end_y = start_y + area_height
        else:
            start_x = 0
            end_x = start_x + area_length
            start_y = 0 - area_height // 2
            end_y = start_y + area_height
        
        logic_grid = self.grids['logic']
        logic_grid.start_x = start_x
        logic_grid.start_y = start_y
        logic_grid.area_length = area_length
        logic_grid.area_width = area_height
        logic_grid.__post_init__()
        
        if self.instances['logic'].num > logic_grid.area:
            ERROR(f"Logic instances num ({self.instances['logic'].num}) exceeds logic grid area ({logic_grid.area})")
        
        utilization = self.instances['logic'].num / (area_length * area_height)
        
        INFO(f"Logic grid {area_length} x {area_height}, start=({start_x}, {start_y}), utilization {utilization:.3f}")

        # --- DSP grid: place DSPs below the logic grid (if 'dsp' region is present) ---
        if 'dsp' in self.instances and 'dsp' in self.grids:
            dsp_grid = self.grids['dsp']
            dsp_grid.start_x = start_x
            dsp_grid.start_y = start_y - max(1, self.instances['dsp'].num)
            dsp_grid.area_length = max(1, area_length)
            dsp_grid.area_width = max(1, self.instances['dsp'].num)
            dsp_grid.__post_init__()

            if self.instances['dsp'].num > dsp_grid.area:
                ERROR(f"DSP instances num ({self.instances['dsp'].num}) exceeds DSP grid area ({dsp_grid.area})")
            INFO(f"DSP grid {dsp_grid.area_length} x {dsp_grid.area_width}, start=({dsp_grid.start_x}, {dsp_grid.start_y})")

    def _init_io_area(self):
        
        if self.place_mode == IoMode.VIRTUAL_NODE:
            # IO area surrounds logic area
            io_grid = self.grids['io']
            thickness = io_grid.thick
            
            logic_start_x = self.grids['logic'].start_x
            logic_start_y = self.grids['logic'].start_y

            io_grid.start_x = logic_start_x - thickness
            io_grid.start_y = logic_start_y - thickness
            
            dim_file = os.path.join(self.result_dir, self.instance_name, 'io_dimensions.txt')
            if os.path.exists(dim_file):
                with open(dim_file, 'r') as f:
                    parts = f.read().strip().split()
                    if len(parts) >= 2:
                        io_grid.area_length = int(parts[0])
                        io_grid.area_width = int(parts[1])
                        if len(parts) >= 3:
                            io_grid.thick = int(parts[2])
                        INFO(f"Loaded IO boundary dimensions loaded from {dim_file}: {io_grid.area_length}x{io_grid.area_width} with thickness {io_grid.thick}")
                    else:
                        io_grid.area_length = self.grids['logic'].area_length + 2 * thickness
                        io_grid.area_width = self.grids['logic'].area_width + 2 * thickness
            else:
                io_grid.area_length = self.grids['logic'].area_length + 2 * thickness
                io_grid.area_width = self.grids['logic'].area_width + 2 * thickness
            
            # For virtual node mode, io grid is essentially a hollow ring around logic grid
            # and we will enforce boundary constraints in the optimizer/legalizer
            io_grid.__post_init__()
            
            if self.instances['io'].num > io_grid.area:
                ERROR(f"Virtual IO instances num ({self.instances['io'].num}) exceeds Virtual IO grid area ({io_grid.area})")
            
            INFO(f"Virtual IO area (Boundary) - position: ({io_grid.start_x}, {io_grid.start_y}) to ({io_grid.end_x}, {io_grid.end_y})")
            return

        num_pins = self.instances['io'].num
        
        io_length = 1
        io_width = num_pins + num_pins // 15
        
        io_start_x = self.grids['logic'].start_x - io_length       
        bbox_center_y = self.grids['logic'].center_y
        io_start_y = bbox_center_y - io_width // 2

        io_grid = self.grids['io']
        io_grid.start_x = io_start_x
        io_grid.start_y = io_start_y
        io_grid.area_length = io_length
        io_grid.area_width = io_width
        io_grid.__post_init__()
        
        if self.instances['io'].num > io_grid.area:
            ERROR(f"IO instances num ({self.instances['io'].num}) exceeds IO grid area ({io_grid.area})")
        
        INFO(f"Left IO area - position: ({io_start_x}, {io_start_y}) to ({io_grid.end_x}, {io_grid.end_y})")

    def _init_clock_buffer_area(self):
        num_clk_buf = self.instances['clock'].num
        clock_length = 1
        clock_height = num_clk_buf
        
        clock_start_x = self.grids['clock'].start_x
        bbox_center_y = self.grids['clock'].center_y
        clock_start_y = bbox_center_y - clock_height // 2
        
        clock_grid = self.grids['clock']
        clock_grid.start_x = clock_start_x
        clock_grid.start_y = clock_start_y
        clock_grid.area_length = clock_length
        clock_grid.area_width = clock_height
        clock_grid.__post_init__()
        
        INFO(f"Clock buffer area - position: ({clock_start_x}, {clock_start_y}) to ({clock_grid.end_x}, {clock_grid.end_y})")

    def random_initial_placement(self, design):
    
        design.unplaceDesign()
        
        INFO(f"Logic instances num: {self.instances['logic'].num}, available sites num: {self.instances['sites'].num}")

        random.shuffle(self.instances['sites'].insts)

        placed_count = 0
        for i, site in enumerate(self.instances['logic'].insts):
            if i < self.instances['sites'].num:
                target_site = self.instances['sites'].insts[i]

                if self.is_site_compatible(site, target_site):
                    # self.place_site(site, target_site)
                    
                    self.unfixed_placements[site.getName()] = {
                        'target_site': target_site,
                        'source_site': site,
                        'target_x': target_site.getInstanceX(),
                        'target_y': target_site.getInstanceY()
                    }
                    placed_count += 1
                else:
                    self.unfixed_placements[site.getName()] = {
                        'target_site': None,
                        'source_site': None,
                        'target_x': -1,
                        'target_y': -1,
                        'bel': None,
                    }
                    ERROR(f"Site {site.getName()}({site.getSiteTypeEnum().name}) is not compatible {target_site.getName()}({target_site.getSiteTypeEnum().name})")
            else:
                self.unfixed_placements[site.getName()] = {
                    'target_site': None,
                    'source_site': None,
                    'target_x': -1,
                    'target_y': -1,
                    'bel': None,
                }
                ERROR(f"No more site for {site.getName()}")

        # Fixed sites
        for fixed_site in (self.instances['io'].insts if 'io' in self.instances else []):
            self.fixed_placements[fixed_site.getName()] = {
                'target_site': None,
                'target_site': fixed_site,
                'target_x': fixed_site.getInstanceX(),
                'target_y': fixed_site.getInstanceY(),
                'bel': None,
            }

        INFO(f"Initial placement {placed_count} for total {len(self.cells)} sites done, {len(self.fixed_placements)} remains.")

    def is_site_compatible(self, source_site, target_site):
        source_type = source_site.getSiteTypeEnum()
        target_type = target_site.getSiteTypeEnum()
        
        if source_type in [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]:
            return target_type in [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]
        
        if source_type in [SiteTypeEnum.IOB33, SiteTypeEnum.IOB33M]:
            return target_type in [SiteTypeEnum.IOB33, SiteTypeEnum.IOB33M]
        
        if source_type in [SiteTypeEnum.IOB18, SiteTypeEnum.IOB18M]:
            return target_type in [SiteTypeEnum.IOB18, SiteTypeEnum.IOB18M]
        
        return source_type == target_type
  
    def init_placement(self, dcp_file = '', edf_file='', dcp_output=''):
        # Design.setAutoGenerateReadableEdif (false)
        with self.suppress_rapidwright_output():
            design = Design.readCheckpoint(dcp_file)
        INFO(f"Reading DCP: {dcp_file} success")
        self.design = design  # Keep the Design object for downstream use (path-based timing, Vivado feedback)
        # routing report
        # design.unrouteDesign()

        # timing_driven = False
        # if timing_driven:
        #     routed_design = RWRoute.routeDesignFullTimingDriven(design)
        # else:
        #     routed_design = RWRoute.routeDesignFullNonTimingDriven(design)
        device = design.getDevice()
        self.available_slices_ml = list(device.getAllCompatibleSites(SiteTypeEnum.SLICEL))
        self.available_slices_ml.extend(list(device.getAllCompatibleSites(SiteTypeEnum.SLICEM)))
        self.cells = design.getCells()
        self.classify_instances(design)
        self._init_place_areas(design)
        if 'io' in self.instances:
            self._init_io_area()
        if 'clock' in self.instances:
            self._init_clock_buffer_area()
        self.get_available_target_sites(device)
        self._map_site_to_id()
        vivado_hpwl = self.net_manager.analyze_design_hpwl(design, 
                                                logic_instances=self.instances['logic'],
                                                io_instances=self.instances.get('io'))
        net_num = self.net_manager.analyze_nets(self.instances['logic'], 
                                                self.instances.get('io'))
        # self.random_initial_placement(design)
        
        # Generate region coordinate tensors
        for r in self.regions:
            getter = getattr(self, f'_get_{r}_area_coords', None)
            if getter is not None:
                getter()
        self._get_combined_coords()

        inst_num = {f'{r}_inst_num': self.instances[r].num for r in self.regions}

        return vivado_hpwl, inst_num, net_num

    def save_init_params(self, instance_name, result_dir='result'):
        os.makedirs(os.path.join(result_dir, instance_name), exist_ok=True)
        self.net_manager.set_debug_path(result_dir, instance_name)
        
        params = {
            'num_inst': self.instances['logic'].num,
            'num_fixed_inst': self.instances['io'].num if 'io' in self.instances else 0,
            'num_site': self.grids['logic'].area,
            'num_fixed_site': self.grids['io'].area if 'io' in self.grids else 0,
            'logic_grid_width': self.grids['logic'].area_width,
            'constraint_alpha': self.constraint_alpha,
            'constraint_beta': self.constraint_beta,
            'device': self.device,
            'place_orientation': self.place_orientation.name if hasattr(self.place_orientation, 'name') else str(self.place_orientation),
            'grid_type': self.grid_type.name if hasattr(self.grid_type, 'name') else str(self.grid_type),
            'utilization_factor': self.utilization_factor,
            'with_io': self.with_io(),
        }
        
        tensor_data = {}
        
        if self.net_manager.insts_matrix is not None:
            tensor_data['coupling_matrix'] = self.net_manager.insts_matrix.cpu().numpy().tolist()
        
        if self.logic_site_coords is not None:
            tensor_data['site_coords_matrix'] = self.logic_site_coords.cpu().numpy().tolist()
        
        if self.net_manager.io_insts_matrix is not None:
            tensor_data['io_site_connect_matrix'] = self.net_manager.io_insts_matrix.cpu().numpy().tolist()
        
        if self.io_site_coords is not None:
            tensor_data['io_site_coords'] = self.io_site_coords.cpu().numpy().tolist()
        
        output_path = os.path.join(result_dir, instance_name, 'init_params.json')
        with open(output_path, 'w') as f:
            json.dump({'params': params, 'tensors': tensor_data}, f, indent=2)
        
        INFO(f"Saved init parameters to {output_path}")
        return output_path

    def map_coords_to_instance(self, *region_coords):
        """Map region coordinates to instance names.
        
        Args:
            *region_coords: One tensor per region, in the order of ``self.regions``.
        """
        instance_coords = {}
        offset = 0
        for r, coords in zip(self.regions, region_coords):
            if coords is None:
                continue
            for instance_id in range(len(coords)):
                site_name = self.get_inst_name_by_id(instance_id + offset)
                instance_coords[site_name] = coords[instance_id]
            offset += self.instances[r].num
        return instance_coords
        
    def _get_logic_area_coords(self):
        place_length = self.grids['logic'].area_length
        place_width = self.grids['logic'].area_width
        
        self.logic_site_coords = self.grids['logic'].to_real_coords_tensor(torch.cartesian_prod(
            torch.arange(place_length, dtype=torch.float32, device=self.device),
            torch.arange(place_width, dtype=torch.float32, device=self.device)
        ))

    def _get_dsp_area_coords(self):
        dsp_grid = self.grids['dsp']
        
        self.dsp_site_coords = self.grids['dsp'].to_real_coords_tensor(torch.cartesian_prod(
            torch.arange(dsp_grid.area_length, dtype=torch.float32, device=self.device),
            torch.arange(dsp_grid.area_width, dtype=torch.float32, device=self.device)
        ))

    def _get_io_area_coords(self):
        if self.place_mode == IoMode.VIRTUAL_NODE:
            io_grid = self.grids['io']
            # _empty_positions stores real coordinates. By default, it's sorted by x then y (column-major)
            self.io_site_coords = torch.tensor(io_grid._empty_positions, dtype=torch.float32, device=self.device)
        else:
            place_width = self.grids['io'].area_width 

            self.io_site_coords = self.grids['io'].to_real_coords_tensor(torch.cartesian_prod(
                torch.tensor([0], dtype=torch.float32, device=self.device),
                torch.arange(place_width, dtype=torch.float32, device=self.device) 
            ))
        
    def _get_combined_coords(self):
        tensors = []
        for r in self.regions:
            attr = f'{r}_site_coords'
            if hasattr(self, attr):
                tensors.append(getattr(self, attr).clone())
        # IO coords may need Y-offset alignment (always last in regions)
        if 'io' in self.regions and hasattr(self, 'io_site_coords'):
            io_coords = self.io_site_coords.clone()
            place_height = self.grids['logic'].area_width
            io_height = self.grids['io'].area_width
            logic_center_y = (place_height - 1) / 2.0
            io_center_y = (io_height - 1) / 2.0
            io_coords[:, 0] = -1.0
            io_coords[:, 1] += logic_center_y - io_center_y
            # Replace the raw io coords in tensors with adjusted version
            io_idx = self.regions.index('io')
            tensors[io_idx] = io_coords
        self.site_coords_all = torch.cat(tensors, dim=0) if tensors else None
        
    def place(self, coords, io_coords=None, include_io=False):
        # TODO replace the legalized logic into FPGA
        min_x = min(site.getInstanceX() for site in self.available_slices_ml)
        max_x = max(site.getInstanceX() for site in self.available_slices_ml)
        min_y = min(site.getInstanceY() for site in self.available_slices_ml)
        max_y = max(site.getInstanceY() for site in self.available_slices_ml)

        device_center_x = (min_x + max_x) // 2
        device_center_y = (min_y + max_y) // 2