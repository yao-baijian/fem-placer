import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .grid import Grid
from .logger import WARNING
from .objectives import get_loss_history, get_placement_history

RECT_WIDTH = 0.6
RECT_HEIGHT = 0.6
INSTANCE_WIDTH = 0.5
INSTANCE_HEIGHT = 0.5

class PlacementDrawer:

    def __init__(self, placer, num_subplots=5, debug_mode=False):
        self.placer = placer

        self.logic_grid: Grid = placer.get_grid('logic')
        self.io_grid: Grid | None = placer.get_grid('io') if 'io' in placer.grids else None
        self._calculate_overall_bbox(False)
        self.debug_mode = debug_mode

        self.site_colors = {
            'logic_empty': {'face': "#B8B8B8", 'edge': "#555454"},
            'logic_placed': {'face': '#64B5F6', 'edge': '#1976D2'},
            'io_empty': {'face': '#FFF176', 'edge': '#FFB300'},
            'io_placed': {'face': '#81C784', 'edge': '#388E3C'},
            'text': "#282727"
        }

        self.wire_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)

        self.figs = None
        self.axes = None

        if debug_mode:
            self._init_debug_interface()

        path = '/usr/share/fonts/opentype/linux-libertine/LinLibertine_RI.otf'
        try:
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
        except (FileNotFoundError, OSError):
            pass  # Use default font if Linux Libertine not available

        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['axes.edgecolor'] = "#C9C4C4"
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['grid.alpha'] = 0.7

    def _calculate_overall_bbox(self, include_io):
        all_grids = [self.logic_grid]
        if include_io:
            all_grids.append(self.io_grid)

        min_x = min([grid.start_x for grid in all_grids])
        max_x = max([grid.end_x for grid in all_grids])
        min_y = min([grid.start_y for grid in all_grids])
        max_y = max([grid.end_y for grid in all_grids])

        self.overall_width = max_x - min_x
        self.overall_height = max_y - min_y
        self.offset_x = -min_x
        self.offset_y = -min_y

        self.grid_bounds = {}
        for grid in all_grids:
            self.grid_bounds[grid.name] = {
                'start_x': grid.start_x + self.offset_x,
                'end_x': grid.end_x + self.offset_x,
                'start_y': grid.start_y + self.offset_y,
                'end_y': grid.end_y + self.offset_y
            }

    def _normalize_coords(self, x, y):
        return x + self.offset_x, y + self.offset_y
    
    def _get_grid_for_position(self, x, y):
        real_x, real_y = x - self.offset_x, y - self.offset_y

        if self.logic_grid.is_within_bounds(real_x, real_y):
            return 'logic'
        
        if self.io_grid and self.io_grid.is_within_bounds(real_x, real_y):
            return 'io'
        
        return None

    def setup_plot(self, ax):
        padding = 2
        ax.set_xlim(-padding, self.overall_width + padding)
        ax.set_ylim(-padding, self.overall_height + padding)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # self._draw_grid_boundaries(ax)

    def _draw_grid_boundaries(self, ax):
        for grid_name, bounds in self.grid_bounds.items():
            width = bounds['end_x'] - bounds['start_x'] + 1
            height = bounds['end_y'] - bounds['start_y'] + 1

            rect = patches.Rectangle(
                (bounds['start_x'] - 0.5, bounds['start_y'] - 0.5),
                width, height,
                linewidth=3, linestyle='--',
                edgecolor=self.site_colors[f'{grid_name}_empty']['edge'],
                facecolor='none', alpha=0.5
            )
            ax.add_patch(rect)

            ax.text(
                bounds['start_x'] + width/2,
                bounds['start_y'] + height/2,
                grid_name.upper(),
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color=self.site_colors[f'{grid_name}_empty']['edge'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

    def draw_placement(self, logic_coords, io_coords=None, include_io=False, iteration=0, title_suffix=""):
        self.ax.clear()
        self.setup_plot(self.ax)
        self._draw_all_base_grids(self.ax, include_io)
        self._draw_instances(self.ax, logic_coords, 'logic', False)

        if include_io:
            self._draw_instances(self.ax, io_coords, 'io', False)

        title = f'Placement - Iteration {iteration}'
        if title_suffix:
            title += f' - {title_suffix}'
        self.ax.set_title(title, fontsize=14)

        self._add_legend()

    def _draw_single_grid_base(self, ax, grid, grid_type):
        for x in range(grid.start_x, grid.end_x + 1):
            for y in range(grid.start_y, grid.end_y + 1):
                plot_x, plot_y = self._normalize_coords(x, y)
                color_config = self.site_colors[f'{grid_type}_empty']

                rect = patches.Rectangle(
                    (plot_x - RECT_WIDTH/2, plot_y - RECT_HEIGHT/2),
                    RECT_WIDTH, RECT_HEIGHT,
                    linewidth= 0.5,
                    edgecolor=color_config['edge'],
                    facecolor=color_config['face'],
                    alpha=0.3
                )
                ax.add_patch(rect)

    def _draw_all_base_grids(self, ax, include_io):
        self._draw_single_grid_base(ax, self.logic_grid, 'logic')

        if include_io:
            self._draw_single_grid_base(ax, self.io_grid, 'io')
        pass

    def _draw_instances(self, ax, coords, grid_type, label=False):
        num_instances = coords.shape[0]
        grid = getattr(self, f'{grid_type}_grid')

        coord_dict = {}
        overlapped_coords = set()

        for i in range(num_instances):
            x = int(coords[i][0].item()) if torch.is_tensor(coords[i][0]) else int(coords[i][0])
            y = int(coords[i][1].item()) if torch.is_tensor(coords[i][1]) else int(coords[i][1])

            if grid.is_within_bounds(x, y):
                coord_key = (x, y)
                if coord_key in coord_dict:
                    coord_dict[coord_key].append(i)
                    overlapped_coords.add(coord_key)
                else:
                    coord_dict[coord_key] = [i]
            else:
                WARNING(f'instance {i} with coords {x, y} out of bounds')

        for i in range(num_instances):
            x = int(coords[i][0].item()) if torch.is_tensor(coords[i][0]) else int(coords[i][0])
            y = int(coords[i][1].item()) if torch.is_tensor(coords[i][1]) else int(coords[i][1])

            if grid.is_within_bounds(x, y):
                plot_x, plot_y = self._normalize_coords(x, y)
                coord_key = (x, y)

                if coord_key in overlapped_coords:
                    rect = patches.Rectangle(
                        (plot_x - INSTANCE_WIDTH/2, plot_y - INSTANCE_HEIGHT/2),
                        INSTANCE_WIDTH, INSTANCE_HEIGHT,
                        linewidth=1, edgecolor='red',
                        facecolor='red', alpha=0.8
                    )
                else:
                    color_config = self.site_colors[f'{grid_type}_placed']
                    rect = patches.Rectangle(
                        (plot_x - INSTANCE_WIDTH/2, plot_y - INSTANCE_HEIGHT/2),
                        INSTANCE_WIDTH, INSTANCE_HEIGHT,
                        linewidth=1, edgecolor=color_config['edge'],
                        facecolor=color_config['face'], alpha=0.8
                    )

                ax.add_patch(rect)

                if label:
                    label = f'{grid_type[0].upper()}{i}'
                    ax.text(plot_x, plot_y, label,
                        ha='center', va='center', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                if coord_key in overlapped_coords and coord_dict[coord_key][0] == i:
                    overlap_count = len(coord_dict[coord_key])
                    if overlap_count > 1:
                        ax.text(plot_x, plot_y, str(overlap_count),
                               ha='center', va='center',
                               fontsize=8, fontweight='bold',
                               color='white')

    def _add_legend(self):
        legend_elements = [
            patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Logic'),
            patches.Patch(facecolor='lightgreen', edgecolor='green', label='IO'),
            patches.Patch(facecolor='red', edgecolor='darkred', label='Overlap')
        ]

        self.ax.legend(handles=legend_elements, loc='upper right',
                      bbox_to_anchor=(0.95, 0.95), fontsize=9)

    def draw_place_and_route(self, logic_coords, routes, io_coords=None,
                             include_io = False, iteration=0, title_suffix=""):
        self.draw_placement(logic_coords, io_coords, include_io, iteration, title_suffix)
        self.draw_routing(routes)

        self.fig.tight_layout()
        self.fig.canvas.draw()

        plt.pause(20)
        plt.savefig(f'final_placement.png', dpi=150, bbox_inches='tight')

    def draw_routing(self, routes, alpha=0.7):
        weights = [route['weight'] for route in routes]
        max_weight = max(weights)
        min_weight = min(weights)
        greens_cmap = cm.Greens

        for route in routes:
            if max_weight > min_weight:
                normalized_weight = (route['weight'] - min_weight) / (max_weight - min_weight)
            else:
                normalized_weight = 0.5
            color = greens_cmap(0.3 + 0.6 * normalized_weight)

            linewidth = 0.8 + normalized_weight * 1.2

            for segment in route['segments']:
                start_x, start_y = segment['start']
                end_x, end_y = segment['end']

                norm_start_x, norm_start_y = self._normalize_coords(start_x, start_y)
                norm_end_x, norm_end_y = self._normalize_coords(end_x, end_y)

                self.ax.plot([norm_start_x, norm_end_x], [norm_start_y, norm_end_y],
                        color=color,
                        linewidth=linewidth,
                        alpha=0.5 * alpha,
                        linestyle='-',
                        zorder=0)

    def _init_debug_interface(self):
        print("Debug mode enabled - interface to be implemented")
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        if event.key == 'd' or event.key == 'D':
            print("Debug command received")

    def add_placement(self, site_coords, step):
        placement_data = {
            'site_coords': site_coords,
            'step': step
        }
        self.placement_history.append(placement_data)

    def draw_multi_step_placement(self, save_path=None):
        step_labels = ['250', '500', '750', '1000']
        placement_history = get_placement_history()
        # Create figure with subplots
        num_plots = len(step_labels)
        self.figs = plt.figure(figsize=(5 * num_plots, 5))
        self.axes = []

        for plot_idx, step_label in enumerate(step_labels):
            ax = self.figs.add_subplot(1, num_plots, plot_idx + 1)
            self.axes.append(ax)
            site_coords = placement_history[plot_idx][0]
            # Setup plot
            
            real_logic_coords = self.logic_grid.to_real_coords_tensor(site_coords)
            self.setup_plot(ax)
            self._draw_all_base_grids(ax, include_io=False)
            self._draw_instances(ax, real_logic_coords, 'logic', label=False)
            
            # 去掉坐标轴边框和刻度
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.set_title(f'Step {step_label}', fontsize=12, fontweight='bold', y=0)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='both', length=0)

        self.figs.subplots_adjust(wspace=0, hspace=0.05, left=0, right=0.8, bottom=0.1, top=0.95)

        # 在最后一张图的右上角添加图例
        last_ax = self.axes[-1]
        
        # 创建自定义图例元素
        from matplotlib.patches import Patch
        import matplotlib.patches as mpatches
        
        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='gray', label='Empty Site'),
            Patch(facecolor='blue', edgecolor='darkblue', label='Placed Site'),
            Patch(facecolor='red', edgecolor='darkred', label='Overlap Site')
        ]
        
        # 在外部添加图例（右上角）
        last_ax.legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(0.95, 0.95), fontsize=10, 
                    frameon=True, fancybox=True, shadow=True)

        # self.figs.tight_layout()
        self.figs.canvas.draw()
        plt.show(block=False)
        plt.pause(5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

    def plot_fpga_placement_loss(self, save_path=None, sigma=1.5):
        """
        Plot FPGA placement loss with Gaussian smoothing.
        
        Args:
            save_path: Path to save the figure
            sigma: Standard deviation for Gaussian filter (default: 1.5)
        """
        loss_data = get_loss_history()
        steps = np.array(list(range(len(loss_data['hpwl_losses']))))
        
        # Convert loss data to numpy arrays
        hpwl_losses = np.array(loss_data['hpwl_losses'])
        constrain_losses = np.array(loss_data['constrain_losses'])
        total_losses = np.array(loss_data['total_losses'])
        
        # Apply Gaussian filter for smoothing
        hpwl_smooth = gaussian_filter1d(hpwl_losses, sigma=sigma)
        constrain_smooth = gaussian_filter1d(constrain_losses, sigma=sigma)
        total_smooth = gaussian_filter1d(total_losses, sigma=sigma)
        
        colors = ["#53D5F9", "#86FAD8", "#DF6FFA"]

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(steps, hpwl_smooth,
                color=colors[0], linewidth=2.5, label='HPWL Loss')
        ax.plot(steps, constrain_smooth,
                color=colors[1], linewidth=2.5, label='Constraint Loss')
        ax.plot(steps, total_smooth,
                color=colors[2], linewidth=2.5, label='Total Loss')

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11)
        
        # Customize grid with fewer lines
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        
        # Format y-axis with scientific notation (10e format)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        plt.tight_layout()
        plt.pause(2)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

    def plot_annealing_comparison(self, annealing_results, save_path=None):
        """
        Plot total loss comparison across different annealing schedules.
        
        Args:
            annealing_results: Dictionary with annealing types as keys and 
                             loss_history dicts as values
                             Example: {
                                 'lin': {'total_losses': [...], ...},
                                 'exp': {'total_losses': [...], ...},
                                 'inverse': {'total_losses': [...], ...}
                             }
            save_path: Path to save the figure
        """
        colors = {'lin': '#FF6B6B', 'exp': '#4ECDC4', 'inverse': '#45B7D1'}
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        for anneal_type, loss_data in annealing_results.items():
            if 'total_losses' in loss_data:
                steps = list(range(len(loss_data['total_losses'])))
                ax.plot(steps, loss_data['total_losses'],
                        color=colors.get(anneal_type, '#000000'),
                        linewidth=2.5, label=f'{anneal_type.upper()} Annealing')
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        # ax.set_title('Total Loss Comparison Across Annealing Schedules', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        
        # Format y-axis with scientific notation (10e format)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        plt.tight_layout()
        plt.pause(2)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show(block=False)