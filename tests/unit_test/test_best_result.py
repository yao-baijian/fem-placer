"""
Multi-worker benchmark script for comparing three FpgaPlacer net modes
across three annealing schedules.

Modes:
  a) baseline inverse-fanout: record_mode='inverse', map_mode='simple'
  b) compare 1 simple:        record_mode='simple',  map_mode='simple'
  c) compare 2 log:           record_mode='simple',  map_mode='log'

Anneals:
  lin, exp, inverse

This script prints best-per-mode results grouped by instance,
and writes a two-panel grouped bar chart:
  - left panel: best final HPWL
  - right panel: runtime (seconds)
"""

import os
import sys
import time
import json
import importlib
import traceback
import threading
import concurrent.futures
from typing import Dict, Any, List

import torch
import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

sys.path.insert(0, '.')

from fem_placer import FpgaPlacer, Legalizer, FPGAPlacementOptimizer
from fem_placer.config import PlaceType, GridType, IoMode
from fem_placer.logger import INFO, WARNING, SET_LEVEL
from ml.dataset import extract_features_from_placer
from ml.predict import predict_alpha


SET_LEVEL('WARNING')

NUM_TRIALS = 5
NUM_STEPS = 400
DEV = os.getenv('FEM_DEV', 'cuda')
MANUAL_GRAD = False
IO_FACTOR = 1
MAX_WORKERS = 8
RESULT_DIR = './result'
SERIALIZE_RAPIDWRIGHT_INIT = os.getenv('FEM_SERIALIZE_RAPIDWRIGHT_INIT', '1') not in ('0', 'false', 'False')
RAPIDWRIGHT_INIT_LOCK = threading.Lock()

# Dict: instance_name -> (alpha_factor, beta_factor, io_factor, net_offset_coeff)
# alpha_factor and beta_factor are multiplied with predicted alpha value
# io_factor controls the IO weight in optimization
# net_offset_coeff controls connectivity offset scaling in NetManager via FpgaPlacer
INSTANCES = {
    # 'c2670_boundary': (1, 1, 700, 0),
    # 'c5315_boundary': (1, 1, 700, 0),
    'c6288_boundary': (1, 1, 1, 0),
    # 'c7552_boundary': (1, 1, 700, 0),
    # 's1488_boundary': (1, 1, 700, 0),
    # 's5378_boundary': (1.5, 1.5, 1, 0),
    # 's9234_boundary': (0.5, 0.5, 1, 0),
    # 's15850_boundary': (0.5, 0.5, 900, 0),
    # 'bgm_boundary': (0.001, 0.001, 1, -1),
    # 'RLE_BlobMerging_boundary': (0.1, 0.1, 1, 1),
    # 'sha1_boundary': (0.1, 0.1, 1, 0),
    # 'FPGA-example1_boundary': (0.01, 0.01, 1, 0),
}

MODES = {
    'inverse-fanout': {'record_mode': 'inverse', 'map_mode': 'no'},
    'inverse-sqr': {'record_mode': 'inverse_sqr', 'map_mode': 'simple'},
    'simple': {'record_mode': 'simple', 'map_mode': 'simple'},
    # 'compare-log': {'record_mode': 'simple', 'map_mode': 'log'},
}

ANNEALS = ['lin', 'exp', 'inverse']


def suppress_java_output_globally() -> bool:
    if os.getenv('FEM_SUPPRESS_JAVA_OUTPUT', '1') in ('0', 'false', 'False'):
        return False

    try:
        java_lang = importlib.import_module('java.lang')
        java_io = importlib.import_module('java.io')
        System = getattr(java_lang, 'System')
        PrintStream = getattr(java_io, 'PrintStream')
        ByteArrayOutputStream = getattr(java_io, 'ByteArrayOutputStream')
        null_stream = PrintStream(ByteArrayOutputStream())
        System.setOut(null_stream)
        System.setErr(null_stream)
        return True
    except Exception:
        return False


def write_all_results_csv(all_csv: str, all_results: List[Dict[str, Any]], all_headers: List[str]) -> None:
    with open(all_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(all_headers) + '\n')
        for row in all_results:
            values = [str(row.get(k, '')) for k in all_headers]
            f.write(','.join(values) + '\n')


def write_best_results_csv(best_csv: str,
                           instances: Dict[str, Any],
                           best_by_instance: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    with open(best_csv, 'w', encoding='utf-8') as f:
        f.write('instance,mode,best_anneal,hpwl_final,overlap,runtime_s,alpha,beta,vivado_hpwl\n')
        for instance in instances:
            best_rows = best_by_instance.get(instance, {})
            vivado_hpwl = ''
            for candidate in best_rows.values():
                if candidate is None:
                    continue
                cand_vivado_hpwl = candidate.get('vivado_hpwl')
                if cand_vivado_hpwl is not None:
                    vivado_hpwl = f'{float(cand_vivado_hpwl):.6f}'
                    break
            for mode_name in MODES.keys():
                row = best_rows.get(mode_name)
                if row is None:
                    f.write(f'{instance},{mode_name},N/A,,,,,,{vivado_hpwl}\n')
                else:
                    f.write(
                        f"{instance},{mode_name},{row['anneal']},{row['hpwl_final']:.6f},{row['overlap']},"
                        f"{row['runtime_s']:.6f},{row['alpha']:.6f},{row['beta']:.6f},{vivado_hpwl}\n"
                    )


def append_failure_jsonl(failure_jsonl: str, row: Dict[str, Any]) -> None:
    failure_record = {
        'timestamp': time.time(),
        'instance': row.get('instance'),
        'mode': row.get('mode'),
        'anneal': row.get('anneal'),
        'runtime_s': row.get('runtime_s'),
        'error': row.get('error'),
        'trace': row.get('trace'),
    }
    with open(failure_jsonl, 'a', encoding='utf-8') as f:
        f.write(json.dumps(failure_record, ensure_ascii=False) + '\n')


def run_single_experiment(instance: str,
                          alpha_factor: float,
                          beta_factor: float,
                          io_factor: float,
                          net_offset_coeff: float,
                          mode_name: str,
                          mode_cfg: Dict[str, str],
                          anneal: str) -> Dict[str, Any]:
    start = time.time()
    place_type = PlaceType.IO

    try:
        if SERIALIZE_RAPIDWRIGHT_INIT:
            with RAPIDWRIGHT_INIT_LOCK:
                fpga_placer = FpgaPlacer(
                    place_orientation=place_type,
                    grid_type=GridType.SQUARE,
                    place_mode=IoMode.VIRTUAL_NODE,
                    utilization_factor=0.4,
                    debug=False,
                    device=DEV,
                    net_offset_coeff=net_offset_coeff,
                    record_mode=mode_cfg['record_mode'],
                    map_mode=mode_cfg['map_mode'],
                )

                fpga_placer.set_instance_name(instance)
                vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(
                    f'./vivado/output_dir/{instance}/post_impl.dcp',
                    f'./vivado/output_dir/{instance}/optimized_placement.pl',
                )
        else:
            fpga_placer = FpgaPlacer(
                place_orientation=place_type,
                grid_type=GridType.SQUARE,
                place_mode=IoMode.VIRTUAL_NODE,
                utilization_factor=0.4,
                debug=False,
                device=DEV,
                net_offset_coeff=net_offset_coeff,
                record_mode=mode_cfg['record_mode'],
                map_mode=mode_cfg['map_mode'],
            )

            fpga_placer.set_instance_name(instance)
            vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(
                f'./vivado/output_dir/{instance}/post_impl.dcp',
                f'./vivado/output_dir/{instance}/optimized_placement.pl',
            )

        row = extract_features_from_placer(fpga_placer, alpha=0, beta=0, with_io=False)
        alpha_pred = predict_alpha(row)
        used_alpha = float(alpha_pred) * alpha_factor
        used_beta = float(alpha_pred) * beta_factor
        fpga_placer.set_alpha(used_alpha)
        fpga_placer.set_beta(used_beta)

        optimizer = FPGAPlacementOptimizer(
            num_inst=fpga_placer.instances['logic'].num,
            num_fixed_inst=fpga_placer.instances['io'].num,
            num_site=fpga_placer.get_grid('logic').area,
            num_fixed_site=fpga_placer.get_grid('io').area,
            coupling_matrix=fpga_placer.net_manager.insts_matrix,
            site_coords_matrix=fpga_placer.logic_site_coords,
            io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
            io_site_coords=fpga_placer.io_site_coords,
            constraint_alpha=fpga_placer.constraint_alpha,
            constraint_beta=fpga_placer.constraint_beta,
            num_trials=NUM_TRIALS,
            num_steps=NUM_STEPS,
            dev=DEV,
            betamin=0.01,
            betamax=0.5,
            anneal=anneal,
            optimizer='adam',
            learning_rate=0.1,
            h_factor=0.01,
            io_factor=io_factor,
            seed=1,
            dtype=torch.float32,
            with_io=True,
            manual_grad=MANUAL_GRAD,
        )

        config, result = optimizer.optimize()
        optimal_inds = torch.argwhere(result == result.min()).reshape(-1)

        legalizer = Legalizer(placer=fpga_placer, device=DEV)
        logic_ids, io_ids = fpga_placer.get_ids()

        real_logic_coords = config[0][optimal_inds[0]]
        real_io_coords = config[1][optimal_inds[0]]

        _, overlap, hpwl_initial, hpwl_final = legalizer.legalize_placement(
            real_logic_coords,
            logic_ids,
            real_io_coords,
            io_ids,
            include_io=True,
        )

        elapsed = time.time() - start
        fpga_placer.close()

        return {
            'ok': True,
            'instance': instance,
            'mode': mode_name,
            'anneal': anneal,
            'record_mode': mode_cfg['record_mode'],
            'map_mode': mode_cfg['map_mode'],
            'alpha': used_alpha,
            'beta': used_beta,
            'hpwl_initial': float(hpwl_initial['hpwl']),
            'hpwl_final': float(hpwl_final['hpwl']),
            'overlap': int(overlap),
            'runtime_s': float(elapsed),
            'logic_inst_num': int(inst_num['logic_inst_num']),
            'io_inst_num': int(inst_num['io_inst_num']),
            'vivado_hpwl': float(vivado_hpwl['hpwl']),
            'logic_net_num': int(net_num['logic_net_num']),
            'total_net_num': int(net_num['total_net_num']),
        }
    except Exception as exc:
        return {
            'ok': False,
            'instance': instance,
            'mode': mode_name,
            'anneal': anneal,
            'error': f'{type(exc).__name__}: {exc}',
            'trace': traceback.format_exc(),
            'runtime_s': float(time.time() - start),
        }


def pick_best_per_mode(results_for_instance: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best = {}
    for mode_name in MODES.keys():
        valid = [r for r in results_for_instance if r.get('ok') and r['mode'] == mode_name]
        if not valid:
            best[mode_name] = None
            continue
        best[mode_name] = min(valid, key=lambda r: (r['hpwl_final'], r['runtime_s']))
    return best


def print_grouped_results(instance: str, best_rows: Dict[str, Dict[str, Any]]):
    print(f'\n=== Instance: {instance} ===', flush=True)
    print(f"{'Mode':<26} {'Best Anneal':<10} {'HPWL Final':<14} {'Overlap':<8} {'Time(s)':<10} {'Alpha':<10} {'Beta':<10}", flush=True)
    print('-' * 96, flush=True)
    for mode_name in MODES.keys():
        row = best_rows.get(mode_name)
        if row is None:
            print(f"{mode_name:<26} {'N/A':<10} {'N/A':<14} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10}", flush=True)
            continue
        print(
            f"{mode_name:<26} {row['anneal']:<10} {row['hpwl_final']:<14.2f} {row['overlap']:<8} "
            f"{row['runtime_s']:<10.2f} {row['alpha']:<10.4f} {row['beta']:<10.4f}",
            flush=True,
        )


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    java_suppressed = suppress_java_output_globally()

    print('Running mode comparison with multi-worker execution', flush=True)
    print(f'  device={DEV}, num_steps={NUM_STEPS}, workers={MAX_WORKERS}', flush=True)
    print(f'  modes={list(MODES.keys())}', flush=True)
    print(f'  anneals={ANNEALS}', flush=True)
    print(f'  java_output_suppressed={java_suppressed}', flush=True)
    print(f'  serialize_rapidwright_init={SERIALIZE_RAPIDWRIGHT_INIT}', flush=True)

    all_results: List[Dict[str, Any]] = []
    best_by_instance: Dict[str, Dict[str, Dict[str, Any]]] = {}

    all_csv = os.path.join(RESULT_DIR, 'mode_anneal_all_results.csv')
    best_csv = os.path.join(RESULT_DIR, 'mode_anneal_best_results.csv')
    failure_jsonl = os.path.join(RESULT_DIR, 'mode_anneal_failures.jsonl')
    fig_path = os.path.join(RESULT_DIR, 'mode_anneal_best_grouped_bars.png')

    failure_count = 0

    all_headers = [
        'ok', 'instance', 'mode', 'anneal', 'record_mode', 'map_mode',
        'alpha', 'beta', 'hpwl_initial', 'hpwl_final', 'overlap', 'runtime_s',
        'logic_inst_num', 'io_inst_num', 'vivado_hpwl', 'logic_net_num', 'total_net_num',
        'error'
    ]

    for instance, (alpha_factor, beta_factor, io_factor, net_offset_coeff) in INSTANCES.items():
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for mode_name, mode_cfg in MODES.items():
                for anneal in ANNEALS:
                    tasks.append(
                        executor.submit(
                            run_single_experiment,
                            instance,
                            alpha_factor,
                            beta_factor,
                            io_factor,
                            net_offset_coeff,
                            mode_name,
                            mode_cfg,
                            anneal,
                        )
                    )

            instance_results = []
            for future in concurrent.futures.as_completed(tasks):
                row = future.result()
                instance_results.append(row)
                all_results.append(row)
                if not row.get('ok', False):
                    WARNING(f"Failed run: instance={row['instance']}, mode={row['mode']}, anneal={row['anneal']}, err={row['error']}")
                    print(f"[FAIL] {row['instance']} | {row['mode']} | {row['anneal']} | err={row['error']}", flush=True)
                    append_failure_jsonl(failure_jsonl, row)
                    failure_count += 1
                else:
                    print(
                        f"[DONE] {row['instance']} | {row['mode']} | {row['anneal']} | "
                        f"hpwl={row['hpwl_final']:.2f} overlap={row['overlap']} t={row['runtime_s']:.2f}s",
                        flush=True,
                    )

        best_rows = pick_best_per_mode(instance_results)
        best_by_instance[instance] = best_rows
        print_grouped_results(instance, best_rows)
        write_all_results_csv(all_csv, all_results, all_headers)
        write_best_results_csv(best_csv, INSTANCES, best_by_instance)
        print(f"[FLUSH] CSV updated after instance={instance}", flush=True)

    print('\nFinished.', flush=True)
    print(f'All results csv:  {all_csv}', flush=True)
    print(f'Best results csv: {best_csv}', flush=True)
    print(f'Failure log jsonl: {failure_jsonl}', flush=True)
    print(f'Failed runs:       {failure_count}', flush=True)
    print(f'Chart output path:{fig_path}', flush=True)
    print('Plot with:', flush=True)
    print(f"  ./.venv/bin/python tests/plot_mode_anneal_results.py --csv {best_csv} --out {fig_path}", flush=True)


if __name__ == '__main__':
    main()
