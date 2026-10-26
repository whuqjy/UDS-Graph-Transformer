# -*- coding: utf-8 -*-
"""Generate event-based SWMM simulation outputs.

Place the baseline INP at data/swmm/network.inp and rainfall event files in
data/rainfall/events/ beside this script. Results are written beneath
data/swmm_outputs/<scenario_name>/. All paths are configured in CONFIG.
Run: python dataset.py

Rainfall files must use date, time, and rain headers by default. Set the
column names in CONFIG if needed. The time-series name, perturbations, and
simulation settings retain their original values.
"""
from __future__ import annotations

import os
import re
import json
import math
import shutil
import random
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd
from pyswmm import Simulation, Nodes, Output
from swmm.toolkit.shared_enum import NodeAttribute
from datetime import datetime



# =========================================================
# 1) Utility functions
# =========================================================

SECTION_PATTERN = re.compile(r'^\s*\[(.+?)\]\s*$')


def fmt_num(x, ndigits=6):
    """Format numbers compactly without unnecessary trailing decimal places."""
    if x is None:
        return ''
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        s = f'{x:.{ndigits}f}'.rstrip('0').rstrip('.')
        return s if s else '0'
    return str(x)


def split_inline_comment(line: str):
    """
    Separate the record content from its semicolon comment.
    Return content and the comment including its semicolon, or an empty string.
    """
    if ';' in line:
        idx = line.find(';')
        return line[:idx].rstrip(), line[idx:]
    return line.rstrip('\n'), ''


def is_data_line(line: str):
    s = line.strip()
    if not s:
        return False
    if s.startswith(';'):
        return False
    if s.startswith('[') and s.endswith(']'):
        return False
    return True


def get_section_ranges(lines):
    """
    Return the start and exclusive end indices of each section:
    {
        'CONDUITS': (start_line_index, end_line_index_exclusive),
        ...
    }
    """
    headers = []
    for i, line in enumerate(lines):
        m = SECTION_PATTERN.match(line.strip())
        if m:
            headers.append((m.group(1).strip().upper(), i))

    ranges = {}
    for idx, (sec_name, start_i) in enumerate(headers):
        end_i = headers[idx + 1][1] if idx + 1 < len(headers) else len(lines)
        ranges[sec_name] = (start_i, end_i)
    return ranges


def replace_options_dates(inp_text, start_dt, end_dt):
    lines = inp_text.splitlines()
    new_lines = []

    replacements = {
        'START_DATE': start_dt.strftime('%m/%d/%Y'),
        'START_TIME': start_dt.strftime('%H:%M:%S'),
        'END_DATE': end_dt.strftime('%m/%d/%Y'),
        'END_TIME': end_dt.strftime('%H:%M:%S'),
        'REPORT_START_DATE': start_dt.strftime('%m/%d/%Y'),
        'REPORT_START_TIME': start_dt.strftime('%H:%M:%S'),
    }

    for line in lines:
        stripped = line.strip()

        if stripped == '' or stripped.startswith(';'):
            new_lines.append(line)
            continue

        parts = stripped.split()
        key = parts[0].upper()

        if key in replacements:
            new_line = f"{parts[0]:<20}{replacements[key]}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines) + '\n'


def read_event_rainfall(event_file, date_col='date', time_col='time', rain_col='rain'):
    """
    Read one rainfall event from a TXT, CSV, or XLSX file.
    Return a DataFrame with datetime and rain columns.
    """
    ext = Path(event_file).suffix.lower()

    if ext == '.xlsx':
        df = pd.read_excel(event_file)
    elif ext == '.csv':
        df = pd.read_csv(event_file)
    elif ext == '.txt':
        df = pd.read_csv(event_file, sep=r'\s+|\t+|,+', engine='python')
    else:
        raise ValueError(f'Unsupported file format: {ext}')

    required_cols = [date_col, time_col, rain_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f'Missing event column: {c}; available columns: {list(df.columns)}')

    df = df[[date_col, time_col, rain_col]].copy()
    df[date_col] = df[date_col].ffill()
    df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0.0)

    df['datetime'] = pd.to_datetime(
        df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip(),
        errors='coerce'
    )

    if df['datetime'].isna().any():
        bad_rows = df[df['datetime'].isna()]
        raise ValueError(f'Invalid event timestamps: {event_file}\nExample rows:\n{bad_rows.head()}')

    df = df.sort_values('datetime').reset_index(drop=True)
    df = df[['datetime', rain_col]].rename(columns={rain_col: 'rain'})
    return df


def rainfall_df_to_swmm_timeseries_lines(df_event, ts_name):
    """
    Convert the rainfall DataFrame to SWMM [TIMESERIES] records.
    Format:
    RainData   01/01/2014   00:00   0.0
    """
    lines = []
    for _, row in df_event.iterrows():
        dt = row['datetime']
        rain = float(row['rain']) * 60
        date_str = dt.strftime('%m/%d/%Y')
        time_str = dt.strftime('%H:%M')
        line = f"{ts_name:<16}{date_str:<14}{time_str:<10}{fmt_num(float(rain), 6)}"
        lines.append(line)
    return lines


def replace_timeseries_in_inp(inp_text, ts_name, new_ts_lines):
    """
    Replace all records for ts_name in the INP [TIMESERIES] section.
    Preserve other time series.
    """
    lines = inp_text.splitlines()
    section_ranges = get_section_ranges(lines)

    if 'TIMESERIES' not in section_ranges:
        raise ValueError('Missing [TIMESERIES] section in the INP file')

    ts_start, ts_end = section_ranges['TIMESERIES']
    before = lines[:ts_start + 1]
    ts_block = lines[ts_start + 1:ts_end]
    after = lines[ts_end:]

    new_ts_block = []
    inserted = False
    inside_target_block = False

    for line in ts_block:
        stripped = line.strip()

        if stripped == '' or stripped.startswith(';'):
            if not inside_target_block:
                new_ts_block.append(line)
            continue

        parts = stripped.split()
        current_name = parts[0] if len(parts) > 0 else None

        if current_name == ts_name:
            if not inserted:
                new_ts_block.extend(new_ts_lines)
                inserted = True
            inside_target_block = True
            continue
        else:
            inside_target_block = False
            new_ts_block.append(line)

    if not inserted:
        if len(new_ts_block) > 0 and new_ts_block[-1].strip() != '':
            new_ts_block.append('')
        new_ts_block.extend(new_ts_lines)

    new_lines = before + new_ts_block + after
    return '\n'.join(new_lines) + '\n'


# =========================================================
# 2) INP parsing and perturbations
# =========================================================

def collect_available_ids_from_inp(inp_text):
    """
    Collect node and conduit identifiers available for selection.
    Nodes: collect from [JUNCTIONS] and [STORAGE].
    Conduits: collect from [CONDUITS].
    """
    lines = inp_text.splitlines()
    section_ranges = get_section_ranges(lines)

    node_ids = []
    conduit_ids = []

    for sec_name in ['JUNCTIONS', 'STORAGE']:
        if sec_name in section_ranges:
            start, end = section_ranges[sec_name]
            for line in lines[start + 1:end]:
                if not is_data_line(line):
                    continue
                content, _ = split_inline_comment(line)
                parts = content.split()
                if len(parts) >= 1:
                    node_ids.append(parts[0])

    if 'CONDUITS' in section_ranges:
        start, end = section_ranges['CONDUITS']
        for line in lines[start + 1:end]:
            if not is_data_line(line):
                continue
            content, _ = split_inline_comment(line)
            parts = content.split()
            if len(parts) >= 1:
                conduit_ids.append(parts[0])

    return {
        'node': sorted(set(node_ids)),
        'conduit': sorted(set(conduit_ids)),
    }


def resolve_selector(available_ids, selector, named_selections=None, base_seed=2026):
    """
    Supported selector modes:
    1) {'mode': 'all'}
    2) {'mode': 'ids', 'ids': ['N1', 'N2']}
    3) {'mode': 'fraction_random', 'fraction': 0.2, 'seed': 123}
    4) {'mode': 'count_random', 'count': 10, 'seed': 123}
    5) {'mode': 'regex', 'pattern': r'^J'}
    6) {'mode': 'same_as', 'ref': 'previous_rule_name'}
    """
    if named_selections is None:
        named_selections = {}

    available_ids = list(sorted(set(available_ids)))
    mode = selector.get('mode', 'all')

    if mode == 'all':
        return set(available_ids)

    if mode == 'ids':
        ids = selector.get('ids', [])
        return set([x for x in ids if x in available_ids])

    if mode == 'fraction_random':
        fraction = float(selector.get('fraction', 0.0))
        fraction = max(0.0, min(1.0, fraction))
        seed = selector.get('seed', base_seed)
        rng = random.Random(seed)

        n = int(round(len(available_ids) * fraction))
        if selector.get('at_least_one', False) and fraction > 0 and len(available_ids) > 0:
            n = max(1, n)
        n = min(n, len(available_ids))

        if n <= 0:
            return set()
        return set(rng.sample(available_ids, n))

    if mode == 'count_random':
        count = int(selector.get('count', 0))
        seed = selector.get('seed', base_seed)
        rng = random.Random(seed)
        count = max(0, min(count, len(available_ids)))
        if count <= 0:
            return set()
        return set(rng.sample(available_ids, count))

    if mode == 'regex':
        pattern = selector.get('pattern', '')
        reg = re.compile(pattern)
        return set([x for x in available_ids if reg.search(x)])

    if mode == 'same_as':
        ref = selector.get('ref')
        if ref not in named_selections:
            raise ValueError(f"selector mode='same_as' references an unknown rule: {ref}")
        return set(named_selections[ref])

    raise ValueError(f'Unsupported selector mode: {mode}')


def get_field_index(target, field, section_name):
    """
    Return the zero-based column index of the requested field.
    Supported fields:
    - CONDUITS: roughness
    - JUNCTIONS / STORAGE: max_depth, init_depth
    """
    section_name = section_name.upper()

    if target == 'conduit':
        field_map = {
            'roughness': 4,   # Name From To Length Roughness ...
        }
        if field not in field_map:
            raise ValueError(f'Unsupported conduit field: {field}')
        return field_map[field]

    if target == 'node':
        if section_name in ['JUNCTIONS', 'STORAGE']:
            field_map = {
                'max_depth': 2,   # Name Elevation MaxDepth InitDepth ...
                'init_depth': 3,
            }
            if field not in field_map:
                raise ValueError(f'Unsupported node field: {field}')
            return field_map[field]
        else:
            # Leave OUTFALLS, DIVIDERS, and other node sections unchanged.
            return None

    raise ValueError(f'Unknown target: {target}')


def apply_method(old_value, method, value, parts=None, section_name=None):
    """
    Supported methods:
    - multiply: new = old * value
    - add: new = old + value
    - set: new = value
    - set_ratio_of_max_depth: new = value * MaxDepth
    - add_ratio_of_max_depth: new = old + value * MaxDepth
    """
    if method == 'multiply':
        return old_value * float(value)

    if method == 'add':
        return old_value + float(value)

    if method == 'set':
        return float(value)

    if method in ['set_ratio_of_max_depth', 'add_ratio_of_max_depth']:
        if parts is None:
            raise ValueError(f'{method} requires the parts argument')

        section_name = (section_name or '').upper()
        if section_name not in ['JUNCTIONS', 'STORAGE']:
            raise ValueError(f'{method} is only supported for JUNCTIONS/STORAGE')

        max_depth = float(parts[2])

        if method == 'set_ratio_of_max_depth':
            return float(value) * max_depth

        if method == 'add_ratio_of_max_depth':
            return old_value + float(value) * max_depth

    raise ValueError(f'Unsupported method: {method}')


def apply_perturbations_to_inp(inp_text, perturbations, base_seed=2026):
    """
    Apply perturbation rules:
    Modify INP text according to the supplied perturbations.

    perturbations is a list of rule dictionaries. Examples:

    {
        'name': 'pipe_roughness_20pct',
        'target': 'conduit',
        'field': 'roughness',
        'selector': {'mode': 'fraction_random', 'fraction': 0.2, 'seed': 100},
        'method': 'multiply',
        'value': 1.30,
        'min': 1e-6,
        'max': None,
        'sections': ['CONDUITS']
    }

    {
        'name': 'node_maxdepth_10pct',
        'target': 'node',
        'field': 'max_depth',
        'selector': {'mode': 'fraction_random', 'fraction': 0.1, 'seed': 101},
        'method': 'multiply',
        'value': 0.90,
        'min': 0.01
    }

    {
        'name': 'node_init_same_group',
        'target': 'node',
        'field': 'init_depth',
        'selector': {'mode': 'same_as', 'ref': 'node_maxdepth_10pct'},
        'method': 'add_ratio_of_max_depth',
        'value': 0.05,
        'min': 0.0
    }
    """
    lines = inp_text.splitlines()
    section_ranges = get_section_ranges(lines)
    available_ids = collect_available_ids_from_inp(inp_text)

    # Resolve the selected objects for each rule.
    resolved_rules = []
    named_selections = {}
    selection_log = []

    for i, rule in enumerate(perturbations, 1):
        r = deepcopy(rule)
        name = r.get('name', f'rule_{i}')
        target = r['target']
        selector = r.get('selector', {'mode': 'all'})

        selected_ids = resolve_selector(
            available_ids=available_ids[target],
            selector=selector,
            named_selections=named_selections,
            base_seed=base_seed
        )

        r['_selected_ids'] = set(selected_ids)
        resolved_rules.append(r)
        named_selections[name] = set(selected_ids)

        selection_log.append({
            'name': name,
            'target': target,
            'field': r.get('field'),
            'selected_count': len(selected_ids),
            'selected_ids': sorted(selected_ids),
            'selector': selector,
            'method': r.get('method'),
            'value': r.get('value')
        })

    # Apply modifications line by line.
    current_section = None
    new_lines = []

    for line in lines:
        sec_match = SECTION_PATTERN.match(line.strip())
        if sec_match:
            current_section = sec_match.group(1).strip().upper()
            new_lines.append(line)
            continue

        if not is_data_line(line):
            new_lines.append(line)
            continue

        content, comment = split_inline_comment(line)
        parts = content.split()

        if len(parts) == 0:
            new_lines.append(line)
            continue

        record_id = parts[0]

        # Apply rules to CONDUITS, JUNCTIONS, and STORAGE records.
        if current_section == 'CONDUITS':
            for rule in resolved_rules:
                if rule['target'] != 'conduit':
                    continue

                allowed_sections = [x.upper() for x in rule.get('sections', ['CONDUITS'])]
                if current_section not in allowed_sections:
                    continue

                if record_id not in rule['_selected_ids']:
                    continue

                field = rule['field']
                col_idx = get_field_index('conduit', field, current_section)
                if col_idx is None or len(parts) <= col_idx:
                    continue

                old_value = float(parts[col_idx])
                new_value = apply_method(
                    old_value=old_value,
                    method=rule['method'],
                    value=rule['value'],
                    parts=parts,
                    section_name=current_section
                )

                if rule.get('min') is not None:
                    new_value = max(float(rule['min']), new_value)
                if rule.get('max') is not None:
                    new_value = min(float(rule['max']), new_value)

                parts[col_idx] = fmt_num(new_value)

        elif current_section in ['JUNCTIONS', 'STORAGE']:
            for rule in resolved_rules:
                if rule['target'] != 'node':
                    continue

                allowed_sections = [x.upper() for x in rule.get('sections', ['JUNCTIONS', 'STORAGE'])]
                if current_section not in allowed_sections:
                    continue

                if record_id not in rule['_selected_ids']:
                    continue

                field = rule['field']
                col_idx = get_field_index('node', field, current_section)
                if col_idx is None or len(parts) <= col_idx:
                    continue

                old_value = float(parts[col_idx])
                new_value = apply_method(
                    old_value=old_value,
                    method=rule['method'],
                    value=rule['value'],
                    parts=parts,
                    section_name=current_section
                )

                if rule.get('min') is not None:
                    new_value = max(float(rule['min']), new_value)
                if rule.get('max') is not None:
                    new_value = min(float(rule['max']), new_value)

                parts[col_idx] = fmt_num(new_value)

            # Enforce the original node-depth bounds:
            # Keep init_depth between zero and max_depth.
            if len(parts) >= 4:
                try:
                    max_depth = max(float(parts[2]), 1e-6)
                    init_depth = max(float(parts[3]), 0.0)
                    if init_depth > max_depth:
                        init_depth = max_depth
                    parts[2] = fmt_num(max_depth)
                    parts[3] = fmt_num(init_depth)
                except Exception:
                    pass

        new_content = '    '.join(parts)
        new_line = new_content + ('' if comment == '' else f'    {comment}')
        new_lines.append(new_line)

    new_text = '\n'.join(new_lines) + '\n'
    return new_text, selection_log


# =========================================================
# 3) Temporary INP generation
# =========================================================

def create_temp_inp_with_event_and_config(
    base_inp_path,
    temp_inp_path,
    event_file,
    ts_name,
    perturbations=None,
    date_col='date',
    time_col='time',
    rain_col='rain',
    report_extend_factor=1.5,
    base_seed=2026
):
    """
    Workflow:
    1. Read the rainfall event.
    2. Replace [TIMESERIES].
    3. Update the start and end dates in [OPTIONS].
    4. Apply perturbation rules.
    5. Write the temporary INP file.
    """
    if perturbations is None:
        perturbations = []

    df_event = read_event_rainfall(
        event_file,
        date_col=date_col,
        time_col=time_col,
        rain_col=rain_col
    )

    new_ts_lines = rainfall_df_to_swmm_timeseries_lines(df_event, ts_name)

    with open(base_inp_path, 'r', encoding='utf-8', errors='ignore') as f:
        inp_text = f.read()

    # 1) Replace the rainfall time series.
    new_inp_text = replace_timeseries_in_inp(inp_text, ts_name, new_ts_lines)

    # 2) Replace simulation start and end times.
    start_dt = df_event['datetime'].iloc[0]
    rain_duration = df_event['datetime'].iloc[-1] - df_event['datetime'].iloc[0]
    end_dt = start_dt + report_extend_factor * rain_duration

    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(minutes=5)

    new_inp_text = replace_options_dates(new_inp_text, start_dt, end_dt)

    # 3) Perturbations
    selection_log = []
    if len(perturbations) > 0:
        new_inp_text, selection_log = apply_perturbations_to_inp(
            new_inp_text,
            perturbations=perturbations,
            base_seed=base_seed
        )

    # 4) Write the file.
    with open(temp_inp_path, 'w', encoding='utf-8') as f:
        f.write(new_inp_text)

    return df_event, selection_log


# =========================================================
# 4) Run SWMM and export depth, head, and flooding
# =========================================================

_WARNED_FLOODING_ATTR = False


def get_node_flooding_value(node):
    """
    Node flooding attribute names may differ between PySWMM versions.
    Try the supported attribute names for compatibility.
    """
    global _WARNED_FLOODING_ATTR

    candidates = [
        'flooding',
        'flooding_losses',
        'flooding_loss',
    ]

    for attr in candidates:
        if hasattr(node, attr):
            return getattr(node, attr)

    if not _WARNED_FLOODING_ATTR:
        warnings.warn(
            "No node.flooding or flooding_losses attribute was found. "
            "Returning zero for flooding. Inspect dir(node) and "
            "adapt get_node_flooding_value() if needed."
        )
        _WARNED_FLOODING_ATTR = True

    return 0.0




def run_swmm_and_save_three_outputs(inp_path, output_prefix, save_format='txt'):
    """
    Example output_prefix:
    data/swmm_outputs/Scenario_A/event_001

    Generated files:
    - event_001_depth.txt
    - event_001_head.txt
    - event_001_flooding.txt

    Data sources:
    - depth / head: extracted during simulation
    - flooding: FLOODING_LOSSES extracted from the OUT file
    """
    from pathlib import Path
    import pandas as pd

    depth_records = []
    head_records = []

    # --------------------------------------------------
    # 1) Run the simulation and extract depth and head.
    # --------------------------------------------------
    with Simulation(inp_path) as sim:
        sim.step_advance(60)   # Advance the Python simulation iterator by 60 seconds.
        nodes = Nodes(sim)
        node_list = list(nodes)
        node_ids = [node.nodeid for node in node_list]

        for _ in sim:
            current_time = sim.current_time

            row_depth = {'datetime': current_time}
            row_head = {'datetime': current_time}

            for node in node_list:
                nid = node.nodeid
                depth_val = float(node.depth)
                head_val = float(node.invert_elevation + node.depth)

                row_depth[nid] = depth_val
                row_head[nid] = head_val

            depth_records.append(row_depth)
            head_records.append(row_head)

    df_depth = pd.DataFrame(depth_records)
    df_head = pd.DataFrame(head_records)


    # --------------------------------------------------
    # 2) Extract flooding losses from the OUT file.
    # --------------------------------------------------
    out_path = str(Path(inp_path).with_suffix('.out'))
    # print("Expected out_path =", out_path)

    with Output(out_path) as out:
        report_step = 1  # Preserve the original one-minute offset from the output start.

        year_s = int(str(out.start)[0:4])
        month_s = int(str(out.start)[5:7])
        day_s = int(str(out.start)[8:10])
        hour_s = int(str(out.start)[11:13])
        minute_s = int(str(out.start)[14:16]) + report_step
        second_s = int(str(out.start)[17:19])

        if minute_s >= 60:
            minute_s -= 60
            hour_s += 1

        year_e = int(str(out.end)[0:4])
        month_e = int(str(out.end)[5:7])
        day_e = int(str(out.end)[8:10])
        hour_e = int(str(out.end)[11:13])
        minute_e = int(str(out.end)[14:16])
        second_e = int(str(out.end)[17:19])

        start_dt = datetime(year_s, month_s, day_s, hour_s, minute_s, second_s)
        end_dt = datetime(year_e, month_e, day_e, hour_e, minute_e, second_e)

        df_flood = None

        for nid in out.nodes:
            ts_flood = out.node_series(
                nid,
                NodeAttribute.FLOODING_LOSSES,
                start_dt,
                end_dt
            )

            s = pd.Series(ts_flood, name=nid)

            if df_flood is None:
                df_flood = s.to_frame()
            else:
                df_flood = pd.concat([df_flood, s], axis=1)

        if df_flood is None:
            raise ValueError("No flooding-loss data could be extracted from the OUT file")

        df_flood = df_flood.reset_index()
        df_flood = df_flood.rename(columns={'index': 'datetime'})

    # --------------------------------------------------
    # 3) Standardize timestamps
    # --------------------------------------------------
    for df in [df_depth, df_head, df_flood]:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.floor('min')
        df.insert(0, 'date', df['datetime'].dt.strftime('%Y-%m-%d'))
        df.insert(1, 'time', df['datetime'].dt.strftime('%H:%M'))

    # Keep the last record when multiple timestamps fall in the same minute.
    df_depth = df_depth.drop_duplicates(subset=['datetime'], keep='last').reset_index(drop=True)
    df_head = df_head.drop_duplicates(subset=['datetime'], keep='last').reset_index(drop=True)
    df_flood = df_flood.drop_duplicates(subset=['datetime'], keep='last').reset_index(drop=True)

    # Drop the original datetime column.
    df_depth.drop(columns=['datetime'], inplace=True)
    df_head.drop(columns=['datetime'], inplace=True)
    df_flood.drop(columns=['datetime'], inplace=True)

    # --------------------------------------------------
    # 4) Write output files
    # --------------------------------------------------
    depth_file = f'{output_prefix}_depth.{save_format}'
    head_file = f'{output_prefix}_head.{save_format}'
    flood_file = f'{output_prefix}_flooding.{save_format}'

    if save_format.lower() == 'xlsx':
        df_depth.to_excel(depth_file, index=False)
        df_head.to_excel(head_file, index=False)
        df_flood.to_excel(flood_file, index=False)
    elif save_format.lower() == 'txt':
        df_depth.to_csv(depth_file, sep='\t', index=False, encoding='utf-8-sig')
        df_head.to_csv(head_file, sep='\t', index=False, encoding='utf-8-sig')
        df_flood.to_csv(flood_file, sep='\t', index=False, encoding='utf-8-sig')
    else:
        raise ValueError("save_format must be 'xlsx' or 'txt'")

    return {
        'depth': df_depth,
        'head': df_head,
        'flooding': df_flood,
        'depth_file': depth_file,
        'head_file': head_file,
        'flood_file': flood_file,
    }


# =========================================================
# 5) Single-event and batch execution
# =========================================================

def save_selection_log(selection_log, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(selection_log, f, ensure_ascii=False, indent=2)


def run_single_event_with_config(config):
    """
    Required configuration keys:
    - scenario_name
    - base_inp_path
    - event_file
    - output_root
    - ts_name
    - perturbations
    """
    scenario_name = config['scenario_name']
    base_inp_path = config['base_inp_path']
    event_file = config['event_file']
    output_root = config['output_root']
    ts_name = config['ts_name']

    date_col = config.get('date_col', 'date')
    time_col = config.get('time_col', 'time')
    rain_col = config.get('rain_col', 'rain')
    save_format = config.get('save_format', 'txt')
    report_extend_factor = config.get('report_extend_factor', 1.5)
    keep_modified_inp = config.get('keep_modified_inp', True)
    delete_temp = config.get('delete_temp', True)
    base_seed = config.get('base_seed', 2026)
    perturbations = config.get('perturbations', [])

    scenario_dir = os.path.join(output_root, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    event_name = Path(event_file).stem

    temp_dir = tempfile.mkdtemp(prefix=f'swmm_{scenario_name}_')
    temp_inp = os.path.join(temp_dir, f'{event_name}_{scenario_name}.inp')

    summary = {
        'scenario_name': scenario_name,
        'event_name': event_name,
        'event_file': event_file,
        'status': 'failed',
    }

    try:
        df_event, selection_log = create_temp_inp_with_event_and_config(
            base_inp_path=base_inp_path,
            temp_inp_path=temp_inp,
            event_file=event_file,
            ts_name=ts_name,
            perturbations=perturbations,
            date_col=date_col,
            time_col=time_col,
            rain_col=rain_col,
            report_extend_factor=report_extend_factor,
            base_seed=base_seed
        )

        selection_log_path = os.path.join(scenario_dir, f'{event_name}_selection_log.json')
        save_selection_log(selection_log, selection_log_path)

        if keep_modified_inp:
            modified_inp_dir = os.path.join(scenario_dir, 'modified_inp')
            os.makedirs(modified_inp_dir, exist_ok=True)
            shutil.copy2(temp_inp, os.path.join(modified_inp_dir, os.path.basename(temp_inp)))

        output_prefix = os.path.join(scenario_dir, event_name)
        result_dict = run_swmm_and_save_three_outputs(
            inp_path=temp_inp,
            output_prefix=output_prefix,
            save_format=save_format
        )

        summary.update({
            'rain_start': df_event['datetime'].iloc[0],
            'rain_end': df_event['datetime'].iloc[-1],
            'rain_steps': len(df_event),
            'sim_steps': len(result_dict['depth']),
            'depth_file': result_dict['depth_file'],
            'head_file': result_dict['head_file'],
            'flooding_file': result_dict['flood_file'],
            'selection_log': selection_log_path,
            'status': 'success'
        })

        print(f"[Completed] scenario={scenario_name} event={event_name}")

    except Exception as e:
        summary['status'] = f'failed: {e}'
        print(f"[Failed] scenario={scenario_name} event={event_name}: {e}")

    finally:
        if delete_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(scenario_dir, f'{event_name}_summary.xlsx')
    # summary_df.to_excel(summary_path, index=False)
    return summary_df


def batch_run_event_dir_with_config(config):
    """
    Run all matching event files in a directory.
    Specify event_dir instead of event_file in the configuration.
    """
    scenario_name = config['scenario_name']
    event_dir = config['event_dir']
    output_root = config['output_root']
    event_suffix = config.get('event_suffix', '.txt')

    scenario_dir = os.path.join(output_root, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    event_files = sorted([
        os.path.join(event_dir, f)
        for f in os.listdir(event_dir)
        if f.lower().endswith(event_suffix.lower())
    ])

    if len(event_files) == 0:
        raise ValueError(f'No {event_suffix} files found in {event_dir}')

    all_summary = []

    for i, event_file in enumerate(event_files, 1):
        print(f'\n========== [{i}/{len(event_files)}] {Path(event_file).stem} ==========')

        cfg_i = deepcopy(config)
        cfg_i['event_file'] = event_file

        summary_df = run_single_event_with_config(cfg_i)
        all_summary.append(summary_df)

    final_summary = pd.concat(all_summary, ignore_index=True)
    final_summary_path = os.path.join(scenario_dir, 'summary_all.xlsx')
    # final_summary.to_excel(final_summary_path, index=False)

    print(f'\nBatch completed. Summary path: {final_summary_path}')
    return final_summary


# =========================================================
# 6) Project configuration
# =========================================================

# Resolve all project paths relative to dataset.py, regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'

CONFIG = {
    # -------- Basic settings --------
    'scenario_name': 'S00_no_perturb',
    'base_inp_path': str(DATA_DIR / 'swmm' / 'network.inp'),

    # Use event_file for a single event and event_dir for batch processing.
    # 'event_file': str(DATA_DIR / 'rainfall' / 'events' / 'event_001.txt'),
    'event_dir': str(DATA_DIR / 'rainfall' / 'events'),

    'output_root': str(DATA_DIR / 'swmm_outputs'),
    'ts_name': 'rain_ser',

    # -------- Rainfall input column names --------
    'date_col': 'date',
    'time_col': 'time',
    'rain_col': 'rain',

    # -------- Output settings --------
    'save_format': 'txt',          # 'txt' or 'xlsx'
    'report_extend_factor': 1,   # End time = start time + report_extend_factor * rainfall duration.
    'keep_modified_inp': False,     # Keep the modified INP file.
    'delete_temp': True,
    'base_seed': 2026,

    'event_suffix': '.txt',

    # -------- Perturbation rules --------
    # Rules are applied in order.
    'perturbations' : []
    # 'perturbations': [
    #     # Example 1: multiply roughness by 3.0 for 30% of conduits.
    #     {
    #         'name': 'pipe_roughness_30pct',
    #         'target': 'conduit',
    #         'field': 'roughness',
    #         'selector': {
    #             'mode': 'fraction_random',
    #             'fraction': 0.3,
    #             'seed': 303
    #         },
    #         'method': 'multiply',
    #         'value': 3.0,
    #         'min': 1e-6,
    #         'sections': ['CONDUITS']
    #     },

        #Example 2: multiply max_depth by 0.90 for 40% of nodes.
        # {
        #     'name': 'node_maxdepth_20pct',
        #     'target': 'node',
        #     'field': 'max_depth',
        #     'selector': {
        #         'mode': 'fraction_random',
        #         'fraction': 0.4,
        #         'seed': 2001
        #     },
        #     'method': 'multiply',
        #     'value': 0.90,
        #     'min': 0.01,
        #     'sections': ['JUNCTIONS', 'STORAGE']
        # },

        # Example 3: add 0.1 * max_depth to initial depth for randomly selected nodes.
        # {
        #     'name': 'node_init_same_as_maxdepth_group',
        #     'target': 'node',
        #     'field': 'init_depth',
        #     'selector': {
        #         'mode': 'fraction_random',
        #         # 'ref': 'node_maxdepth_20pct'
        #         'fraction': 0.4,
        #         'seed': 3001
        #     },
        #     'method': 'add_ratio_of_max_depth',
        #     'value': 0.1,
        #     'min': 0.0,
        #     'sections': ['JUNCTIONS', 'STORAGE']
        # },
#     ]
}


# =========================================================
# 7) Main entry point
# =========================================================

if __name__ == '__main__':

    # ---------- Single rainfall event ----------
    # summary_df = run_single_event_with_config(CONFIG)
    # print(summary_df)

    # ---------- Batch processing of the event directory ----------
    # For batch processing, leave run_single_event_with_config commented out.
    # Set event_dir in CONFIG and use the batch call below.
    #
    CONFIG_BATCH = deepcopy(CONFIG)
    summary_df = batch_run_event_dir_with_config(CONFIG_BATCH)
    print(summary_df.head())

