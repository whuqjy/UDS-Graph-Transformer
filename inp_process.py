"""Extract sewer-network static features from a SWMM INP file.

Usage:
    python Inp_Extraction.py --inp data/network.inp --output data/static_features.xlsx
    python Inp_Extraction.py --inp data/network.inp --output data/static_features.xlsx --encoding gbk

Dependencies: numpy, pandas, openpyxl.
Worksheets and column order are unchanged from the original script.
Node numeric arrays have 20 columns; edge numeric arrays have 13 columns.
Do not pass all node columns to a pipeline configured for 17 static features
without reproducing its feature selection and node ordering.

Scope: junctions, outfalls, storage nodes, conduits, pumps, and Horton input
fields. Other link types and routed subcatchment contributions are not
implemented. No unit conversion or SWMM simulation is performed. The original
hydraulic approximations, zero placeholders, and type-specific feature
suppression are preserved and documented in the relevant functions.
"""

import argparse
import re
import shlex
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTED_COLUMNS = [
    'imperv',
    'subcatch_slope',
    'N_imperv',
    'N_perv',
    'Dstore_imperv',
    'Dstore_perv',
    'pct_zero_imperv',
    'horton_max',
    'horton_min',
    'horton_decay',
    'horton_drying_time',
]

SUBCATCHMENT_COLUMNS = [
    'subcatch_area',
    'subcatch_width',
    'imperv',
    'subcatch_slope',
    'N_imperv',
    'N_perv',
    'Dstore_imperv',
    'Dstore_perv',
    'pct_zero_imperv',
    'horton_max',
    'horton_min',
    'horton_decay',
    'horton_drying_time',
]

NODE_COLUMNS = [
    'node',
    'node_type',
    'node_type_id',
    'is_junction',
    'is_outfall',
    'is_storage',
    'invert_elev',
    'max_depth',
    'initial_depth',
    'subcatch_area',
    'subcatch_width',
    'subcatch_slope',
    'imperv',
    'N_imperv',
    'N_perv',
    'Dstore_imperv',
    'Dstore_perv',
    'pct_zero_imperv',
    'horton_max',
    'horton_min',
    'horton_decay',
    'horton_drying_time',
]

EDGE_COLUMNS = [
    'edge',
    'from_node',
    'to_node',
    'edge_type',
    'edge_type_id',
    'is_conduit',
    'is_pump',
    'shape',
    'shape_id',
    'maxdepth',
    'length',
    'roughness',
    'in_offset',
    'out_offset',
    'slope',
    'hydraulic_radius',
    'velocity_index',
    'retention_time',
]


def split_tokens(line):
    """Tokenize a record, treating semicolons outside quotes as comments."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ";"
    return list(lexer)


def read_inp_sections(inp_path, encoding="utf-8-sig"):
    """Read sections with explicit decoding and contextual quoting errors."""
    sections = {}
    current_section = None
    with Path(inp_path).open(encoding=encoding) as file:
        for line_number, raw_line in enumerate(file, 1):
            try:
                tokens = split_tokens(raw_line)
            except ValueError as exc:
                raise ValueError(f"{inp_path}:{line_number}: {exc}") from exc
            if not tokens:
                continue
            line = shlex.join(tokens)
            match = re.fullmatch(r"\[([^]]+)\]", tokens[0])
            if match:
                current_section = match.group(1).upper()
                sections.setdefault(current_section, [])
            elif current_section is not None:
                sections[current_section].append(line)
    return sections


def get_option(sections, option_name, default=None):
    """Return an uppercase option value, or the supplied default."""
    option_name = option_name.upper()
    for line in sections.get('OPTIONS', []):
        tokens = split_tokens(line)
        if len(tokens) >= 2 and tokens[0].upper() == option_name:
            return tokens[1].upper()
    return default


def parse_subcatchments(sections):
    """Read direct subcatchment outlets, area, width, imperviousness, and slope."""
    records = []
    for line in sections.get('SUBCATCHMENTS', []):
        tokens = split_tokens(line)
        if len(tokens) < 7:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'subcatchment': tokens[0],
            'rain_gage': tokens[1],
            'outlet': tokens[2],
            'area': float(tokens[3]),
            'imperv': float(tokens[4]),
            'width': float(tokens[5]),
            'subcatch_slope': float(tokens[6]),
        })
    return pd.DataFrame(records, columns=[
        'subcatchment',
        'rain_gage',
        'outlet',
        'area',
        'imperv',
        'width',
        'subcatch_slope',
    ])


def parse_subareas(sections):
    """Read surface roughness and depression-storage properties."""
    records = []
    for line in sections.get('SUBAREAS', []):
        tokens = split_tokens(line)
        if len(tokens) < 6:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'subcatchment': tokens[0],
            'N_imperv': float(tokens[1]),
            'N_perv': float(tokens[2]),
            'Dstore_imperv': float(tokens[3]),
            'Dstore_perv': float(tokens[4]),
            'pct_zero_imperv': float(tokens[5]),
        })
    return pd.DataFrame(records, columns=[
        'subcatchment',
        'N_imperv',
        'N_perv',
        'Dstore_imperv',
        'Dstore_perv',
        'pct_zero_imperv',
    ])


def parse_infiltration(sections):
    """Read the four Horton parameters used by the original feature schema."""
    records = []
    for line in sections.get('INFILTRATION', []):
        tokens = split_tokens(line)
        if len(tokens) < 5:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'subcatchment': tokens[0],
            'horton_max': float(tokens[1]),
            'horton_min': float(tokens[2]),
            'horton_decay': float(tokens[3]),
            'horton_drying_time': float(tokens[4]),
        })
    return pd.DataFrame(records, columns=[
        'subcatchment',
        'horton_max',
        'horton_min',
        'horton_decay',
        'horton_drying_time',
    ])


def parse_junctions(sections):
    """Read junction elevations and depths; initial depth defaults to zero."""
    records = []
    for line in sections.get('JUNCTIONS', []):
        tokens = split_tokens(line)
        if len(tokens) < 3:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'node': tokens[0],
            'node_type': 'junction',
            'invert_elev': float(tokens[1]),
            'max_depth': float(tokens[2]),
            'initial_depth': float(tokens[3]) if len(tokens) >= 4 else 0.0,
        })
    return pd.DataFrame(records, columns=['node', 'node_type', 'invert_elev', 'max_depth', 'initial_depth'])


def parse_outfalls(sections):
    """Read outfall elevations; depth features are zero placeholders."""
    records = []
    for line in sections.get('OUTFALLS', []):
        tokens = split_tokens(line)
        if len(tokens) < 2:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'node': tokens[0],
            'node_type': 'outfall',
            'invert_elev': float(tokens[1]),
            'max_depth': 0.0,
            'initial_depth': 0.0,
        })
    return pd.DataFrame(records, columns=['node', 'node_type', 'invert_elev', 'max_depth', 'initial_depth'])


def parse_storage(sections):
    """Read storage-node elevations, maximum depths, and initial depths."""
    records = []
    for line in sections.get('STORAGE', []):
        tokens = split_tokens(line)
        if len(tokens) < 4:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'node': tokens[0],
            'node_type': 'storage',
            'invert_elev': float(tokens[1]),
            'max_depth': float(tokens[2]),
            'initial_depth': float(tokens[3]),
        })
    return pd.DataFrame(records, columns=['node', 'node_type', 'invert_elev', 'max_depth', 'initial_depth'])


def build_node_static_features(sections):
    """Build node features in junction, outfall, then storage order.

    Aggregate directly connected subcatchments by outlet: sum area and width,
    and area-weight the remaining properties. Missing properties become zero.
    Preserve the original convention of zero subcatchment features for storage
    and outfalls. Routing through another subcatchment is not accumulated."""
    infiltration = get_option(sections, "INFILTRATION", "HORTON")
    if sections.get("INFILTRATION") and infiltration not in {"HORTON", "MODIFIED_HORTON"}:
        raise ValueError("This feature schema supports Horton infiltration fields only.")
    df_sub = parse_subcatchments(sections)
    df_subarea = parse_subareas(sections)
    df_infil = parse_infiltration(sections)
    df_junc = parse_junctions(sections)
    df_outfall = parse_outfalls(sections)
    df_storage = parse_storage(sections)
    node_tables = [table for table in (df_junc, df_outfall, df_storage) if not table.empty]
    if not node_tables:
        raise ValueError("No supported nodes found in the INP file.")
    df_nodes = pd.concat(node_tables, ignore_index=True)
    node_type_map = {'junction': 0, 'outfall': 1, 'storage': 2}
    if df_nodes["node"].duplicated().any():
        raise ValueError("Duplicate node identifiers found.")
    df_nodes['node_type_id'] = df_nodes['node_type'].map(node_type_map)
    df_nodes['is_junction'] = (df_nodes['node_type'] == 'junction').astype(int)
    df_nodes['is_outfall'] = (df_nodes['node_type'] == 'outfall').astype(int)
    df_nodes['is_storage'] = (df_nodes['node_type'] == 'storage').astype(int)
    if len(df_sub) > 0:
        df_sc = df_sub.copy()
        if not df_sc["outlet"].isin(df_nodes["node"]).all():
            warnings.warn("Some subcatchment outlets are not supported nodes; "
                          "their contributions will not be accumulated.", stacklevel=2)
        if len(df_subarea) > 0:
            df_sc = df_sc.merge(df_subarea, on='subcatchment', how='left')
        if len(df_infil) > 0:
            df_sc = df_sc.merge(df_infil, on='subcatchment', how='left')
        for column in WEIGHTED_COLUMNS:
            if column not in df_sc:
                df_sc[column] = 0.0
        df_sc = df_sc.fillna(0.0)
        grouped_records = []
        for outlet, g in df_sc.groupby('outlet'):
            area_sum = g['area'].sum()
            width_sum = g['width'].sum()
            record = {'node': outlet, 'subcatch_area': area_sum, 'subcatch_width': width_sum}
            for col in WEIGHTED_COLUMNS:
                if area_sum > 0:
                    record[col] = np.average(g[col], weights=g['area'])
                else:
                    record[col] = 0.0
            grouped_records.append(record)
        df_node_sc = pd.DataFrame(grouped_records)
        df_nodes = df_nodes.merge(df_node_sc, on='node', how='left')
    for col in SUBCATCHMENT_COLUMNS:
        if col not in df_nodes.columns:
            df_nodes[col] = 0.0
    df_nodes[SUBCATCHMENT_COLUMNS] = df_nodes[SUBCATCHMENT_COLUMNS].fillna(0.0)
    outfall_mask = df_nodes['node_type'] == 'outfall'
    # Preserve the original type-specific suppression of continuous features.
    df_nodes.loc[outfall_mask, ['max_depth', 'initial_depth'] + SUBCATCHMENT_COLUMNS] = 0.0
    storage_mask = df_nodes['node_type'] == 'storage'
    df_nodes.loc[storage_mask, SUBCATCHMENT_COLUMNS] = 0.0
    return df_nodes[NODE_COLUMNS]


def parse_conduits(sections):
    """Read conduit endpoints, length, roughness, and optional offsets."""
    records = []
    for line in sections.get('CONDUITS', []):
        tokens = split_tokens(line)
        if len(tokens) < 5:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'edge': tokens[0],
            'edge_type': 'conduit',
            'from_node': tokens[1],
            'to_node': tokens[2],
            'length': float(tokens[3]),
            'roughness': float(tokens[4]),
            'in_offset': float(tokens[5]) if len(tokens) >= 6 else 0.0,
            'out_offset': float(tokens[6]) if len(tokens) >= 7 else 0.0,
        })
    return pd.DataFrame(records, columns=[
        'edge',
        'edge_type',
        'from_node',
        'to_node',
        'length',
        'roughness',
        'in_offset',
        'out_offset',
    ])


def parse_pumps(sections):
    """Read pump endpoints; conduit-specific attributes remain zero placeholders."""
    records = []
    for line in sections.get('PUMPS', []):
        tokens = split_tokens(line)
        if len(tokens) < 3:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'edge': tokens[0],
            'edge_type': 'pump',
            'from_node': tokens[1],
            'to_node': tokens[2],
            'length': 0.0,
            'roughness': 0.0,
            'in_offset': 0.0,
            'out_offset': 0.0,
        })
    return pd.DataFrame(records, columns=[
        'edge',
        'edge_type',
        'from_node',
        'to_node',
        'length',
        'roughness',
        'in_offset',
        'out_offset',
    ])


def parse_xsections(sections):
    """Read cross-section shape and geometry; unused geometry fields are omitted."""
    records = []
    for line in sections.get('XSECTIONS', []):
        tokens = split_tokens(line)
        if len(tokens) < 3:
            warnings.warn(f"Skipping incomplete record: {line}", stacklevel=2)
            continue
        records.append({
            'edge': tokens[0],
            'shape': tokens[1].upper(),
            'maxdepth': float(tokens[2]),
            'geom2': float(tokens[3]) if len(tokens) >= 4 else 0.0,
        })
    return pd.DataFrame(records, columns=['edge', 'shape', 'maxdepth', 'geom2'])


def hydraulic_radius_full(shape, maxdepth, geom2=0.0):
    """Return the original approximate radius feature, not a general section solver.

    Circular: depth / 4. Both rectangular types: width * depth / (width +
    2 * depth). All other types also use depth / 4. These legacy approximations
    are retained for numerical reproducibility; RECT_CLOSED uses the same
    perimeter expression as RECT_OPEN. Nonpositive dimensions return zero."""
    if maxdepth <= 0:
        return 0.0
    shape = str(shape).upper()
    if shape == 'CIRCULAR':
        return maxdepth / 4.0
    elif shape in ['RECT_OPEN', 'RECT_CLOSED']:
        width = geom2
        depth = maxdepth
        if width <= 0:
            return 0.0
        area = width * depth
        wetted_perimeter = width + 2.0 * depth
        if wetted_perimeter <= 0:
            return 0.0
        return area / wetted_perimeter
    else:
        return maxdepth / 4.0


def compute_manning_velocity(roughness, hydraulic_radius, slope):
    """Return the original Manning-style velocity index with coefficient 1.

    The caller supplies absolute slope. Nonpositive inputs return zero.
    No unit conversion is applied; this is a static proxy, not simulated flow."""
    if roughness <= 0 or hydraulic_radius <= 0 or slope <= 0:
        return 0.0
    return 1.0 / roughness * hydraulic_radius ** (2.0 / 3.0) * slope ** 0.5


def build_edge_static_features(sections, df_node_features):
    """Build conduit/pump features in the original order and units.

    Slope is unsigned. In ELEVATION mode, a zero offset falls back to the
    node invert, matching the original script. Retention time is length /
    velocity_index, or a zero placeholder when the index is nonpositive.
    Shape IDs are assigned from sorted shapes present in this file, so the
    mapping must be checked when combining different networks."""
    for section in ("ORIFICES", "WEIRS", "OUTLETS"):
        if sections.get(section):
            warnings.warn(f"[{section}] links are not included in this feature schema.",
                          stacklevel=2)
    df_conduit = parse_conduits(sections)
    df_pump = parse_pumps(sections)
    df_xsec = parse_xsections(sections)
    edge_tables = [table for table in (df_conduit, df_pump) if not table.empty]
    if not edge_tables:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    df_edges = pd.concat(edge_tables, ignore_index=True)
    if df_edges["edge"].duplicated().any():
        raise ValueError("Duplicate edge identifiers found.")
    if not (df_edges["from_node"].isin(df_node_features["node"]).all()
            and df_edges["to_node"].isin(df_node_features["node"]).all()):
        raise ValueError("Some link endpoints do not match the extracted nodes.")
    if len(df_xsec) > 0:
        df_edges = df_edges.merge(df_xsec, on='edge', how='left')
    else:
        df_edges['shape'] = 'NONE'
        df_edges['maxdepth'] = 0.0
        df_edges['geom2'] = 0.0
    df_edges[['shape', 'maxdepth', 'geom2']] = df_edges[['shape', 'maxdepth', 'geom2']].fillna({'shape': 'NONE', 'maxdepth': 0.0, 'geom2': 0.0})
    edge_type_map = {'conduit': 0, 'pump': 1}
    df_edges['edge_type_id'] = df_edges['edge_type'].map(edge_type_map)
    df_edges['is_conduit'] = (df_edges['edge_type'] == 'conduit').astype(int)
    df_edges['is_pump'] = (df_edges['edge_type'] == 'pump').astype(int)
    # IDs are local to this file; the shape column records their meaning.
    unique_shapes = sorted(df_edges['shape'].fillna('NONE').unique())
    shape_map = {s: i for i, s in enumerate(unique_shapes)}
    df_edges['shape_id'] = df_edges['shape'].map(shape_map)
    invert_dict = dict(zip(df_node_features['node'], df_node_features['invert_elev']))
    link_offsets = get_option(sections, 'LINK_OFFSETS', default='DEPTH')
    slopes = []
    velocities = []
    retention_times = []
    hydraulic_radii = []
    for _, row in df_edges.iterrows():
        if row['edge_type'] != 'conduit':
            slopes.append(0.0)
            hydraulic_radii.append(0.0)
            velocities.append(0.0)
            retention_times.append(0.0)
            continue
        from_node = row['from_node']
        to_node = row['to_node']
        z_from = invert_dict.get(from_node, np.nan)
        z_to = invert_dict.get(to_node, np.nan)
        if np.isnan(z_from) or np.isnan(z_to) or row['length'] <= 0:
            slope = 0.0
        else:
            if link_offsets == 'ELEVATION':
                upstream_elev = row['in_offset'] if row['in_offset'] != 0 else z_from
                downstream_elev = row['out_offset'] if row['out_offset'] != 0 else z_to
            else:
                upstream_elev = z_from + row['in_offset']
                downstream_elev = z_to + row['out_offset']
            slope = abs((upstream_elev - downstream_elev) / row['length'])
        R = hydraulic_radius_full(shape=row['shape'], maxdepth=row['maxdepth'], geom2=row['geom2'])
        velocity = compute_manning_velocity(roughness=row['roughness'], hydraulic_radius=R, slope=slope)
        if velocity > 0:
            retention_time = row['length'] / velocity
        else:
            retention_time = 0.0
        slopes.append(slope)
        hydraulic_radii.append(R)
        velocities.append(velocity)
        retention_times.append(retention_time)
    df_edges['slope'] = slopes
    df_edges['hydraulic_radius'] = hydraulic_radii
    df_edges['velocity_index'] = velocities
    df_edges['retention_time'] = retention_times
    return df_edges[EDGE_COLUMNS]


def extract_static_features_from_inp(inp_path, output_excel_path, encoding="utf-8-sig"):
    """Save feature tables and return tables plus float64 numeric arrays.

    Array columns follow the worksheet order with text columns excluded.
    Arrays are returned, not written to NPY files. Nodes are ordered by
    junctions, outfalls, then storage; edges by conduits, then pumps.
    """
    inp_path = Path(inp_path)
    output_excel_path = Path(output_excel_path)
    if inp_path.resolve() == output_excel_path.resolve():
        raise ValueError("The output path must differ from the input path.")
    sections = read_inp_sections(inp_path, encoding=encoding)
    df_node = build_node_static_features(sections)
    df_edge = build_edge_static_features(sections, df_node)
    node_numeric_cols = [c for c in NODE_COLUMNS if c not in {"node", "node_type"}]
    edge_numeric_cols = [c for c in EDGE_COLUMNS
                         if c not in {"edge", "from_node", "to_node", "edge_type", "shape"}]
    node_array = df_node[node_numeric_cols].to_numpy(dtype=float)
    edge_array = df_edge[edge_numeric_cols].to_numpy(dtype=float)
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        df_node.to_excel(writer, sheet_name="node_static_features", index=False)
        df_edge.to_excel(writer, sheet_name="edge_static_features", index=False)
    print(f"Saved {output_excel_path}")
    print(f"Node numeric features: {node_array.shape}; edge numeric features: {edge_array.shape}")
    return df_node, df_edge, node_array, edge_array


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp", type=Path, required=True, help="Input SWMM INP file.")
    parser.add_argument("--output", type=Path, required=True, help="Output XLSX workbook.")
    parser.add_argument("--encoding", default="utf-8-sig", help="Input encoding, e.g. gbk.")
    args = parser.parse_args()
    extract_static_features_from_inp(args.inp, args.output, encoding=args.encoding)


if __name__ == "__main__":
    main()
