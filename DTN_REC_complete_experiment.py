import os
import ast
import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tkinter import Tk, filedialog

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456, 789, 2026]

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

HIDDEN_DIM = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
MAX_EPOCHS = 500
PATIENCE = 40
EPSILON = 1e-12

# Display figures directly in Spyder's Plots pane / IPython console.
SHOW_PLOTS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_ORDER = ["AvgDelay", "Jitter", "Packet Loss"]
REPRESENTATION_ORDER = ["Traffic", "Traffic + Path", "Topology-aware"]

# Explicit feature groups from the converted GNN Challenge 2023 CSV.
TRAFFIC_CANDIDATES = [
    "AvgBw",
    "PktsGen",
    "TotalPktsGen",
    "AvgPktSize",
    "p10PktSize",
    "p20PktSize",
    "p50PktSize",
    "p80PktSize",
    "p90PktSize",
    "VarPktSize",
]

PATH_CANDIDATES = [
    "num_hops",
    "physical_path_length",
    "min_path_bandwidth",
    "max_path_bandwidth",
    "mean_path_bandwidth",
]


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def select_csv_file():
    root = Tk()
    root.withdraw()

    csv_file = filedialog.askopenfilename(
        title="datasets_v0",
        filetypes=[
            ("TAR files", "*.tar"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not csv_file:
        raise RuntimeError("No CSV file was selected.")

    return os.path.normpath(csv_file)

def normalize_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "")
    )


def find_column(df, candidates, required=False):
    normalized = {normalize_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            "Could not find required column. Tried: "
            + ", ".join(candidates)
        )

    return None


def find_available_features(df, candidates):
    result = []

    for candidate in candidates:
        col = find_column(df, [candidate], required=False)

        if col is not None and col not in result:
            result.append(col)

    return result

def parse_route(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        route = list(value)

    else:
        if pd.isna(value):
            return []

        text = str(value).strip()

        if not text:
            return []

        route = None

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple)):
                route = list(parsed)

        except Exception:
            pass

        if route is None:
            cleaned = (
                text
                .replace("->", ",")
                .replace(";", ",")
                .replace("|", ",")
            )

            # Do not blindly replace "-" because node labels may contain "-".
            if "," in cleaned:
                route = [
                    item.strip()
                    for item in cleaned.split(",")
                    if item.strip()
                ]
            else:
                route = [
                    item.strip()
                    for item in cleaned.split()
                    if item.strip()
                ]

    cleaned_route = []

    for node in route:
        try:
            cleaned_route.append(int(float(node)))
        except Exception:
            cleaned_route.append(str(node))

    return cleaned_route


def route_to_links(route):
    if route is None or len(route) < 2:
        return []

    return [
        (route[i], route[i + 1])
        for i in range(len(route) - 1)
    ]

def load_and_prepare_dataframe(csv_file):
    print("\n" + "=" * 100)
    print("LOADING DATASET")
    print("=" * 100)
    print(csv_file)

    df = pd.read_csv(csv_file, low_memory=False)

    print(f"\nRows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    sample_col = find_column(
        df,
        ["sample_id", "sample", "sampleid", "simulation_id"],
        required=True,
    )

    src_col = find_column(
        df,
        ["src", "source", "source_node", "src_node"],
        required=True,
    )

    dst_col = find_column(
        df,
        ["dst", "destination", "destination_node", "dst_node"],
        required=True,
    )

    route_col = find_column(
        df,
        ["route", "routing", "routing_path", "route_nodes"],
        required=False,
    )

    delay_col = find_column(
        df,
        ["AvgDelay", "avg_delay", "delay", "average_delay"],
        required=True,
    )

    jitter_col = find_column(
        df,
        ["Jitter", "jitter", "avg_jitter", "average_jitter"],
        required=True,
    )

    packet_loss_col = find_column(
        df,
        [
            "packet_loss_ratio",
            "packet_loss",
            "Packet Loss",
            "loss_ratio",
            "pkt_loss",
        ],
        required=False,
    )

    pkts_drop_col = find_column(
        df,
        ["PktsDrop", "pkts_drop", "packets_dropped", "dropped_packets"],
        required=False,
    )

    total_pkts_col = find_column(
        df,
        [
            "TotalPktsGen",
            "total_pkts_gen",
            "PktsGen",
            "pkts_gen",
            "packets_generated",
        ],
        required=False,
    )

    traffic_features = find_available_features(df, TRAFFIC_CANDIDATES)
    path_features = find_available_features(df, PATH_CANDIDATES)

    if len(traffic_features) == 0:
        raise RuntimeError(
            "No traffic features were found. Expected columns such as "
            "AvgBw, PktsGen, TotalPktsGen, AvgPktSize, etc."
        )

    if len(path_features) == 0:
        raise RuntimeError(
            "No path-summary features were found. Expected columns such as "
            "num_hops, physical_path_length, mean_path_bandwidth, etc."
        )

    # -------------------------------------------------------------------------
    # Targets
    # -------------------------------------------------------------------------

    df["__target_delay__"] = pd.to_numeric(
        df[delay_col],
        errors="coerce",
    )

    df["__target_jitter__"] = pd.to_numeric(
        df[jitter_col],
        errors="coerce",
    )

    if packet_loss_col is not None:
        df["__target_packet_loss__"] = pd.to_numeric(
            df[packet_loss_col],
            errors="coerce",
        )

    elif pkts_drop_col is not None and total_pkts_col is not None:
        dropped = pd.to_numeric(
            df[pkts_drop_col],
            errors="coerce",
        )

        generated = pd.to_numeric(
            df[total_pkts_col],
            errors="coerce",
        )

        df["__target_packet_loss__"] = (
            dropped / generated.replace(0, np.nan)
        )

    else:
        raise RuntimeError(
            "Packet Loss cannot be constructed. The CSV needs either "
            "packet_loss_ratio/packet_loss or both PktsDrop and "
            "TotalPktsGen/PktsGen."
        )

    # -------------------------------------------------------------------------
    # IDs
    # -------------------------------------------------------------------------

    for col in [sample_col, src_col, dst_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[sample_col, src_col, dst_col]).copy()

    df[sample_col] = df[sample_col].astype(int)
    df[src_col] = df[src_col].astype(int)
    df[dst_col] = df[dst_col].astype(int)

    all_features = list(dict.fromkeys(traffic_features + path_features))

    for col in all_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nDetected columns")
    print("Sample ID    :", sample_col)
    print("Source       :", src_col)
    print("Destination  :", dst_col)
    print("Route        :", route_col)
    print("AvgDelay     :", delay_col)
    print("Jitter       :", jitter_col)
    print("Packet Loss  :", packet_loss_col or "constructed from dropped/generated")

    print("\nTraffic features:")
    for x in traffic_features:
        print("  -", x)

    print("\nPath features:")
    for x in path_features:
        print("  -", x)

    metadata = {
        "sample": sample_col,
        "src": src_col,
        "dst": dst_col,
        "route": route_col,
    }

    feature_sets = {
        "Traffic": traffic_features,
        "Traffic + Path": list(
            dict.fromkeys(traffic_features + path_features)
        ),
        "Topology-aware": list(
            dict.fromkeys(traffic_features + path_features)
        ),
    }

    target_columns = {
        "AvgDelay": "__target_delay__",
        "Jitter": "__target_jitter__",
        "Packet Loss": "__target_packet_loss__",
    }

    return df, metadata, feature_sets, target_columns

def create_sample_split(df, sample_col, seed):
    sample_ids = np.asarray(
        sorted(df[sample_col].dropna().unique())
    )

    rng = np.random.RandomState(seed)
    rng.shuffle(sample_ids)

    n = len(sample_ids)

    if n < 5:
        raise RuntimeError(
            "At least five network samples are required."
        )

    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train:n_train + n_val]
    test_ids = sample_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


# =============================================================================
# TRAIN-ONLY FEATURE PREPROCESSING
# =============================================================================

class FeaturePreprocessor:

    def __init__(self, feature_columns):
        self.feature_columns = list(feature_columns)
        self.medians = None
        self.scaler = StandardScaler()

    def fit(self, train_df):
        X = train_df[self.feature_columns].copy()

        medians = X.median(axis=0, skipna=True)
        medians = medians.fillna(0.0)

        self.medians = medians

        X = X.fillna(self.medians)

        self.scaler.fit(
            X.to_numpy(dtype=np.float64)
        )

        return self

    def transform(self, df):
        X = df[self.feature_columns].copy()
        X = X.fillna(self.medians)

        return self.scaler.transform(
            X.to_numpy(dtype=np.float64)
        ).astype(np.float32)


class TargetScaler:

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)

        self.mean = float(np.mean(y))
        self.std = float(np.std(y))

        if not np.isfinite(self.std) or self.std < EPSILON:
            self.std = 1.0

        return self

    def transform(self, y):
        return (
            (np.asarray(y, dtype=np.float64) - self.mean)
            / self.std
        ).astype(np.float32)

    def inverse_transform(self, y):
        return (
            np.asarray(y, dtype=np.float64) * self.std
            + self.mean
        )


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_flow_adjacency(group, metadata, topology_aware):
    """
    One node = one source-destination flow.

    Traffic / Traffic+Path:
        identity adjacency. This preserves the same GNN architecture while
        preventing topology information from entering E1/E2.

    Topology-aware:
        connect two flows if their routes share a directed link or a route node.
        Self-loops are always retained.
    """

    n = len(group)

    if not topology_aware:
        return np.eye(n, dtype=np.float32)

    routes = []

    for _, row in group.iterrows():
        src = int(row[metadata["src"]])
        dst = int(row[metadata["dst"]])

        if metadata["route"] is not None:
            route = parse_route(
                row[metadata["route"]]
            )
        else:
            route = []

        if len(route) < 2:
            route = [src, dst]

        routes.append(route)

    link_sets = [
        set(route_to_links(route))
        for route in routes
    ]

    node_sets = [
        set(route)
        for route in routes
    ]

    A = np.eye(n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            shared_directed_link = bool(
                link_sets[i] & link_sets[j]
            )

            # Prefer RouteNet-like structural evidence: shared directed links.
            # A weaker connection is used only when two flows are consecutive
            # at an endpoint (destination of one is source of the other).
            endpoint_continuity = (
                routes[i][-1] == routes[j][0]
                or routes[j][-1] == routes[i][0]
            )

            if shared_directed_link:
                union = len(link_sets[i] | link_sets[j])
                overlap = len(link_sets[i] & link_sets[j]) / max(union, 1)
                weight = 1.0 + overlap
                A[i, j] = weight
                A[j, i] = weight
            elif endpoint_continuity:
                A[i, j] = 0.25
                A[j, i] = 0.25

    degree = A.sum(axis=1)
    degree = np.maximum(degree, EPSILON)

    inv_sqrt = 1.0 / np.sqrt(degree)

    A_norm = (
        inv_sqrt[:, None]
        * A
        * inv_sqrt[None, :]
    )

    return A_norm.astype(np.float32)


# =============================================================================
# 10. BUILD GRAPH OBJECTS
# =============================================================================

def build_graphs(
    structure_df,
    supervised_df,
    metadata,
    feature_columns,
    preprocessor,
    target_col,
    target_scaler,
    topology_aware,
):
    """
    Graph structure is built from ALL rows in structure_df for each sample.
    Supervision is applied only to rows in supervised_df with a valid target.

    This prevents invalid target rows from accidentally deleting route/topology
    information from the graph.
    """

    graphs = []

    supervised_index_set = set(
        supervised_df.index.tolist()
    )

    for sample_id, group in structure_df.groupby(
        metadata["sample"],
        sort=True,
    ):
        group = group.copy()

        # Preserve original dataframe indices for the supervision mask.
        original_indices = group.index.to_numpy()

        X = preprocessor.transform(group)

        A = build_flow_adjacency(
            group,
            metadata,
            topology_aware=topology_aware,
        )

        mask = np.asarray(
            [
                idx in supervised_index_set
                for idx in original_indices
            ],
            dtype=bool,
        )

        if mask.sum() == 0:
            continue

        y_raw = pd.to_numeric(
            group[target_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        y_scaled = np.zeros(
            len(group),
            dtype=np.float32,
        )

        valid_positions = np.where(mask)[0]

        y_scaled[valid_positions] = target_scaler.transform(
            y_raw[valid_positions]
        )

        graphs.append(
            {
                "sample_id": int(sample_id),
                "X": torch.tensor(
                    X,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                "A": torch.tensor(
                    A,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                "mask": torch.tensor(
                    mask,
                    dtype=torch.bool,
                    device=DEVICE,
                ),
                "y": torch.tensor(
                    y_scaled,
                    dtype=torch.float32,
                    device=DEVICE,
                ),
                "y_raw": y_raw,
                "row_indices": original_indices,
            }
        )

    return graphs

class GNNRegressor(nn.Module):
    """Residual two-layer message-passing regressor.

    The predictor is identical across representations. For Traffic and
    Traffic+Path the adjacency is identity; for Topology-aware it is the
    normalized route-derived adjacency. Self and neighbor information are
    concatenated to reduce over-smoothing.
    """

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, adjacency):
        h0 = self.encoder(x)

        n1 = torch.matmul(adjacency, h0)
        h1 = self.conv1(torch.cat([h0, n1], dim=-1))
        h1 = self.norm1(h1)
        h1 = torch.relu(h1)
        h1 = self.dropout(h1)
        h1 = h1 + h0

        n2 = torch.matmul(adjacency, h1)
        h2 = self.conv2(torch.cat([h1, n2], dim=-1))
        h2 = self.norm2(h2)
        h2 = torch.relu(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h1

        return self.readout(h2).squeeze(-1)


def evaluate_graph_loss(model, graphs):
    model.eval()

    total_loss = 0.0
    total_count = 0

    criterion = nn.MSELoss(
        reduction="sum"
    )

    with torch.no_grad():
        for graph in graphs:
            pred = model(
                graph["X"],
                graph["A"],
            )

            mask = graph["mask"]

            loss = criterion(
                pred[mask],
                graph["y"][mask],
            )

            total_loss += float(loss.item())
            total_count += int(mask.sum().item())

    if total_count == 0:
        return np.inf

    return total_loss / total_count


def train_gnn(train_graphs, val_graphs, input_dim):
    model = GNNRegressor(
        input_dim=input_dim
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    criterion = nn.MSELoss(
        reduction="sum"
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    best_val = np.inf
    patience_counter = 0

    train_history = []
    val_history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        order = np.random.permutation(
            len(train_graphs)
        )

        total_loss = 0.0
        total_count = 0

        for graph_index in order:
            graph = train_graphs[
                int(graph_index)
            ]

            optimizer.zero_grad()

            pred = model(
                graph["X"],
                graph["A"],
            )

            mask = graph["mask"]

            loss_sum = criterion(
                pred[mask],
                graph["y"][mask],
            )

            count = int(
                mask.sum().item()
            )

            loss = (
                loss_sum
                / max(count, 1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += float(
                loss_sum.item()
            )

            total_count += count

        train_loss = (
            total_loss
            / max(total_count, 1)
        )

        val_loss = evaluate_graph_loss(
            model,
            val_graphs,
        )
        scheduler.step(val_loss)

        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0

        else:
            patience_counter += 1

        if (
            epoch == 1
            or epoch % 25 == 0
            or patience_counter >= PATIENCE
        ):
            print(
                f"      Epoch {epoch:4d} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f}"
            )

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(
        best_state
    )

    return model, train_history, val_history

def predict_graphs(model, graphs, target_scaler):
    model.eval()

    true_values = []
    pred_values = []
    sample_ids = []
    row_indices = []

    with torch.no_grad():
        for graph in graphs:
            pred_scaled = model(
                graph["X"],
                graph["A"],
            ).detach().cpu().numpy()

            mask = (
                graph["mask"]
                .detach()
                .cpu()
                .numpy()
            )

            positions = np.where(mask)[0]

            pred_raw = target_scaler.inverse_transform(
                pred_scaled[positions]
            )

            true_raw = graph["y_raw"][
                positions
            ]

            true_values.extend(
                true_raw.tolist()
            )

            pred_values.extend(
                pred_raw.tolist()
            )

            sample_ids.extend(
                [graph["sample_id"]] * len(positions)
            )

            row_indices.extend(
                graph["row_indices"][
                    positions
                ].tolist()
            )

    return (
        np.asarray(true_values, dtype=np.float64),
        np.asarray(pred_values, dtype=np.float64),
        np.asarray(sample_ids),
        np.asarray(row_indices),
    )


# =============================================================================
# STANDARD REGRESSION METRICS
# =============================================================================

def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }


# =============================================================================
# 15. REC / PRED METRICS
# =============================================================================

def calculate_relative_errors(y_true, y_pred):
    """
    Absolute relative error:
        |y_true - y_pred| / |y_true|

    Zero-valued ground truth is excluded because percentage relative error
    is undefined at y_true == 0.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(y_true)
        &
        np.isfinite(y_pred)
    )

    nonzero = (
        np.abs(y_true) > EPSILON
    )

    valid = finite & nonzero

    if valid.sum() == 0:
        return np.asarray(
            [],
            dtype=np.float64,
        ), valid

    relative_error = (
        np.abs(
            y_true[valid]
            -
            y_pred[valid]
        )
        /
        np.abs(
            y_true[valid]
        )
    )

    return relative_error, valid


def calculate_rec_metrics(y_true, y_pred):
    relative_error, valid_mask = (
        calculate_relative_errors(
            y_true,
            y_pred,
        )
    )

    if len(relative_error) == 0:
        return {
            "PRED@5%": np.nan,
            "PRED@10%": np.nan,
            "PRED@20%": np.nan,
            "90%-Error Tolerance": np.nan,
            "REC_Valid_N": 0,
            "REC_Excluded_Zero_N": int(
                len(y_true)
            ),
        }

    pred_5 = (
        np.mean(
            relative_error <= 0.05
        )
        * 100.0
    )

    pred_10 = (
        np.mean(
            relative_error <= 0.10
        )
        * 100.0
    )

    pred_20 = (
        np.mean(
            relative_error <= 0.20
        )
        * 100.0
    )

    sorted_error = np.sort(relative_error)
    k90 = max(0, int(np.ceil(0.90 * len(sorted_error))) - 1)
    tolerance_90 = sorted_error[k90] * 100.0

    return {
        "PRED@5%": float(pred_5),
        "PRED@10%": float(pred_10),
        "PRED@20%": float(pred_20),
        "90%-Error Tolerance": float(tolerance_90),
        "REC_Valid_N": int(
            len(relative_error)
        ),
        "REC_Excluded_Zero_N": int(
            len(y_true)
            -
            len(relative_error)
        ),
    }

def save_training_curve(
    train_history,
    val_history,
    target_name,
    representation,
    seed,
    output_dir,
):
    plt.figure(figsize=(7, 5))

    plt.plot(
        np.arange(1, len(train_history) + 1),
        train_history,
        label="Training",
    )

    plt.plot(
        np.arange(1, len(val_history) + 1),
        val_history,
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE Loss")
    plt.title(
        f"{target_name} | {representation} | Seed {seed}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    safe_target = (
        target_name
        .replace(" ", "_")
        .replace("/", "_")
    )

    safe_rep = (
        representation
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )

    filename = os.path.join(
        output_dir,
        f"training_{safe_target}_{safe_rep}_seed_{seed}.png",
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close()


# =============================================================================
# 17. REC CURVES
# =============================================================================

def create_rec_plots(predictions_df, output_dir):
    """Plot empirical REC curves and ensure the 90% crossing is visible."""
    for target_name in TARGET_ORDER:
        plt.figure(figsize=(7.2, 5.2))
        target_subset = predictions_df[predictions_df["Target"] == target_name]

        errors_by_rep = {}
        tau90_by_rep = {}
        for representation in REPRESENTATION_ORDER:
            subset = target_subset[target_subset["Representation"] == representation]
            if len(subset) == 0:
                continue
            relative_error, _ = calculate_relative_errors(
                subset["y_true"].to_numpy(), subset["y_pred"].to_numpy()
            )
            if len(relative_error) == 0:
                continue
            errors_by_rep[representation] = relative_error
            ordered = np.sort(relative_error)
            k90 = max(0, int(np.ceil(0.90 * len(ordered))) - 1)
            tau90_by_rep[representation] = ordered[k90] * 100.0

        if not errors_by_rep:
            plt.close()
            continue

        # Unlike the old 200% cap, this range includes every empirical 90% crossing.
        x_max = max(20.0, 1.10 * max(tau90_by_rep.values()))
        # Keep pathological tails from making the plot unreadable while never
        # hiding tau90 itself.
        thresholds_pct = np.linspace(0.0, x_max, 700)

        for representation in REPRESENTATION_ORDER:
            if representation not in errors_by_rep:
                continue
            err_pct = errors_by_rep[representation] * 100.0
            pred_values = np.array([
                100.0 * np.mean(err_pct <= t) for t in thresholds_pct
            ])
            plt.plot(thresholds_pct, pred_values, linewidth=2, label=representation)

        plt.axhline(90.0, linestyle="--", linewidth=1.2, label="90% coverage")
        for x in (5.0, 10.0, 20.0):
            plt.axvline(x, linestyle=":", linewidth=1.0)

        plt.xlabel("Relative Error Tolerance (%)")
        plt.ylabel("PRED / Cumulative Predictions (%)")
        plt.title(f"REC Curve — {target_name}")
        plt.xlim(0, x_max)
        plt.ylim(0, 100)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        safe_target = target_name.replace(" ", "_").replace("/", "_")
        plt.savefig(
            os.path.join(output_dir, f"REC_{safe_target}.png"),
            dpi=300, bbox_inches="tight"
        )
        if SHOW_PLOTS:
            plt.show()
        plt.close()

def main():
    print("\n" + "=" * 100)
    print("GNN REC EXPERIMENT")
    print("=" * 100)

    print("\nDevice :", DEVICE)
    print("Seeds  :", SEEDS)
    print("Split  : 60% train / 20% validation / 20% test")
    print("Epochs :", MAX_EPOCHS)
    print("Patience:", PATIENCE)

    csv_file = select_csv_file()

    output_dir = os.path.join(
        os.path.dirname(csv_file),
        "GNN_REC_Results",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    (
        df,
        metadata,
        feature_sets,
        target_columns,
    ) = load_and_prepare_dataframe(
        csv_file
    )

    # Save feature design for reproducibility.
    feature_rows = []

    for representation in REPRESENTATION_ORDER:
        for feature in feature_sets[
            representation
        ]:
            feature_rows.append(
                {
                    "Representation": representation,
                    "Feature": feature,
                }
            )

    pd.DataFrame(
        feature_rows
    ).to_csv(
        os.path.join(
            output_dir,
            "GNN_REC_Feature_Design.csv",
        ),
        index=False,
    )

    per_seed_rows = []
    prediction_frames = []

    # =========================================================================
    # FIVE SEEDS
    # =========================================================================

    for run_number, seed in enumerate(
        SEEDS,
        start=1,
    ):
        print("\n\n" + "#" * 100)
        print(
            f"RUN {run_number}/{len(SEEDS)} | SEED = {seed}"
        )
        print("#" * 100)

        set_seed(seed)

        (
            train_ids,
            val_ids,
            test_ids,
        ) = create_sample_split(
            df,
            metadata["sample"],
            seed,
        )

        split_df = pd.DataFrame(
            {
                "sample_id": np.concatenate(
                    [
                        train_ids,
                        val_ids,
                        test_ids,
                    ]
                ),
                "partition": (
                    ["train"] * len(train_ids)
                    +
                    ["validation"] * len(val_ids)
                    +
                    ["test"] * len(test_ids)
                ),
            }
        )

        split_df.to_csv(
            os.path.join(
                output_dir,
                f"split_seed_{seed}.csv",
            ),
            index=False,
        )

        structure_train = df[
            df[metadata["sample"]].isin(
                train_ids
            )
        ].copy()

        structure_val = df[
            df[metadata["sample"]].isin(
                val_ids
            )
        ].copy()

        structure_test = df[
            df[metadata["sample"]].isin(
                test_ids
            )
        ].copy()

        print(
            "\nSample counts:",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )

        for target_name in TARGET_ORDER:
            target_col = target_columns[
                target_name
            ]

            print("\n" + "=" * 100)
            print("TARGET:", target_name)
            print("=" * 100)

            # Valid target rows for this target only.
            train_target = pd.to_numeric(
                structure_train[target_col],
                errors="coerce",
            )

            val_target = pd.to_numeric(
                structure_val[target_col],
                errors="coerce",
            )

            test_target = pd.to_numeric(
                structure_test[target_col],
                errors="coerce",
            )

            supervised_train = structure_train[
                train_target.notna()
                &
                np.isfinite(train_target)
                &
                (train_target >= 0)
            ].copy()

            supervised_val = structure_val[
                val_target.notna()
                &
                np.isfinite(val_target)
                &
                (val_target >= 0)
            ].copy()

            supervised_test = structure_test[
                test_target.notna()
                &
                np.isfinite(test_target)
                &
                (test_target >= 0)
            ].copy()

            if (
                len(supervised_train) == 0
                or len(supervised_val) == 0
                or len(supervised_test) == 0
            ):
                print(
                    "Skipping target because one split "
                    "contains no valid observations."
                )
                continue

            target_scaler = TargetScaler().fit(
                supervised_train[
                    target_col
                ].to_numpy(
                    dtype=np.float64
                )
            )

            for representation in REPRESENTATION_ORDER:
                print(
                    f"\n  Representation: {representation}"
                )

                set_seed(seed)

                feature_columns = feature_sets[
                    representation
                ]

                preprocessor = FeaturePreprocessor(
                    feature_columns
                ).fit(
                    structure_train
                )

                topology_aware = (
                    representation
                    ==
                    "Topology-aware"
                )

                train_graphs = build_graphs(
                    structure_train,
                    supervised_train,
                    metadata,
                    feature_columns,
                    preprocessor,
                    target_col,
                    target_scaler,
                    topology_aware,
                )

                val_graphs = build_graphs(
                    structure_val,
                    supervised_val,
                    metadata,
                    feature_columns,
                    preprocessor,
                    target_col,
                    target_scaler,
                    topology_aware,
                )

                test_graphs = build_graphs(
                    structure_test,
                    supervised_test,
                    metadata,
                    feature_columns,
                    preprocessor,
                    target_col,
                    target_scaler,
                    topology_aware,
                )

                print(
                    f"      Graphs: "
                    f"train={len(train_graphs)}, "
                    f"val={len(val_graphs)}, "
                    f"test={len(test_graphs)}"
                )

                model, train_history, val_history = (
                    train_gnn(
                        train_graphs,
                        val_graphs,
                        input_dim=len(
                            feature_columns
                        ),
                    )
                )

                save_training_curve(
                    train_history,
                    val_history,
                    target_name,
                    representation,
                    seed,
                    output_dir,
                )

                (
                    y_true,
                    y_pred,
                    sample_ids,
                    row_indices,
                ) = predict_graphs(
                    model,
                    test_graphs,
                    target_scaler,
                )

                regression = (
                    calculate_regression_metrics(
                        y_true,
                        y_pred,
                    )
                )

                rec = calculate_rec_metrics(
                    y_true,
                    y_pred,
                )

                print(
                    f"      MSE={regression['MSE']:.6g} | "
                    f"RMSE={regression['RMSE']:.6g} | "
                    f"MAE={regression['MAE']:.6g} | "
                    f"R2={regression['R2']:.6f}"
                )

                print(
                    f"      PRED@5%={rec['PRED@5%']:.2f}% | "
                    f"PRED@10%={rec['PRED@10%']:.2f}% | "
                    f"PRED@20%={rec['PRED@20%']:.2f}% | "
                    f"90%-Tolerance="
                    f"{rec['90%-Error Tolerance']:.2f}%"
                )

                per_seed_rows.append(
                    {
                        "Run": run_number,
                        "Seed": seed,
                        "Target": target_name,
                        "Representation": representation,
                        "MSE": regression["MSE"],
                        "RMSE": regression["RMSE"],
                        "MAE": regression["MAE"],
                        "R2": regression["R2"],
                        "PRED@5%": rec["PRED@5%"],
                        "PRED@10%": rec["PRED@10%"],
                        "PRED@20%": rec["PRED@20%"],
                        "90%-Error Tolerance": rec[
                            "90%-Error Tolerance"
                        ],
                        "REC_Valid_N": rec[
                            "REC_Valid_N"
                        ],
                        "REC_Excluded_Zero_N": rec[
                            "REC_Excluded_Zero_N"
                        ],
                    }
                )

                relative_error = np.full(
                    len(y_true),
                    np.nan,
                    dtype=np.float64,
                )

                valid_rel = (
                    np.abs(y_true) > EPSILON
                )

                relative_error[
                    valid_rel
                ] = (
                    np.abs(
                        y_true[valid_rel]
                        -
                        y_pred[valid_rel]
                    )
                    /
                    np.abs(
                        y_true[valid_rel]
                    )
                )

                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "Run": run_number,
                            "Seed": seed,
                            "Target": target_name,
                            "Representation": representation,
                            "sample_id": sample_ids,
                            "row_index": row_indices,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "absolute_error": np.abs(
                                y_true - y_pred
                            ),
                            "relative_error": relative_error,
                            "relative_error_percent": (
                                relative_error * 100.0
                            ),
                        }
                    )
                )

                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    per_seed_file = os.path.join(
        output_dir,
        "GNN_REC_Per_Seed.csv",
    )

    per_seed_df.to_csv(
        per_seed_file,
        index=False,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_file = os.path.join(
        output_dir,
        "GNN_REC_All_Test_Predictions.csv",
    )

    predictions_df.to_csv(
        predictions_file,
        index=False,
    )

    # =========================================================================
    # AGGREGATE FIVE SEEDS
    #
    # PRED and tolerance are calculated per seed first, then mean ± std.
    # This treats the five random-seed runs as the experimental replicates.
    # =========================================================================

    metrics_to_aggregate = [
        "MSE",
        "RMSE",
        "MAE",
        "R2",
        "PRED@5%",
        "PRED@10%",
        "PRED@20%",
        "90%-Error Tolerance",
    ]

    summary_rows = []

    for target_name in TARGET_ORDER:
        for representation in REPRESENTATION_ORDER:
            subset = per_seed_df[
                (
                    per_seed_df["Target"]
                    ==
                    target_name
                )
                &
                (
                    per_seed_df["Representation"]
                    ==
                    representation
                )
            ]

            if len(subset) == 0:
                continue

            row = {
                "Target": target_name,
                "Representation": representation,
                "Runs": int(len(subset)),
            }

            for metric in metrics_to_aggregate:
                values = subset[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )

                row[
                    f"{metric}_Mean"
                ] = float(
                    np.nanmean(values)
                )

                row[
                    f"{metric}_STD"
                ] = float(
                    np.nanstd(
                        values,
                        ddof=1,
                    )
                )

            row["REC_Valid_N_Mean"] = float(
                subset[
                    "REC_Valid_N"
                ].mean()
            )

            row[
                "REC_Excluded_Zero_N_Mean"
            ] = float(
                subset[
                    "REC_Excluded_Zero_N"
                ].mean()
            )

            summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    numeric_summary_file = os.path.join(
        output_dir,
        "GNN_REC_Numeric_Summary.csv",
    )

    summary_df.to_csv(
        numeric_summary_file,
        index=False,
    )

    paper_rows = []

    for target_name in TARGET_ORDER:
        for representation in REPRESENTATION_ORDER:
            subset = summary_df[
                (
                    summary_df["Target"]
                    ==
                    target_name
                )
                &
                (
                    summary_df["Representation"]
                    ==
                    representation
                )
            ]

            if len(subset) == 0:
                continue

            row = subset.iloc[0]

            paper_rows.append(
                {
                    "Target": target_name,
                    "Representation": representation,
                    "PRED@5%": (
                        f"{row['PRED@5%_Mean']:.2f} "
                        f"± {row['PRED@5%_STD']:.2f}%"
                    ),
                    "PRED@10%": (
                        f"{row['PRED@10%_Mean']:.2f} "
                        f"± {row['PRED@10%_STD']:.2f}%"
                    ),
                    "PRED@20%": (
                        f"{row['PRED@20%_Mean']:.2f} "
                        f"± {row['PRED@20%_STD']:.2f}%"
                    ),
                    "90%-Error Tolerance": (
                        f"{row['90%-Error Tolerance_Mean']:.2f} "
                        f"± {row['90%-Error Tolerance_STD']:.2f}%"
                    ),
                }
            )

    paper_df = pd.DataFrame(
        paper_rows
    )

    paper_file = os.path.join(
        output_dir,
        "GNN_REC_Results.csv",
    )

    paper_df.to_csv(
        paper_file,
        index=False,
    )

    create_rec_plots(
        predictions_df,
        output_dir,
    )

    print("\n\n" + "=" * 120)
    print(
        "FINAL GNN REC RESULTS — FIVE SEEDS: MEAN ± STANDARD DEVIATION"
    )
    print("=" * 120)

    print(
        paper_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 100)

    print("\nResults directory:")
    print(output_dir)

    print("\nGenerated files:")
    print("  1. GNN_REC_Per_Seed.csv")
    print("  2. GNN_REC_Numeric_Summary.csv")
    print("  3. GNN_REC_Results.csv")
    print("  4. GNN_REC_All_Test_Predictions.csv")
    print("  5. GNN_REC_Feature_Design.csv")
    print("  6. REC_AvgDelay.png")
    print("  7. REC_Jitter.png")
    print("  8. REC_Packet_Loss.png")
    print("  9. training curves")
    print(" 10. split_seed_<seed>.csv")

if __name__ == "__main__":
    main()
