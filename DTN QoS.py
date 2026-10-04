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
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tkinter import Tk, filedialog


warnings.filterwarnings("ignore")



SEEDS = [
    42,
    123,
    456,
    789,
    2026
]

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
HIDDEN_DIM = 64

LEARNING_RATE = 0.01

WEIGHT_DECAY = 5e-4

DROPOUT = 0.5

MAX_EPOCHS = 500

PATIENCE = 50

NUM_TRANSFORMER_LAYERS = 2

NUM_ATTENTION_HEADS = 4

ROUTENET_ITERATIONS = 8

EPSILON = 1e-8


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_NAMES = [
    "MLP",
    "GNN",
    "Graph Transformer",
    "RouteNet-F"
]


TARGET_NAMES = [
    "Delay",
    "Jitter",
    "Packet Loss"
]


METRIC_NAMES = [
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
    "R2"
]


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


# =============================================================================
# 4. SELECT CSV FILE
# =============================================================================

def select_csv_file():

    root = Tk()

    root.withdraw()

    csv_file = filedialog.askopenfilename(

        title="Select combined network-performance CSV",

        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not csv_file:

        raise RuntimeError(
            "No CSV file was selected."
        )

    return os.path.normpath(
        csv_file
    )

def find_column(
    df,
    candidates,
    required=False
):

    lower_map = {
        str(column).lower().strip():
            column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().strip()

        if key in lower_map:

            return lower_map[key]

    # -------------------------------------------------------------------------
    # Normalized matching
    # -------------------------------------------------------------------------

    def normalize(name):

        return (
            str(name)
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "")
        )

    normalized_map = {

        normalize(column):
            column

        for column in df.columns
    }

    for candidate in candidates:

        key = normalize(
            candidate
        )

        if key in normalized_map:

            return normalized_map[
                key
            ]

    if required:

        raise ValueError(

            "\nCould not find one of the required columns:\n"

            + "\n".join(
                f"  - {candidate}"
                for candidate in candidates
            )
        )

    return None

def parse_route(value):

    if value is None:

        return []

    if isinstance(
        value,
        (list, tuple, np.ndarray)
    ):

        route = list(
            value
        )

    else:

        if pd.isna(value):

            return []

        text = str(
            value
        ).strip()

        if not text:

            return []

        route = None

        # ---------------------------------------------------------------------
        # Python-list representation:
        # [0, 1, 4]
        # ---------------------------------------------------------------------

        try:

            parsed = ast.literal_eval(
                text
            )

            if isinstance(
                parsed,
                (list, tuple)
            ):

                route = list(
                    parsed
                )

        except Exception:

            pass

        # ---------------------------------------------------------------------
        # Other representations:
        # 0-1-4
        # 0,1,4
        # 0 1 4
        # 0;1;4
        # ---------------------------------------------------------------------

        if route is None:

            cleaned = (
                text
                .replace("->", ",")
                .replace("-", ",")
                .replace(";", ",")
                .replace("|", ",")
            )

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

            cleaned_route.append(
                int(float(node))
            )

        except Exception:

            cleaned_route.append(
                str(node)
            )

    return cleaned_route


# =============================================================================
#  ROUTE TO LINKS
# =============================================================================

def route_to_links(route):

    if route is None:

        return []

    if len(route) < 2:

        return []

    return [

        (
            route[index],
            route[index + 1]
        )

        for index in range(
            len(route) - 1
        )
    ]


# =============================================================================
# NETWORK SAMPLE
# =============================================================================

class NetworkSample:

    def __init__(
        self,
        sample_id,
        flow_features,
        targets,
        routes,
        flow_pairs
    ):

        self.sample_id = sample_id

        self.flow_features = flow_features

        self.targets = targets

        self.routes = routes

        self.flow_pairs = flow_pairs




def load_csv(csv_file):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LOADING COMBINED CSV DATASET"
    )

    print(
        "=" * 100
    )

    print(
        "\nCSV:"
    )

    print(
        csv_file
    )

    df = pd.read_csv(
        csv_file,
        low_memory=False
    )

    print(
        "\nRows:",
        f"{len(df):,}"
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "\nCSV columns:"
    )

    for index, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{index:3d}. {column}"
        )

    return df


def identify_columns(df):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "IDENTIFYING DATASET COLUMNS"
    )

    print(
        "=" * 100
    )

    sample_column = find_column(

        df,

        [
            "sample_id",
            "sample",
            "sampleid",
            "simulation_id",
            "simulation"
        ],

        required=True
    )


    src_column = find_column(

        df,

        [
            "src",
            "source",
            "source_node",
            "src_node"
        ],

        required=True
    )


    dst_column = find_column(

        df,

        [
            "dst",
            "destination",
            "destination_node",
            "dst_node"
        ],

        required=True
    )


    route_column = find_column(

        df,

        [
            "route",
            "routing",
            "path",
            "routing_path",
            "route_nodes",
            "routing_nodes"
        ],

        required=False
    )


    delay_column = find_column(

        df,

        [
            "delay",
            "avg_delay",
            "AvgDelay",
            "average_delay",
            "mean_delay"
        ],

        required=True
    )


    jitter_column = find_column(

        df,

        [
            "jitter",
            "Jitter",
            "avg_jitter",
            "average_jitter"
        ],

        required=True
    )


    packet_loss_column = find_column(

        df,

        [
            "packet_loss",
            "packet loss",
            "packet_loss_ratio",
            "loss",
            "loss_ratio",
            "pkt_loss",
            "packetloss"
        ],

        required=False
    )


    packets_dropped_column = find_column(

        df,

        [
            "pkts_drop",
            "PktsDrop",
            "packets_dropped",
            "dropped_packets"
        ],

        required=False
    )


    packets_generated_column = find_column(

        df,

        [
            "pkts_gen",
            "PktsGen",
            "packets_generated",
            "generated_packets"
        ],

        required=False
    )


    print(
        "\nDetected:"
    )

    print(
        "Sample ID       :",
        sample_column
    )

    print(
        "Source          :",
        src_column
    )

    print(
        "Destination     :",
        dst_column
    )

    print(
        "Route           :",
        route_column
    )

    print(
        "Delay           :",
        delay_column
    )

    print(
        "Jitter          :",
        jitter_column
    )

    print(
        "Packet Loss     :",
        packet_loss_column
    )

    print(
        "Packets Dropped :",
        packets_dropped_column
    )

    print(
        "Packets Gen.    :",
        packets_generated_column
    )


    if (
        packet_loss_column is None
        and
        (
            packets_dropped_column is None
            or
            packets_generated_column is None
        )
    ):

        raise ValueError(

            "\nPacket loss cannot be constructed.\n"

            "The CSV must contain either:\n"
            "  packet_loss / packet_loss_ratio\n\n"
            "or both:\n"
            "  packets dropped\n"
            "  packets generated"
        )


    return {

        "sample":
            sample_column,

        "src":
            src_column,

        "dst":
            dst_column,

        "route":
            route_column,

        "delay":
            delay_column,

        "jitter":
            jitter_column,

        "packet_loss":
            packet_loss_column,

        "packets_dropped":
            packets_dropped_column,

        "packets_generated":
            packets_generated_column
    }

def numeric_series(
    df,
    column
):

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )

def construct_targets(
    df,
    columns
):

    delay = numeric_series(
        df,
        columns["delay"]
    )

    jitter = numeric_series(
        df,
        columns["jitter"]
    )

    if columns[
        "packet_loss"
    ] is not None:

        packet_loss = numeric_series(
            df,
            columns["packet_loss"]
        )

    else:

        dropped = numeric_series(
            df,
            columns[
                "packets_dropped"
            ]
        )

        generated = numeric_series(
            df,
            columns[
                "packets_generated"
            ]
        )

        packet_loss = (
            dropped
            /
            generated.replace(
                0,
                np.nan
            )
        )


    targets = pd.DataFrame(

        {
            "__target_delay__":
                delay,

            "__target_jitter__":
                jitter,

            "__target_packet_loss__":
                packet_loss
        }
    )


    return targets


def select_feature_columns(
    df,
    columns
):


    excluded = {

        columns["sample"],
        columns["src"],
        columns["dst"],
        columns["route"],
        columns["delay"],
        columns["jitter"],
        columns["packet_loss"],
        columns["packets_dropped"]
    }

    excluded = {
        column
        for column in excluded
        if column is not None
    }


    leakage_keywords = [

        "target",
        "predicted",

        "avgdelay",
        "average_delay",
        "mean_delay",

        "jitter",

        "packetloss",
        "packet_loss",

        "pktsdrop",
        "packets_dropped",

        "loss_ratio",

        "performance"
    ]


    feature_columns = []


    for column in df.columns:

        if column in excluded:

            continue

        normalized = (
            str(column)
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )


        if any(
            keyword.replace("_", "") in normalized
            for keyword in leakage_keywords
        ):

            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )


        valid_fraction = (
            converted.notna().mean()
        )


        if valid_fraction >= 0.90:

            feature_columns.append(
                column
            )


    if len(feature_columns) == 0:

        raise RuntimeError(
            "No usable numerical input features were found."
        )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "INPUT FEATURES"
    )

    print(
        "=" * 100
    )

    print(
        "\nNumber of features:",
        len(feature_columns)
    )


    for index, column in enumerate(
        feature_columns,
        start=1
    ):

        print(
            f"{index:3d}. {column}"
        )


    return feature_columns


def prepare_dataframe(
    df,
    columns,
    feature_columns
):

    df = df.copy()


    targets = construct_targets(
        df,
        columns
    )


    for column in targets.columns:

        df[column] = targets[
            column
        ]

    df[
        columns["sample"]
    ] = pd.to_numeric(

        df[
            columns["sample"]
        ],

        errors="coerce"
    )


    df[
        columns["src"]
    ] = pd.to_numeric(

        df[
            columns["src"]
        ],

        errors="coerce"
    )


    df[
        columns["dst"]
    ] = pd.to_numeric(

        df[
            columns["dst"]
        ],

        errors="coerce"
    )

    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    required = [

        columns["sample"],
        columns["src"],
        columns["dst"],

        "__target_delay__",
        "__target_jitter__",
        "__target_packet_loss__"
    ]


    before = len(
        df
    )


    df = df.dropna(
        subset=required
    )

    df = df[

        (
            df[
                "__target_delay__"
            ] >= 0
        )

        &

        (
            df[
                "__target_jitter__"
            ] >= 0
        )

        &

        (
            df[
                "__target_packet_loss__"
            ] >= 0
        )
    ]


    after = len(
        df
    )


    print(
        "\nRows removed because of invalid targets/IDs:",
        before - after
    )

    for column in feature_columns:

        median = df[
            column
        ].median()

        if not np.isfinite(
            median
        ):

            median = 0.0

        df[column] = df[
            column
        ].fillna(
            median
        )


    df[
        columns["sample"]
    ] = df[
        columns["sample"]
    ].astype(int)


    df[
        columns["src"]
    ] = df[
        columns["src"]
    ].astype(int)


    df[
        columns["dst"]
    ] = df[
        columns["dst"]
    ].astype(int)


    return df

def build_samples_from_csv(
    df,
    columns,
    feature_columns
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONSTRUCTING NETWORK SAMPLES FROM CSV"
    )

    print(
        "=" * 100
    )


    samples = []


    grouped = df.groupby(
        columns["sample"],
        sort=True
    )


    for sample_id, group in grouped:

        group = group.reset_index(
            drop=True
        )


        features = (
            group[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        targets = group[

            [
                "__target_delay__",
                "__target_jitter__",
                "__target_packet_loss__"
            ]

        ].to_numpy(
            dtype=np.float32
        )


        src_values = group[
            columns["src"]
        ].to_numpy()


        dst_values = group[
            columns["dst"]
        ].to_numpy()


        flow_pairs = [

            (
                int(src),
                int(dst)
            )

            for src, dst
            in zip(
                src_values,
                dst_values
            )
        ]


        routes = []


        for row_index in range(
            len(group)
        ):

            src = int(
                src_values[
                    row_index
                ]
            )

            dst = int(
                dst_values[
                    row_index
                ]
            )

            if columns[
                "route"
            ] is not None:

                route = parse_route(

                    group.loc[
                        row_index,
                        columns["route"]
                    ]
                )

            else:

                route = []

            if len(route) < 2:

                route = [
                    src,
                    dst
                ]


            routes.append(
                route
            )


        if (
            len(features) == 0
            or
            len(targets) == 0
        ):

            continue


        samples.append(

            NetworkSample(

                sample_id=int(
                    sample_id
                ),

                flow_features=features,

                targets=targets,

                routes=routes,

                flow_pairs=flow_pairs
            )
        )


    if len(samples) < 5:

        raise RuntimeError(

            "Fewer than five valid network samples were constructed.\n"

            "A 60/20/20 sample-level experiment cannot be performed."
        )


    total_flows = sum(

        len(
            sample.flow_features
        )

        for sample in samples
    )


    print(
        "\nNetwork samples:",
        len(samples)
    )

    print(
        "Total flows:",
        f"{total_flows:,}"
    )

    print(
        "Input dimension:",
        samples[
            0
        ].flow_features.shape[
            1
        ]
    )

    print(
        "Output dimension:",
        samples[
            0
        ].targets.shape[
            1
        ]
    )


    return samples


def split_samples(
    samples,
    seed
):

    rng = np.random.RandomState(
        seed
    )


    indices = np.arange(
        len(samples)
    )


    rng.shuffle(
        indices
    )


    n = len(
        indices
    )


    n_train = int(
        TRAIN_RATIO * n
    )


    n_val = int(
        VAL_RATIO * n
    )


    train_indices = indices[
        :n_train
    ]


    val_indices = indices[
        n_train:
        n_train + n_val
    ]


    test_indices = indices[
        n_train + n_val:
    ]


    train_samples = [

        samples[index]

        for index
        in train_indices
    ]


    val_samples = [

        samples[index]

        for index
        in val_indices
    ]


    test_samples = [

        samples[index]

        for index
        in test_indices
    ]


    return (
        train_samples,
        val_samples,
        test_samples
    )


# =============================================================================
#  FEATURE SCALER
# =============================================================================

def fit_feature_scaler(
    train_samples
):

    X_train = np.vstack(

        [
            sample.flow_features

            for sample
            in train_samples
        ]
    )


    scaler = StandardScaler()


    scaler.fit(
        X_train
    )


    return scaler

def fit_target_scaler(
    train_samples
):

    y_train = np.vstack(

        [
            sample.targets

            for sample
            in train_samples
        ]
    )


    scaler = StandardScaler()


    scaler.fit(
        y_train
    )


    return scaler

def build_flow_adjacency(
    sample
):
    
    number_flows = len(
        sample.routes
    )


    route_link_sets = [

        set(
            route_to_links(
                route
            )
        )

        for route
        in sample.routes
    ]


    route_node_sets = [

        set(
            route
        )

        for route
        in sample.routes
    ]


    adjacency = np.eye(

        number_flows,

        dtype=np.float32
    )


    for i in range(
        number_flows
    ):

        for j in range(
            i + 1,
            number_flows
        ):

            shared_link = bool(

                route_link_sets[
                    i
                ]

                &

                route_link_sets[
                    j
                ]
            )


            shared_node = bool(

                route_node_sets[
                    i
                ]

                &

                route_node_sets[
                    j
                ]
            )


            if (
                shared_link
                or
                shared_node
            ):

                adjacency[
                    i,
                    j
                ] = 1.0


                adjacency[
                    j,
                    i
                ] = 1.0


    degree = adjacency.sum(
        axis=1
    )


    degree = np.maximum(
        degree,
        EPSILON
    )


    inverse_sqrt_degree = (

        1.0
        /
        np.sqrt(
            degree
        )
    )


    normalized = (

        inverse_sqrt_degree[
            :,
            None
        ]

        *

        adjacency

        *

        inverse_sqrt_degree[
            None,
            :
        ]
    )


    return torch.tensor(

        normalized,

        dtype=torch.float32,

        device=DEVICE
    )


# =============================================================================
#  ROUTENET FLOW-LINK GRAPH
# =============================================================================

def build_routenet_graph(
    sample
):

    flow_routes = [

        route_to_links(
            route
        )

        for route
        in sample.routes
    ]


    all_links = []


    for route_links in flow_routes:

        all_links.extend(
            route_links
        )


    unique_links = list(
        dict.fromkeys(
            all_links
        )
    )


    if len(
        unique_links
    ) == 0:

        unique_links = [
            (-1, -1)
        ]


    link_index = {

        link:
            index

        for index, link
        in enumerate(
            unique_links
        )
    }


    incidence = np.zeros(

        (
            len(
                flow_routes
            ),
            len(
                unique_links
            )
        ),

        dtype=np.float32
    )


    for flow_id, links in enumerate(
        flow_routes
    ):

        for link in links:

            if link in link_index:

                incidence[
                    flow_id,
                    link_index[
                        link
                    ]
                ] = 1.0


    # -------------------------------------------------------------------------
    # Link receives messages from flows
    # -------------------------------------------------------------------------

    link_degree = incidence.sum(
        axis=0
    )


    link_degree[
        link_degree == 0
    ] = 1.0


    link_from_flow = (

        incidence.T

        /

        link_degree[
            :,
            None
        ]
    )


    # -------------------------------------------------------------------------
    # Flow receives messages from links
    # -------------------------------------------------------------------------

    flow_degree = incidence.sum(
        axis=1
    )


    flow_degree[
        flow_degree == 0
    ] = 1.0


    flow_from_link = (

        incidence

        /

        flow_degree[
            :,
            None
        ]
    )


    return (

        torch.tensor(

            link_from_flow,

            dtype=torch.float32,

            device=DEVICE
        ),

        torch.tensor(

            flow_from_link,

            dtype=torch.float32,

            device=DEVICE
        )
    )


# =============================================================================
#  PREPARE SAMPLE
# =============================================================================

def prepare_sample(
    sample,
    feature_scaler,
    target_scaler
):

    X = feature_scaler.transform(
        sample.flow_features
    )


    y = target_scaler.transform(
        sample.targets
    )


    X = torch.tensor(

        X,

        dtype=torch.float32,

        device=DEVICE
    )


    y = torch.tensor(

        y,

        dtype=torch.float32,

        device=DEVICE
    )


    adjacency = build_flow_adjacency(
        sample
    )


    routenet_graph = build_routenet_graph(
        sample
    )


    return (
        X,
        y,
        adjacency,
        routenet_graph
    )


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


# =============================================================================
#  GNN
# =============================================================================

class GNN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()


        self.conv1 = nn.Linear(
            input_dim,
            hidden_dim
        )


        self.conv2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        self.output = nn.Linear(
            hidden_dim,
            output_dim
        )


        self.dropout = nn.Dropout(
            DROPOUT
        )


    def forward(
        self,
        x,
        adjacency
    ):

        # ---------------------------------------------------------------------
        # Layer 1
        # ---------------------------------------------------------------------

        h = torch.matmul(
            adjacency,
            x
        )


        h = self.conv1(
            h
        )


        h = torch.relu(
            h
        )


        h = self.dropout(
            h
        )


        # ---------------------------------------------------------------------
        # Layer 2
        # ---------------------------------------------------------------------

        h = torch.matmul(
            adjacency,
            h
        )


        h = self.conv2(
            h
        )


        h = torch.relu(
            h
        )


        h = self.dropout(
            h
        )


        return self.output(
            h
        )


# =============================================================================
# GRAPH TRANSFORMER LAYER
# =============================================================================

class GraphTransformerLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_heads
    ):

        super().__init__()


        self.attention = nn.MultiheadAttention(

            embed_dim=hidden_dim,

            num_heads=num_heads,

            dropout=DROPOUT,

            batch_first=True
        )


        self.norm1 = nn.LayerNorm(
            hidden_dim
        )


        self.norm2 = nn.LayerNorm(
            hidden_dim
        )


        self.feed_forward = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
        )


        self.dropout = nn.Dropout(
            DROPOUT
        )


    def forward(
        self,
        x,
        adjacency
    ):

        # ---------------------------------------------------------------------
        # True = blocked attention
        # ---------------------------------------------------------------------

        attention_mask = (
            adjacency <= 0
        )


        h = x.unsqueeze(
            0
        )


        attention_output, _ = self.attention(

            h,
            h,
            h,

            attn_mask=attention_mask,

            need_weights=False
        )


        h = self.norm1(

            h

            +

            self.dropout(
                attention_output
            )
        )


        feed_forward_output = (
            self.feed_forward(
                h
            )
        )


        h = self.norm2(

            h

            +

            self.dropout(
                feed_forward_output
            )
        )


        return h.squeeze(
            0
        )


# =============================================================================
#  GRAPH TRANSFORMER
# =============================================================================

class GraphTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()


        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim
        )


        self.layers = nn.ModuleList(

            [

                GraphTransformerLayer(
                    hidden_dim,
                    NUM_ATTENTION_HEADS
                )

                for _ in range(
                    NUM_TRANSFORMER_LAYERS
                )
            ]
        )


        self.output = nn.Linear(
            hidden_dim,
            output_dim
        )


    def forward(
        self,
        x,
        adjacency
    ):

        h = self.input_projection(
            x
        )


        for layer in self.layers:

            h = layer(
                h,
                adjacency
            )


        return self.output(
            h
        )


# =============================================================================
#  ROUTENET-F-STYLE MODEL
# =============================================================================

class RouteNetFAdapted(nn.Module):

    """
    RouteNet-F-style flow-link message passing model.

    IMPORTANT:
    This is an adapted implementation for this experiment,
    not the official RouteNet-Fermi source implementation.

    Flow states -> link states -> flow states
    using recurrent GRU updates.
    """


    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        iterations
    ):

        super().__init__()


        self.hidden_dim = hidden_dim

        self.iterations = iterations


        # ---------------------------------------------------------------------
        # Flow encoder
        # ---------------------------------------------------------------------

        self.flow_encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU()
        )

        self.initial_link_state = nn.Parameter(

            torch.zeros(
                1,
                hidden_dim
            )
        )

        self.flow_message = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        self.link_update = nn.GRUCell(
            hidden_dim,
            hidden_dim
        )


        self.link_message = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        self.flow_update = nn.GRUCell(
            hidden_dim,
            hidden_dim
        )


        self.dropout = nn.Dropout(
            DROPOUT
        )


        self.readout = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )


    def forward(
        self,
        x,
        route_graph
    ):

        (
            link_from_flow,
            flow_from_link
        ) = route_graph

        h_flow = self.flow_encoder(
            x
        )


        number_links = (
            link_from_flow.shape[
                0
            ]
        )


        h_link = (

            self.initial_link_state

            .expand(
                number_links,
                -1
            )

            .clone()
        )


        for _ in range(
            self.iterations
        ):


            flow_message = (
                self.flow_message(
                    h_flow
                )
            )


            aggregated_flows = torch.matmul(

                link_from_flow,

                flow_message
            )


            h_link = self.link_update(

                aggregated_flows,

                h_link
            )


            h_link = self.dropout(
                h_link
            )

            link_message = (
                self.link_message(
                    h_link
                )
            )


            aggregated_links = torch.matmul(

                flow_from_link,

                link_message
            )


            h_flow = self.flow_update(

                aggregated_links,

                h_flow
            )


            h_flow = self.dropout(
                h_flow
            )


        return self.readout(
            h_flow
        )


def create_model(
    model_name,
    input_dim
):

    output_dim = len(
        TARGET_NAMES
    )


    if model_name == "MLP":

        model = MLP(

            input_dim,

            HIDDEN_DIM,

            output_dim
        )


    elif model_name == "GNN":

        model = GNN(

            input_dim,

            HIDDEN_DIM,

            output_dim
        )


    elif model_name == "Graph Transformer":

        model = GraphTransformer(

            input_dim,

            HIDDEN_DIM,

            output_dim
        )


    elif model_name == "RouteNet-F":

        model = RouteNetFAdapted(

            input_dim,

            HIDDEN_DIM,

            output_dim,

            ROUTENET_ITERATIONS
        )


    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    return model.to(
        DEVICE
    )


def forward_model(
    model,
    model_name,
    X,
    adjacency,
    routenet_graph
):

    if model_name == "MLP":

        return model(
            X
        )


    if model_name in [

        "GNN",
        "Graph Transformer"

    ]:

        return model(
            X,
            adjacency
        )


    if model_name == "RouteNet-F":

        return model(
            X,
            routenet_graph
        )


    raise ValueError(
        model_name
    )


def train_one_epoch(
    model,
    model_name,
    samples,
    feature_scaler,
    target_scaler,
    optimizer,
    criterion
):

    model.train()


    total_loss = 0.0

    total_flows = 0


    sample_order = np.random.permutation(
        len(samples)
    )


    for sample_index in sample_order:

        sample = samples[
            sample_index
        ]


        (
            X,
            y,
            adjacency,
            routenet_graph

        ) = prepare_sample(

            sample,

            feature_scaler,

            target_scaler
        )


        optimizer.zero_grad()


        prediction = forward_model(

            model,

            model_name,

            X,

            adjacency,

            routenet_graph
        )


        loss = criterion(
            prediction,
            y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            5.0
        )


        optimizer.step()


        number_flows = len(
            sample.flow_features
        )


        total_loss += (
            loss.item()
            *
            number_flows
        )


        total_flows += (
            number_flows
        )


    return (

        total_loss

        /

        max(
            total_flows,
            1
        )
    )

def evaluate_loss(
    model,
    model_name,
    samples,
    feature_scaler,
    target_scaler,
    criterion
):

    model.eval()


    total_loss = 0.0

    total_flows = 0


    with torch.no_grad():

        for sample in samples:

            (
                X,
                y,
                adjacency,
                routenet_graph

            ) = prepare_sample(

                sample,

                feature_scaler,

                target_scaler
            )


            prediction = forward_model(

                model,

                model_name,

                X,

                adjacency,

                routenet_graph
            )


            loss = criterion(
                prediction,
                y
            )


            number_flows = len(
                sample.flow_features
            )


            total_loss += (
                loss.item()
                *
                number_flows
            )


            total_flows += (
                number_flows
            )


    return (

        total_loss

        /

        max(
            total_flows,
            1
        )
    )

def train_model(
    model,
    model_name,
    train_samples,
    val_samples,
    feature_scaler,
    target_scaler
):

    optimizer = optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    criterion = nn.MSELoss()
    best_val_loss = float(
        "inf"
    )

    best_state = None

    patience_counter = 0

    training_history = []

    validation_history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        train_loss = train_one_epoch(

            model,
            model_name,

            train_samples,

            feature_scaler,
            target_scaler,

            optimizer,
            criterion
        )


        val_loss = evaluate_loss(

            model,
            model_name,

            val_samples,

            feature_scaler,
            target_scaler,

            criterion
        )


        training_history.append(
            train_loss
        )


        validation_history.append(
            val_loss
        )


        if val_loss < best_val_loss:

            best_val_loss = val_loss


            best_state = copy.deepcopy(
                model.state_dict()
            )


            patience_counter = 0


        else:

            patience_counter += 1


        if (
            epoch == 1
            or
            epoch % 25 == 0
        ):

            print(

                f"Epoch "
                f"{epoch:03d}/{MAX_EPOCHS}"

                f" | Train MSE: "
                f"{train_loss:.6f}"

                f" | Val MSE: "
                f"{val_loss:.6f}"
            )


        if patience_counter >= PATIENCE:

            print(

                f"Early stopping at "
                f"epoch {epoch}."
            )

            break


    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    return (

        model,

        training_history,

        validation_history
    )

def predict_dataset(
    model,
    model_name,
    samples,
    feature_scaler,
    target_scaler
):

    model.eval()

    true_values = []

    predicted_values = []

    with torch.no_grad():

        for sample in samples:

            (
                X,
                _,
                adjacency,
                routenet_graph

            ) = prepare_sample(

                sample,

                feature_scaler,

                target_scaler
            )


            prediction_scaled = forward_model(

                model,

                model_name,

                X,

                adjacency,

                routenet_graph
            )


            prediction_scaled = (

                prediction_scaled

                .detach()

                .cpu()

                .numpy()
            )


            prediction = (

                target_scaler

                .inverse_transform(
                    prediction_scaled
                )
            )


            true_values.append(
                sample.targets
            )


            predicted_values.append(
                prediction
            )


    y_true = np.vstack(
        true_values
    )


    y_pred = np.vstack(
        predicted_values
    )


    return (
        y_true,
        y_pred
    )

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )


    y_pred = np.asarray(
        y_pred,
        dtype=float
    )


    valid = (

        np.isfinite(
            y_true
        )

        &

        np.isfinite(
            y_pred
        )
    )


    y_true = y_true[
        valid
    ]


    y_pred = y_pred[
        valid
    ]


    if len(
        y_true
    ) == 0:

        return {

            "MSE": np.nan,

            "RMSE": np.nan,

            "MAE": np.nan,

            "MAPE": np.nan,

            "R2": np.nan
        }

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    nonzero = (

        np.abs(
            y_true
        )

        > EPSILON
    )


    if np.any(
        nonzero
    ):

        mape = (

            np.mean(

                np.abs(

                    (

                        y_true[
                            nonzero
                        ]

                        -

                        y_pred[
                            nonzero
                        ]

                    )

                    /

                    np.abs(

                        y_true[
                            nonzero
                        ]
                    )
                )
            )

            *

            100.0
        )


    else:

        mape = np.nan

    if (

        len(
            y_true
        ) > 1

        and

        np.var(
            y_true
        ) > EPSILON

    ):

        r2 = r2_score(
            y_true,
            y_pred
        )


    else:

        r2 = np.nan


    return {

        "MSE":
            float(
                mse
            ),

        "RMSE":
            float(
                rmse
            ),

        "MAE":
            float(
                mae
            ),

        "MAPE":
            float(
                mape
            ),

        "R2":
            float(
                r2
            )
    }

def evaluate_predictions(
    y_true,
    y_pred
):

    results = {}


    for target_index, target_name in enumerate(
        TARGET_NAMES
    ):

        results[
            target_name
        ] = calculate_metrics(

            y_true[
                :,
                target_index
            ],

            y_pred[
                :,
                target_index
            ]
        )


    return results


def save_training_curve(
    train_history,
    val_history,
    model_name,
    seed,
    output_directory
):

    plt.figure(
        figsize=(
            8,
            5
        )
    )


    plt.plot(
        train_history,
        label="Training"
    )


    plt.plot(
        val_history,
        label="Validation"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Scaled MSE Loss"
    )


    plt.title(

        f"{model_name} "
        f"(Seed = {seed})"
    )


    plt.legend()


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    filename = os.path.join(

        output_directory,

        (
            f"Training_"
            f"{model_name.replace(' ', '_')}_"
            f"Seed_{seed}.png"
        )
    )


    plt.savefig(

        filename,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


def create_metric_plots(
    summary_df,
    output_directory
):

    for target in TARGET_NAMES:

        target_data = summary_df[

            summary_df[
                "Target"
            ] == target
        ]


        for metric in METRIC_NAMES:

            means = []

            stds = []


            for model in MODEL_NAMES:

                row = target_data[

                    target_data[
                        "Model"
                    ] == model
                ]


                if len(
                    row
                ) == 0:

                    means.append(
                        np.nan
                    )

                    stds.append(
                        np.nan
                    )


                else:

                    means.append(

                        row[
                            f"{metric}_Mean"
                        ].iloc[
                            0
                        ]
                    )


                    stds.append(

                        row[
                            f"{metric}_STD"
                        ].iloc[
                            0
                        ]
                    )


            x = np.arange(
                len(
                    MODEL_NAMES
                )
            )


            plt.figure(
                figsize=(
                    10,
                    6
                )
            )


            plt.bar(

                x,

                means,

                yerr=stds,

                capsize=5
            )


            plt.xticks(

                x,

                MODEL_NAMES,

                rotation=15
            )


            plt.xlabel(
                "Model"
            )


            if metric == "MAPE":

                plt.ylabel(
                    "MAPE (%)"
                )

            elif metric == "R2":

                plt.ylabel(
                    "R-squared"
                )

            else:

                plt.ylabel(
                    metric
                )


            plt.title(

                f"{target}: "
                f"{metric} Comparison"
            )


            plt.grid(
                axis="y",
                alpha=0.3
            )


            plt.tight_layout()


            filename = os.path.join(

                output_directory,

                (
                    f"{target.replace(' ', '_')}_"
                    f"{metric}_comparison.png"
                )
            )


            plt.savefig(

                filename,

                dpi=300,

                bbox_inches="tight"
            )


            plt.close()


def create_prediction_plots(
    predictions_df,
    output_directory
):

    for model_name in MODEL_NAMES:

        model_data = predictions_df[

            predictions_df[
                "Model"
            ] == model_name
        ]


        for target_name in TARGET_NAMES:

            true_column = (
                f"{target_name}_True"
            )


            predicted_column = (
                f"{target_name}_Predicted"
            )


            if (
                true_column
                not in model_data.columns
                or
                predicted_column
                not in model_data.columns
            ):

                continue


            true_values = model_data[
                true_column
            ].to_numpy()


            predicted_values = model_data[
                predicted_column
            ].to_numpy()


            valid = (

                np.isfinite(
                    true_values
                )

                &

                np.isfinite(
                    predicted_values
                )
            )


            true_values = true_values[
                valid
            ]


            predicted_values = predicted_values[
                valid
            ]


            if len(
                true_values
            ) == 0:

                continue


            minimum = min(

                np.min(
                    true_values
                ),

                np.min(
                    predicted_values
                )
            )


            maximum = max(

                np.max(
                    true_values
                ),

                np.max(
                    predicted_values
                )
            )


            plt.figure(
                figsize=(
                    7,
                    7
                )
            )


            plt.scatter(

                true_values,

                predicted_values,

                alpha=0.4,

                s=18
            )


            plt.plot(

                [
                    minimum,
                    maximum
                ],

                [
                    minimum,
                    maximum
                ],

                linestyle="--"
            )


            plt.xlabel(
                f"True {target_name}"
            )


            plt.ylabel(
                f"Predicted {target_name}"
            )


            plt.title(

                f"{model_name}: "
                f"{target_name}"
            )


            plt.grid(
                alpha=0.3
            )


            plt.tight_layout()


            filename = os.path.join(

                output_directory,

                (
                    f"Prediction_"
                    f"{model_name.replace(' ', '_')}_"
                    f"{target_name.replace(' ', '_')}.png"
                )
            )


            plt.savefig(

                filename,

                dpi=300,

                bbox_inches="tight"
            )


            plt.close()

def create_combined_metric_figure(
    summary_df,
    output_directory
):

    for target in TARGET_NAMES:

        subset = summary_df[

            summary_df[
                "Target"
            ] == target
        ].copy()


        plot_data = []


        for model in MODEL_NAMES:

            row = subset[

                subset[
                    "Model"
                ] == model
            ]


            if len(
                row
            ) == 0:

                continue


            record = {
                "Model":
                    model
            }


            for metric in METRIC_NAMES:

                record[
                    metric
                ] = row[
                    f"{metric}_Mean"
                ].iloc[
                    0
                ]


            plot_data.append(
                record
            )


        plot_df = pd.DataFrame(
            plot_data
        )


        if len(
            plot_df
        ) == 0:

            continue
        
        normalized = plot_df.copy()


        for metric in METRIC_NAMES:

            values = plot_df[
                metric
            ].to_numpy(
                dtype=float
            )


            finite = np.isfinite(
                values
            )


            if not np.any(
                finite
            ):

                normalized[
                    metric
                ] = 0.0

                continue


            minimum = np.nanmin(
                values
            )


            maximum = np.nanmax(
                values
            )


            if abs(
                maximum - minimum
            ) < EPSILON:

                normalized[
                    metric
                ] = 1.0


            else:

                normalized[
                    metric
                ] = (

                    values - minimum

                ) / (

                    maximum - minimum
                )


        x = np.arange(
            len(
                MODEL_NAMES
            )
        )


        width = 0.15


        plt.figure(
            figsize=(
                13,
                7
            )
        )


        for metric_index, metric in enumerate(
            METRIC_NAMES
        ):

            values = []


            for model in MODEL_NAMES:

                row = normalized[

                    normalized[
                        "Model"
                    ] == model
                ]


                if len(
                    row
                ) == 0:

                    values.append(
                        np.nan
                    )

                else:

                    values.append(

                        row[
                            metric
                        ].iloc[
                            0
                        ]
                    )


            offset = (

                metric_index

                -

                (
                    len(
                        METRIC_NAMES
                    ) - 1
                ) / 2

            ) * width


            plt.bar(

                x + offset,

                values,

                width=width,

                label=metric
            )


        plt.xticks(
            x,
            MODEL_NAMES
        )


        plt.xlabel(
            "Model"
        )


        plt.ylabel(
            "Normalized Metric Value"
        )


        plt.title(

            f"{target}: "
            f"Overall Model-Metric Visualization"
        )


        plt.legend()


        plt.grid(
            axis="y",
            alpha=0.3
        )


        plt.tight_layout()


        filename = os.path.join(

            output_directory,

            (
                f"{target.replace(' ', '_')}_"
                f"all_metrics.png"
            )
        )


        plt.savefig(

            filename,

            dpi=300,

            bbox_inches="tight"
        )


        plt.close()


if __name__ == "__main__":

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NETWORK PERFORMANCE PREDICTION FROM COMBINED CSV"
    )

    print(
        "MLP | GNN | GRAPH TRANSFORMER | ROUTENET-F"
    )

    print(
        "=" * 100
    )
    
    print(
        "\nDevice:",
        DEVICE
    )


    print(
        "Hidden dimension:",
        HIDDEN_DIM
    )


    print(
        "Learning rate:",
        LEARNING_RATE
    )


    print(
        "Weight decay:",
        WEIGHT_DECAY
    )


    print(
        "Dropout:",
        DROPOUT
    )


    print(
        "Maximum epochs:",
        MAX_EPOCHS
    )


    print(
        "Early-stopping patience:",
        PATIENCE
    )


    print(
        "Train/Validation/Test:",
        "60% / 20% / 20%"
    )


    print(
        "Random seeds:",
        SEEDS
    )


    CSV_FILE = select_csv_file()


    print(
        "\nSelected CSV:"
    )


    print(
        CSV_FILE
    )


    OUTPUT_DIRECTORY = os.path.join(

        os.path.dirname(
            CSV_FILE
        ),

        "prediction_results"
    )


    os.makedirs(

        OUTPUT_DIRECTORY,

        exist_ok=True
    )


    print(
        "\nResults directory:"
    )


    print(
        OUTPUT_DIRECTORY
    )


    df = load_csv(
        CSV_FILE
    )


    columns = identify_columns(
        df
    )

    feature_columns = select_feature_columns(

        df,

        columns
    )

    df = prepare_dataframe(

        df,

        columns,

        feature_columns
    )

    samples = build_samples_from_csv(

        df,

        columns,

        feature_columns
    )


    input_dim = (

        samples[
            0
        ]

        .flow_features

        .shape[
            1
        ]
    )


    print(
        "\nFinal input feature dimension:",
        input_dim
    )

    if columns[
        "route"
    ] is None:

        print(
            "\nWARNING:"
        )

        print(
            "No explicit route/path column was detected."
        )

        print(
            "The graph models will therefore use source-destination "
            "connectivity as the routing fallback."
        )

        print(
            "For a stronger RouteNet-F experiment, the combined CSV "
            "should contain the complete route for every flow."
        )


    else:

        print(
            "\nRouting information detected:"
        )

        print(
            columns[
                "route"
            ]
        )

    metric_rows = []

    prediction_frames = []


    for run_number, seed in enumerate(

        SEEDS,

        start=1
    ):

        print(
            "\n\n"
            + "=" * 100
        )


        print(

            f"RUN "
            f"{run_number}/{len(SEEDS)}"

            f" | SEED = {seed}"
        )


        print(
            "=" * 100
        )


        set_seed(
            seed
        )

        (
            train_samples,
            val_samples,
            test_samples

        ) = split_samples(

            samples,

            seed
        )


        print(
            "\nNetwork sample split:"
        )


        print(
            "Training:",
            len(
                train_samples
            )
        )


        print(
            "Validation:",
            len(
                val_samples
            )
        )


        print(
            "Testing:",
            len(
                test_samples
            )
        )


        train_flows = sum(

            len(
                sample.flow_features
            )

            for sample
            in train_samples
        )


        val_flows = sum(

            len(
                sample.flow_features
            )

            for sample
            in val_samples
        )


        test_flows = sum(

            len(
                sample.flow_features
            )

            for sample
            in test_samples
        )


        print(
            "\nFlow split:"
        )


        print(
            "Training flows:",
            f"{train_flows:,}"
        )


        print(
            "Validation flows:",
            f"{val_flows:,}"
        )


        print(
            "Testing flows:",
            f"{test_flows:,}"
        )


        feature_scaler = fit_feature_scaler(
            train_samples
        )


        target_scaler = fit_target_scaler(
            train_samples
        )

        for model_name in MODEL_NAMES:

            print(
                "\n"
                + "-" * 100
            )


            print(
                f"MODEL: {model_name}"
            )


            print(
                "-" * 100
            )


            set_seed(
                seed
            )


            model = create_model(

                model_name,

                input_dim
            )


            parameter_count = sum(

                parameter.numel()

                for parameter
                in model.parameters()

                if parameter.requires_grad
            )


            print(
                "Trainable parameters:",
                f"{parameter_count:,}"
            )

            (
                model,
                train_history,
                val_history

            ) = train_model(

                model,

                model_name,

                train_samples,

                val_samples,

                feature_scaler,

                target_scaler
            )

            save_training_curve(

                train_history,

                val_history,

                model_name,

                seed,

                OUTPUT_DIRECTORY
            )

            (
                y_true,
                y_pred

            ) = predict_dataset(

                model,

                model_name,

                test_samples,

                feature_scaler,

                target_scaler
            )

            results = evaluate_predictions(

                y_true,

                y_pred
            )


            print(
                "\nTest performance:"
            )


            for target_name in TARGET_NAMES:

                metric = results[
                    target_name
                ]


                print(

                    f"\n{target_name}"

                    f"\n  MSE  = "
                    f"{metric['MSE']:.8f}"

                    f"\n  RMSE = "
                    f"{metric['RMSE']:.8f}"

                    f"\n  MAE  = "
                    f"{metric['MAE']:.8f}"

                    f"\n  MAPE = "
                    f"{metric['MAPE']:.4f}%"

                    f"\n  R2   = "
                    f"{metric['R2']:.6f}"
                )


                metric_rows.append(

                    {
                        "Run":
                            run_number,

                        "Seed":
                            seed,

                        "Model":
                            model_name,

                        "Target":
                            target_name,

                        "MSE":
                            metric[
                                "MSE"
                            ],

                        "RMSE":
                            metric[
                                "RMSE"
                            ],

                        "MAE":
                            metric[
                                "MAE"
                            ],

                        "MAPE":
                            metric[
                                "MAPE"
                            ],

                        "R2":
                            metric[
                                "R2"
                            ]
                    }
                )

            prediction_frame = pd.DataFrame(

                {
                    "Run":
                        np.repeat(
                            run_number,
                            len(
                                y_true
                            )
                        ),

                    "Seed":
                        np.repeat(
                            seed,
                            len(
                                y_true
                            )
                        ),

                    "Model":
                        np.repeat(
                            model_name,
                            len(
                                y_true
                            )
                        ),

                    "Delay_True":
                        y_true[
                            :,
                            0
                        ],

                    "Delay_Predicted":
                        y_pred[
                            :,
                            0
                        ],

                    "Jitter_True":
                        y_true[
                            :,
                            1
                        ],

                    "Jitter_Predicted":
                        y_pred[
                            :,
                            1
                        ],

                    "Packet Loss_True":
                        y_true[
                            :,
                            2
                        ],

                    "Packet Loss_Predicted":
                        y_pred[
                            :,
                            2
                        ]
                }
            )


            prediction_frames.append(
                prediction_frame
            )

            del model


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


    results_df = pd.DataFrame(
        metric_rows
    )


    individual_file = os.path.join(

        OUTPUT_DIRECTORY,

        "individual_run_results.csv"
    )


    results_df.to_csv(

        individual_file,

        index=False
    )


    summary_rows = []


    for model_name in MODEL_NAMES:

        for target_name in TARGET_NAMES:

            subset = results_df[

                (
                    results_df[
                        "Model"
                    ]

                    ==

                    model_name
                )

                &

                (
                    results_df[
                        "Target"
                    ]

                    ==

                    target_name
                )
            ]


            summary = {

                "Model":
                    model_name,

                "Target":
                    target_name
            }


            for metric in METRIC_NAMES:

                values = subset[
                    metric
                ].to_numpy(
                    dtype=float
                )


                summary[
                    f"{metric}_Mean"
                ] = np.nanmean(
                    values
                )


                summary[
                    f"{metric}_STD"
                ] = np.nanstd(
                    values,
                    ddof=1
                )


            summary_rows.append(
                summary
            )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_file = os.path.join(

        OUTPUT_DIRECTORY,

        "mean_std_results.csv"
    )


    summary_df.to_csv(

        summary_file,

        index=False
    )

    paper_rows = []


    for _, row in summary_df.iterrows():

        paper_rows.append(

            {
                "Model":
                    row[
                        "Model"
                    ],

                "Target":
                    row[
                        "Target"
                    ],

                "MSE":
                    (
                        f"{row['MSE_Mean']:.6f}"
                        f" ± "
                        f"{row['MSE_STD']:.6f}"
                    ),

                "RMSE":
                    (
                        f"{row['RMSE_Mean']:.6f}"
                        f" ± "
                        f"{row['RMSE_STD']:.6f}"
                    ),

                "MAE":
                    (
                        f"{row['MAE_Mean']:.6f}"
                        f" ± "
                        f"{row['MAE_STD']:.6f}"
                    ),

                "MAPE (%)":
                    (
                        f"{row['MAPE_Mean']:.4f}"
                        f" ± "
                        f"{row['MAPE_STD']:.4f}"
                    ),

                "R-squared":
                    (
                        f"{row['R2_Mean']:.6f}"
                        f" ± "
                        f"{row['R2_STD']:.6f}"
                    )
            }
        )


    paper_df = pd.DataFrame(
        paper_rows
    )


    paper_file = os.path.join(

        OUTPUT_DIRECTORY,

        "paper_ready_results.csv"
    )


    paper_df.to_csv(

        paper_file,

        index=False
    )

    predictions_df = pd.concat(

        prediction_frames,

        ignore_index=True
    )


    predictions_file = os.path.join(

        OUTPUT_DIRECTORY,

        "all_test_predictions.csv"
    )


    predictions_df.to_csv(

        predictions_file,

        index=False
    )

    feature_file = os.path.join(

        OUTPUT_DIRECTORY,

        "input_features.csv"
    )


    pd.DataFrame(

        {
            "Feature":
                feature_columns
        }

    ).to_csv(

        feature_file,

        index=False
    )

    print(
        "\n\n"
        + "=" * 150
    )


    print(
        "FINAL RESULTS — FIVE RUNS: MEAN ± STANDARD DEVIATION"
    )


    print(
        "=" * 150
    )


    print(

        paper_df.to_string(
            index=False
        )
    )

    print(
        "\nCreating metric comparison figures..."
    )


    create_metric_plots(

        summary_df,

        OUTPUT_DIRECTORY
    )


    print(
        "Creating combined metric figures..."
    )


    create_combined_metric_figure(

        summary_df,

        OUTPUT_DIRECTORY
    )


    print(
        "Creating true-vs-predicted figures..."
    )


    create_prediction_plots(

        predictions_df,

        OUTPUT_DIRECTORY
    )

    print(
        "\n"
        + "=" * 100
    )


    print(
        "EXPERIMENT COMPLETED SUCCESSFULLY"
    )


    print(
        "=" * 100
    )


    print(
        "\nResults directory:"
    )


    print(
        OUTPUT_DIRECTORY
    )


    print(
        "\nGenerated CSV files:"
    )


    print(
        "  1. individual_run_results.csv"
    )


    print(
        "  2. mean_std_results.csv"
    )


    print(
        "  3. paper_ready_results.csv"
    )


    print(
        "  4. all_test_predictions.csv"
    )


    print(
        "  5. input_features.csv"
    )


    print(
        "\nGenerated figures:"
    )


    print(
        "  - Training/validation curves"
    )


    print(
        "  - MSE comparison plots"
    )


    print(
        "  - RMSE comparison plots"
    )


    print(
        "  - MAE comparison plots"
    )


    print(
        "  - MAPE comparison plots"
    )


    print(
        "  - R-squared comparison plots"
    )


    print(
        "  - Combined model/metric plots"
    )


    print(
        "  - True-vs-predicted plots"
    )