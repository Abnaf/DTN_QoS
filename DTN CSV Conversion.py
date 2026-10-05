import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

DATASET_DIR = (
    r"C:\Users\user\DTN Challenge 2023\gnnet-ch23-dataset-mb"
)

TARGET_ARCHIVE = "results_mb_0075-0099.tar.gz"

OUTPUT_CSV = os.path.join(
    DATASET_DIR,
    "network_performance_0075-0099.csv"
)


# ============================================================
# 2. CHECK DATASET DIRECTORY
# ============================================================

print("=" * 80)
print("GNN CHALLENGE 2023 -> CSV CONVERTER")
print("=" * 80)

print("\nDataset directory:")
print(DATASET_DIR)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset directory does not exist:\n{DATASET_DIR}"
    )

print("\nDataset directory found successfully.")


# ============================================================
# 3. SET PYTHON PATH
# ============================================================

os.chdir(DATASET_DIR)

if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)


# ============================================================
# 4. IMPORT OFFICIAL DATANET API
# ============================================================

try:
    from datanetAPI import DatanetAPI

    print("\nDatanetAPI imported successfully.")

except ImportError as e:

    print("\nERROR: Could not import datanetAPI.py")
    print(e)

    raise


# ============================================================
# 5. INITIALIZE DATASET
# ============================================================

print("\nInitializing dataset...")

dataset = DatanetAPI(
    DATASET_DIR,
    shuffle=False
)

print("Dataset initialized successfully.")


# ============================================================
# 6. GET AVAILABLE ARCHIVES
# ============================================================

available_files = dataset.get_available_files()

print("\nNumber of archives detected:")
print(len(available_files))

if len(available_files) == 0:
    raise RuntimeError(
        "DatanetAPI did not detect any dataset archives."
    )


# ============================================================
# 7. FIND TARGET ARCHIVE
# ============================================================

selected_files = [
    item
    for item in available_files
    if item[1] == TARGET_ARCHIVE
]

print("\n" + "=" * 80)
print("TARGET ARCHIVE")
print("=" * 80)

if len(selected_files) == 0:

    raise FileNotFoundError(
        f"Target archive was not found:\n{TARGET_ARCHIVE}"
    )

print("\nSelected archive:")
print(selected_files[0])


# ============================================================
# 8. SELECT ARCHIVE
# ============================================================
#
# IMPORTANT:
#
# The supplied DatanetAPI contains a bug inside
# set_files_to_process().
#
# Therefore, instead of:
#
# dataset.set_files_to_process(selected_files)
#
# we directly populate the internal selected-file list.
#
# ============================================================

dataset._selected_tuple_files = selected_files.copy()

print("\nArchive selected successfully.")


# ============================================================
# 9. STORAGE
# ============================================================

rows = []

sample_counter = 0

error_counter = 0


# ============================================================
# 10. HELPER FUNCTION
# ============================================================

def safe_float(value):
    """
    Convert a value to float safely.
    Returns NaN when conversion is impossible.
    """

    try:

        if value is None:
            return np.nan

        return float(value)

    except (TypeError, ValueError):
        return np.nan


# ============================================================
# 11. ITERATE THROUGH DATASET SAMPLES
# ============================================================

print("\n" + "=" * 80)
print("PROCESSING SAMPLES")
print("=" * 80)

for sample in dataset:

    sample_counter += 1

    # ========================================================
    # SAMPLE IDENTIFICATION
    # ========================================================

    try:
        sample_identifier = sample.get_sample_id()
    except Exception:
        sample_identifier = ("unknown", sample_counter)

    try:

        if (
            isinstance(sample_identifier, tuple)
            and len(sample_identifier) >= 2
        ):

            dataset_file = sample_identifier[0]
            sample_id = sample_identifier[1]

        else:

            dataset_file = TARGET_ARCHIVE
            sample_id = sample_identifier

    except Exception:

        dataset_file = TARGET_ARCHIVE
        sample_id = sample_counter


    # ========================================================
    # NETWORK SIZE
    # ========================================================

    try:
        network_size = int(
            sample.get_network_size()
        )

    except Exception as e:

        print(
            f"\nCould not obtain network size "
            f"for sample {sample_id}: {e}"
        )

        continue


    # ========================================================
    # SAMPLE-LEVEL INFORMATION
    # ========================================================

    try:
        capture_time = sample.get_capture_time()
    except Exception:
        capture_time = np.nan

    try:
        global_packets = sample.get_global_packets()
    except Exception:
        global_packets = np.nan

    try:
        global_losses = sample.get_global_losses()
    except Exception:
        global_losses = np.nan

    try:
        global_delay = sample.get_global_delay()
    except Exception:
        global_delay = np.nan

    try:
        max_link_load = sample.get_max_link_load()
    except Exception:
        max_link_load = np.nan


    # ========================================================
    # MATRICES
    # ========================================================

    try:
        performance_matrix = (
            sample.get_performance_matrix()
        )
    except Exception:
        performance_matrix = None

    try:
        traffic_matrix = (
            sample.get_traffic_matrix()
        )
    except Exception:
        traffic_matrix = None

    try:
        routing_matrix = (
            sample.get_routing_matrix()
        )
    except Exception:
        routing_matrix = None

    try:
        physical_path_matrix = (
            sample.get_physical_path_matrix()
        )
    except Exception:
        physical_path_matrix = None


    # ========================================================
    # PROCESS SOURCE-DESTINATION PAIRS
    # ========================================================

    sample_rows_before = len(rows)

    for src in range(network_size):

        for dst in range(network_size):

            # ------------------------------------------------
            # Ignore self-communication
            # ------------------------------------------------

            if src == dst:
                continue

            try:

                # ============================================
                # PERFORMANCE INFORMATION
                # ============================================

                perf_agg = {}

                if performance_matrix is not None:

                    try:
                        perf = performance_matrix[src, dst]
                    except Exception:
                        perf = None

                    if isinstance(perf, dict):

                        if "AggInfo" in perf:

                            if isinstance(
                                perf["AggInfo"],
                                dict
                            ):

                                perf_agg = perf["AggInfo"]

                        else:

                            # Fallback in case performance
                            # dictionary is already aggregate
                            perf_agg = perf


                # ============================================
                # TRAFFIC INFORMATION
                # ============================================

                traffic_agg = {}

                if traffic_matrix is not None:

                    try:
                        traffic = traffic_matrix[src, dst]
                    except Exception:
                        traffic = None

                    if isinstance(traffic, dict):

                        if "AggInfo" in traffic:

                            if isinstance(
                                traffic["AggInfo"],
                                dict
                            ):

                                traffic_agg = (
                                    traffic["AggInfo"]
                                )

                        else:

                            traffic_agg = traffic


                # ============================================
                # ROUTING INFORMATION
                # ============================================

                route_list = []

                if routing_matrix is not None:

                    try:

                        route = routing_matrix[src, dst]

                        if route is not None:

                            route_list = list(route)

                    except Exception:

                        route_list = []


                # Number of nodes in route
                path_nodes = len(route_list)

                # Number of graph hops
                num_hops = max(
                    path_nodes - 1,
                    0
                )

                # Readable route
                route_string = "->".join(
                    str(node)
                    for node in route_list
                )


                # ============================================
                # PHYSICAL PATH INFORMATION
                # ============================================

                physical_path_list = []

                if physical_path_matrix is not None:

                    try:

                        physical_path = (
                            physical_path_matrix[src, dst]
                        )

                        if physical_path is not None:

                            physical_path_list = list(
                                physical_path
                            )

                    except Exception:

                        physical_path_list = []


                physical_path_length = len(
                    physical_path_list
                )

                physical_path_string = "->".join(
                    str(node)
                    for node in physical_path_list
                )


                # ============================================
                # PATH BANDWIDTH
                # ============================================

                link_bandwidths = []

                if len(route_list) >= 2:

                    for u, v in zip(
                        route_list[:-1],
                        route_list[1:]
                    ):

                        try:

                            bandwidth = (
                                sample
                                .get_srcdst_link_bandwidth(
                                    u,
                                    v
                                )
                            )

                            bandwidth = safe_float(
                                bandwidth
                            )

                            if np.isfinite(bandwidth):

                                link_bandwidths.append(
                                    bandwidth
                                )

                        except Exception:
                            pass


                if len(link_bandwidths) > 0:

                    min_path_bandwidth = float(
                        np.min(link_bandwidths)
                    )

                    max_path_bandwidth = float(
                        np.max(link_bandwidths)
                    )

                    mean_path_bandwidth = float(
                        np.mean(link_bandwidths)
                    )

                else:

                    min_path_bandwidth = np.nan

                    max_path_bandwidth = np.nan

                    mean_path_bandwidth = np.nan


                # ============================================
                # OPTIONAL PACKET-LOSS RATE
                # ============================================
                #
                # We retain the original PktsDrop field.
                #
                # A normalized packet-loss ratio is also
                # calculated when TotalPktsGen is available.
                #
                # ============================================

                packets_drop = safe_float(
                    perf_agg.get(
                        "PktsDrop",
                        np.nan
                    )
                )

                total_packets_generated = safe_float(
                    traffic_agg.get(
                        "TotalPktsGen",
                        np.nan
                    )
                )

                if (
                    np.isfinite(packets_drop)
                    and
                    np.isfinite(total_packets_generated)
                    and
                    total_packets_generated > 0
                ):

                    packet_loss_ratio = (
                        packets_drop
                        /
                        total_packets_generated
                    )

                else:

                    packet_loss_ratio = np.nan


                # ============================================
                # CREATE ROW
                # ============================================

                row = {

                    # ----------------------------------------
                    # IDENTIFIERS
                    # ----------------------------------------

                    "sample_id":
                        sample_id,

                    "archive":
                        os.path.basename(
                            str(dataset_file)
                        ),

                    "src":
                        src,

                    "dst":
                        dst,


                    # ----------------------------------------
                    # NETWORK-LEVEL FEATURES
                    # ----------------------------------------

                    "network_size":
                        network_size,

                    "capture_time":
                        safe_float(
                            capture_time
                        ),

                    "global_packets":
                        safe_float(
                            global_packets
                        ),

                    "global_losses":
                        safe_float(
                            global_losses
                        ),

                    "global_delay":
                        safe_float(
                            global_delay
                        ),

                    "max_link_load":
                        safe_float(
                            max_link_load
                        ),


                    # ----------------------------------------
                    # TRAFFIC FEATURES
                    # ----------------------------------------

                    "AvgBw":
                        safe_float(
                            traffic_agg.get(
                                "AvgBw",
                                np.nan
                            )
                        ),

                    "PktsGen":
                        safe_float(
                            traffic_agg.get(
                                "PktsGen",
                                np.nan
                            )
                        ),

                    "TotalPktsGen":
                        total_packets_generated,

                    "AvgPktSize":
                        safe_float(
                            traffic_agg.get(
                                "AvgPktSize",
                                np.nan
                            )
                        ),

                    "p10PktSize":
                        safe_float(
                            traffic_agg.get(
                                "p10PktSize",
                                np.nan
                            )
                        ),

                    "p20PktSize":
                        safe_float(
                            traffic_agg.get(
                                "p20PktSize",
                                np.nan
                            )
                        ),

                    "p50PktSize":
                        safe_float(
                            traffic_agg.get(
                                "p50PktSize",
                                np.nan
                            )
                        ),

                    "p80PktSize":
                        safe_float(
                            traffic_agg.get(
                                "p80PktSize",
                                np.nan
                            )
                        ),

                    "p90PktSize":
                        safe_float(
                            traffic_agg.get(
                                "p90PktSize",
                                np.nan
                            )
                        ),

                    "VarPktSize":
                        safe_float(
                            traffic_agg.get(
                                "VarPktSize",
                                np.nan
                            )
                        ),


                    # ----------------------------------------
                    # ROUTING / TOPOLOGY FEATURES
                    # ----------------------------------------

                    "num_hops":
                        num_hops,

                    "path_nodes":
                        path_nodes,

                    "route":
                        route_string,

                    "physical_path_length":
                        physical_path_length,

                    "physical_path":
                        physical_path_string,

                    "min_path_bandwidth":
                        min_path_bandwidth,

                    "max_path_bandwidth":
                        max_path_bandwidth,

                    "mean_path_bandwidth":
                        mean_path_bandwidth,


                    # ----------------------------------------
                    # PERFORMANCE TARGETS
                    # ----------------------------------------

                    "PktsDrop":
                        packets_drop,

                    "packet_loss_ratio":
                        packet_loss_ratio,

                    "AvgDelay":
                        safe_float(
                            perf_agg.get(
                                "AvgDelay",
                                np.nan
                            )
                        ),

                    "AvgLnDelay":
                        safe_float(
                            perf_agg.get(
                                "AvgLnDelay",
                                np.nan
                            )
                        ),

                    "p10Delay":
                        safe_float(
                            perf_agg.get(
                                "p10Delay",
                                np.nan
                            )
                        ),

                    "p20Delay":
                        safe_float(
                            perf_agg.get(
                                "p20Delay",
                                np.nan
                            )
                        ),

                    "p50Delay":
                        safe_float(
                            perf_agg.get(
                                "p50Delay",
                                np.nan
                            )
                        ),

                    "p80Delay":
                        safe_float(
                            perf_agg.get(
                                "p80Delay",
                                np.nan
                            )
                        ),

                    "p90Delay":
                        safe_float(
                            perf_agg.get(
                                "p90Delay",
                                np.nan
                            )
                        ),

                    "Jitter":
                        safe_float(
                            perf_agg.get(
                                "Jitter",
                                np.nan
                            )
                        )
                }


                # ============================================
                # ADD ROW
                # ============================================

                rows.append(row)


            except Exception as e:

                error_counter += 1

                # Print only first few errors so Spyder
                # output is not flooded.
                if error_counter <= 10:

                    print(
                        "\nWarning:"
                        f" sample={sample_id},"
                        f" src={src},"
                        f" dst={dst}"
                    )

                    print(
                        "Reason:",
                        str(e)
                    )


    # ========================================================
    # SAMPLE PROGRESS
    # ========================================================

    rows_added = (
        len(rows)
        -
        sample_rows_before
    )

    print(
        f"Sample {sample_counter:04d} | "
        f"ID={sample_id} | "
        f"nodes={network_size} | "
        f"rows added={rows_added} | "
        f"total rows={len(rows)}"
    )


# ============================================================
# 12. VERIFY DATA WAS EXTRACTED
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTION COMPLETED")
print("=" * 80)

print("\nSamples processed:")
print(sample_counter)

print("\nSource-destination errors:")
print(error_counter)

print("\nRows extracted:")
print(len(rows))


if len(rows) == 0:

    raise RuntimeError(
        "No rows were extracted from the archive."
    )


# ============================================================
# 13. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("DATAFRAME CREATED")
print("=" * 80)

print("\nOriginal dataframe shape:")
print(df.shape)


# ============================================================
# 14. DISPLAY COLUMNS
# ============================================================

print("\nColumns:")

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i:02d}. {column}"
    )


# ============================================================
# 15. CHECK PERFORMANCE TARGETS
# ============================================================

primary_targets = [
    "AvgDelay",
    "Jitter",
    "PktsDrop"
]

print("\n" + "=" * 80)
print("TARGET AVAILABILITY")
print("=" * 80)

for target in primary_targets:

    available = (
        df[target]
        .notna()
        .sum()
    )

    print(
        f"{target:20s}: "
        f"{available:,} non-missing values"
    )


# ============================================================
# 16. REMOVE ROWS WHERE ALL PRIMARY TARGETS ARE MISSING
# ============================================================

before_filter = len(df)

df = df.dropna(
    subset=primary_targets,
    how="all"
).copy()

after_filter = len(df)

print("\nRows before target filtering:")
print(before_filter)

print("\nRows after target filtering:")
print(after_filter)

print("\nRows removed:")
print(
    before_filter
    -
    after_filter
)


# ============================================================
# 17. REMOVE INFINITE NUMERIC VALUES
# ============================================================

numeric_columns = (
    df.select_dtypes(
        include=[np.number]
    ).columns
)

df[numeric_columns] = (
    df[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# 18. TARGET STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("TARGET STATISTICS")
print("=" * 80)

targets_to_check = [
    "AvgDelay",
    "Jitter",
    "PktsDrop",
    "packet_loss_ratio"
]

for target in targets_to_check:

    print(
        f"\n----- {target} -----"
    )

    print(
        df[target].describe()
    )


# ============================================================
# 19. TRAFFIC FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("TRAFFIC FEATURE STATISTICS")
print("=" * 80)

traffic_features = [
    "AvgBw",
    "PktsGen",
    "TotalPktsGen",
    "AvgPktSize",
    "VarPktSize"
]

existing_traffic_features = [
    column
    for column in traffic_features
    if column in df.columns
]

print(
    df[
        existing_traffic_features
    ].describe()
)


# ============================================================
# 20. MISSING VALUE REPORT
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing_values = (
    df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_values = (
    missing_values[
        missing_values > 0
    ]
)

if len(missing_values) == 0:

    print(
        "No missing values."
    )

else:

    print(
        missing_values
    )


# ============================================================
# 21. DISPLAY FIRST 10 ROWS
# ============================================================

print("\n" + "=" * 80)
print("FIRST 10 ROWS")
print("=" * 80)

display_columns = [
    "sample_id",
    "src",
    "dst",
    "network_size",
    "AvgBw",
    "PktsGen",
    "AvgPktSize",
    "num_hops",
    "min_path_bandwidth",
    "PktsDrop",
    "packet_loss_ratio",
    "AvgDelay",
    "Jitter"
]

display_columns = [
    column
    for column in display_columns
    if column in df.columns
]

print(
    df[
        display_columns
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 22. CHECK UNIQUE SAMPLES
# ============================================================

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(
    "\nUnique sample IDs:"
)

print(
    df["sample_id"]
    .nunique()
)

print(
    "\nUnique source nodes:"
)

print(
    df["src"]
    .nunique()
)

print(
    "\nUnique destination nodes:"
)

print(
    df["dst"]
    .nunique()
)


# ============================================================
# 23. SAVE CSV
# ============================================================

print("\n" + "=" * 80)
print("SAVING CSV")
print("=" * 80)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 24. VERIFY OUTPUT FILE
# ============================================================

if os.path.isfile(OUTPUT_CSV):

    file_size_mb = (
        os.path.getsize(
            OUTPUT_CSV
        )
        /
        (1024 ** 2)
    )

    print(
        "\nCSV saved successfully."
    )

    print(
        "\nOutput file:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nCSV size:"
    )

    print(
        f"{file_size_mb:.2f} MB"
    )

else:

    raise RuntimeError(
        "CSV file was not created."
    )


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(
    f"\nArchive processed : "
    f"{TARGET_ARCHIVE}"
)

print(
    f"Samples processed : "
    f"{sample_counter}"
)

print(
    f"Final CSV rows    : "
    f"{len(df):,}"
)

print(
    f"Final CSV columns : "
    f"{len(df.columns)}"
)

print(
    f"Pair-level errors : "
    f"{error_counter}"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_CSV
)

print(
    "\nConversion completed."
)
