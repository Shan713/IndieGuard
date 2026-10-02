import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer


SOURCE = Path(r"..\IndieGuard\data\processed\games_clean.parquet")
OUTPUT_DIR = Path("dimensionality_reduction_output")
OUTPUT_DIR.mkdir(exist_ok=True)

N_COMPONENTS = 154
RANDOM_STATE = 42


def build_feature_lists(df):
    prefixes = {
        "genres": "genre__",
        "categories": "category__",
        "tags": "tag__",
    }

    feature_lists = []

    for _, row in df.iterrows():
        features = []

        for column, prefix in prefixes.items():
            value = row[column]

            if pd.isna(value):
                continue

            for item in str(value).split(";"):
                item = item.strip()

                if item:
                    features.append(prefix + item)

        feature_lists.append(sorted(set(features)))

    return feature_lists


def main():
    print("Loading cleaned dataset...")
    df = pd.read_parquet(SOURCE)

    print(f"Games: {len(df)}")

    feature_lists = build_feature_lists(df)

    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(feature_lists)

    print(f"Original dimensions: {X.shape[1]}")
    print(f"Matrix shape: {X.shape}")
    print(f"Non-zero values: {X.nnz}")
    print(
        f"Sparsity: "
        f"{(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%"
    )

    print(f"\nRunning Truncated SVD with {N_COMPONENTS} components...")

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        n_iter=10,
        random_state=RANDOM_STATE,
    )

    X_reduced = svd.fit_transform(X)

    explained = svd.explained_variance_ratio_
    cumulative = explained.cumsum()

    print(
        f"Variance retained: "
        f"{cumulative[-1] * 100:.2f}%"
    )

    # Reduced dataset
    component_columns = [
        f"svd_component_{i:03d}"
        for i in range(1, N_COMPONENTS + 1)
    ]

    reduced_df = pd.DataFrame(
        X_reduced,
        columns=component_columns,
    )

    reduced_df.insert(0, "name", df["name"].values)
    reduced_df.insert(0, "appid", df["appid"].values)

    reduced_path = OUTPUT_DIR / "games_svd_154.parquet"
    reduced_df.to_parquet(reduced_path, index=False)

    # Explained variance table
    variance_df = pd.DataFrame(
        {
            "component": range(1, N_COMPONENTS + 1),
            "explained_variance_ratio": explained,
            "explained_variance_percent": explained * 100,
            "cumulative_explained_variance": cumulative,
            "cumulative_explained_variance_percent": cumulative * 100,
        }
    )

    variance_path = OUTPUT_DIR / "svd_variance.csv"
    variance_df.to_csv(variance_path, index=False)

    # Feature-to-component loadings
    loadings_df = pd.DataFrame(
        svd.components_.T,
        index=mlb.classes_,
        columns=component_columns,
    )

    loadings_df.index.name = "original_feature"

    loadings_path = OUTPUT_DIR / "svd_feature_loadings.csv"
    loadings_df.to_csv(loadings_path)

    # Metadata for reproducibility
    metadata = {
        "source": str(SOURCE),
        "games": int(X.shape[0]),
        "original_dimensions": int(X.shape[1]),
        "reduced_dimensions": N_COMPONENTS,
        "variance_retained_percent": float(cumulative[-1] * 100),
        "random_state": RANDOM_STATE,
        "n_iter": 10,
        "method": "Truncated SVD",
        "representation": "Multi-hot encoding of genres, categories and tags",
        "selection_rule": (
            "Minimum number of components required to retain "
            "at least 90% cumulative explained variance"
        ),
    }

    metadata_path = OUTPUT_DIR / "svd_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSUCCESS")
    print("=" * 50)
    print(f"Original dimensions : {X.shape[1]}")
    print(f"Reduced dimensions  : {N_COMPONENTS}")
    print(
        f"Variance retained   : "
        f"{cumulative[-1] * 100:.2f}%"
    )
    print(f"\nOutput folder: {OUTPUT_DIR.resolve()}")
    print("\nCreated files:")

    for path in [
        reduced_path,
        variance_path,
        loadings_path,
        metadata_path,
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()