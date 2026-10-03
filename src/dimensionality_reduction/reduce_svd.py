"""First tag reduction (Truncated SVD, fixed 154 components), fitted on training games only.

Superseded by src/features/tags.py (PCA on filtered columns, k at the variance elbow): see
docs/feature_engineering.md. Kept for history; its outputs are not used by the feature tables.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/processed/games_clean.parquet"
SPLIT_SOURCE = ROOT / "data/processed/splits/game_split.csv"
OUTPUT_DIR = ROOT / "data/processed/dimensionality_reduction"
DOCS_DIR = ROOT / "docs/dimensionality_reduction"

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

    print("Loading train/test split...")
    split_df = pd.read_csv(SPLIT_SOURCE)

    # ---------------------------------------------------------
    # Validate split
    # ---------------------------------------------------------

    if split_df["appid"].duplicated().any():
        raise ValueError("Duplicate AppIDs found in game_split.csv.")

    if not df["appid"].isin(split_df["appid"]).all():
        missing = (~df["appid"].isin(split_df["appid"])).sum()
        raise ValueError(
            f"{missing} games from games_clean.parquet "
            "are missing from game_split.csv."
        )

    if not split_df["appid"].isin(df["appid"]).all():
        extra = (~split_df["appid"].isin(df["appid"])).sum()
        raise ValueError(
            f"{extra} AppIDs in game_split.csv "
            "are missing from games_clean.parquet."
        )

    # Keep split information aligned with games_clean
    df = df.merge(
        split_df[["appid", "split", "cv_fold"]],
        on="appid",
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        raise ValueError("Some games do not have a train/test assignment.")

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Total games: {len(df)}")
    print(f"Training games: {len(train_df)}")
    print(f"Test games: {len(test_df)}")

    # ---------------------------------------------------------
    # Build feature representation
    # ---------------------------------------------------------

    train_feature_lists = build_feature_lists(train_df)
    test_feature_lists = build_feature_lists(test_df)

    # Fit the encoder ONLY on training data
    mlb = MultiLabelBinarizer(sparse_output=True)

    X_train = mlb.fit_transform(train_feature_lists)
    X_test = mlb.transform(test_feature_lists)

    print(f"\nOriginal dimensions: {X_train.shape[1]}")
    print(f"Training matrix shape: {X_train.shape}")
    print(f"Test matrix shape: {X_test.shape}")

    print(f"Training non-zero values: {X_train.nnz}")
    print(f"Test non-zero values: {X_test.nnz}")

    print(
        f"Training sparsity: "
        f"{(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])) * 100:.2f}%"
    )

    print(
        f"Test sparsity: "
        f"{(1 - X_test.nnz / (X_test.shape[0] * X_test.shape[1])) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Fit SVD ONLY on training data
    # ---------------------------------------------------------

    print(
        f"\nFitting Truncated SVD with "
        f"{N_COMPONENTS} components on TRAINING data..."
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        n_iter=10,
        random_state=RANDOM_STATE,
    )

    X_train_reduced = svd.fit_transform(X_train)

    # Apply the already-fitted SVD to test data
    X_test_reduced = svd.transform(X_test)

    explained = svd.explained_variance_ratio_
    cumulative = explained.cumsum()

    print(
        f"Training variance retained: "
        f"{cumulative[-1] * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Component names
    # ---------------------------------------------------------

    component_columns = [
        f"svd_component_{i:03d}"
        for i in range(1, N_COMPONENTS + 1)
    ]

    # ---------------------------------------------------------
    # Save reduced TRAIN dataset
    # ---------------------------------------------------------

    train_reduced_df = pd.DataFrame(
        X_train_reduced,
        columns=component_columns,
    )

    train_reduced_df.insert(0, "name", train_df["name"].values)
    train_reduced_df.insert(0, "appid", train_df["appid"].values)

    train_reduced_path = (
        OUTPUT_DIR / "games_svd_train_154.parquet"
    )

    train_reduced_df.to_parquet(
        train_reduced_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save reduced TEST dataset
    # ---------------------------------------------------------

    test_reduced_df = pd.DataFrame(
        X_test_reduced,
        columns=component_columns,
    )

    test_reduced_df.insert(0, "name", test_df["name"].values)
    test_reduced_df.insert(0, "appid", test_df["appid"].values)

    test_reduced_path = (
        OUTPUT_DIR / "games_svd_test_154.parquet"
    )

    test_reduced_df.to_parquet(
        test_reduced_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Explained variance
    # ---------------------------------------------------------

    variance_df = pd.DataFrame(
        {
            "component": range(1, N_COMPONENTS + 1),
            "explained_variance_ratio": explained,
            "explained_variance_percent": explained * 100,
            "cumulative_explained_variance": cumulative,
            "cumulative_explained_variance_percent": cumulative * 100,
        }
    )

    variance_path = DOCS_DIR / "svd_variance.csv"
    variance_df.to_csv(variance_path, index=False)

    # ---------------------------------------------------------
    # Feature-to-component loadings
    # ---------------------------------------------------------

    loadings_df = pd.DataFrame(
        svd.components_.T,
        index=mlb.classes_,
        columns=component_columns,
    )

    loadings_df.index.name = "original_feature"

    loadings_path = DOCS_DIR / "svd_feature_loadings.csv"
    loadings_df.to_csv(loadings_path)

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "source": "data/processed/games_clean.parquet",
        "split_source": "data/processed/splits/game_split.csv",
        "total_games": int(len(df)),
        "training_games": int(len(train_df)),
        "test_games": int(len(test_df)),
        "original_dimensions": int(X_train.shape[1]),
        "reduced_dimensions": N_COMPONENTS,
        "training_variance_retained_percent": float(
            cumulative[-1] * 100
        ),
        "random_state": RANDOM_STATE,
        "n_iter": 10,
        "method": "Truncated SVD",
        "representation": (
            "Multi-hot encoding of genres, categories and tags"
        ),
        "fit_strategy": (
            "MultiLabelBinarizer and TruncatedSVD fitted only "
            "on training games; test games transformed using "
            "the fitted objects"
        ),
        "selection_rule": (
            "154 components retained to remain consistent with "
            "the original dimensionality reduction implementation"
        ),
    }

    metadata_path = DOCS_DIR / "svd_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\nSUCCESS")
    print("=" * 50)
    print(f"Total games        : {len(df)}")
    print(f"Training games     : {len(train_df)}")
    print(f"Test games         : {len(test_df)}")
    print(f"Original dimensions: {X_train.shape[1]}")
    print(f"Reduced dimensions : {N_COMPONENTS}")
    print(
        f"Train variance     : "
        f"{cumulative[-1] * 100:.2f}%"
    )

    print("\nCreated files:")
    print(f"  - {train_reduced_path}")
    print(f"  - {test_reduced_path}")
    print(f"  - {variance_path}")
    print(f"  - {loadings_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()