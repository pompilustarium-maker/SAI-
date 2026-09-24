"""
WikiArt Multi-Task Dataset, Data Cleaning, and Label Encoding.
Handles joint parsing for Genre and Historical Era/Century classification,
mitigating severe class imbalance via grouping, filtering, and balanced weighting.
"""

import os
import re
import ast
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from data.transforms import get_train_transforms, get_val_transforms


# Canonical subgenre aggregation mapping
GENRE_AGGREGATION_MAP = {
    'Synthetic Cubism': 'Cubism',
    'Analytical Cubism': 'Cubism',
    'Contemporary Realism': 'Realism',
    'New Realism': 'Realism',
    'Mannerism Late Renaissance': 'Renaissance',
    'Early Renaissance': 'Renaissance',
    'High Renaissance': 'Renaissance',
    'Northern Renaissance': 'Renaissance',
    'Action painting': 'Abstract Expressionism',
    'Post Impressionism': 'Post-Impressionism',
    'Art Nouveau Modern': 'Art Nouveau',
    'Naive Art Primitivism': 'Primitivism',
    'Color Field Painting': 'Color Field',
}

# Standard Century Bins
CENTURY_ORDER = ['Pre-XV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI']


def clean_raw_genre(genre_raw: Any) -> str:
    """Parses raw CSV string (e.g. \"['Impressionism']\") into a primary clean genre string."""
    if pd.isna(genre_raw):
        return "Unknown"
    
    text = str(genre_raw).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            genre_name = str(parsed[0]).strip()
            return GENRE_AGGREGATION_MAP.get(genre_name, genre_name)
    except Exception:
        pass

    # Fallback regex / text strip
    cleaned = re.sub(r"[\[\]'\"']", "", text).split(",")[0].strip()
    return GENRE_AGGREGATION_MAP.get(cleaned, cleaned)


def extract_year_and_century(desc: Any, filename: Any) -> Tuple[Optional[int], str]:
    """
    Extracts publication year from description or filename and maps it
    to a valid historical Century/Era.
    Eliminates corrupted values (e.g., '56 век', negative numbers, or invalid years).
    """
    year = None
    
    # 1. Search in description for 3 or 4 digit year
    desc_str = str(desc) if not pd.isna(desc) else ""
    match_end = re.search(r'-(\d{3,4})$', desc_str.strip())
    if match_end:
        year = int(match_end.group(1))
    else:
        match_any = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', desc_str)
        if match_any:
            year = int(match_any.group(1))

    # 2. Fallback to filename
    if year is None:
        file_str = str(filename) if not pd.isna(filename) else ""
        match_file = re.search(r'-(\d{4})\.(?:jpg|png|jpeg)', file_str, re.IGNORECASE)
        if match_file:
            year = int(match_file.group(1))

    if year is None or year < 800 or year > 2026:
        return None, "Unknown"

    # Map to Century
    if year <= 1400:
        return year, "Pre-XV"
    elif year <= 1500:
        return year, "XV"
    elif year <= 1600:
        return year, "XVI"
    elif year <= 1700:
        return year, "XVII"
    elif year <= 1800:
        return year, "XVIII"
    elif year <= 1900:
        return year, "XIX"
    elif year <= 2000:
        return year, "XX"
    else:
        return year, "XXI"


class WikiArtDataset(Dataset):
    """
    PyTorch Dataset for WikiArt Multi-Task Learning.
    Returns:
        (image_tensor, genre_label, century_label, sample_info_dict)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str,
        transform=None,
        genre_col: str = "genre_idx",
        century_col: str = "century_idx"
    ):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
        self.genre_col = genre_col
        self.century_col = century_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = str(row['filename']).lstrip('/\\')
        img_path = os.path.join(self.base_dir, rel_path)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Safe placeholder to prevent data loader crash on disk read error
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        genre_label = int(row[self.genre_col])
        century_label = int(row[self.century_col])

        info = {
            'filename': row['filename'],
            'artist': row.get('artist', 'Unknown'),
            'genre': row.get('clean_genre', 'Unknown'),
            'century': row.get('clean_century', 'Unknown'),
            'year': row.get('year', -1),
        }

        return image, genre_label, century_label, info


def compute_class_weights(df: pd.DataFrame, target_col: str) -> torch.Tensor:
    """Computes balanced inverse-frequency weights for CrossEntropyLoss."""
    classes = np.sort(df[target_col].unique())
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=df[target_col].values
    )
    return torch.tensor(weights, dtype=torch.float32)


def load_and_preprocess_wikiart(
    csv_path: str,
    min_genre_samples: int = 50,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, LabelEncoder, torch.Tensor, torch.Tensor]:
    """
    Loads raw metadata CSV, performs canonical cleaning, groups rare classes,
    encodes labels for both Genre and Century tasks, and computes class weights.
    """
    df = pd.read_csv(csv_path)

    # 1. Clean Genre
    df['clean_genre'] = df['genre'].apply(clean_raw_genre)

    # Filter out very rare genres below threshold
    genre_counts = df['clean_genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= min_genre_samples].index
    df = df[df['clean_genre'].isin(valid_genres)].copy()

    # 2. Extract Year and Century
    year_century_pairs = [
        extract_year_and_century(desc, fn)
        for desc, fn in zip(df['description'], df['filename'])
    ]
    df['year'] = [p[0] for p in year_century_pairs]
    df['clean_century'] = [p[1] for p in year_century_pairs]

    # Filter out records where century could not be reliably determined
    df = df[df['clean_century'] != "Unknown"].copy()

    # 3. Label Encoders
    genre_encoder = LabelEncoder()
    df['genre_idx'] = genre_encoder.fit_transform(df['clean_genre'])

    century_encoder = LabelEncoder()
    # Fit with predefined canonical century ordering present in data
    present_centuries = [c for c in CENTURY_ORDER if c in df['clean_century'].values]
    # Ensure remaining found centuries are included
    extra_centuries = [c for c in df['clean_century'].unique() if c not in present_centuries]
    all_century_classes = present_centuries + extra_centuries
    century_encoder.fit(all_century_classes)
    df['century_idx'] = century_encoder.transform(df['clean_century'])

    # 4. Train / Validation Split
    # Stratify by combined task representation for balanced distribution across both tasks
    df['stratify_key'] = df['clean_genre'].astype(str) + "_" + df['clean_century'].astype(str)
    # If any stratify key has < 2 members, fallback to stratifying by genre
    key_counts = df['stratify_key'].value_counts()
    rare_keys = key_counts[key_counts < 2].index
    if len(rare_keys) > 0:
        strat_col = df['genre_idx']
    else:
        strat_col = df['stratify_key']

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_col
    )

    # 5. Compute Balanced Class Weights
    genre_weights = compute_class_weights(train_df, 'genre_idx')
    century_weights = compute_class_weights(train_df, 'century_idx')

    return train_df, val_df, genre_encoder, century_encoder, genre_weights, century_weights


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_dir: str,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Instantiates PyTorch DataLoaders with specialized augmentations."""
    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)

    train_dataset = WikiArtDataset(train_df, base_dir, transform=train_transform)
    val_dataset = WikiArtDataset(val_df, base_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
