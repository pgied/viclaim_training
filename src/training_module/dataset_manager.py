
import math
import os
import random
import re
from typing import List, Tuple
import pandas as pd
import torch


# Clean sentences: drop NaN sentences; strip whitespace
def clean_sentence(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    # collapsing internal whitespace/newlines
    s = re.sub(r'\s+', ' ', s)
    return s if s != '' else None


def load_and_filter_data(dataset_path):    
    print("PWD:", os.getcwd())
    # Read CSV into DataFrame
    df = pd.read_csv(dataset_path)
    
    # Make sure row_index is numeric and sort by clip_id then row_index to define ordering
    df['row_index'] = pd.to_numeric(df['row_index'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values(['clip_id', 'row_index'])
    
    df['sentence_clean'] = df['sentence'].apply(clean_sentence)
    df['sentence'] = df['sentence_clean'].fillna('').astype(str)
    df = df.drop(columns=['sentence_clean'])
    
    # Group by clip_id and join sentences preserving order
    grouped_by_clip_df = (
        df.dropna(subset=['sentence'])
        .groupby('clip_id', sort=False)
        .agg(
            fulltext = ('sentence', lambda s: ' '.join(s.tolist())),
        )
        .reset_index()
    )

    # Normalize spaces (in case joining created double spaces)
    grouped_by_clip_df['fulltext'] = grouped_by_clip_df['fulltext'].apply(lambda t: re.sub(r'\s+', ' ', t).strip())
    
    clip_to_fulltext = dict(zip(grouped_by_clip_df['clip_id'], grouped_by_clip_df['fulltext']))
    df['fulltext'] = df['clip_id'].map(clip_to_fulltext)
    df['fulltext'] = df['fulltext'].fillna('')
                
    return df


def create_stratified_train_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.15,
    eval_size: float = 0.15,
    split_by_se_id: bool = False,
    lang_col: str = "language",
    topic_col: str = "topic",
    clip_col: str = "clip_id",
    sentence_id_col: str = "row_index",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by (language, topic). For each (lang, topic) group:
      - collect unique item ids (clip_id or sentence id depending on split_by_se_id)
      - shuffle the ids
      - allocate ids to test / eval / train according to test_size / eval_size
      - include all rows that belong to an allocated id in that split

    Returns: (train_df, eval_df, test_df)
    """
    # Defensive copy
    df = df.copy()

    # choose the id column used for splitting
    item_col = sentence_id_col if split_by_se_id else clip_col

    rng = random.Random(random_seed)

    train_parts = []
    eval_parts = []
    test_parts = []

    # group by (language, topic)
    grouped = df.groupby([lang_col, topic_col], sort=False)

    for (lang, topic), group_df in grouped:
        # get unique item ids for this group (dropna items)
        ids = group_df[item_col].dropna().unique().tolist()
        if len(ids) == 0:
            # no items to split in this group
            continue

        ids = sorted(ids)  # preserve deterministic ordering before shuffling
        rng.shuffle(ids)

        n_items = len(ids)
        n_test = int(test_size * n_items)
        n_eval = int(eval_size * n_items)

        test_ids = ids[:n_test]
        eval_ids = ids[n_test : n_test + n_eval]
        train_ids = ids[n_test + n_eval :]

        # select rows for each split for this (lang,topic)
        if train_ids:
            train_parts.append(group_df[group_df[item_col].isin(train_ids)])
        if eval_ids:
            eval_parts.append(group_df[group_df[item_col].isin(eval_ids)])
        if test_ids:
            test_parts.append(group_df[group_df[item_col].isin(test_ids)])

    # concat all pieces (if empty, create empty DataFrame with same columns)
    def _concat_or_empty(parts):
        if parts:
            return pd.concat(parts, ignore_index=True)
        else:
            return pd.DataFrame(columns=df.columns)

    train_df = _concat_or_empty(train_parts)
    eval_df = _concat_or_empty(eval_parts)
    test_df = _concat_or_empty(test_parts)

    # final global shuffle of train set only (to mix examples across groups)
    if len(train_df) > 0:
        train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # reset indices for eval/test as well
    if len(eval_df) > 0:
        eval_df = eval_df.reset_index(drop=True)
    if len(test_df) > 0:
        test_df = test_df.reset_index(drop=True)

    return train_df, eval_df, test_df


def k_fold_cv_df(
    df: pd.DataFrame,
    n_folds: int = 5,
    id_col: str = "row_index",
    lang_col: str = "language",
    topic_col: str = "topic",
    random_seed: int | None = None,
) -> List[pd.DataFrame]:
    """
    K-fold cross validation split on a DataFrame, stratified by (language, topic).
    For each (lang, topic) group we:
      - collect unique ids (clip_id or sentence id),
      - shuffle them,
      - split them into n_folds parts (balanced),
      - assign all rows belonging to ids in each part to the corresponding fold.

    Returns: list of length n_folds of DataFrames (each fold).
    """
    rng = random.Random(random_seed)

    def split_arr(a, n):
        """Split list a into n nearly-equal parts (first parts get the extra items)."""
        k, m = divmod(len(a), n)
        parts = []
        start = 0
        for i in range(n):
            size = k + (1 if i < m else 0)
            parts.append(a[start:start + size])
            start += size
        return parts

    folds_parts: List[List[pd.DataFrame]] = [[] for _ in range(n_folds)]

    # Group by (language, topic)
    grouped = df.groupby([lang_col, topic_col], sort=False)

    for (lang, topic), group_df in grouped:
        # get unique IDs for this (lang, topic)
        ids = group_df[id_col].dropna().unique().tolist()
        if len(ids) == 0:
            continue

        ids = sorted(ids)               # deterministic order before shuffle
        rng.shuffle(ids)               # randomize
        splits = split_arr(ids, n_folds)  # list of id-lists for each fold

        for i, id_split in enumerate(splits):
            if not id_split:
                continue
            # pick rows in this (lang,topic) group that match any id in id_split
            part = group_df[group_df[id_col].isin(id_split)]
            if not part.empty:
                folds_parts[i].append(part)

    # concatenate parts per fold (or return empty DataFrame with same columns)
    folds: List[pd.DataFrame] = []
    for parts in folds_parts:
        if parts:
            fold_df = pd.concat(parts, ignore_index=True)
        else:
            fold_df = pd.DataFrame(columns=df.columns)
        folds.append(fold_df)

    return folds


def process_sentences(df: pd.DataFrame, use_full_context = True):
    cids = df['clip_id'].tolist()
    row_index = df['row_index'].tolist()
    if use_full_context:
        x = (
            (df['sentence'].fillna('') + ' ' + df['fulltext'].fillna(''))
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .tolist()
        )
    else:
        x = df['sentence'].tolist()
    cols = ["label_fcw", "label_fnc", "label_opn"]
    y = [torch.tensor(row, dtype=torch.float32) for row in df[cols].fillna(0).to_numpy()]
        
    return cids, row_index, x, y