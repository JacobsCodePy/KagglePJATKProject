"""
Contains helpers for model's learning process. Methods like target encoding, data filtering
and possibly more.
"""
from typing import Union, List, Tuple
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm


def cat_wrap(df: pd.DataFrame, column: str, threshold: int):
    """
    Converts all the categories of a categorical features, which are less frequent
    than the `threshold` into the 'Other' category.
    """
    counts = df[column].value_counts()
    mask = df[column].isin(counts[counts < threshold].index)
    df.loc[mask, column] = 'Other'
    return df


def cat_filter(df, column, threshold):
    """
    Sets all the categories of a categorical features, which are less frequent
    than the `threshold` to Nan.
    """
    counts = df[column].value_counts()
    mask = df[column].isin(counts[counts < threshold].index)
    df.loc[mask, column] = np.nan
    return df


def target_encode(df: pd.DataFrame, df_outer: Union[List[pd.DataFrame], pd.DataFrame],
                  features: List, high_cardinality_features: List[str], low_cardinality_features: List[str],
                  high_cardinality_threshold: int = 10, low_cardinality_threshold: int = 50) -> Tuple:
    """
    Encodes the target features based on cardinality.
    :param df: Dataframe, to which on based on which the target encoding should be performed
    :param df_outer: Dataframe or dataframes, to which target encoding should be done based on `df`. This is
    to target encode features without target leakage.
    :param features: List of current features. Target encoding will return the extended version with new features.
    :param high_cardinality_features: List of current high cardinality features.
    :param low_cardinality_features: List of current low cardinality features.
    :param high_cardinality_threshold: Frequency threshold under which high cardinality categories are ignored.
    :param low_cardinality_threshold: Frequency threshold under which low cardinality categories are ignored.
    """
    features = copy.deepcopy(features)

    # Target encoding for high cardinality features
    for cat in tqdm(high_cardinality_features):
        df = cat_filter(df, cat, high_cardinality_threshold)
        engineered_stats = ['median', 'max', 'min', 'nunique']
        table_of_cat_stats = df.groupby(cat).agg({'Cena': engineered_stats})
        engineered_features = []
        for stat in engineered_stats:
            stat_series = table_of_cat_stats[('Cena', stat)]
            df.loc[:, f'{cat}_{stat}'] = df.loc[:, cat].map(stat_series)
            if df_outer is not None:
                if isinstance(df_outer, pd.DataFrame):
                    df_outer.loc[:, f'{cat}_{stat}'] = df_outer.loc[:, cat].map(stat_series)
                elif isinstance(df_outer, list):
                    for dfu in df_outer:
                        dfu.loc[:, f'{cat}_{stat}'] = dfu.loc[:, cat].map(stat_series)
                else:
                    raise ValueError('Expected unknown data frames to be list or a single data frame.')
            engineered_features.append(f'{cat}_{stat}')
        features += engineered_features

    # Target encoding for low cardinality features
    for cat in tqdm(low_cardinality_features):
        df = cat_filter(df, cat, low_cardinality_threshold)
        engineered_stats = ['median', 'mean', 'std']
        table_of_cat_stats = df.groupby(cat).agg({'Cena': engineered_stats})
        engineered_features = []
        for stat in engineered_stats:
            stat_series = table_of_cat_stats[('Cena', stat)]
            df.loc[:, f'{cat}_{stat}'] = df.loc[:, cat].map(stat_series)
            if df_outer is not None:
                if isinstance(df_outer, pd.DataFrame):
                    df_outer.loc[:, f'{cat}_{stat}'] = df_outer.loc[:, cat].map(stat_series)
                elif isinstance(df_outer, list):
                    for dfu in df_outer:
                        dfu.loc[:, f'{cat}_{stat}'] = dfu.loc[:, cat].map(stat_series)
                else:
                    raise ValueError('Expected unknown data frames to be list or a single data frame.')
            engineered_features.append(f'{cat}_{stat}')
        features += engineered_features
    return df, df_outer, features
