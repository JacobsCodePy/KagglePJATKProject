"""
Tools for preprocessing the data of the car sales. Both fundamental for analysis
and special for training.
"""
import ast
from typing import Callable, List, TypeAlias, Tuple, Sequence, Any, Union
from utils import logger
import numpy as np
import pandas as pd

PreprocessMethod: TypeAlias = Union[
    Tuple[Callable[[pd.DataFrame, ...], pd.DataFrame], Sequence[Any]],
    Callable[[pd.DataFrame], pd.DataFrame]
]


class PreprocessPipeline:
    """
    A shortcut for creating a list of preprocessing steps for the data.
    """

    def __init__(self, methods: List[PreprocessMethod]):
        self._methods = methods

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for methodset in self._methods:
            if isinstance(methodset, tuple):
                method, args = methodset
                df = method(df, *args)
            elif callable(methodset):
                df = methodset(df)
        return df


def preprocess_equipment_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts `Wyposazenie` from string typed list to the python list.
    """
    column = 'Wyposazenie'

    def str_to_arr(row: str) -> np.ndarray:
        if type(row) is str:
            return ast.literal_eval(row)
        return None

    df[column] = df[column].apply(str_to_arr)
    return df


def preprocess_publication_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts `Data_publikacji_oferty` from string to the datatime format.
    Additionally, adds `Wiek_oferty` column counted in days.
    """
    column = 'Data_publikacji_oferty'
    name = 'Wiek_oferty'
    df[column] = pd.to_datetime(df[column], format="mixed")
    logger.info(f"Date range : {df[column].min()}, {df[column].max()}")
    df[name] = (pd.Timestamp.now() - df[column]).dt.days
    df[name] = df[name] - df[name].min()
    df[name] = df[name].astype(float)
    return df


def preprocess_currency(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    """
    Convert currency in `EUR` to `PLN` using rate 4.3.
    If other currrency occurs raises an ValueError.
    """
    euro_constant = 4.3
    if 'Cena' not in df.columns:
        logger.warning("Preprocessing currency did not found `Cena` column. Skipping.")
        return df
    column = 'Waluta'
    df['Cena'] = df['Cena'].astype(float)
    mask = df[column] == 'EUR'
    if not invert:
        df.loc[mask, 'Cena'] *= euro_constant
    else:
        df.loc[mask, 'Cena'] /= euro_constant
    logger.info(f"Amount of samples with foreign currency  : {mask.sum()}")
    if not np.all(np.isin(df.loc[~df[column].isna(), column].unique(), ['PLN', 'EUR'])):
        raise ValueError(f"At least one of the currencies {df[column].unique()} is unknown.")
    return df


def preprocess_equipment_list_binary(df: pd.DataFrame) -> pd.DataFrame:
    def parse_equipment(equip_str):
        """
        Prases equipment from string format to a list
        :param equip_str: String equipment
        :return: Equipment list
        """
        if pd.isna(equip_str) or equip_str == '' or equip_str == 'nan':
            return []

        if isinstance(equip_str, list):
            return equip_str

        try:
            cleaned_str = equip_str.replace("'", '"')
            evaluated = ast.literal_eval(cleaned_str)
            return evaluated if isinstance(evaluated, list) else []
        except Exception as e:
            return []

    df['Lista_wyposazenia'] = df['Wyposazenie'].apply(parse_equipment)
    df['Count_wyposazenia'] = df['Lista_wyposazenia'].apply(len)

    # Finds the equipment observable in at least 5% of the rows
    all_equipment = {}
    for equip_list in df['Lista_wyposazenia']:
        for item in equip_list:
            all_equipment[item] = all_equipment.get(item, 0) + 1

    min_count = int(0.05 * len(df))
    common_equipment = [item for item, count in all_equipment.items() if count >= min_count]

    # Add equipment as binary features
    for item in common_equipment:
        feature_name = f"wyp_{item.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        df[feature_name] = df['Lista_wyposazenia'].apply(lambda x: 1 if item in x else 0)

    return df


BasicPreprocessPipeline = PreprocessPipeline([
    preprocess_equipment_list,
    preprocess_publication_date,
    preprocess_currency,
])


ExtendedPreprocessPipeline = PreprocessPipeline([
    preprocess_equipment_list_binary,
    preprocess_publication_date,
    preprocess_currency,
])
