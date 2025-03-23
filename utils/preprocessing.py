"""
Tools for preprocessing the data of the car sales. Both fundamental for analysis
and special for training.
"""
import ast
from typing import Callable, List, TypeAlias, Tuple, Sequence, Any, Union, Optional
from scipy.stats import iqr
from utils import logger
import numpy as np
import pandas as pd
import utils.constants as constants

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


def preprocessing_horsepower_cleanup(df: pd.DataFrame, popular_threshold: int = 100, rare_threshold: int = 10,
                                     sigma_scalar: float = 5) -> pd.DataFrame:
    """
    Based on the average brands horsepower tries to fix horsepower outliers in the data, by capping
    their value to the maximal possible for the given brand.
    Popular brands are capped by the value of `sigma_scalar` * `standard deviation` of the brand. Whereas
    less common brands are capped by using simple IQR, since their deviation can't be trusted. IQR should works
    since after analysis of the data it can be seen that outliers are single examples
    (which means they can't get to the Q1 and Q3 value).
    """

    # Function for capping the horsepower
    def cap_function(brands: List[str], max_table: pd.DataFrame, min_table: Optional[pd.DataFrame] = None):
        def cap_power_based_on_brand(row: pd.Series):
            row = row.copy()
            if row.Marka_pojazdu not in brands:
                return row
            if row.Moc_KM > max_table[row.Marka_pojazdu]:
                row.Moc_KM = max_table[row.Marka_pojazdu]
            elif min_table is not None and row.Moc_KM < min_table[row.Marka_pojazdu]:
                row.Moc_KM = min_table[row.Marka_pojazdu]
            return row

        return cap_power_based_on_brand

    # Cap the horsepower of the popular brands
    funcs = ('mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75))
    power_table = ((df.groupby('Marka_pojazdu').agg({'Moc_KM': funcs}))
                   .sort_values(('Moc_KM', 'mean'), ascending=False))
    power_table.columns = ('mean', 'std', 'Q1', 'Q3')
    value_counts = df.Marka_pojazdu.value_counts()[power_table.index]
    common_brands = power_table[value_counts >= popular_threshold]
    max_values = common_brands.loc[:, 'mean'] + common_brands.loc[:, 'std'] * sigma_scalar
    df.loc[:, 'Moc_KM'] = df.loc[:, ['Moc_KM', 'Marka_pojazdu']].apply(cap_function(common_brands, max_values), axis=1)

    # Cap the horsepower of the rare brands
    uncommon_brands = power_table[(value_counts < popular_threshold) & (value_counts >= rare_threshold)]
    inter_quantile_range = uncommon_brands.loc[:, 'Q3'] - uncommon_brands.loc[:, 'Q1']
    max_values = uncommon_brands.loc[:, 'Q3'] + 1.5 * inter_quantile_range
    min_values = uncommon_brands.loc[:, 'Q1'] - 1.5 * inter_quantile_range
    df.loc[:, 'Moc_KM'] = df.loc[:, ['Moc_KM', 'Marka_pojazdu']].apply(cap_function(uncommon_brands, max_values,
                                                                                    min_values), axis=1)
    return df


def preprocessing_engine_volume_cleanup(df: pd.DataFrame, error_threshold: int = -2_001,
                                        volume_threshold: int = 1_001) -> pd.DataFrame:
    """
    Using linear regression and natural correlation between engine's volume and its horsepower
    fixes the engine's volume towards the value implied by linear regression. Applies only to very small engines
    with high horsepower, as such engines are near impossible in real life.
    :param df: String equipment
    :param error_threshold: How big difference there must be between predicted volume and actual volume to act
    upon it.
    :param volume_threshold: How small the engine must be to act upon its volume.
    """
    mask = ~(df.Pojemnosc_cm3.isna() | df.Moc_KM.isna())
    slope, bias = np.polyfit(df.Moc_KM[mask].to_numpy(),
                             df.Pojemnosc_cm3[mask].to_numpy(), 1)
    errors = (df.Pojemnosc_cm3[mask].to_numpy() - (bias + slope * df.Moc_KM[mask].to_numpy()))
    args = np.argwhere((df.Pojemnosc_cm3[mask].to_numpy() < volume_threshold) & (errors < error_threshold))
    indices = df.Moc_KM[mask].index[args.flatten()]
    df.loc[indices, 'Pojemnosc_cm3'] = (df.loc[indices, 'Pojemnosc_cm3'] + bias + slope * df.Moc_KM[mask].to_numpy()[
        indices]) / 2
    return df


def preprocessing_mileage_cleanup(df: pd.DataFrame, scalar: float = 5) -> pd.DataFrame:
    """
    Cleans the mileage based on Q3 + `scalar` * IQR. Is alternative for using log transformation
    for the mileage to get rid of extreme outliers (n-magnitudes higher than normal values). Therefore,
    suggested `scalar` is higher than normal 1,5 constant. Mileage above threshold is set to Nan.
    """
    max_value = df.Przebieg_km.quantile(0.75) + scalar * iqr(df.Przebieg_km.to_numpy(), nan_policy='omit')
    df.loc[df.Przebieg_km > max_value, 'Przebieg_km'] = np.nan
    return df


def preprocessing_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the city and voivodeship from the location column.
    """

    # Extract important keywords from the location strings
    to_lower = lambda words: np.array([word.lower() for word in words]) if isinstance(words, list) else np.nan
    locations = df.Lokalizacja_oferty.str.findall(r'\b[A-Za-z]+\b').apply(to_lower)

    # Set up location columns
    df['Miasto'] = np.nan
    df['Wojewodztwo'] = np.nan
    df.Miasto = df.Miasto.astype('string')
    df.Wojewodztwo = df.Wojewodztwo.astype('string')

    # Fill the locations based on keywords
    is_city = lambda locs: np.any(np.isin(locs, constants.CITIES)) \
        if isinstance(locs, np.ndarray) and list(locs) else False
    city_to_category = lambda locs: locs[np.isin(locs, constants.CITIES)][0] \
        if isinstance(locs, np.ndarray) else np.nan
    is_voivodeship = lambda locs: np.any(np.isin(locs, constants.VOIVODESHIPS)) \
        if isinstance(locs, np.ndarray) and list(locs) else False
    voivodeship_to_category = lambda locs: locs[np.isin(locs, constants.VOIVODESHIPS)][0] \
        if isinstance(locs, np.ndarray) else np.nan
    is_city_mask = locations.apply(is_city)
    is_voivodeship_mask = locations.apply(is_voivodeship)
    df.loc[is_city_mask, 'Miasto'] = locations.loc[is_city_mask].apply(city_to_category)
    df.loc[is_voivodeship_mask, 'Wojewodztwo'] = locations.loc[is_voivodeship_mask].apply(voivodeship_to_category)
    return df


def preprocess_car_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts car state into the binary feature.
    """
    df['Stan_binary'] = df.Stan.map({'Used': 0, 'New': 1})
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


FeaturePreprocessPipeline = PreprocessPipeline([
    preprocessing_horsepower_cleanup,
    preprocessing_engine_volume_cleanup,
    preprocessing_mileage_cleanup,
    preprocess_car_state,
    preprocessing_location,
])
