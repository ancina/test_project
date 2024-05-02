import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
import numpy as np

def preprocess(df, numerical_columns, categorical_columns, ordinal_columns, outcome_column):
    """
    Preprocesses the input DataFrame by scaling numerical features,
    encoding categorical features, and transforming ordinal features.

    Args:
    - df (DataFrame): Input DataFrame to preprocess.
    - numerical_columns (list): List of numerical column names.
    - categorical_columns (list): List of categorical column names.
    - ordinal_columns (list): List of ordinal column names.

    Returns:
    - Preprocessed DataFrame.
    """
    # Initialize scalers
    sc = StandardScaler()
    minmax = MinMaxScaler()

    # Remove the outcome column from numerical columns because I don't want to rescale it
    numerical_columns_no_delta = numerical_columns.copy()
    numerical_columns_no_delta.remove(outcome_column)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

    # Encode ordinal columns
    df[ordinal_columns] = OrdinalEncoder().fit_transform(df[ordinal_columns])

    # Scale numerical columns except the outcome
    df[numerical_columns_no_delta] = sc.fit_transform(df[numerical_columns_no_delta])
    df[ordinal_columns] = minmax.fit_transform(df[ordinal_columns])

    return df

def columns_of_interest(dictionary):
    """
    Extracts a list of variables of interest from a dictionary.

    Args:
    - dictionary (dict): Dictionary containing variable names.

    Returns:
    - List of variables of interest.
    """
    # Flatten dictionary values
    vars_of_interest = [var for sublist in dictionary.values() for var in sublist]
    return vars_of_interest

def extract_dataframe(df, dictionary):
    """
    Extracts a DataFrame containing columns specified in a dictionary.

    Args:
    - df (DataFrame): Input DataFrame.
    - dictionary (dict): Dictionary containing column names.

    Returns:
    - DataFrame containing columns of interest.
    """
    # apply function above to extract dataframe
    vars_of_interest = columns_of_interest(dictionary=dictionary)
    return df[vars_of_interest]

def include_exclude_nan_removal(df, columns = 'all'):
    """
    Removes rows containing NaN values and returns included and excluded DataFrames.

    Args:
    - df (DataFrame): Input DataFrame.

    Returns:
    - included (DataFrame): DataFrame with rows containing NaN values removed.
    - excluded (DataFrame): DataFrame containing only rows with NaN values.
    """
    if columns == 'all':
        mask = ~df.isna().any(axis=1)
    else:
        mask = ~df[columns].isna().any(axis=1)
    included = df[mask]
    excluded = df[~mask]
    return included, excluded

def assign_type(df, dict_types):
    """
    Assigns data types to DataFrame columns as per the specified dictionary.

    Args:
    - df (DataFrame): Input DataFrame.
    - dict_types (dict): Dictionary specifying data types for columns.

    Returns:
    - DataFrame with assigned data types.
    """
    for column in df.columns:
        if column in dict_types['number']:
            df[column] = df[column].astype(float)
        elif column in dict_types['categorical']:
            df[column] = df[column].astype('category')
        else:
            order = np.sort(np.unique((df[column])))
            df[column] = pd.Categorical(df[column], categories=order, ordered=True)
    return df

def dichotomise_YF(df, mapping):
    """
    Dichotomizes a column in the DataFrame based on the specified mapping.
    Done for Yellow Flags (YFs)

    Args:
    - df (DataFrame): Input DataFrame.
    - mapping (dict): Dictionary specifying thresholds for dichotomization.

    Returns:
    - DataFrame with dichotomized column.
    """
    for key, value in mapping.items():
        df[key] = df[key].apply(lambda x: 'negative' if x >= value else 'positive')
    return df

def dichotomize_comi_items(df, mapping):
    """
    Dichotomizes columns in the DataFrame based on the specified mapping.

    Args:
    - df (DataFrame): Input DataFrame.
    - mapping (dict): Dictionary specifying thresholds for dichotomization.

    Returns:
    - DataFrame with dichotomized columns.
    """
    for key, value in mapping.items():
        df[key] = df[key].apply(lambda x: 'bad' if x >= value else 'good')
    return df

def rename_column_values(df, col, mapping):
    """
    Renames column values in the DataFrame based on the specified mapping.

    Args:
    - df (DataFrame): Input DataFrame.
    - col (str): Column name to rename values.
    - mapping (dict): Dictionary specifying new values.

    Returns:
    - DataFrame with renamed column values.
    """
    df[col] = df[col].map(mapping)
    return df

def impute_within_group(group):
    """
    Imputes missing values within each group using forward fill (ffill) and backward fill (bfill).

    Parameters:
    - group (pandas.DataFrame): Group of data containing missing values.

    Returns:
    - pandas.DataFrame: Group with missing values imputed.
    """
    # Fill missing values within each group using forward fill (ffill) and then backward fill (bfill)
    return group.fillna(method='ffill').fillna(method='bfill')

def missing_values(df, columns):
    """
    Calculates and summarizes missing values in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame.

    Returns:
    - pandas.DataFrame: DataFrame summarizing missing values.
    """
    # Calculate the total number of missing values per column
    total_missing = df[columns].isna().sum()

    # Calculate the percentage of missing values per column
    percent_missing = (total_missing / len(df)) * 100

    # Create a new DataFrame to hold the missing data statistics
    missing_data_df = pd.DataFrame({
        "Total number missing": total_missing,  # Total number of missing values
        "Percentage missing": percent_missing   # Percentage of missing values
    })

    # Sort the DataFrame by the total number of missing values in ascending order
    return missing_data_df.sort_values(by="Total number missing", ascending=True)

def prepare_data_for_plotting_duplicates(df, id_column, hue):
    """
    Prepare the data by including the first occurrence of duplicate patients
    and all patients who have only one occurrence.

    Parameters:
    df (pd.DataFrame): The dataframe containing the surgery data.
    id_column (str): The name of the column containing patient identifiers.

    Returns:
    pd.DataFrame: The prepared dataframe for plotting.
    """
    # Mark all duplicates
    df_ = df.copy()
    df_[hue] = df_.duplicated(subset=id_column, keep=False)

    # Get the first occurrence of duplicates
    first_occurrences = df_.loc[~df_.duplicated(subset=id_column, keep='first')]

    return first_occurrences


def map_dataframe_values(df, mappings, ordinal_vars):
    """
    Maps values in specified columns of the DataFrame according to the given mappings.

    For ordinal variables, it renames the column values and specifies their order.
    For 'YFs' column, it dichotomizes the values based on the provided mapping.
    For other columns, it renames the column values.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be processed.
        mappings (dict): A dictionary containing mappings for each column.
        ordinal_vars (list): A list of column names that are ordinal variables.

    Returns:
        pd.DataFrame: The DataFrame with mapped values according to the given mappings.
    """
    for var, mapping in mappings.items():
        # Check if the variable is an ordinal variable
        if var in ordinal_vars:
            # Rename column values
            df = rename_column_values(df, col=var, mapping=mapping)
            # Specify ordering
            custom_order = [value for value in mapping.values()]
            df[var] = pd.Categorical(df[var], categories=custom_order, ordered=True)
        # For 'YFs' column, dichotomize the values
        elif var == 'YFs':
            df = dichotomise_YF(df=df, mapping=mapping)
        # For other columns, rename column values
        else:
            df = rename_column_values(df, col=var, mapping=mapping)

    return df

def prepare_data_for_plotting_missing(df, id_column, hue):
    """
    Prepare the data by including the first occurrence of duplicate patients
    and all patients who have only one occurrence.

    Parameters:
    df (pd.DataFrame): The dataframe containing the surgery data.
    id_column (str): The name of the column for which I want to find missing values.

    Returns:
    pd.DataFrame: The prepared dataframe for plotting.
    """
    # Mark all duplicates
    df_ = df.copy()
    df_[hue] = df_[id_column].isna()

    mapping = {True : 'Yes', False : 'No'}

    df_ = rename_column_values(df_, col=hue, mapping=mapping)

    return df_


