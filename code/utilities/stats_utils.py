import pandas as pd
from scipy import stats

def test_continuous(df1, df2):
    """
    Performs a statistical test on two continuous variables.

    Parameters:
    - df1, df2 (pandas.DataFrame): The input DataFrames.

    Returns:
    - p_val_1, p_val_2, p_val (float): The p-values of the Shapiro-Wilk test for df1 and df2, and the p-value of the t-test or Mann-Whitney U test.
    """
    # Perform the Shapiro-Wilk test for normality on df1 and df2
    _, p_val_1 = stats.shapiro(df1)
    _, p_val_2 = stats.shapiro(df2)

    # If both distributions are normal (p > 0.05), perform a t-test
    if True:
        t_stat, p_val = stats.ttest_ind(df1, df2, nan_policy='omit')
    else:
        # If either distribution is not normal, perform a Mann-Whitney U test
        u_stat, p_val = stats.ranksums(df1, df2, nan_policy='omit')

    return p_val_1, p_val_2, p_val

def test_categorical(df1, df2):
    """
    Performs a chi-square test of independence on two categorical variables.

    Parameters:
    - df1, df2 (pandas.DataFrame): The input DataFrames.

    Returns:
    - p_val (float): The p-value of the chi-square test.
    """
    # Get the counts of each category in the 'sex' column for both datasets
    counts_df1 = df1.value_counts()
    counts_df2 = df2.value_counts()

    # Create a DataFrame for the contingency matrix
    contingency_table = pd.DataFrame({'Dataset1': counts_df1, 'Dataset2': counts_df2})

    # Fill NaN values with 0 (in case some categories are not present in one of the datasets)
    contingency_table.fillna(0, inplace=True)

    # Perform a chi-square test of independence
    chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)

    return p_val

def demographics_statistics(df, group1, group2, columns_of_interest, name_group_1, name_group_2):
    """
    Computes demographic statistics for a dataset.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - group1, group2 (pandas.DataFrame): DataFrames containing the two groups for comparison.
    - columns_of_interest (list): A list of column names to include in the analysis.
    - incl_excl (bool): Boolean indicating whether to include or exclude data.

    Returns:
    - summary_df (pandas.DataFrame): A DataFrame containing the computed statistics.
    """

    # Use only columns of interest
    df = df[columns_of_interest]

    # Get column names for float64 and object type variables
    float_cols = df.select_dtypes(include=['float64']).columns
    object_cols = df.select_dtypes(include=['object', 'category']).columns

    # Initialize summary dataframe
    summary_df = pd.DataFrame()
    pvals = pd.DataFrame()

    # Compute statistics for float64 type variables
    for col in float_cols:
        mean1, std1 = df[col].mean(), df[col].std()
        mean2, std2 = group1[col].mean(), group1[col].std()
        mean3, std3 = group2[col].mean(), group2[col].std()

        summary_df.loc[f'mean_all (N = {len(df)})', col] = f'{mean1:.2f} ± {std1:.2f}'
        summary_df.loc[f'mean_{name_group_1} (N = {len(group1)})', col] = f'{mean2:.2f} ± {std2:.2f}'
        summary_df.loc[f'mean_{name_group_2} (N = {len(group2)})', col] = f'{mean3:.2f} ± {std3:.2f}'

        p_val_1, p_val_2, p_val = test_continuous(group1[col], group2[col])

        pvals.loc[f'p-value', col] = f'{p_val:.5f}'

    # Compute statistics for object type variables
    for col in object_cols:
        categories = df[col].value_counts().index.values
        proportions_tot = df[col].value_counts() / len(df)
        proportions_tot = '/'.join(round(proportions_tot, 2).astype(str))
        summary_df.loc['proportions all', f'{col} ({"/".join(categories)})'] = proportions_tot
        proportions_included = group1[col].value_counts() / len(group1)
        proportions_included = '/'.join(round(proportions_included, 2).astype(str))
        summary_df.loc[f'proportions {name_group_1}', f'{col} ({"/".join(categories)})'] = proportions_included
        proportions_excluded = group2[col].value_counts() / len(group2)
        proportions_excluded = '/'.join(round(proportions_excluded, 2).astype(str))
        summary_df.loc[f'proportions {name_group_2}', f'{col} ({"/".join(categories)})'] = proportions_excluded

        p_val = test_categorical(group1[col], group2[col])

        pvals.loc['p-value', f'{col} ({"/".join(categories)})'] = p_val

    summary_df = pd.concat([summary_df, pvals])

    return summary_df
