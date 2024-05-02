import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import seaborn as sns
from utilities.utils import prepare_data_for_plotting_duplicates, missing_values, prepare_data_for_plotting_missing
import pandas as pd

def plot_duplicates_trend(duplicates, dir, filename):
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Iterate over unique patients among duplicates
    colors = plt.cm.tab10.colors[:len(duplicates['APROPOS_Patient_No'].unique())]
    legend_handles = []
    for i, (patient, pdata) in enumerate(duplicates.groupby('APROPOS_Patient_No')):
        for therapy, tdata in pdata.groupby('Therapy_Main'):
            # Plotting lines for each duplicate patient
            line, = plt.plot(tdata[['COMI_Pre_Final', 'COMI_3mo_Final', 'COMI_12mo_Final']].values[0],
                             linestyle='-' if therapy == 'Conservative' else '--',
                             color=colors[i],  # Use a different color for each patient
                             label=f'Patient {i + 1} - {patient} - Therapy {therapy}')
            # Add the line to the legend handles
            legend_handles.append(line)

    # Adding x-axis labels
    plt.xticks([0, 1, 2], ['Baseline', '3 Months', '12 Months'])
    # Create legend handles with desired line styles
    legend_handles.append(mlines.Line2D([], [], color='black', linestyle='-', label='Conservative'))
    legend_handles.append(mlines.Line2D([], [], color='black', linestyle='--', label='Surgical'))

    # Adding labels and legend
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.legend(handles=legend_handles, loc = 'center', bbox_to_anchor=(1.05, 0.5))
    plt.title('Trend of Values for Each Patient (Duplicates Only)')

    # Save plot
    plt.grid(True)
    plt.savefig(os.path.join(dir, filename), bbox_inches="tight")

def plot_missing(df, columns, dir, filename):
    mv_variables = missing_values(df, columns)
    # Set the figure size to (6,8)
    plt.figure(figsize=(6, 8))

    # Create a variable to hold the barplot
    fig_mv_variables = sns.barplot(

        # Set x axis to be the percentage of missing values
        x=mv_variables["Percentage missing"].values,

        # Set y axis to be the variables
        y=mv_variables["Percentage missing"].index
    )

    # Set the title and labels
    fig_mv_variables.set(
        title="Missing values per variable",
        xlabel="Percentage of Missing Values",
        ylabel="Variables"
    )

    # Save the figure in the output folder as `11c_mv_variables_barblot.png`
    plt.savefig(os.path.join(dir, filename), bbox_inches="tight")

def plot_variable_with_hue_duplicates(df, id_column, variable, dir, hue='is_duplicate'):
    """
    Plot a given variable for patients, distinguishing between those with multiple surgeries and those with only one.

    Parameters:
    df (pd.DataFrame): The dataframe containing the surgery data.
    id_column (str): The name of the column containing patient identifiers.
    variable (str): The variable/column to plot.
    hue (str): The column to use for color encoding.
    folder (str): Folder path where the plots will be saved.
    """
    prepared_df = prepare_data_for_plotting_duplicates(df, id_column, hue)

    # Check if the variable is categorical or continuous
    if pd.api.types.is_numeric_dtype(prepared_df[variable]):
        # The variable is continuous
        plt.figure(figsize=(10, 6))
        sns.histplot(data=prepared_df, x=variable, hue=hue, kde=True, bins=30, element="step", palette="Set2")
        plt.title(f'Distribution of {variable} by Surgery Count')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
    else:
        # The variable is categorical
        plt.figure(figsize=(10, 6))
        sns.countplot(data=prepared_df, y=variable, hue=hue, palette="Set2")
        plt.title(f'Count of {variable} by Surgery Count')
        plt.xlabel('Count')
        plt.ylabel(variable)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{dir}/{variable}_distribution_by_surgery_count.png')
    plt.close()

def analyze_and_plot_patient_surgeries(df, dir, filename, id_column='KWSno MRN'):
    # Analysis as before
    total_entries = df.shape[0]
    unique_patients = df[id_column].nunique()
    print(f"Total entries: {total_entries}")
    print(f"Unique patients: {unique_patients}")

    duplicate_entries = df[df[id_column].duplicated(keep=False)]
    duplicate_summary = duplicate_entries.groupby(id_column).size().reset_index(name='Count')
    duplicate_summary = duplicate_summary.sort_values(by='Count', ascending=False)

    print(f"Total duplicate entries (patients with more than one surgery): {duplicate_summary.shape[0]}")

    # Plot showing the number of surgeries per patient
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Count', data=duplicate_summary)
    plt.title('Number of Surgeries Per Patient')
    plt.xlabel('Number of Surgeries')
    plt.ylabel('Number of Patients')
    plt.savefig(os.path.join(dir, filename))

    return duplicate_summary

def plot_variable_with_hue_missing(df, id_column, variable, dir, hue):
    """
    Plot a given variable for patients, distinguishing between those with multiple surgeries and those with only one.

    Parameters:
    df (pd.DataFrame): The dataframe containing the surgery data.
    id_column (str): The name of the column containing patient identifiers.
    variable (str): The variable/column to plot.
    hue (str): The column to use for color encoding.
    folder (str): Folder path where the plots will be saved.
    """
    prepared_df = prepare_data_for_plotting_missing(df, id_column, hue)

    # Check if the variable is categorical or continuous
    if pd.api.types.is_numeric_dtype(prepared_df[variable[0]]):
        # The variable is continuous
        plt.figure(figsize=(10, 6))
        medianprops = {'color': 'black', 'linewidth': 2}
        my_pal = {"Yes": "b", "No": "r"}
        # Create a boxplot
        ax = sns.boxplot(data=prepared_df,
                         y = variable[0],
                         x = 'Therapy_Main',
                         hue = hue,
                         linecolor='black',
                         fill=False,
                         gap=0.2,
                         palette=my_pal,
                         showfliers=True,
                         medianprops=medianprops
                         )

        # Change the edge color of the boxes to black
        for i, artist in enumerate(ax.artists):
            artist.set_facecolor('white')

        # iterate over boxes
        # Create a stripplot on the same axes
        # add stripplot with dodge=True
        sns.stripplot(data=prepared_df,
                      y = variable[0],
                      x = 'Therapy_Main',
                      hue = hue,
                      palette=my_pal,
                      dodge=True, ax=ax, size=5, alpha=0.5)

        # remove extra legend handles
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title='COMI 12 months missing')  # , bbox_to_anchor=(1, 1.02), loc='upper left')

        plt.title(f'Distribution of {variable[0]} missing')
        plt.xlabel('Therapy_Main')
        plt.ylabel(variable[1])
    else:
        # The variable is categorical
        plt.figure(figsize=(10, 6))
        sns.countplot(data=prepared_df, x = variable[0], hue=hue, palette="Set2")
        #sns.barplot(data=prepared_df, x=variable, y='Therapy_Main', hue=hue, palette="Set2")
        plt.title(f'Count of {variable[0]} missing')
        plt.ylabel('Count')
        plt.xlabel(variable[0])

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{dir}/{variable[0]}_distribution_missing COMI 12 months.png')
    plt.close()