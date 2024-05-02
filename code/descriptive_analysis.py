from psmpy.plotting import *
import os
import shutil
from utilities.plotting import (plot_duplicates_trend,
                                plot_variable_with_hue_duplicates,
                                plot_missing,
                                plot_variable_with_hue_missing)
from utilities.utils import (extract_dataframe,
                             include_exclude_nan_removal,
                             map_dataframe_values)
from utilities.stats_utils import demographics_statistics
from statsmodels.formula.api import ols

if __name__ == '__main__':
    # get current working directory
    cwd = os.getcwd()
    # Move one directory up
    parent_dir = os.path.dirname(cwd)
    # create results directory
    results_dir = os.path.join(parent_dir, 'results_matching')

    os.makedirs(results_dir, exist_ok=True)
    # Read the main dataset
    df = pd.read_excel('../data/Subset_new.xlsx',
                       index_col=0,
                       header=0)
    # replace column names filling the space or dot characters with underscores
    df.columns = df.columns.str.replace(' ', '_').str.replace('.', '_')
    # Filter duplicates. Keep both duplicates
    duplicates = df[df.duplicated(subset=['APROPOS_Patient_No'], keep=False)]

    plot_duplicates_trend(duplicates, dir = results_dir, filename = 'duplicates trend')

    df_copy = df.copy()

    df_copy['KWSno MRN'] = df_copy.index

    plot_variable_with_hue_duplicates(df=df_copy,
                                       id_column='KWSno MRN',
                                       variable='Age_at_OP',
                                       hue='More than 1 surg',
                                       dir = results_dir)


    # Assign the new column names to the DataFrame
    # dictionary to specify data types
    types = {'number': ['Age_at_OP',
                        'COMI_Pre_Final',
                        'COMI_3mo_Final',
                        'COMI_12mo_Final',
                        #'Duration of Episode',
                        'DeltaCOMI',
                        'DeltaCOMI_3mo',
                        'BMI'
                        ],
             'categorical': ['COMI_Problem_loc_pre',
                             'Sex',
                             #'Age_category',
                             #'CenterCode',
                             'Neurol__Abnormality',
                             'Neurogenic_Claudication',
                             'Radicular_Pain',
                             'Foraminal_Stenosis',
                             'Central_Stenosis',
                             'Clinically_Relevant_Instability',
                             'AUC_comorbidity_ALL',
                             'AUC_disability',
                             'AUC_lowbackpain',
                             # 'AUC_legpain',
                             # 'functional',
                             # 'PASS',
                             # 'QoL',
                             'Current_Smoker',
                             'Therapy_Main',
                             'YF_catastrophising',
                             'YF_depression',
                             'YF_anxiety',
                             'YF_fear_avoidance'
                             ],
            'ordinal': ['COMI_back_pain_pre',
                        'COMI_leg_pain_pre',
                        'COMI_function_work_pre',
                        'COMI_symptoms_pre',
                        'COMI_QoL_pre',
                        'COMI_cutdown_pre',
                        'COMI_offwork_pre',
                        'BMI_Category'
                        ]
             }

    # specify order for BMI category
    custom_order = ['<20', '20-25', '26-30', '31-35', '>35']
    df['BMI_Category'] = pd.Categorical(df['BMI_Category'], categories=custom_order,
                                             ordered=True)

    # dataframe with variables of interest
    df = extract_dataframe(df=df, dictionary=types)

    descriptive_dir = os.path.join(results_dir, 'descriptive plots and dataframes')
    # if I have already the directory, I delete it
    if (os.path.isdir(descriptive_dir)):
        shutil.rmtree(descriptive_dir)

    os.makedirs(descriptive_dir, exist_ok=True)

    plot_missing(df, columns = df.columns, dir = descriptive_dir, filename = 'missing_vals')
    #
    mappings = {'YFs' : {'YF_catastrophising':3,
                         'YF_depression':4,
                         'YF_anxiety':4,
                         'YF_fear_avoidance':3},
                'Neurol__Abnormality' : {1:'mild', 2:'moderate', 3:'severe'},
                'AUC_comorbidity_ALL' : {1: 'mild', 2: 'severe'},
                'AUC_disability' : {1: 'mild', 2: 'moderate', 3: 'severe'},
                'AUC_lowbackpain' : {1: 'no', 2: 'yes'},
                'COMI_Problem_loc_pre' : {1: 'back', 2: 'leg', 3:'sensory dist', 4:'None'}
                }

    df = map_dataframe_values(df = df,
                              mappings = mappings,
                              ordinal_vars = ['Neurol__Abnormality', 'AUC_comorbidity_ALL', 'AUC_disability'])

    variable_to_test = {'Age_at_OP':'years', 'Sex':[], 'DeltaCOMI_3mo':'points', 'COMI_3mo_Final':'points'}
    for variable in variable_to_test.items():
        plot_variable_with_hue_missing(df, id_column = 'COMI_12mo_Final', variable = variable, dir = descriptive_dir, hue = 'COMI_12_months_missing')


    # drop NaNs
    no_missing_comi_12, missing_comi_12 = include_exclude_nan_removal(df, columns = ['COMI_12mo_Final'])

    plot_missing(no_missing_comi_12, columns=['COMI_Pre_Final', 'COMI_3mo_Final'], dir=descriptive_dir, filename='missing_vals_COMI_12_NO_missing')
    plot_missing(missing_comi_12, columns=['COMI_Pre_Final', 'COMI_3mo_Final'], dir=descriptive_dir,filename='missing_vals_COMI_12_missing')

    variable_to_test = ['Age_at_OP',
                        'Sex',
                        'COMI_Pre_Final',
                        'BMI',
                        'DeltaCOMI_3mo',
                        'COMI_3mo_Final',
                        'YF_catastrophising',
                        'YF_depression',
                        'YF_anxiety',
                        'YF_fear_avoidance']

    stats_incl_excl = demographics_statistics(df,
                                              no_missing_comi_12,
                                              missing_comi_12,
                                              columns_of_interest = variable_to_test,
                                              name_group_1 = 'NO missing COMI 12mo',
                                              name_group_2 = 'Missing COMI')

    df['COMI_12_months_missing'] = df['COMI_12mo_Final'].isna().astype(int)


    stats_incl_excl.to_csv(os.path.join(descriptive_dir, 'statistics_COMI_12mo_missing_no_missing.csv'))


    surgical = no_missing_comi_12[no_missing_comi_12['Therapy_Main'] == 'Surgical']
    conservative = no_missing_comi_12[no_missing_comi_12['Therapy_Main'] == 'Conservative']

    stats_cons_surg = demographics_statistics(no_missing_comi_12,
                                              surgical,
                                              conservative,
                                              columns_of_interest = variable_to_test,
                                              name_group_1 = 'Surgical NO missing',
                                              name_group_2 = 'Conservative NO missing')

    stats_cons_surg.to_csv(os.path.join(descriptive_dir, 'statistics_surgical_conservative_COMI_12mo_no_missing.csv'))