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

    # Assign the new column names to the DataFrame
    # dictionary to specify data types
    types = {'number': ['Age_at_OP',
                        'COMI_Pre_Final',
                        'COMI_3mo_Final',
                        'COMI_12mo_Final',
                        # 'Duration of Episode',
                        'DeltaCOMI',
                        'DeltaCOMI_3mo',
                        'BMI'
                        ],
             'categorical': ['COMI_Problem_loc_pre',
                             'Sex',
                             # 'Age_category',
                             # 'CenterCode',
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

    print()