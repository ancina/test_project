from psmpy.plotting import *
import os
import seaborn as sns
import shutil
from matplotlib import style
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from sklearn.metrics import recall_score, auc, roc_curve
from utilities.plotting import (plot_duplicates_trend,
                                plot_variable_with_hue_duplicates,
                                plot_missing,
                                plot_variable_with_hue_missing)
from utilities.utils import (extract_dataframe,
                             dichotomize_comi_items,
                             include_exclude_nan_removal,
                             preprocess,
                             map_dataframe_values)
from utilities.stats_utils import demographics_statistics

style.use("fivethirtyeight")

# Set the font for all elements
plt.rcParams['font.family'] = 'Times New Roman'

def fit_plot_psm(psm_instance, results_dir, replacement, caliper):
    psm_instance.logistic_ps(balance=True)

    #psm_instance.knn_matched_12n(matcher='propensity_score', how_many=2)
    psm_instance.knn_matched(matcher='propensity_score', replacement=replacement, caliper=caliper)

    plt.figure()
    psm_instance.plot_match(Title='Side by side matched controls', Ylabel='Number of patients', Xlabel='Propensity logit',
                   names=['Surgical', 'Conservative'], save=False)
    plt.savefig(os.path.join(results_dir, 'propensity_match.png'))
    plt.figure()
    psm_instance.effect_size_plot(save=False)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.vlines(x = 0.2, ymin = ymin, ymax=ymax, linestyles='--', color = 'r')
    plt.title('Effect size before/after matching')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'effect_size.png'))

    return psm_instance

def extract_matched_groups(df, psm_object):
    indices_for_match = psm_object.matched_ids

    id_conservative = indices_for_match.iloc[:, 0]
    id_surgical = indices_for_match.iloc[:, 1]

    conservative = df.loc[id_conservative]
    surgical = df.loc[id_surgical]

    matched = df.loc[id_conservative.tolist() + id_surgical.to_list()]
    matched['patID'] = matched.index
    matched['Treatment'] = matched['patID'].apply(lambda x: 'Conservative' if x in id_conservative.tolist() else 'Surgical')

    prop_scores = psm_object.df_matched#['propensity_score']

    prop_scores.index = prop_scores['patID']

    matched.loc[prop_scores.index, 'propensity_score'] = prop_scores.loc[prop_scores.index, 'propensity_score']

    prop_scores = prop_scores.loc[id_conservative.tolist() + id_surgical.to_list()]

    return matched, conservative, surgical, id_conservative, id_surgical, prop_scores

def bootstrap_ci(df, variable, n_bootstrap=1000):
    means = []
    for _ in range(n_bootstrap):
        sample = df.sample(len(df), replace=True)
        mean = sample.groupby(['Treatment', variable])['DeltaCOMI'].mean().unstack('Treatment').diff(axis=1).iloc[:, -1].mean()
        means.append(mean)
    return np.percentile(means, [2.5, 97.5])

if __name__ == '__main__':
    # get current working directory
    cwd = os.getcwd()
    # Move one directory up
    parent_dir = os.path.dirname(cwd)
    # create results directory
    results_dir = os.path.join(parent_dir, 'results_matching')
    if (os.path.isdir(results_dir)):
        shutil.rmtree(results_dir)

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
                        #'Duration of Episode',
                        'DeltaCOMI',
                        'DeltaCOMI_3mo',
                        'BMI'
                        ],
             'categorical': ['COMI_Problem_loc_pre',
                             'Sex',
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

    # set the outcome of interest
    outcome = 'DeltaCOMI'
    # specify order for BMI category
    custom_order = ['<20', '20-25', '26-30', '31-35', '>35']
    df['BMI_Category'] = pd.Categorical(df['BMI_Category'], categories=custom_order,
                                             ordered=True)

    # dataframe with variables of interest
    df = extract_dataframe(df=df, dictionary=types)

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

    # drop NaNs
    included, excluded = include_exclude_nan_removal(df, columns = 'all')


    # # preprocess the dataset
    df_final_processed = preprocess(df = included,
                                    numerical_columns = types['number'],
                                    categorical_columns = types['categorical'],
                                    ordinal_columns = types['ordinal']
                                    )
    # add patient id to the dataset to perform PS matching
    df_final_processed['patID'] = df_final_processed.index
    # exclude the 2 possible outcomes
    psm = PsmPy(df_final_processed,
                treatment='Therapy_Main_Surgical',
                indx='patID',
                exclude=['COMI_12mo_Final',
                         'DeltaCOMI',
                         'BMI_Category',
                         'AUC_disability_moderate',
                         'AUC_disability_severe',
                         'AUC_lowbackpain_yes',
                         'COMI_Pre_Final',
                         'COMI_back_pain_pre',
                         'COMI_leg_pain_pre',
                         'COMI_function_work_pre',
                         'COMI_symptoms_pre',
                         'COMI_QoL_pre',
                         'COMI_cutdown_pre',
                         'COMI_offwork_pre'])

    psm = fit_plot_psm(psm_instance = psm,
                       replacement = False,
                       caliper = 0.4,
                       results_dir = results_dir)

    matched, conservative, surgical, id_conservative, id_surgical, prop_scores = extract_matched_groups(included, psm_object=psm)

    stats_after_matching = demographics_statistics(matched,
                                              surgical,
                                              conservative,
                                              columns_of_interest=included.columns,
                                              incl_excl=False)

    for col in matched.columns:
        if col != 'Therapy_Main':
            if matched[col].dtype in ['category', object]:

                sns.boxplot(x=col, y="propensity_score", data=matched)
                plt.title(f"Confounding Evidence for {col}")
                plt.tight_layout()
                plt.show()

    stats_after_matching.to_csv(os.path.join(results_dir, 'statistics_after_matching.csv'))

    median_before = np.median(included['COMI_Pre_Final'])
    median_after = np.median(matched['COMI_Pre_Final'])

    plt.figure()
    sns.boxplot(included, x = "Therapy_Main", y = 'COMI_Pre_Final', fill = False)
    plt.show()

    plt.figure()
    sns.boxplot(matched, x="Treatment", y='COMI_Pre_Final', fill = False)
    plt.show()

    matched['Age_category'] = matched['Age_at_OP']
    median = int(np.median(matched['Age_at_OP']))
    matched['Age_category'] = matched['Age_category'].apply(lambda x: f'<{median}' if x < median else f'>{median}').astype('category')

    matched['Obese'] = matched['BMI']
    matched['Obese'] = matched['Obese'].apply(
        lambda x: 'No' if x < 30 else 'Yes').astype('category')

    matched['COMI_baseline_dichot'] = matched['COMI_Pre_Final']
    median = int(np.median(matched['COMI_Pre_Final']))
    matched['COMI_baseline_dichot'] = matched['COMI_baseline_dichot'].apply(
        lambda x: f'<{median}' if x < median else f'>{median}').astype('category')

    mapping_comi = {'COMI_leg_pain_pre': 4,
                    'COMI_function_work_pre': 4,
                    'COMI_symptoms_pre': 3,
                    'COMI_QoL_pre': 4,
                    }
    matched = dichotomize_comi_items(df=matched, mapping=mapping_comi)

    matched.rename(columns={'COMI_leg_pain_pre': 'AUC_legpain',
                            'COMI_function_work_pre': 'functional',
                            'COMI_symptoms_pre': 'PASS',
                            'COMI_QoL_pre': 'QoL'},
                            inplace=True)

    # media = int(np.nanmedian(matched['COMI_Pre_Final']))
    # matched['COMI_baseline_category'] = matched['COMI_baseline_category'].apply(
    #     lambda x: f'<{media}' if x < media else f'>{media}')

    plt.figure()
    sns.displot(matched, x=outcome, hue='COMI_baseline_dichot')
    plt.show()

    from psmpy.functions import cohenD

    ate_before = np.mean(included.loc[included['Therapy_Main'] == 'Surgical', 'DeltaCOMI']) - np.mean(included.loc[included['Therapy_Main'] == 'Conservative', outcome])

    included_plot = included[['Therapy_Main', outcome]]
    included_plot['Matching'] = 'Before'

    matched_plot = matched[['Treatment', outcome]]
    matched_plot.rename(columns = {'Treatment':'Therapy_Main'}, inplace = True)
    matched_plot['Matching'] = 'After'

    df_incl_match = pd.concat([included_plot, matched_plot], axis = 0)
    plt.figure()
    box = sns.boxplot(x= 'Matching',
                      y=outcome,
                      data=df_incl_match,
                      hue = 'Therapy_Main',
                      fill=False)
    strip = sns.stripplot(x='Matching', y=outcome, hue='Therapy_Main', data=df_incl_match, dodge=True, linewidth=0.5)

    handles, labels = strip.get_legend_handles_labels()
    strip.legend(handles[:2], labels[:2], title='Therapy_Main')

    box.set_title('Comparison before/after matching')
    box.grid(True)
    plt.savefig(os.path.join(results_dir, 'TE_comparison'))

    # plt.figure()
    # sns.boxplot(x='Treatment',
    #             y=outcome,
    #             data=matched,
    #             fill = False)
    # plt.grid()
    # plt.savefig(os.path.join(results_dir, 'TE_after_matching'))

    ate_after = np.mean(surgical[outcome]) - np.mean(conservative[outcome])
    #matched['Age category'] = included.loc[matched.index, 'Age category']



    preds = prop_scores['propensity_score'].apply(lambda x: 0 if x <0.5 else 1)
    true = prop_scores['Therapy_Main_Surgical']

    acc = recall_score(preds, true)
    fpr, tpr, thresholds = roc_curve(true, prop_scores['propensity_score'])
    auc = auc(fpr, tpr)

    prop_scores['COMI_12_months'] = df.loc[prop_scores.index, 'COMI_12mo_Final']
    prop_scores['treatment'] = prop_scores['Therapy_Main_Surgical']
    prop_scores['treatment'] = prop_scores['treatment'].apply(lambda x: 'Surgical' if x == 1 else 'Conservative')
    plt.figure()
    sns.displot(prop_scores, x="propensity_score", hue="treatment")
    plt.xlim([0, 1])
    plt.savefig(os.path.join(results_dir, 'histogram_PS'))
    # sns.lmplot(x='propensity_score',
    #              y='COMI 12 months',
    #              hue='Therapy Main_Surgical',
    #              palette=['b', 'r'],
    #              data=prop_scores)
    #
    # plt.savefig(os.path.join(results_dir, 'regression_plot'))

    # cons_img_features = df_xray.loc[id_conservative]
    # surg_img_features = df_xray.loc[id_surgical]
    #
    # for col in cons_img_features.columns:
    #     t_stat, p_val = stats.ttest_ind(cons_img_features[col], surg_img_features[col], nan_policy='omit')
    #     print(f'For column {col} the p is {p_val}')


    # Group by 'Sex' and calculate the mean 'DeltaCOMI' for the conservative dataset
    #grouping = matched.groupby(['Treatment', 'Sex'])['DeltaCOMI'].mean()

    # Iterate over the categorical variables
    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame()
    for var in matched.columns:
        if matched[var].dtype == 'category' or matched[var].dtype == object:
            if var != 'Therapy_Main' or var != 'Treatment':
                print(var)
                # Group by 'Treatment' and the categorical variable, and calculate the mean and std of 'DeltaCOMI'
                grouping = matched.groupby(['Treatment', var])[outcome].agg(['mean', 'std', 'count'])
                grouping['statistics'] = grouping.apply(lambda row: f'{row["mean"]:.1f} ({row["std"]:.1f}) N = {row["count"]:.0f}', axis=1)

                # Pivot the DataFrame to get 'Treatment' as columns
                pivot = grouping['statistics'].unstack(level=0)

                # Add the name of the column to the row names
                pivot.index = [f'{var}_{i}' for i in pivot.index]

                # Concatenate the results
                results = pd.concat([results, pivot], axis=0)

                plt.figure(figsize=(10, 6))

                # Create a boxplot
                box = sns.boxplot(x=var, y=outcome, hue='Treatment', data=matched, palette="Set1", fill = False)

                # Add a stripplot
                strip = sns.stripplot(x=var, y=outcome, hue='Treatment', data=matched, dodge=True, linewidth=0.5, palette="Set1")

                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                # remove extra legend handles
                handles, labels = strip.get_legend_handles_labels()
                strip.legend(handles[:2], labels[:2], title='Treatment')  # , bbox_to_anchor=(1, 1.02), loc='upper left')

                # Place the legend outside the plot
                #
                plt.grid()
                plt.title(f'{var}')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'plot_{var.split(".")[0]}'))

        results.to_csv(os.path.join(results_dir, 'Effects_categories.csv'))

    # Initialize an empty string to store the formula
    formula = f'{outcome} ~ '

    # Iterate over each predictor variable and add it to the formula along with its interaction with 'Treatment'
    for var in matched.columns:
        if matched[var].dtype == 'category' or matched[var].dtype == object:
            formula += f'{var} + {var}:Treatment + '

    # Remove the extra ' + ' and spaces at the end of the formula
    formula = formula[:-3]

    # Fit the OLS model using the formula
    model = smf.ols(formula=formula, data=matched).fit()

    # Print the summary of the model
    print(model.summary())

    results_all = pd.DataFrame()
    for var in matched.columns:
        if matched[var].dtype == 'category' or matched[var].dtype == object:
            covariates_list = ['COMI_Pre_Final', 'Problem_location_back', 'CenterCode_3', 'CenterCode_6',
                               'Neurogenic_Claudication_Yes', 'Clinically_Relevant_Intstability_Yes',
                               'AUC_legpain_good', 'functional_good', 'QoL_good']
            if var != 'Therapy_Main':
                # if var in covariates_list:
                #     covariates_list.remove(var)
                #     # Perform two-way ANOVA
                #     model = ols(f'DeltaCOMI ~ C({var}) + C({covariates_list[0]}) +C({covariates_list[1]}) +C({covariates_list[2]}) +C({covariates_list[3]}) +C({covariates_list[4]}) +C({covariates_list[5]}) +C({covariates_list[6]}) +C({covariates_list[7]}) + C(Treatment) +  C({var}):C(Treatment)', data=matched).fit()
                # else:
                #     model = ols(
                #         f'DeltaCOMI ~ C({var}) + C({covariates_list[0]}) + C({covariates_list[1]}) + C({covariates_list[2]}) + C({covariates_list[3]}) + C({covariates_list[4]}) + C({covariates_list[5]}) + C({covariates_list[6]}) + C({covariates_list[7]}) + C({covariates_list[8]}) + C(Treatment) +  C({var}):C(Treatment)',
                #         data=matched).fit()
                model = ols(
                    f'{outcome} ~ 1 + C({var}) + C(Treatment) +  C({var}):C(Treatment)',
                    data=matched).fit()

                if var == 'Age_category':
                    print(model.summary())
                    print(matched.groupby(['Age_category', 'Treatment'])[outcome].mean())
                # Extract interaction terms and their corresponding p-values
                interaction_coeffs = model.params.filter(like=':').rename('TE')
                interaction_pvalues = model.pvalues.filter(like=':').rename('P-value')

                # Extract variable names for creating readable interaction names
                var_name = var.split('_')
                var_name = '_'.join(var_name)

                # Extract Treatment levels
                treatment_levels = matched['Treatment'].unique()
                treatment_names = [t.split('_')[-1].lower() for t in treatment_levels]

                # Create readable interaction names
                readable_interactions = []
                for coeff_name in interaction_coeffs.index:
                    var1, var2 = coeff_name.split(':')
                    var1_level = var1.split('[')[1].split(']')[0].split('.')[1]
                    var2_level = var2.split('[')[1].split(']')[0].split('.')[1]
                    readable_interaction = f"{var_name}_{var1_level.lower()}_{var2_level.lower()}"
                    readable_interactions.append(readable_interaction)

                # Combine coefficients and p-values into a DataFrame
                partial = pd.concat([interaction_coeffs, interaction_pvalues], axis=1)
                partial.index = readable_interactions  # Set index to readable interaction names

                # Append to results DataFrame
                results_all = pd.concat([results_all, partial])

        # Reset index of the results DataFrame
        # results_all.reset_index(inplace=True)

        # Reset index of the results DataFrame
        # results_all.reset_index(drop=True, inplace=True)

    print()
    results_all.to_csv(os.path.join(results_dir, 'results_all_anova.csv'))