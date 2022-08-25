import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import matplotlib
from shap import plots


from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, balanced_accuracy_score

# load the font properties
font = matplotlib.font_manager.FontProperties(fname="/local/home/sschallmoser/.env_complications/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf")
font1 = matplotlib.font_manager.FontProperties(fname="/local/home/sschallmoser/.env_complications/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf", size=16)

def get_results(population,
                outcome, 
                model, 
                file_ending,
                random_seeds, 
                method, 
                metric='auroc',
                directory='results', 
                multitask=False, 
                return_aurocs=False):
    
    outcomes = np.array(['eyes', 'renal', 'nerves', 'pvd', 'cevd', 'cavd'])
    multitask_idx = np.argmax(outcomes == outcome)
    if multitask == 'real_multitask':
        outcome = 'any'
    scores = []
    scores_all = []
    for random_seed in random_seeds:
        scores1 = []
        results = np.load(f'../prediction/{directory}/results_{population}_{outcome}_{model}_{random_seed}_{file_ending}.npy', 
                          allow_pickle=True).flatten()[0]
        for i in range(5):
            if multitask == 'multitask':
                score = roc_auc_score(results['y_test'][i][:, 0], results['y_pred'][i][:, 0])
            elif multitask == 'real_multitask':
                if len(np.unique(results['y_test'][i][:, multitask_idx])) == 1:
                    print('No outcomes for seed', random_seed, ' and outcome', outcomes[multitask_idx])
                else:
                    score = roc_auc_score(results['y_test'][i][:, multitask_idx], results['y_pred'][i][:, multitask_idx])
            else:
                if metric == 'auroc':
                    score = roc_auc_score(results['y_test'][i], results['y_pred'][i])
                elif metric == 'auprc':
                    precision, recall, thresholds = precision_recall_curve(results['y_test'][i], results['y_pred'][i])
                    score = auc(recall, precision)
                elif metric == 'sensitivity':
                    fpr, tpr, thresholds = roc_curve(results['y_test'][i], results['y_pred'][i])
                    score = tpr[np.argmax(tpr-fpr)]
                elif metric == 'specificity':
                    fpr, tpr, thresholds = roc_curve(results['y_test'][i], results['y_pred'][i])
                    score = 1 - fpr[np.argmax(tpr-fpr)]
                elif metric == 'bacc':
                    fpr, tpr, thresholds = roc_curve(results['y_test'][i], results['y_pred'][i])
                    threshold_optimal = thresholds[np.argmax(tpr-fpr)]
                    y_pred_rounded = [1 if i > threshold_optimal else 0 for i in results['y_pred'][i]]
                    score = balanced_accuracy_score(results['y_test'][i], y_pred_rounded)
            scores1.append(score)
            scores_all.append(score)
        if method == 'minmax':
            scores.append(np.mean(scores1, axis=0))
    
    mean = np.mean(scores_all, axis=0)
    std = np.std(scores_all, axis=0)
    if method == 'minmax':
        min_ = np.min(scores, axis=0)
        max_ = np.max(scores, axis=0)
    if return_aurocs:
        return mean, min_, max_, scores_all
    
    return mean, std

def mann_whitney_u_test(aurocs_1, aurocs_2):
    
    from scipy.stats import mannwhitneyu
    
    p_value = mannwhitneyu(aurocs_1, aurocs_2).pvalue
    
    if p_value >= 0.05:
        stars = 'ns'
        p_value = '=' + str(np.round(p_value, 2))[1:]
    elif 0.01 < p_value < 0.05:
        stars = '*'
        p_value_ = '=' + str(np.round(p_value, 2))[1:]
        if p_value_ == '=.05':
            p_value = '=' + str(np.round(p_value, 3))[1:]
        else:
            p_value = p_value_
    elif 0.001 < p_value < 0.01:
        stars = '**'
        p_value_ = '=' + str(np.round(p_value, 3))[1:]
        if p_value_ == '=.01':
            p_value = '=' + str(np.round(p_value, 4))[1:]
        else:
            p_value = p_value_
    elif p_value < 0.001:
        stars = '***'
        p_value = '<.001'
        
    return stars, p_value

def plot_results(random_seeds,
                 outcomes, 
                 outcomes_long,
                 file_ending,
                 method='minmax',
                 figsize=(14, 8),
                 capsize=4,
                 store=False,
                 y_lim_up=0.84,
                 y_lim_low=0.46):
    
    # Change font type to Arial
    plt.figure(figsize=figsize, facecolor='white')
    
    for population, population_title in zip(['prediabetes', 'diabetes'], ['Prediabetes', 'Diabetes']):
        
        if population == 'prediabetes':
            plt.subplot(2, 1, 1)
            text = 'A'
        else:
            plt.subplot(2, 1, 2)
            text = 'B'
        for model, model_text in zip(['logreg', 'catboost'], ['Logistic regression', 'GBDT']):
            if model == 'logreg':
                offset = -0.1
                marker = 'o'
                color = '#505353'
            else:
                offset = 0.1
                marker = 's'
                color = '#0909EF'
            for idx, outcome in enumerate(outcomes):
                if idx == 5 and population == 'diabetes':
                    label = model_text
                else:
                    label = None
                mean_, min_, max_, aurocs_ = get_results(population=population, 
                                                         outcome=outcome,
                                                         model=model, 
                                                         file_ending=file_ending, 
                                                         random_seeds=random_seeds,
                                                         method=method, 
                                                         multitask=False,
                                                         return_aurocs=True)
                
                mean_logreg, min_logreg, max_logreg, aurocs_logreg = get_results(population=population, 
                                                                                 outcome=outcome,
                                                                                 model='logreg', 
                                                                                 file_ending=file_ending, 
                                                                                 random_seeds=random_seeds,
                                                                                 method=method, 
                                                                                 multitask=False,
                                                                                 return_aurocs=True)

                plt.errorbar(idx+1+offset, mean_, yerr=[[mean_-min_], 
                                                            [max_-mean_]],
                             marker=marker,
                             color=color, capsize=capsize, label=label)
                if model == 'catboost':
                    if mean_ > mean_logreg:
                        result_upper = mean_
                        result_lower = mean_logreg
                    else:
                        result_upper = mean_logreg
                        result_lower = mean_
                    plt.vlines(idx+1+2.7*offset, result_lower, result_upper, color='black')
                    plt.hlines(result_lower, idx+1+2.2*offset, idx+1+2.7*offset+0.007, color='black')
                    plt.hlines(result_upper, idx+1+2.2*offset, idx+1+2.7*offset+0.007, color='black')
                    star, p_value = mann_whitney_u_test(aurocs_logreg, aurocs_)
                    if star == 'ns':
                        y_offset = -0.008
                        y_offset1 = 0.024
                        fontsize_star = 14
                    else:
                        y_offset = 0.008
                        y_offset1 = 0.024
                        fontsize_star = 18
                    plt.text(idx+1+3*offset, (result_lower+result_upper)/2-y_offset,
                             star, fontsize=fontsize_star, fontproperties=font)
                    plt.text(idx+1+3*offset, (result_lower+result_upper)/2-y_offset1,
                             '$\it{P}$'+p_value, fontsize=12, fontproperties=font)

            #plt.ylim(0.48, 0.92)
            plt.ylim(y_lim_low, y_lim_up)
            plt.xticks(np.arange(1, 7), outcomes_long, fontsize=16, alpha=1, rotation=0, fontproperties=font)
            plt.title(population_title, fontsize=18, fontproperties=font)
            plt.grid(True)
            plt.yticks([0.5, 0.6, 0.7, 0.8], alpha=1, fontsize=16, fontproperties=font)
            plt.ylabel('AUROC', fontsize=16, alpha=1, fontproperties=font)
            plt.xlim(0.5, 6.8)
    
        plt.text(0.02, 0.85, text, fontsize=25, fontproperties=font)
    plt.legend(bbox_to_anchor=(0.411, -0.2), fontsize=16, framealpha=1, edgecolor='black', fancybox=False,
               prop=font1, ncol=2)
    plt.subplots_adjust(hspace=0.4)
    
    if store:
        plt.savefig('plots/results.png', dpi=600, bbox_inches='tight')
        plt.savefig('plots/results.pdf', dpi=600, bbox_inches='tight')
        
    return 0

def rename_data(X):
    X_renamed = X.rename({'test=creatinineserum': 'SCr',
                          'test=hba1c(%)': 'HbA1c',
                          'test=glucose': 'Glucose', 
                          'test=calciumserum': 'SCa',
                          'test=eosinophils': 'Eosinophils',
                          'icd9_prefix=250': 'ICD-9: 250',
                          'test=alt(gpt)': 'ALT',
                          'test=potassiumserum': 'SPo',
                          'test=mpv': 'MPV', 
                          'test=vitaminb12': 'Vitamin B12',
                          'icd9_prefix=401': 'ICD-9: 401',
                          'age': 'Age',
                          'sex': 'Sex', 
                          'bmi': 'BMI', 
                          'systolic_bp': 'SBP',
                          'diastolic_bp': 'DBP',
                          'test=mchc': 'MCHC', 
                          'test=bun': 'BUN', 
                          'test=basophils': 'Basophils',
                          'test=hdl-cholest.': 'HDL',
                          'test=ldl-cholest.': 'LDL', 
                          'test=albuminserum': 'SAl',
                          'test=triglycerides': 'Triglycerides',
                          'test=mch': 'MCH',
                          'test=mcv': 'MCV',
                          'test=esr': 'ESR',
                          'test=leucocytesur.': 'Leucocytes',
                          'test=ast(got)': 'AST',
                          'test=cholesterol': 'Cholesterol',
                          'test=ldh': 'LDH',
                          'test=microalbuminconc.': 'Microalbumin',
                          'test=creatinineuconc.': 'UCr',
                          'test=folicacid': 'Folate',
                          'icd9_prefix=786': 'ICD-9: 786',
                          '(med_class = 5)': 'CCB',
                          'test=totalprotein': 'Total protein',
                          'test=pt(seconds)': 'PT (s)',
                          'test=pt(%)': 'PT (%)',
                          'test=urea': 'Urea',
                          'icd9_prefix=780': 'ICD-9: 780',
                          'icd9_prefix=782': 'ICD-9: 782',
                          'test=hemoglobin': 'Hemoglobin',
                          'test=platelets': 'Platelets',
                          'test=hematocrit': 'Hematocrit',
                          '(med_class = 4)': 'BB',
                          'icd9_prefix=724': 'ICD-9: 724',
                          'test=albumin/creatineratio-u': 'UACR', 
                          'test=crp': 'CRP',
                          'test=sodiumserum': 'SSo',
                          'test=ferritin': 'Ferritin', 
                          'test=cpk': 'CPK',
                          '(med_class = 2)': 'Metformin',
                          'test=neutrophils': 'Neutrophils',
                          'icd9_prefix=788': 'ICD-9: 788',
                          'icd9_prefix=466': 'ICD-9: 466',
                          'icd9_prefix=463': 'ICD-9: 463',
                          'icd9_prefix=723': 'ICD-9: 723',
                          'icd9_prefix=729': 'ICD-9: 729',
                          'icd9_prefix=389': 'ICD-9: 389',
                          'icd9_prefix=535': 'ICD-9: 535',
                          'icd9_prefix=272': 'ICD-9: 272',
                          'icd9_prefix=682': 'ICD-9: 682',
                          'icd9_prefix=564': 'ICD-9: 564',
                          'icd9_prefix=719': 'ICD-9: 719',
                          'icd9_prefix=465': 'ICD-9: 465',
                          'test=uricacid': 'UA', 
                          'test=rdw': 'RDW',
                          'test=wbc': 'WBC',
                          '(med_class = 11)': 'Thyroid drug',
                          '(med_class = 15)': 'Alpha-1 blocker',
                          'test=proteineur': 'Proteinuria',
                          }, axis=1)
    return X_renamed

labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Predictor value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}

colors = plots.colors

def summary_plot(population, outcomes, outcomes_long, features_all, file_ending=None, store=False, 
                 feature_names=None, max_display=5, plot_type=None,
                 color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                 color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                 class_inds=None,
                 color_bar_label=labels["FEATURE_VALUE"],
                 # depreciated
                 auto_size_plot=None,
                 use_log_scale=False):
    
    # Change font type to Arial
    plt.rcParams.update({'font.sans-serif':'Arial'})

    plt.figure(figsize=(16, 10), facecolor='white')
    for outcome_long, (idx, outcome) in zip(outcomes_long, enumerate(outcomes)):
        plt.subplot(2, 3, idx+1)
        if file_ending is None:
            shap_values = np.load(f'shap_data/shap_values_{population}_{outcome}.npy', allow_pickle=True)
        else:
            shap_values = np.load(f'shap_data/shap_values_{population}_{outcome}_{file_ending}.npy', allow_pickle=True)
        features = rename_data(features_all[outcome])
        if file_ending == 0:
            features = features.drop('ICD-9: 250', axis=1)
        multi_class = False
        if isinstance(shap_values, list):
            multi_class = True
            if plot_type is None:
                plot_type = "bar" # default for multi-output explanations
            assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
        else:
            if plot_type is None:
                plot_type = "dot" # default for single output explanations
            assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

        # default color:
        if color is None:
            color = colors.blue_rgb

        # convert from a DataFrame or other types
        if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
            if feature_names is None:
                feature_names = features.columns
            features = features.values
        elif isinstance(features, list):
            if feature_names is None:
                feature_names = features
            features = None
        elif (features is not None) and len(features.shape) == 1 and feature_names is None:
            feature_names = features
            features = None

        num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

        if feature_names is None:
            feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

        if use_log_scale:
            plt.xscale('symlog')


        if max_display is None:
            max_display = 20

        if sort:
            # order features by the sum of their effect magnitudes
            if multi_class:
                feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
            else:
                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
            feature_order = feature_order[-min(max_display, len(feature_order)):]
        else:
            feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

        row_height = 0.4
        plt.axvline(x=0, color="#999999", zorder=-1)

        if plot_type == "dot":
            for pos, i in enumerate(feature_order):
                plt.axhline(y=pos, color="grey", lw=0.8, dashes=(1, 5), zorder=-1)
                shaps = shap_values[:, i]
                values = None if features is None else features[:, i]
                inds = np.arange(len(shaps))
                np.random.shuffle(inds)
                if values is not None:
                    values = values[inds]
                shaps = shaps[inds]
                colored_feature = True
                try:
                    values = np.array(values, dtype=np.float64)  # make sure this can be numeric
                except:
                    colored_feature = False
                N = len(shaps)
                # hspacing = (np.max(shaps) - np.min(shaps)) / 200
                # curr_bin = []
                nbins = 100
                quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
                inds = np.argsort(quant + np.random.randn(N) * 1e-6)
                layer = 0
                last_bin = -1
                ys = np.zeros(N)
                for ind in inds:
                    if quant[ind] != last_bin:
                        layer = 0
                    ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                    layer += 1
                    last_bin = quant[ind]
                ys *= 0.9 * (row_height / np.max(ys + 1))

                if features is not None and colored_feature:
                    # trim the color range, but prevent the color range from collapsing
                    vmin = np.nanpercentile(values, 5)
                    vmax = np.nanpercentile(values, 95)
                    if vmin == vmax:
                        vmin = np.nanpercentile(values, 1)
                        vmax = np.nanpercentile(values, 99)
                        if vmin == vmax:
                            vmin = np.min(values)
                            vmax = np.max(values)
                    if vmin > vmax: # fixes rare numerical precision issues
                        vmin = vmax

                    assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                    # plot the nan values in the interaction feature as grey
                    nan_mask = np.isnan(values)
                    plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                               vmax=vmax, s=16, alpha=alpha, linewidth=0,
                               zorder=3, rasterized=len(shaps) > 500)

                    # plot the non-nan values colored by the trimmed feature value
                    cvals = values[np.invert(nan_mask)].astype(np.float64)
                    cvals_imp = cvals.copy()
                    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                    cvals[cvals_imp > vmax] = vmax
                    cvals[cvals_imp < vmin] = vmin
                    plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                               cmap=colors.red_blue, vmin=vmin, vmax=vmax, s=16,
                               c=cvals, alpha=alpha, linewidth=0,
                               zorder=3, rasterized=len(shaps) > 500)
                else:

                    plt.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                               color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)


        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().tick_params(color='black', labelcolor='black')
        plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=14,
                  ha='right', fontproperties=font)
        if plot_type != "bar":
            plt.gca().tick_params('y', length=20, width=0.5, which='major')
        #plt.gca().tick_params('x', labelsize=14)
        #plt.xticks(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], labels=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], fontsize=14)
        plt.ylim(-1, len(feature_order))
        plt.xticks(fontsize=14, fontproperties=font)

        plt.xlabel('SHAP value', fontsize=14, alpha=1, fontproperties=font)
        plt.title(outcome_long, fontsize=16, loc='left', fontproperties=font)
    
    # Draw color bar
    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=colors.red_blue if plot_type != "layered_violin" else plt.get_cmap(color))
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=10)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    for t in cb.ax.get_yticklabels():
        t.set_fontproperties(font)
        t.set_alpha(0.8)
    cb.set_label(color_bar_label, size=14, labelpad=0, alpha=1, fontproperties=font)
    cb.ax.tick_params(labelsize=14, length=0, grid_alpha=1)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    
    
    plt.subplots_adjust(wspace=0.62, hspace=0.25)
    #if population == 'diabetes':
    #    plt.text(-6.4, 12.9, 'B', fontsize=20)
    #else:
    #    plt.text(-6.4, 12.9, 'C', fontsize=20)
    if show:
        plt.show()
        
    if store:
        plt.savefig(f'plots/shap_{population}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'plots/shap_{population}.pdf', dpi=600, bbox_inches='tight')