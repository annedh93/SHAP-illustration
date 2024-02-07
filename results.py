import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss
import string
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def bootstrap_AUC(n, y, prob_y, curve='ROC'):
    y.index = range(0,len(y))
    prob_y.index = range(0,len(y))
    n_bootstraps = n
    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, prob_y.shape[0], prob_y.shape[0])
        if len(np.unique(y[indices])) < 2:
             # We need at least one positive and one negative sample for ROC AUC
             # to be defined: reject the sample
             continue
        if curve=='ROC':
            x_axis, y_axis, thresholds = roc_curve(y[indices], prob_y[indices])
        elif curve=='PRC':
            y_axis, x_axis, thresholds = precision_recall_curve(y[indices], prob_y[indices])
        score = auc(x_axis, y_axis)
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return(confidence_lower, confidence_upper)

def performance_table(y, p):
    table = pd.DataFrame(columns=['AUROC','AUPRC','calibration int','calibration slope'])
    p[p==1] = 0.9999
    p[p==0] = 0.0001
    # AUROC
    auroc_CI = bootstrap_AUC(1000, y, p, curve='ROC')
    fpr, tpr, _ = roc_curve(y, p)
    # AUPRC
    auprc_CI = bootstrap_AUC(1000, y, p, curve='PRC')
    precision, recall, _ = precision_recall_curve(y, p)
    # calibration
    logit = np.log(p/(1-p))
    df = pd.DataFrame(np.transpose([y,logit]),columns=['y','logit'])
    mod_slope = smf.glm('y~logit', data=df, family=sm.families.Binomial()).fit()
    mod_interc = smf.glm('y~1', data=df, offset=logit, family=sm.families.Binomial()).fit()
    # form table
    table = table.append({
        'AUROC': str(round(auc(fpr, tpr),2)) + ' ' + str(tuple(round(i,2) for i in auroc_CI)),
        'AUPRC': str(round(auc(recall, precision),2)) + ' ' + str(tuple(round(i,2) for i in auprc_CI)),
        'calibration int': str(round(mod_interc.params[0],2)) + ' ' + str(tuple(round(i,2) for i in list(np.array(mod_interc.conf_int(alpha=0.05))[0,:]))),
        'calibration slope': str(round(mod_slope.params[1],2)) + ' ' + str(tuple(round(i,2) for i in list(np.array(mod_slope.conf_int(alpha=0.05))[1,:]))),
        },ignore_index=True)
    return table

def plot_calibration(probabilities):
    group_n = len(probabilities)
    colors = ['#6082B6','#B660AD','#B69460','#60B669']
    fig, axs = plt.subplots(1+group_n, 1, figsize=(6,7), sharex=True, gridspec_kw={'height_ratios': [6]+[2/group_n]*group_n, 'hspace': 0})
    # plot 45 degree line
    axs[0].plot((0,1), (0,1), ls="--", c="#dbdbdb")

    i = 1   
    for c in probabilities:
        index = probabilities[c]['index']
        df = probabilities[c]['data'].loc[index,:]
        y = probabilities[c]['outcome']
        p = probabilities[c]['predictions']
        index = probabilities[c]['index']
        
        # plot calibraiton curves, lowess smoothing
        lowess = sm.nonparametric.lowess
        z = lowess(endog=df[y], exog=df[p], frac=0.5, it=0, return_sorted=True)
        axs[0].plot(z[:,0], z[:,1], color=colors[i-1], label=c)
        
        # plot histograms
        df_event = df[df[y]==1]
        axs[i].hist(df_event[p], range=(0,1), color=colors[i-1], bins=100, bottom=1)
        axs[i].get_yaxis().set_ticks([])
        axs[i].spines['top'].set_color("#dbdbdb")
        i+=1

    # fit the axis
    axs[0].legend(fontsize=14)
    axs[0].set_ylabel("Observed frequency", fontsize=16, labelpad=10)
    axs[int(group_n/2)].set_ylabel("Counts", fontsize=16, labelpad=10)
    axs[int(group_n/2)].yaxis.set_label_coords(-0.1,0)
    plt.xlim(0,1)
    plt.xlabel("Predicted probability", fontsize=18, labelpad=10)
    
    return plt

def decision_curve(probabilities, xlim=[0,1]):
    # make nb table
    thresholds = np.hstack((np.arange(0.01,0.25,0.01),np.arange(0.25,1,0.1)))
    nb = pd.DataFrame(thresholds,columns=['threshold'])

    # cycling through each predictor and calculating net benefit
    for m in probabilities:
        data = probabilities[m]['data']
        N = data.shape[0]
        nb[m] = 0
        y = data.loc[:,probabilities[m]['outcome']]
        event_rate = np.mean(y)
        p = data.loc[:,probabilities[m]['predictions']]
        for ind,t in enumerate(nb.threshold):
            tp = np.sum(y.loc[p>=t]==True)
            fp = np.sum(y.loc[p>=t]==False)
            if np.sum(p>=t)==0:
                tp=fp=0
            nb.iloc[ind,nb.columns.get_indexer([m])] = tp/N-(fp/N)*(t/(1-t))
    
    nb['treat_all'] = event_rate - (1-event_rate)*nb.threshold/(1-nb.threshold)
    nb['treat_none'] = 0

    # Make plot
    ymax = np.max(np.max(nb.loc[:,nb.columns!='threshold']))
    ymin = np.min(np.min(nb.loc[:,(nb.columns!='threshold')&(nb.columns!='treat_all')]))
    plt.figure(figsize=(8,6))
    plt.plot(nb.threshold, nb.treat_all)
    plt.plot(nb.threshold, nb.treat_none)
    for m in probabilities:
        plt.plot(nb.threshold, nb.loc[:,m])
    plt.ylim(bottom=ymin,top=ymax)
    plt.xlim(left=xlim[0],right=xlim[1])
    plt.legend(title='Predictors', labels=['Treat everyone as becoming at risk of depression','Treat no one as becoming at risk of depression']+list(probabilities.keys()))
    plt.xlabel('Decision probability threshold')
    plt.ylabel('Net benefit (% reduction unidentified depression risk)')

    return plt

def performance_table_thresholds(probabilities, thresholds):  
    # Define table characteristics
    index = ['AUC'] + [str(t) + ' sensitivity' for t in thresholds] + [str(t) + ' specificity' for t in thresholds] + [str(t) + ' PPV' for t in thresholds] + [str(t) + ' NPV' for t in thresholds]
    columns = probabilities.keys()
    table = pd.DataFrame(index=index, columns=columns)
    for group in probabilities.keys():
        outcome = probabilities[group]['outcome']
        predictions = probabilities[group]['predictions']
        index = probabilities[group]['index']
        true_y = probabilities[group]['data'].loc[index,outcome]
        prob_y = probabilities[group]['data'].loc[index,predictions]
        # Calculate AUC
        auroc_CI = bootstrap_AUC(1000, true_y, prob_y, curve='ROC')
        fpr, tpr, _ = roc_curve(true_y, prob_y)
        table.loc['AUC',group] = str(round(auc(fpr, tpr),2)) + ' ' + str(tuple(round(i,2) for i in auroc_CI))
        for t in thresholds:
            pred_y = prob_y>t
            table.loc[str(t)+' sensitivity',group] = str(np.round(compute_sensitivity(true_y,pred_y),2)) + ' (' + str((true_y[pred_y==1]==1).sum()) + '/' + str((true_y==1).sum()) + ')'
            table.loc[str(t)+' specificity',group] = str(np.round(compute_specificity(true_y,pred_y),2)) + ' (' + str((true_y[pred_y==0]==0).sum()) + '/' + str((true_y==0).sum()) + ')'
            table.loc[str(t)+' PPV',group] = str(np.round(compute_PPV(true_y,pred_y),2)) + ' (' + str((true_y[pred_y==1]==1).sum()) + '/' + str((pred_y==1).sum()) + ')'
            table.loc[str(t)+' NPV',group] = str(np.round(compute_NPV(true_y,pred_y),2)) + ' (' + str((true_y[pred_y==0]==0).sum()) + '/' + str((pred_y==0).sum()) + ')'
    return table

def compute_sensitivity(y, predictions):
    """Computes the sensitivity of predictions againts the gold labels, y."""
    if sum(y==1)>0: return sum(y[predictions==1]==1)/sum(y==1)
    else: return np.Inf

def compute_specificity(y, predictions):
    """Computes the specificity of predictions againts the gold labels, y."""
    if sum(y==0)>0: return sum(y[predictions==0]==0)/sum(y==0)
    else: return np.Inf

def compute_PPV(y, predictions):
    """Computes the precision of predictions againts the gold labels, y."""
    if sum(predictions==1)>0: return sum(y[predictions==1]==1)/sum(predictions==1)
    else: return np.Inf

def compute_NPV(y, predictions):
    """Computes the precision of predictions againts the gold labels, y."""
    if sum(predictions==0)>0: return sum(y[predictions==0]==0)/sum(predictions==0)
    else: return np.Inf