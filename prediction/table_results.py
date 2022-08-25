import numpy as np
import pandas as pd

from docx import Document
from docx.shared import Cm

from plot_functions import get_results


def results_table(random_seeds,
                  method, 
                  file_ending,
                  directory):
    
    if method == 'minmax':
        return_aurocs = True
    else:
        return_aurocs = False
    populations = ['prediabetes', 'diabetes']
    populations_text = ['Prediabetes', 'Diabetes']
    models = ['logreg', 'catboost']
    model_texts = ['Logistic regression', 'GBDT']
    outcomes = ['eyes', 'renal', 'nerves', 'pvd', 'cevd', 'cavd']
    outcomes_text = ['Retinopathy', 'Nephropathy', 'Neuropathy', 'PVD', 'CeVD', 'CVD']
    document = Document()
    table = document.add_table(0, 9)
    table.style = 'TableGrid'
    table.add_row()
    row = table.rows[0]
    row.cells[0].text = 'Metric'
    row.cells[1].text = 'Population'
    row.cells[2].text = 'Model | Outcome'
    for idx, outcome_text in enumerate(outcomes_text):
        row.cells[idx+3].text = outcome_text
    for metric, metric_text in zip(['auroc', 'auprc', 'sensitivity', 'specificity', 'bacc'],
                                   ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Balanced accuracy']):
        
        for idx, (population, population_text) in enumerate(zip(populations, populations_text)):
            for idx1, (model, model_text) in enumerate(zip(models, model_texts)):
                row_idx = len(table.rows)
                table.add_row()
                row = table.rows[row_idx]
                if idx == 0 and idx1 == 0:
                    row.cells[0].text = metric_text
                row.cells[1].text = population_text
                row.cells[2].text = model_text
                for idx2, outcome in enumerate(outcomes):
                    mean, min_, max_, _ = get_results(population, 
                                                      outcome, 
                                                      model, 
                                                      file_ending,
                                                      random_seeds,
                                                      method=method,
                                                      metric=metric,
                                                      directory=directory,
                                                      return_aurocs=return_aurocs)
                    mean = np.round(mean, 3)
                    min_ = np.round(min_, 3)
                    max_ = np.round(max_, 3)
                    row.cells[idx2+3].text = "%0.3f" % mean + ' (' + "%0.3f" % min_ + ', ' + "%0.3f" % max_ + ')'
    document.save('results.docx')

    return 0