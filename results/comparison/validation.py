import pandas as pd

results = pd.read_csv('./results.csv')

summary_results = results.groupby(['Model', 'Prototypes', 'Dataset']).agg({ 
    'test_RSq': ['mean', 'std', 'max'],
    'test_SEP': ['mean', 'std', 'max'],
    'test_Err10': ['mean', 'std', 'max'],
    } 
    )

summary = summary_results.round(decimals=4)
summary.to_csv('./summary.csv')