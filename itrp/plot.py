import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
sns.set(style = 'white', font_scale=1.5)

def plot_batch(dfp, batch_col = 'cohort'):
    '''
    dfp: dataframe of samples x genes, with a column contains batch information
    ''' 
    glist = dfp.columns[~dfp.columns.isin([batch_col])]

    if len(glist) == 2:
        df2d = dfp[glist]
    else:
        mp = UMAP(random_state=42, spread=2, verbose=1) #, metric='correlation'
        embed = mp.fit_transform(dfp[glist])
        df2d = pd.DataFrame(embed, columns=['UMAP1', 'UMAP2'],index=dfp.index)
    col1, col2 = df2d.columns
    df2d = df2d.join(dfp[batch_col])
    cohort_order = df2d.groupby(batch_col).mean().mean(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(5,5))
    for bt in cohort_order:
        dfp1 = df2d[df2d[batch_col] == bt]
        ax.scatter(dfp1[col1], dfp1[col2], label = bt, s = 10)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(bottom='on', left='off',  labelleft='on', labelbottom='on', pad=-.6,)
    sns.despine(top=True, right=True, left=False, bottom=False)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    return ax
