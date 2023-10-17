import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
sns.set(style = 'white', font_scale=1.5)

def plot_embed_with_label(dfp, label_col = ['cancer_type'],  label_type = ['c'], 
                           figsize=(10,10), metric='euclidean',  n_neighbors=32, spread=1, s = 5,
                           cmap = 'hsv'):
    '''
    dfp: dataframe of samples x genes, with a column contains batch information
    label_col: list of labels to be used
    label_type: list of the label types, 'c' for categorical label, 'r' for continous label
    ''' 

    glist = dfp.columns[~dfp.columns.isin(label_col)]
    if len(glist) == 2:
        df2d = dfp[glist]
    else:
        mp = UMAP(random_state=42, spread=spread, n_neighbors=n_neighbors, metric=metric, verbose=1) #, metric='correlation'
        embed = mp.fit_transform(dfp[glist])
        df2d = pd.DataFrame(embed, columns=['UMAP1', 'UMAP2'],index=dfp.index)
    col1, col2 = df2d.columns
    df2d = df2d.join(dfp[label_col])
    
    figs = []
    for label, t in zip(label_col, label_type):
        fig, ax = plt.subplots(figsize=figsize) 
        if t == 'c': 
            cohort_order = df2d.groupby(label).size().sort_values().index
            colors = sns.color_palette("bright", len(cohort_order)).as_hex()
            for bt, c in zip(cohort_order, colors):
                dfp1 = df2d[df2d[label] == bt]
                ax.scatter(dfp1[col1], dfp1[col2], label = bt, s = s,  c = c)
                #print(color)
            if len(cohort_order) <= 10:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend(loc='center left', ncol=3, prop={'size':7},  bbox_to_anchor=(1, 0.5))
        else:
            ax.scatter(df2d[col1], df2d[col2], label = label, s = s, c = df2d[label], cmap = cmap)        
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
        ax.tick_params(bottom='on', left='off',  labelleft='on', labelbottom='on', pad=-.6,)
        sns.despine(top=True, right=True, left=False, bottom=False)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(label)
        
        figs.append(fig)
    return figs