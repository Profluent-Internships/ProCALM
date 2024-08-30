
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

### Useful functions for plotting and processing data and results###

def make_hierarchical_onehot(levels, maxes):
    """
    levels: list of levels to encode
    maxes: list of maximum values for each level
    Outputs a onehot encoding of the levels using concatenation/
    """
    for i, level in enumerate(levels):
        onehot = torch.zeros(maxes[i], dtype=torch.float32)
        if level != 0: #0 refers to no information at that level, don't encode anything, skips the lowest level which is -1 in taxonomy and - for an empty EC
            onehot[level-1] = 1
        if i == 0:
            out = onehot
        else:
            out = torch.concat((out, onehot), 0)
    return out

def get_average_prediction_similarity(query_ec, reference_ECs, ec2encoding):
    """
    Calculates the average prediction similarity between a query EC and a list of reference ECs.
    The similarity is a cosine similarity using the encodings from ec2encoding.
    """
    #split predicted_ECs if there is a ; character in it
    if query_ec == "no-ec":
        return None
    else:
        #remove predicted_ECs that are not in the keys of ecs
        #predicted_ECs = [ec for ec in predicted_ECs if ec in ec2drfp]
        
        predicted_ec2encoding = {ec: ec2encoding[ec] for ec in reference_ECs if ec in ec2encoding}
        if len(predicted_ec2encoding) == 0:
            return 0
        else:
            fingerprints = torch.stack(list(predicted_ec2encoding.values()))
            return torch.cosine_similarity(ec2encoding[query_ec], fingerprints).max().item()
        
def make_pie_charts(title_list, df_list):
    """
    title_list: list of each title for each pie chart
    df_list: list ofeach dataframe with EC numbers
    Makes pie charts for the distribution of level 2 ECs for each level 1 EC.
    """

    fig, axs = plt.subplots(2, 3, figsize=(7, 6))

    for index, (title, test_df) in enumerate(zip(title_list, df_list)):

        df = pd.DataFrame()
        df['EC'] = test_df['EC number']

        df['EC1'] = df['EC'].str.split('.').str[0]
        df['EC2'] = df['EC'].str.split('.').str[:2].str.join('.')

        distribution = [list(df[df['EC1'] == ec]['EC2'].value_counts().values) for ec in np.sort(df['EC1'].unique())]
        level2_labels = [list(df[df['EC1'] == ec]['EC2'].value_counts().keys()) for ec in np.sort(df['EC1'].unique())]

        #concatenate zeros so that each list is the same length
        max_len = max([len(l) for l in distribution])
        for i, l in enumerate(distribution):
            distribution[i] = l + [0] * (max_len - len(l))
            level2_labels[i] = level2_labels[i] + [''] * (max_len - len(level2_labels[i]))
        #distribution

        sum = np.sum(np.sum(distribution))
        #replace level2 labels with an empty string if the corresponding entry in distribution is too small
        for i, l in enumerate(distribution):
            for j, count in enumerate(l):
                if count/sum < 0.05:
                    level2_labels[i][j] = ''


        size = 0.3
        #vals = np.array([[1000., 32.], [37., 40.], [29., 10.]])
        vals = np.array(distribution)
        i = index // 3
        j = index % 3

        #cmap = plt.colormaps["tab10"]
        #outer_colors = cmap(np.arange(vals.shape[0])*3.7)
        #inner_colors = cmap(np.arange(vals.shape[0]*vals.shape[1]))

        mylabels = np.arange(1, vals.shape[0]+1)

        outer_colors = sns.color_palette("tab10", vals.shape[0])
        axs[i, j].pie(vals.sum(axis=1), radius=1, colors=outer_colors,
            wedgeprops=dict(width=size, edgecolor='w'), labels=mylabels)
        #repeat the outer colors as the inner colors
        inner_colors = []
        for k in range(vals.shape[0]):
            inner_colors = inner_colors + [outer_colors[k]] * vals.shape[1]

        
        # axs[i, j].pie(vals.flatten(), radius=1-size, colors=inner_colors,
        #     wedgeprops=dict(width=size, edgecolor='w'), labels=np.array(level2_labels).flatten(), labeldistance=0.65, )
        #don't include level 2 labels for now
        axs[i, j].pie(vals.flatten(), radius=1-size, colors=inner_colors,
            wedgeprops=dict(width=size, edgecolor='w'))

        axs[i, j].set(aspect="equal", title=title)
    #plt.savefig('figs/pie_charts_task2.png', dpi=500)
    plt.show()
    return fig