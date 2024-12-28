import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import copy

SCRIPT_DIR = os.path.dirname(__file__)
DATASET_LOGS = pd.read_csv(SCRIPT_DIR + "/data/dataset_logs.csv")

ordered_palette = sns.color_palette(['#B0B1B6', '#BEB1A8', '#8A95A9',
                                     '#99857E', '#686789', '#B77F70',
                                     '#B57C82', '#9FABB9', '#ECCED0',
                                     '#91A0A5', '#E5E2B9', '#88878D',
                                     '#E8D3C0', '#7D7465', '#789798',
                                     '#7A8A71', '#9AA690'])

METHOD_ORDER = ['smt_wa', 'smt_nw', 'jrs_wa', 'at', 'jrs_nw_l', 'ls', 'jrs_mc', 'i_ilp', 'i_omt', 'cg', 'jrs_nw',
                'smt_frag', 'cp_wa', 'ls_tb', 'ls_pl', 'smt_pre', 'dt']

marker_dict = {'jrs_wa': 'o', 'jrs_mc': 'o', 'jrs_nw_l': '^', 'jrs_nw': '^', 'ls': 's', 'i_ilp': 's', 'cg': 'v',
               'smt_wa': 'D', 'at': 'D', 'cp_wa': '*', 'smt_pre': '*', 'i_omt': 'p', 'ls_tb': 'p', 'ls_pl': 'p',
               'smt_nw': 'X', 'smt_frag': 'X', 'dt': '.'}

dash_dict = {name: (2, 2) for name in METHOD_ORDER}

ALPHA_REJ = 0.5


def get_schedulability(data: pd.DataFrame, var: str):
    data = copy.deepcopy(data)
    data[var] = data["data_id"].map(dict(zip(DATASET_LOGS["id"], DATASET_LOGS[var])))

    # count and group data by the flags: "successful", "infeasible", and "unknown"
    group_index = ['name', 'flag', var]
    grouped_data = data.groupby(group_index, as_index=False)['data_id'].count().groupby('flag')

    # merge the groups
    schedulability = pd.merge(
        left=grouped_data.get_group('successful')[['name', var, 'data_id']].rename(
            columns={'data_id': 'num_successful'}),
        right=grouped_data.get_group('infeasible')[['name', var, 'data_id']].rename(
            columns={'data_id': 'num_infeasible'}),
        how='outer',
        on=['name', var]
    )
    schedulability = pd.merge(
        left=schedulability,
        right=grouped_data.get_group('unknown')[['name', var, 'data_id']].rename(
            columns={'data_id': 'num_unknown'}),
        how='outer',
        on=['name', var]
    )
    schedulability = schedulability.fillna(0)

    # calculate lower bound schedulability
    schedulability['schedulability'] = (schedulability['num_successful']) / (
            schedulability['num_successful'] + schedulability['num_infeasible'] + schedulability['num_unknown'])

    return schedulability


def test_evidence_thres(stat: pd.DataFrame, var: str):
    stat_pass = stat[
        stat['num_unknown'] <= (stat['num_successful'] + stat['num_infeasible'] + stat["num_unknown"]) * 0.9]
    stat_rej = stat[
        stat['num_unknown'] > (stat['num_successful'] + stat['num_infeasible'] + stat["num_unknown"]) * 0.9]

    if stat[var].dtype == 'int' or stat[var].dtype == 'int64':
        var_range = stat[var].unique()
        var_range.sort()
        stat_rej = stat_rej.sort_values(['name', var]).reset_index(drop=True)
        addition_points = []
        for i, row in stat_rej.iterrows():
            var_index = np.where(var_range == row[var])[0][0]
            if (var_index - 1 >= 0
                    and var_range[var_index - 1] not in stat_rej[stat_rej['name'] == row['name']][var].unique()):
                addition_points.append(
                    stat_pass.loc[
                        (stat_pass['name'] == row['name']) &
                        (stat_pass[var] == var_range[var_index - 1])
                        ]
                )
            if var_index + 1 < len(var_range) and var_range[var_index + 1] not in \
                    stat_rej[stat_rej['name'] == row['name']][var].unique():
                addition_points.append(
                    stat_pass.loc[
                        (stat_pass['name'] == row['name']) &
                        (stat_pass[var] == var_range[var_index + 1])
                        ]
                )
        stat_rej = pd.concat([stat_rej] + addition_points)

    stat_pass = stat_pass.fillna(0).reset_index(drop=True)
    stat_rej = stat_rej.fillna(0).reset_index(drop=True)

    return stat_pass, stat_rej


def remove_duplicate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    unique_labels = []
    unique_handles = []
    for i in range(len(labels)):
        label = labels[i]
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handles[i])
    return unique_handles[::-1], unique_labels[::-1]


def draw_streams(df: pd.DataFrame):
    draw(df, "num_stream", "Number of Streams")


def draw_bridges(df: pd.DataFrame):
    draw(df, "num_sw", "Number of Bridges")


def draw_frames(df: pd.DataFrame):
    frames_list = []
    for piid in DATASET_LOGS["id"]:
        task = pd.read_csv(SCRIPT_DIR + "/data/" + str(piid) + "_task.csv")
        cycle = np.lcm.reduce(task["period"])
        frames = 0
        for period in task["period"]:
            frames += cycle / period
        frames = np.power(2, np.log2(frames).astype(int))
        frames_list.append(frames)
    DATASET_LOGS["num_frame"] = frames_list
    draw(df, "num_frame", "Number of Frames")


def draw_links(df: pd.DataFrame):
    links_list = []
    for piid in DATASET_LOGS["id"]:
        topo = pd.read_csv(SCRIPT_DIR + "/data/" + str(piid) + "_topo.csv")
        links = len(topo["link"])
        links = (links // 50 + 1) * 50  # discreet
        links_list.append(links)
    DATASET_LOGS["num_link"] = links_list
    draw(df, "num_link", "Number of Links")


def draw(df: pd.DataFrame, var: str, graph_name: str):
    plt.rcParams['axes.axisbelow'] = True
    plt.figure(figsize=(3, 2))

    schedulability = get_schedulability(df, var)
    stat_pass, stat_rej = test_evidence_thres(schedulability, var)

    # plot rejected points
    ax = sns.lineplot(data=stat_rej,
                      x=var,
                      y='schedulability',
                      hue="name",
                      style="name",
                      palette=ordered_palette,
                      hue_order=METHOD_ORDER,
                      markers=marker_dict,
                      dashes=dash_dict,
                      alpha=ALPHA_REJ,
                      markeredgecolor=None,
                      fillstyle='none',
                      linewidth=1.2,
                      markersize=6)

    for stat in list([x[1].reset_index(drop=True) for x in stat_pass.groupby('name')]):
        ax = sns.lineplot(ax=ax,
                          data=stat.fillna(0),
                          x=var,
                          y="schedulability",
                          hue="name",
                          style="name",
                          palette=ordered_palette,
                          hue_order=METHOD_ORDER,
                          markers=marker_dict,
                          dashes=False,
                          markeredgecolor=None,
                          fillstyle='none',
                          linewidth=1.2,
                          markersize=6,
                          )
    ax.grid(axis='y')
    ax.set_ylim(0, 1)
    ax.set_xlabel(graph_name, fontsize=12)
    ax.set_ylabel('Schedulable Ratio', fontsize=12)
    legend = ax.legend(*remove_duplicate_legend(ax), ncol=3, loc="upper center", prop={'size': 10}, mode="expand",
                       bbox_to_anchor=(0.0, 1.4, 1.0, 0), frameon=False)
    legend.remove()
    plt.savefig(var + '.pdf', bbox_inches="tight")
