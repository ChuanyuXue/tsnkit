import warnings

import pandas as pd
import numpy as np
import os
import copy

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_LOGS = pd.read_csv(SCRIPT_DIR + "/data/dataset_logs.csv")

METHOD_ORDER = ['smt_wa', 'smt_nw', 'jrs_wa', 'at', 'jrs_nw_l', 'ls', 'jrs_mc', 'i_ilp', 'i_omt', 'cg', 'jrs_nw',
                'smt_fr', 'cp_wa', 'ls_tb', 'ls_pl', 'smt_pr', 'dt']

marker_dict = {'jrs_wa': 'o', 'jrs_mc': 'o', 'jrs_nw_l': '^', 'jrs_nw': '^', 'ls': 's', 'i_ilp': 's', 'cg': 'v',
               'smt_wa': 'D', 'at': 'D', 'cp_wa': '*', 'smt_pr': '*', 'i_omt': 'p', 'ls_tb': 'p', 'ls_pl': 'p',
               'smt_nw': 'X', 'smt_fr': 'X', 'dt': '.'}

dash_dict = {name: (2, 2) for name in METHOD_ORDER}

ALPHA_REJ = 0.5

_temp_morandi = ["#F0F0F0", "#E0E0E0", "#C0C0C0", "#8B8680", "#808080"]

def get_palette(palette: int):
    import seaborn as sns
    if palette == 0:
        return sns.color_palette(['#B0B1B6', '#BEB1A8', '#8A95A9',
                                  '#99857E', '#686789', '#B77F70',
                                  '#B57C82', '#9FABB9', '#ECCED0',
                                  '#91A0A5', '#E5E2B9', '#88878D',
                                  '#E8D3C0', '#7D7465', '#789798',
                                  '#7A8A71', '#9AA690'])

    return sns.color_palette([
        '#686789', '#B77F70', '#E5E2B9', '#BEB1A8', '#A79A89', '#8A95A9',
        '#ECCED0', '#7D7465', '#E8D3C0', '#7A8A71', '#789798', '#B57C82',
        '#9FABB9', '#B0B1B6', '#99857E', '#88878D', '#91A0A5',
    ])


def get_schedulability(data: pd.DataFrame, var: str):
    data = copy.deepcopy(data)
    data[var] = data["data_id"].map(dict(zip(DATASET_LOGS["id"], DATASET_LOGS[var])))

    group_index = ['name', 'flag', var]
    grouped_data = data.groupby(group_index, as_index=False)['data_id'].count().groupby('flag')

    schedulability = pd.merge(
        left=grouped_data.get_group('successful')[['name', var, 'data_id']].rename(columns={'data_id': 'num_successful'}),
        right=grouped_data.get_group('infeasible')[['name', var, 'data_id']].rename(columns={'data_id': 'num_infeasible'}),
        how='outer',
        on=['name', var]
    )
    schedulability = pd.merge(
        left=schedulability,
        right=grouped_data.get_group('unknown')[['name', var, 'data_id']].rename(columns={'data_id': 'num_unknown'}),
        how='outer',
        on=['name', var]
    )
    schedulability = schedulability.fillna(0)

    # lower bound schedulability
    schedulability['schedulability'] = (schedulability['num_successful']) / (
            schedulability['num_successful'] + schedulability['num_infeasible'] + schedulability['num_unknown'])

    return schedulability


def test_evidence_thres(stat: pd.DataFrame, var: str, confidence=0.9):
    stat_pass = stat[
        stat['num_unknown'] <= (stat['num_successful'] + stat['num_infeasible'] + stat["num_unknown"]) * confidence]
    stat_rej = stat[
        stat['num_unknown'] > (stat['num_successful'] + stat['num_infeasible'] + stat["num_unknown"]) * confidence]

    if stat[var].dtype == 'int' or stat[var].dtype == 'int64':
        var_range = stat[var].unique()
        var_range.sort()
        stat_rej = stat_rej.sort_values(["name", var]).reset_index(drop=True)
        addition_points = []
        for i, row in stat_rej.iterrows():
            var_index = np.where(var_range == row[var])[0][0]
            if (var_index - 1 >= 0
                    and var_range[var_index - 1] not in stat_rej[stat_rej["name"] == row["name"]][var].unique()):
                addition_points.append(
                    stat_pass.loc[
                        (stat_pass["name"] == row["name"]) &
                        (stat_pass[var] == var_range[var_index - 1])
                        ]
                )
            if (var_index + 1 < len(var_range)
                    and var_range[var_index + 1] not in stat_rej[stat_rej["name"] == row["name"]][var].unique()):
                addition_points.append(
                    stat_pass.loc[
                        (stat_pass["name"] == row["name"]) &
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


def draw_streams(df: pd.DataFrame, file_name: str):
    draw_fig4(df, "num_stream", "Number of Streams", file_name)


def draw_bridges(df: pd.DataFrame, file_name: str):
    draw_fig4(df, "num_sw", "Number of Bridges", file_name)


def draw_frames(df: pd.DataFrame, file_name: str):
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
    draw_fig4(df, "num_frame", "Number of Frames", file_name)


def draw_links(df: pd.DataFrame, file_name: str):
    links_list = []
    for piid in DATASET_LOGS["id"]:
        topo = pd.read_csv(SCRIPT_DIR + "/data/" + str(piid) + "_topo.csv")
        links = len(topo["link"])
        links = (links // 50 + 1) * 50  # discreet
        links_list.append(links)
    DATASET_LOGS["num_link"] = links_list
    draw_fig4(df, "num_link", "Number of Links", file_name)


def draw_fig4(df: pd.DataFrame, var: str, graph_name: str, file_name: str):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.rcParams['axes.axisbelow'] = True
    plt.figure(figsize=(3, 2))

    schedulability = get_schedulability(df, var)
    stat_pass, stat_rej = test_evidence_thres(schedulability, var)
    palette = get_palette(0)

    # plot rejected points
    ax = sns.lineplot(data=stat_rej,
                      x=var,
                      y='schedulability',
                      hue="name",
                      style="name",
                      palette=palette,
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
                          data=stat,
                          x=var,
                          y="schedulability",
                          hue="name",
                          style="name",
                          palette=palette,
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
    plt.savefig(file_name + ".pdf", bbox_inches="tight")


def draw_period(df: pd.DataFrame, file_name: str):
    period_dict = {3: "Harmonic Sparse", 4: "Harmonic Dense"}
    DATASET_LOGS["period"] = DATASET_LOGS["period"].apply(lambda x: period_dict[x])
    schedulability = get_schedulability(df, "period")
    draw_fig5(schedulability, "period", list(period_dict.values()), file_name)


def draw_payload(df: pd.DataFrame, file_name: str):
    size_dict = {2: "Small"}
    DATASET_LOGS["size"] = DATASET_LOGS["size"].apply(lambda x: size_dict[x])
    schedulability = get_schedulability(df, "size")
    draw_fig5(schedulability, "size", list(size_dict.values()), file_name)


def draw_deadline(df: pd.DataFrame, file_name: str):
    deadline_dict = {1: "Implicit"}
    DATASET_LOGS["deadline"] = DATASET_LOGS["deadline"].apply(lambda x: deadline_dict[x])
    schedulability = get_schedulability(df, "deadline")
    draw_fig5(schedulability, "deadline", list(deadline_dict.values()), file_name)


def draw_topo(df: pd.DataFrame, file_name: str):
    topo_dict = {0: "Line", 1: "Ring", 2: "Tree", 3: "Mesh"}
    DATASET_LOGS["topo"] = DATASET_LOGS["topo"].apply(lambda x: topo_dict[x])
    schedulability = get_schedulability(df, "topo")
    draw_fig5(schedulability, "topo", ["Line", "Ring"], file_name)


def draw_fig5(df: pd.DataFrame, var: str, hue_order: list, file_name: str):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 1))
    plt.rc('xtick', labelsize=8)
    plt.rcParams['axes.axisbelow'] = True
    ax = sns.barplot(
        data=df,
        y="schedulability",
        x="name",
        hue=var,
        hue_order=hue_order,
        palette=get_palette(1),
        order=METHOD_ORDER
    )
    plt.xlabel('')
    plt.grid(axis='y')
    plt.yticks(np.arange(0, 1.00001, step=0.2))
    plt.ylabel('Schedulable Ratio')
    ax.legend(*remove_duplicate_legend(ax), ncol=6, loc="upper center", prop={'size': 10}, mode="expand",
              bbox_to_anchor=(0.0, 1.4, 1.0, 0), frameon=False)
    plt.savefig(f"{file_name}.pdf", bbox_inches="tight")


def get_comparison_matrix(df: pd.DataFrame):
    index_map = {method: i for i, method in enumerate(df["name"].unique())}
    num_methods = len(index_map)
    group_index = ["data_id"]

    single_data = df[["name", "data_id", "flag"]]
    paired_data = pd.merge(
        left=single_data[single_data["flag"] != "unknown"],
        right=single_data[single_data["flag"] != "unknown"],
        on=group_index
    ).dropna()

    comparison_matrix = np.zeros((num_methods, num_methods))
    all_result_matrix = np.zeros((num_methods, num_methods))

    for i, row in paired_data.iterrows():
        x = row["name_x"]
        y = row["name_y"]
        if (row["flag_x"] == "successful") and (row["flag_y"] == "infeasible"):
            comparison_matrix[index_map[x], index_map[y]] += 1
        all_result_matrix[index_map[x], index_map[y]] += 1

    return comparison_matrix, all_result_matrix


def draw_comparison_matrix(df: pd.DataFrame, file_name: str):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    default_color = "#FFEBCD"
    morandi_cmap = mcolors.LinearSegmentedColormap.from_list("morandi_cmap", _temp_morandi)
    morandi_cmap.set_bad(color=default_color)
    extended_colors = [default_color] + _temp_morandi
    extended_cmap = mcolors.LinearSegmentedColormap.from_list("morandi_cmap", extended_colors)
    comparison_matrix, all_result_matrix = get_comparison_matrix(df)
    comparison_matrix[np.where(comparison_matrix == 0)] = np.nan

    methods = df["name"].unique()
    num_methods = len(methods)
    dominate_matrix = np.empty([num_methods, num_methods], dtype=str)

    for i in range(num_methods):
        for j in range(num_methods):
            sa_ij = comparison_matrix[i][j]
            sa_ji = comparison_matrix[j][i]
            if sa_ij > 0 and np.isnan(sa_ji):
                dominate_matrix[i][j] = '✗'

    means = np.nanmean(np.nan_to_num(comparison_matrix / all_result_matrix), axis=0)
    sorted_indices = np.argsort(means)
    sorted_comparison_matrix = (comparison_matrix / all_result_matrix)[:, sorted_indices][sorted_indices, :]
    sorted_dominate_matrix = dominate_matrix[:, sorted_indices][sorted_indices, :]

    plt.figure()
    sns.heatmap(data=sorted_comparison_matrix,
                xticklabels=[methods[x] for x in sorted_indices],
                yticklabels=[methods[x] for x in sorted_indices],
                cmap=extended_cmap,
                cbar_kws={"label": "Schedulability Advantage"},
                linewidths=1,
                linecolor="white",
                vmin=0
                )
    sns.heatmap(data=sorted_comparison_matrix,
                xticklabels=[methods[x] for x in sorted_indices],
                yticklabels=[methods[x] for x in sorted_indices],
                cmap=morandi_cmap,
                cbar=False,
                linewidths=1,
                linecolor='white',
                vmin=0,
                annot=sorted_dominate_matrix,
                fmt=''
                )

    plt.xticks(rotation=45, ha='right')
    plt.savefig(f"{file_name}.pdf", bbox_inches="tight")


def get_runtime_stat(data: pd.DataFrame, var: str):
    data = data[(data["flag"] != "unknown") | (data["total_mem"] < 4000)]
    data.loc[:, ["total_time"]] = data["total_time"] / 60
    return data.groupby([var, "name"], as_index=False)["total_time"].mean()


def get_memory_stat(data: pd.DataFrame, var: str):
    data = data[(data["flag"] != "unknown") | (data["total_time"] < 7200)]
    return data.groupby([var, "name"], as_index=False)["total_mem"].mean()


def draw_scalability(df: pd.DataFrame, x: str, y: str, x_label: str, y_label: str, file_name: str):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = copy.deepcopy(df)
    df[x] = df["data_id"].map(dict(zip(DATASET_LOGS["id"], DATASET_LOGS[x])))

    plt.rcParams['axes.axisbelow'] = True
    plt.figure(figsize=(3, 2))

    stat = get_runtime_stat(df, x) if y == "total_time" else get_memory_stat(df, x)

    schedulability = get_schedulability(df, x)
    pass_rej = test_evidence_thres(schedulability, x)
    stat_pass = pd.merge(stat, pass_rej[0], on=["name", x])
    stat_rej = pd.merge(stat, pass_rej[1], on=["name", x])
    palette = get_palette(0)

    ax = sns.lineplot(data=stat_pass,
                      x=x,
                      y=y,
                      hue="name",
                      style="name",
                      palette=palette,
                      hue_order=METHOD_ORDER,
                      markers=marker_dict,
                      dashes=False,
                      markeredgecolor=None,
                      fillstyle="none",
                      linewidth=1.2,
                      markersize=6, )

    ax = sns.lineplot(ax=ax,
                      data=stat_rej,
                      x=x,
                      y=y,
                      hue="name",
                      style="name",
                      palette=palette,
                      hue_order=METHOD_ORDER,
                      markers=marker_dict,
                      dashes=dash_dict,
                      alpha=ALPHA_REJ,
                      markeredgecolor=None,
                      fillstyle="none",
                      linewidth=1.2,
                      markersize=6, )

    ax.grid(axis="y")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    legend = plt.legend(prop={"size": 7})
    legend.remove()
    plt.savefig(f"{file_name}.pdf", bbox_inches="tight")


def draw_runtime(df: pd.DataFrame, file_name: str):
    draw_scalability(df, "num_stream", "total_time", "Number of streams", "Runtime (Mins)", f"{file_name}_stream")
    draw_scalability(df, "num_sw", "total_time", "Number of bridges", "Runtime (Mins)", f"{file_name}_bridge")


def draw_mem(df: pd.DataFrame, file_name: str):
    draw_scalability(df, "num_stream", "total_mem", "Number of streams", "Memory (MB)", f"{file_name}_stream")
    draw_scalability(df, "num_sw", "total_mem", "Number of bridges", "Memory (MB)", f"{file_name}_bridge")

def draw_legend():
    import matplotlib.pyplot as plt
    legend_fig = plt.figure(figsize=(10, 2))
    handles = [plt.Line2D([0], [0], color="none", marker=marker_dict[METHOD_ORDER[i]], linestyle="",
                          markersize=7, markeredgecolor=get_palette(0)[i]) for i in range(len(METHOD_ORDER))]
    legend_fig.legend(handles, METHOD_ORDER, loc="center", ncol=9, prop={"size": 8},
                               numpoints=1, handletextpad=0)
    legend_fig.savefig("legend.pdf", bbox_inches="tight")


def draw(path: str, output_affix="./"):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(path)
    draw_streams(df, f"{output_affix}stream")
    draw_bridges(df, f"{output_affix}bridge")
    draw_links(df, f"{output_affix}link")
    draw_frames(df, f"{output_affix}frame")
    draw_topo(df, f"{output_affix}topo")
    draw_period(df, f"{output_affix}period")
    draw_payload(df, f"{output_affix}payload")
    draw_deadline(df, f"{output_affix}deadline")
    draw_comparison_matrix(df, f"{output_affix}comparison_matrix")
    draw_runtime(df, f"{output_affix}runtime")
    draw_mem(df, f"{output_affix}mem")
    draw_legend()


if __name__ == "__main__":
    draw("./results.csv")
