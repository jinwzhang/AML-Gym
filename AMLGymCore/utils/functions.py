import networkx as nx
import os
import datetime as dt
from pandas import Series
import random as rd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.image import imread
from AMLGymCore.config import conf

def string_to_list(string1: str):
    assert isinstance(string1, str), 'Wrong config input'
    return_list = string1.replace('，', ',').split(sep=',')
    return [str(i).strip().lower() for i in return_list if str(i).strip()]


def column_to_discrete_number(pd_column):
    unique = list(set(pd_column.to_list()))
    unique.sort()
    pair = {i: unique.index(i) for i in unique}
    return pair


def cell_to_discrete_number(cell, pair: dict):
    return pair.get(cell)


# def monetary_diff(ingest_dic: dict, assets_dic: dict):
#     ingested_asset = sum(ingest_dic.values())
#     income = assets_dic.get('income', 0)
#     other_assets = sum(assets_dic.values()) - income
#     total_cost_of_income = ingested_asset - other_assets
#     return total_cost_of_income


def mid_lable(xy, text, xytext):
    xmid = 2/3 * xy[0] + 1/3 + xytext[0]
    ymid = 2/3 * xy[1] + 1/3 * xytext[1]
    return [xmid, ymid]


def nx_plt(points_and_weights, info):

    DG = nx.DiGraph()
    plt.figure(figsize=(15, 10))
    #colors = ['#a6cee3','#1f78b4','#b2df8a']
    colors = ['#708283','#1f78b4','#d2b48c','#b2df8a']
    color_dict = {'placement': colors[0], 'layering': colors[1], 'industry': colors[2], 'integration': colors[3]}
    if info:
        for k, v in points_and_weights.items():
            start, end = str(k).split('->')
            DG.add_node(start, label=node_label(start), color=color_dict.get(node_label(start)))
            DG.add_node(end, label=node_label(end), color=color_dict.get(node_label(end)))

            DG.add_weighted_edges_from([(start, end, int(round(v/1000, 0)))])

        pos = nx.spring_layout(DG, seed=6)
        new_pos = node_position_by_lable(DG, pos)
        colors = []
        for node in DG.nodes():
            colors.append(DG.nodes[node]['color'])

        nx.draw_networkx_nodes(DG, new_pos, node_color=colors)
        nx.draw_networkx_labels(DG, new_pos)
        edge_labels1 = nx.get_edge_attributes(DG, 'weight')
        # weight
        try:
            norm = Normalize(vmin=min(edge_labels1.values()), vmax=max(edge_labels1.values()))
            widths = [max(0.1, norm(weight) * 3) for weight in edge_labels1.values()]
            nx.draw_networkx_edges(DG, new_pos, width=widths, edge_color='k', alpha=0.5, arrowsize=20)
            nx.draw_networkx_edge_labels(DG, new_pos, label_pos=0.2, edge_labels=edge_labels1, rotate=True)
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            plt.text(0.05, 0.95, info, fontdict={'fontsize': 7}, transform=plt.gcf().transFigure)
            if not info.startswith('_'):  # Avoid to save so many repeated pics.
                log_name = info.split('Log Name:')[-1].replace('.log', '')
                plt.savefig((conf.LOG_DIR + os.sep + log_name + os.sep + log_name + '_%s.svg' % dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")), format='svg')
                #plt.savefig((conf.LOG_DIR + os.sep + log_name + os.sep + 'last_img.png'), format='png')
        except ValueError:
            pass




def node_label(node_name):
    if node_name in ('cash'):
        label = 'placement'
    elif node_name in ('income', 'reportable profits', 'bank balance', 'cost'):
        label = 'integration'
    else:
        label = 'layering'
    return label


def node_position_by_lable(DG, pos):
    max_degree = 6   # this is for separating the plt dot
    degrees = {node: min(DG.degree(node), max_degree) for node in DG.nodes}
    try:
        x_min = min(pos[node][0] for node in DG.nodes)
        x_max = max(pos[node][0] for node in DG.nodes)
        same_label_nodes = [node for node in DG.nodes if DG.nodes[node]['label'] == 'layering']
        same_label_nodes.sort(key=lambda x: degrees[x])
        n = len(same_label_nodes)
        x_range = x_max - x_min
        loc_n = 1
        for i, node in enumerate(DG.nodes):
            lable = str(DG.nodes[node]['label']).strip().lower()
            if lable == 'placement':
                pos[node] = (x_min + x_range / 8, 1/2)
            elif lable == 'layering' and n:
                pos[node] = (x_min + x_range * (1/6 + 1/2 * degrees.get(node)/max_degree + 1/10*rd.random(
                ) * rd.choice([-1, 1])), i / n)
            elif lable == 'integration' and n:
                pos[node] = (x_max + 1/5 + (1 + rd.random())/8 * (-1)**loc_n, 1/4 * loc_n)
                loc_n += 1
    except ValueError:
        print('Pos has some error!')
    return pos


def multi_target_loader(string: str) -> dict:
    if string.startswith("{"):
        # data cleaning for Full-width -> Half-width English characters to avoid json decode error
        string = string.replace('“', '"').replace('”', '"').replace("，", ',').replace('：', ':')
        try:
            multi_target_dict = json.loads(string)
        except json.JSONDecodeError:
            raise ValueError('Wrong exchage_taget \"%s\" config, should be JSON' % string)
        return multi_target_dict
    else:
        return {string: 1}


def combine_paid_and_returned(paid:dict, returned:dict):
    the_vector = {}
    total_paid = sum(paid.values())
    for k1, v1 in paid.items():
        for k2, v2 in returned.items():
            key = k1 + '->' + k2
            the_vector[key] = v1/total_paid * v2
    return the_vector


def combine_full_path(the_path_list):
    full_path_and_value = {}
    for tup in the_path_list:
        the_vector = combine_paid_and_returned(tup[0], tup[1])
        for k, v in the_vector.items():
            if k in full_path_and_value.keys():
                full_path_and_value[k] += v
            else:
                full_path_and_value[k] = v

    full_path_and_value = {k: v for k, v in full_path_and_value.items() if v > conf.TRANSACTION_VALUE * 4} # ignore non-significant value

    return full_path_and_value


def show_latest_image(log_name):
    log_name = log_name.replace('.log', '')
    image_dir = conf.LOG_DIR + os.sep + log_name
    img_path = os.path.join(image_dir, 'last_img.png')
    if os.path.isfile(img_path):
        svg = imread(img_path)
        plt.imshow(svg, aspect='auto')
        plt.show()


def obs_value_flat(number):
    if number > 20:
        n = 2
    elif number == 0:
        n = -1
    else:
        n = 1
    return n


def combine_dict_keys(list1):
    l1 = []
    for dic in list1:
        l1.extend(dic.keys())
    set1 = set(l1)
    return Series(list(set1))


def trade_table_preprocess(trade_config_df, group):
    trade_config_df = trade_config_df.copy()
    trade_config_df = trade_config_df[
        trade_config_df['group'].astype(int).isin([0] + group)]  # filter out irrelevant trade_ids
    trade_id_pair, product_name_pair = column_to_discrete_number(
        trade_config_df['trade_id']), column_to_discrete_number(trade_config_df['product_name'])
    trade_config_df.loc[trade_config_df.index, '_trade_id_descre'] = trade_config_df['trade_id'].apply(cell_to_discrete_number, pair=trade_id_pair)
    trade_config_df.loc[trade_config_df.index, 'exchangeable_target'] = trade_config_df['exchangeable_target'].apply(
        lambda s: str(s).lower().strip())


    return trade_config_df