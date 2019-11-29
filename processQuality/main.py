from pm4py.algo.simulation.tree_generator import factory as tree_gen_factory
from pm4py.objects.process_tree import semantics
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.evaluation import factory as evaluation_factory
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.conversion.log import factory as conversion_factory
from processQuality.TTC import TTC
from pm4py.objects.conversion.process_tree import factory as pt_conv_factory
import os
import pandas as pd
import numpy as np
import time
import copy

def rec_node(stack, leaf, i=0):
    '''
    Recursing the process tree to get the size of the trees and the leafs
    e.g., rec_nodes([nodes],[])
    :param stack: stack of nodes that needs to be visited next. During first call, simply provide the root: [root]
    :param leaf: List of leaf. During first call, simply provide an empty list
    :param i: current number of leaf. 0 at first call
    :return: number_of_leafs (int), leafs (list)
    '''
    c = stack.pop(0)
    stack.extend(c._children)
    if len(c._children) == 0:
        leaf.append(c)
    i += 1

    if len(stack) == 0:
        return i, leaf
    else:
        return rec_node(stack, leaf, i)

def randomly_remove_log(df, noise):
    '''
    Return a reduced version of the dataframe by removing {noise}% of traces.
    It will always cut the last event of the traces and will never entirely cut a trace
    (the number of cases cannot decrease)
    :param df: pandas dataframe with pm4py naming convention
    :param noise: ratio of trace to truncate
    :return: same df with less record
    '''

    # Virtually cutting all the traces
    # i.e., choosing a index to cut for each traces, even if they will not be truncated.
    df['to_remove'] = False
    df['cumcount'] = df.groupby('case:concept:name').cumcount()
    d = df.groupby('case:concept:name')['concept:name'].count()
    df['cut'] = df['case:concept:name'].map(pd.Series([np.random.randint(1,x) for x in d.values], index=d.index).to_dict())

    # Select case index to be truncated
    case = df['case:concept:name'].unique()
    case = np.random.choice(case, int(case.shape[0]*noise))
    df.loc[(df['case:concept:name'].isin(case)) & (df['cumcount']>=df['cut']), 'to_remove'] = True
    df = df.loc[df['to_remove']==False, ['case:concept:name', 'concept:name']].reset_index()
    return df


def run_inductive_and_evaluate(log_for_pd, log_for_eval):
    '''
    Given 2 event logs, it runs the inductive miner and evaluate the result of the process models
    :param log_for_pd: logs that will be used to discover the process models (e.g., might be truncated)
    :param log_for_eval: logs that will be replayed on top of the discovered process model (e.g., will not be truncated = ground truth)
    :return: The evaluation of the process model in a dictionary.
    '''

    # Discover petri net from log
    tree = inductive_miner.apply_tree(log_for_pd)
    net, initial_marking, final_marking = pt_conv_factory.apply(tree, variant=pt_conv_factory.TO_PETRI_NET)

    # Replay the log on top of the discovered process model
    inductive_evaluation_result = evaluation_factory.apply(log_for_eval, net, initial_marking, final_marking)
    inductive_evaluation_result.update(inductive_evaluation_result['fitness'])
    inductive_evaluation_result['discovered_pt'] = str(tree)
    del inductive_evaluation_result['fitness']
    return inductive_evaluation_result

number_of_experiment_to_run = 100
results = []
for _ in range(number_of_experiment_to_run):

    # Generate a random process tree and get traces from the process tree
    tree = tree_gen_factory.apply()
    log = semantics.generate_log(tree, no_traces=1000)
    size_tree, leafs = rec_node([tree], [])
    leafs = set(leafs).union(set('τ'))
    tree = {
        'tree': str(tree),
        'tree_size': size_tree
    }
    csv_exporter.export(log, 'temp.csv')
    original_df = pd.read_csv('temp.csv')
    os.remove('temp.csv')
    log_for_eval = copy.deepcopy(log)

    # Try different level of noise
    for noise_ratio in np.arange(0, 1.01, 0.05):
        noise_ratio = round(noise_ratio,2)

        # Alter the event log
        df = original_df.copy()
        df = randomly_remove_log(df, noise_ratio)

        # The classic approach is simply to apply the inductive miner
        # on the traces that might contain truncated traces.
        result = {'type':'classic', 'noise':noise_ratio}
        log = conversion_factory.apply(df)
        result['time_ttc'] = 0
        result.update(tree)
        now = time.time()
        result.update(run_inductive_and_evaluate(log, log_for_eval, tree['tree']))
        result['time_run_inductive_and_evaluate'] = time.time() - now
        results.append(result)

        # The TTC approach first apply a TTC to the event log
        # before discovering a process model
        now = time.time()
        ttc = TTC(df)
        ttc.train()
        caseTruncated = ttc.predictTruncated()
        ttc_df = df[~df['case:concept:name'].isin(caseTruncated)].copy()
        time_ttc = time.time()-now
        log = conversion_factory.apply(ttc_df)
        result = {'type':'ttc', 'noise':noise_ratio, 'time_ttc':time_ttc}
        result.update(tree)
        now = time.time()
        result.update(run_inductive_and_evaluate(log, log_for_eval, tree['tree']))
        result['time_run_inductive_and_evaluate'] = time.time() - now
        results.append(result)

    # Saving the results
    results_df = pd.DataFrame(results)
    print (results_df.tail(100).to_string())
    results_df.to_csv('results.csv')


