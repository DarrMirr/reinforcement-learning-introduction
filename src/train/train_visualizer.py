import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from train.class_train_metrics import TrainMetrics

total_rows = 4
x_label = 'episodes'

def visualize(train_metrics: TrainMetrics):
    df_train_metrics = pd.DataFrame(train_metrics.to_list_of_dictionaries())

    figure = plt.figure(figsize=(10,8))

    __plot_positive_actions(df_train_metrics, figure)
    __plot_action_nums(df_train_metrics, figure)
    __plot_successfully_terminated(df_train_metrics, figure)
    __plot_loss_output(df_train_metrics, figure)
    
    plt.tight_layout()
    plt.show()


def __plot_positive_actions(df_train_metrics: pd.DataFrame, figure: plt.Figure):
    axes = figure.add_subplot(total_rows, 1, 1)

    axes.bar(df_train_metrics.index, df_train_metrics['exploration_action_num'], label='exploration_action_num', width=0.8, align='center')
    axes.bar(df_train_metrics.index, df_train_metrics['exploration_action_positive'], label='exploration_action_positive', width=0.8, align='center')
    axes.set_xlabel(x_label)
    axes.legend()


def __plot_action_nums(df_train_metrics: pd.DataFrame, figure: plt.Figure):
    axes = figure.add_subplot(total_rows, 1, 2)

    axes.bar(df_train_metrics.index, df_train_metrics['explotation_action_num'], label='explotation_action_num', width=0.8, align='center')
    axes.bar(df_train_metrics.index, df_train_metrics['explotation_action_positive'], label='explotation_action_positive', width=0.8, align='center')
    axes.set_xlabel(x_label)
    axes.legend()


def __plot_successfully_terminated(df_train_metrics: pd.DataFrame, figure: plt.Figure):
    df_train_metrics['is_successfully_terminated'] = df_train_metrics['is_successfully_terminated'].map({ True:1, False:0 })
    df_train_metrics['successfully_terminated'] = df_train_metrics['is_successfully_terminated'].cumsum()

    axes = figure.add_subplot(total_rows, 1, 3)

    axes.plot(df_train_metrics.index, df_train_metrics['successfully_terminated'], label='successfully_terminated')
    axes.set_xlabel(x_label)
    axes.legend()


def __plot_loss_output(df_train_metrics: pd.DataFrame, figure: plt.Figure):
    df_train_metrics['loss_fn_output_mean'] = df_train_metrics['loss_fn_output'].apply(np.mean)
    axes = figure.add_subplot(total_rows, 1, 4)

    axes.bar(df_train_metrics.index, df_train_metrics['loss_fn_output_mean'], label='loss_fn_output', width=0.8, align='center')
    axes.set_xlabel(x_label)
    axes.legend()