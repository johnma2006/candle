"""A simplified Tensorboard for metric visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import IPython.display
from enum import Enum
from typing import Dict


class ChartType(Enum):
    SCALAR = 0
    MULTI_SCALAR = 1
    HISTOGRAM = 2
    

class Dashboard:
    
    def __init__(self):
        # Dictionary mapping `chart_label` str to (chart_type, chart_data)
        # chart_type is a ChartType enum
        # chart_data is a dictionary mapping `step` to `value`
        self.data = {}
        self.settings = ChartSettings()
    
    
    def add_scalar(self,
                   chart_label: str,
                   scalar: float,
                   step: int = None):
        """Adds scalar to scalar plot.
        
        Parameters
        ----------
        chart_label
            Identifier of chart.
        scalar
            y-value to plot.
        step
            x-value to plot. If None, then appends to next step.
            
        """
        self._add_data_for_chart_type(ChartType.SCALAR, chart_label, scalar, step)

    
    def add_scalars(self,
                    chart_label: str,
                    scalar_dict: Dict[str, float],
                    step: int = None):
        """Adds scalars to multi-scalar plot.
        
        Parameters
        ----------
        chart_label
            Identifier of chart.
        scalar
            Dict mapping legend label to scalar.
        step
            x-value to plot. If None, then appends to next step.
            
        """
        if chart_label in self.data:
            (chart_type, chart_data) = self.data[chart_label]
            assert chart_type == ChartType.MULTI_SCALAR
            if step in chart_data:
                existing_scalar_dict = chart_data[step]
                scalar_dict = existing_scalar_dict | scalar_dict  # Merge dictionaries
            
        self._add_data_for_chart_type(ChartType.MULTI_SCALAR, chart_label, scalar_dict, step)
    
    
    def add_histogram(self,
                      chart_label: str,
                      values: np.array,
                      step: int = None):
        """Adds histogram to histogram plot.
        
        Parameters
        ----------
        chart_label
            Identifier of chart.
        values
            Values to histogram.
        step
            x-value to plot. If None, then appends to next step.
            
        """
        # Compress values into histograms to save memory
        xlim = self.settings.get('xlim', chart_label)
        if xlim is not None:
            values = [x for x in values if xlim[0] <= x <= xlim[1]]

        (hist, bins) = np.histogram(values,
                                    bins=self.settings.get('histogram_nbins', chart_label))

        self._add_data_for_chart_type(ChartType.HISTOGRAM, chart_label, (hist, bins), step)
        
        
    def plot(self,
             chart_label: str = None,
             clear_output: bool = True):
        """Plots all charts.
        
        Parameters
        ----------
        chart_label
            Label of chart to plot.
            If None, plots all charts.
        clear_output
            Clears output of cells before plotting. Useful for live dashboards.
            
        """
        plt.style.use('seaborn-v0_8-white')

        if chart_label is None:
            chart_labels_to_plot = self.data.keys()
        else:
            chart_labels_to_plot = [chart_label]

        figsize = self.settings.get('figsize', None)
        ncols = self.settings.get('ncols', None)
        fig = plt.figure(figsize=(figsize[0] * ncols, figsize[1] * len(chart_labels_to_plot)))
        for (chart_i, chart_label) in enumerate(chart_labels_to_plot):

            xlim = self.settings.get('xlim', chart_label)
            ylim = self.settings.get('ylim', chart_label)

            (chart_type, chart_data) = self.data[chart_label]

            if chart_type in [ChartType.SCALAR, ChartType.MULTI_SCALAR]:

                ax = fig.add_subplot(len(chart_labels_to_plot), ncols, chart_i + 1)

                if chart_type == ChartType.SCALAR:
                    data_to_plot = pd.Series(chart_data, name=chart_label).to_frame()
                else:  # ChartType.MULTI_SCALAR
                    data_to_plot = pd.DataFrame(chart_data).T

                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for (label, color) in zip(data_to_plot.columns, color_cycle):
                    smoothness = self.settings.get('smoothness', chart_label)
                    rolling_mean = data_to_plot[label].ffill().rolling(smoothness, min_periods=1).mean()

                    rolling_mean.plot(ax=ax, color=color, linewidth=1.0, label=label, alpha=0.8)
                    data_to_plot[label].dropna().plot(ax=ax, color=color, linewidth=1.5,
                                                      alpha=0.1, label='_nolegend_')
                
                xscale = self.settings.get('xscale', chart_label)
                yscale = self.settings.get('yscale', chart_label)
                if xscale == 'log':
                    ax.set_xscale(xscale, base=2)
                else:
                    ax.axvline(0, color='black', linewidth=0.8, zorder=0)
                    ax.set_xscale(xscale)

                if yscale == 'log':
                    ax.set_yscale(yscale, base=2)
                else:
                    ax.axhline(0, color='black', linewidth=0.8, zorder=0)
                    ax.set_yscale(yscale)

                if xlim is not None:
                    ax.set_xlim(*xlim)
                if ylim is not None:
                    ax.set_ylim(*ylim)

                xlabel = self.settings.get('xlabel', chart_label)
                ylabel = self.settings.get('ylabel', chart_label)
                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                ax.grid(linewidth=0.5)
                ax.set_title(chart_label)
                if chart_type == ChartType.MULTI_SCALAR:
                    ax.legend(loc='lower right')

            elif chart_type == ChartType.HISTOGRAM:

                ax = fig.add_subplot(len(chart_labels_to_plot), ncols, chart_i + 1, projection='3d', computed_zorder=False)
                ax.set_box_aspect((figsize[0] / figsize[1] * 2, 2, 1), zoom=1.3)
                ax.view_init(140, 90, 180)

                steps = sorted(list(chart_data.keys()))
                step_size = max(len(steps) // self.settings.get('max_histograms', chart_label), 1)
                steps = steps[::step_size]

                for step in steps:
                    (hist, bins) = chart_data[step]
                    xs = (bins[:-1] + bins[1:]) / 2
                    ax.bar(xs, hist, zs=step, zdir='y', width=1.05 * (xs[1] - xs[0]), alpha=0.8)
                    
                ax.invert_yaxis()
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.zaxis.line.set_color((1.0, 1.0, 1.0, 1.0))
                ax.set_yticks(steps)
                ax.set_zticks([])
                ax.set_title(chart_label)
                if ylim is not None:
                    ax.set_ylim(*ylim)

                hist_xlabel = self.settings.get('hist_xlabel', chart_label)
                hist_ylabel = self.settings.get('hist_ylabel', chart_label)
                if hist_xlabel is not None:
                    ax.set_xlabel(hist_xlabel)
                if hist_ylabel is not None:
                    ax.set_ylabel(hist_ylabel)

        try:  # Sometimes this fails for mysterious reasons when there are too many plots
            plt.tight_layout()
        except:
            pass
            
        try:
            if clear_output:
                IPython.display.clear_output(wait=True)
            plt.show()
        except KeyboardInterrupt:
            pass  # Need to catch this, otherwise dashboard disappears upon KeyboardInterrupt
        
    
    def change_settings(self,
                        setting_name: str,
                        value,
                        chart_label: str = None):
        self.settings.set(setting_name,
                          value,
                          chart_label)
    
    
    def _add_data_for_chart_type(self,
                                 chart_type: ChartType,
                                 chart_label: str,
                                 value: np.array,
                                 step: int):
    
        if chart_label not in self.data:
            self.data[chart_label] = (chart_type, {})
            
        (existing_chart_type, chart_data) = self.data[chart_label]
        if chart_type != existing_chart_type:
            raise ValueError(f'Tried to add {chart_type} data to chart label "{chart_label}", but '
                             f'chart already contains data of type {existing_chart_type}.')
        
        if step is None:
            if len(chart_data) == 0:
                step = 0
            else:
                step = max(chart_data.keys()) + 1
            
        chart_data[step] = value

        
class ChartSettings:
    
    def __init__(self):
        self.global_settings = {
            # General settings
            
            'figsize': (6, 4),
            'ncols': 2,
            'xlim': None,
            'ylim': None,
            
            # SCALAR and MULTI_SCALAR settings
            
            'xscale': 'linear',
            'yscale': 'linear',
            'smoothness': 10,
            'xlabel': 'Step',
            'ylabel': None,
            
            # HISTOGRAM settings
            
            'histogram_nbins': 80,
            'max_histograms': 10,
            'hist_xlabel': None,
            'hist_ylabel': None,
        }
        
        # Dictionary mapping `chart_label` to `settings`
        # `settings` is a dictionary mapping `setting_name` to `value`
        self.chart_settings = {}
        
    
    def set(self,
            setting_name: str,
            value,
            chart_label: str = None):
        """Set setting.
        
        Parameters
        ----------
        setting_name
            Name of setting.
        chart_label
            Chart to set for. If None, then sets global setting.
            
        """
        if setting_name not in self.global_settings:
            raise ValueError(f'\'{setting_name}\' is not a valid setting. Must be one of '
                             f'{list(self.global_settings.keys())}.')
            
        if chart_label is None:
            self.global_settings[setting_name] = value
        else:
            if chart_label not in self.chart_settings:
                self.chart_settings[chart_label] = {}
            self.chart_settings[chart_label][setting_name] = value
    
    
    def get(self,
            setting_name: str,
            chart_label: str):
        """Get setting.
        
        Parameters
        ----------
        setting_name
            Name of setting.

        """
        if chart_label in self.chart_settings and setting_name in self.chart_settings[chart_label]:
            return self.chart_settings[chart_label][setting_name]
        
        return self.global_settings[setting_name]
    