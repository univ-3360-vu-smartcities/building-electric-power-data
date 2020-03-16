from typing import Iterable, Tuple, Dict, Any, Union, List
from itertools import zip_longest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_dataframes(frames: Iterable[pd.DataFrame], ax: plt.Axes,
                       lseries: Iterable[str], rseries: Iterable[str]=(),
                       xlim: Tuple[float]=None, ylim: Tuple[float]=None,
                       xlabel: str='', ylabel: str='', labels: Iterable[str]=None,
                       xscale: str='linear', yscale: str='linear', legend: bool=True,
                       fmt: Union[Iterable[Iterable[str]], Iterable, str]=(('-',), ('--',)),
                       anim_args: Dict[str, Any]={}) \
                       -> FuncAnimation:
    """
    Make an animated line plot from a list of DataFrames.

    Parameters
    ----------
    `frames`:
        A list of DataFrames to plot,
    `ax`:
        The axis on which to animate,
    `lseries`:
        List of column names to plot on the left,
    `rseries`:
        List of column names to plot on the right,
    `xlim, ylim`:
        Tuples of min/max axis values,
    `xlabel, ylabel`:
        String names of each axis. `xlabel` is a single string.
        If `rseries` provided, `ylabel` must be a tuple of two labels,
    `labels`:
        Iterable of string labels for each frame in `frames`,
    `xscale, yscale`:
        Axis scales ('log', 'linear' etc.). If `rseries` provided, yscale
        must be a tuple of two strings,
    `legend`:
        Whether to show legend on plot.
    `fmt`:
        A format string for lines. Either a single string for all lines, or
        an iterable of strings applying to lseries and rseries in order, or an iterable
        of iterables of strings - one iterable for each lseries and rseries.
    `anim_args`:
        A dictionary of arguments to pass to `matplotlib.animation.FuncAnimation`
        class. See https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.animation.FuncAnimation.html
        for more details.
    """
    # generate defaults
    if ylim is None:
        min1 = min(f[lseries].values.min() for f in frames)
        max1 = max(f[lseries].values.max() for f in frames)
        if rseries:
            min2 = min(f[rseries].values.min() for f in frames)
            max2 = max(f[rseries].values.max() for f in frames)
            ylim = ((min1, max1), (min2, max2))
        else:
            ylim = ((min1, max1),)
    elif isinstance(ylim, Iterable):
        if len(ylim) == 2 and not any(map(lambda x: isinstance(x, Iterable), ylim)):
            ylim = (ylim, ylim)

    if isinstance(yscale, str): yscale = (yscale, yscale)
    if isinstance(ylabel, str): ylabel = (ylabel, ylabel)

    if xlim is None:
        xlim = (min(f.index.values.min() for f in frames),
                max(f.index.values.max() for f in frames))

    if rseries:
        if isinstance(fmt, str):
            raise TypeError('Provide format string for rseries.')
        elif isinstance(fmt, Iterable) and all((map(lambda x: isinstance(x, str), fmt))):
            fmt = ((*fmt[:len(lseries)]), (*fmt[len(lseries):]))
    else:
        if isinstance(fmt, str):
            fmt = [fmt] * len(lseries)
        elif isinstance(fmt, Iterable) and all(map(lambda x: isinstance(x, Iterable), fmt)):
            fmt = fmt[0] * len(lseries)

    # set up figure and axes
    fig = ax.get_figure()
    ax1 = ax        # left axis
    ax1.set(ylabel=ylabel[0], yscale=yscale[0], ylim=ylim[0],
            xlabel=xlabel, xscale=xscale, xlim=xlim)
    fig.autofmt_xdate()
    if rseries:     # right axis
        ax2 = ax1.twinx()
        ax2.set(ylabel=ylabel[1], yscale=yscale[1], ylim=ylim[1])

    lines1 = [ax1.plot([], [], f, label=s)[0] for s, f in \
                zip_longest(lseries, fmt[0], fillvalue=fmt[0][-1])]
    if rseries:
        lines2 = [ax2.plot([], [], f, label=s)[0] for s, f in \
                zip_longest(rseries, fmt[1], fillvalue=fmt[1][-1])]
    else:
        lines2 = []

    if legend:
        ax1.legend(lines1+lines2, (*lseries, *rseries), loc='upper right')

    # define animation start and frames
    def init_func():
        return (*lines1, *lines2)

    def plot_func(i):
        frame = frames[i]
        if labels is not None:
            ax1.set_title(labels[i])
        for lines, series in zip((lines1, lines2), (lseries, rseries)):
            for line, s in zip(lines, series):
                line.set_data(frame.index, frame[s])
        return (*lines1, *lines2)

    anim = FuncAnimation(fig, func=plot_func, frames=len(frames),
                         init_func=init_func, **anim_args)
    return anim



def aggregate_over_time(df: pd.DataFrame, period: str='W') \
    -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Convert a dataframe indexed by datetimes into a list of dataframes over a
    period such that each dataframe is contains mean values for that period
    at each time of day. So a datadrame over a year can be converted into a list
    of dataframes divided by week, where each dataframe contains average values
    across days in the week.
    
    Parameters
    ----------
    `df`:
        DataFrame where the index is DatetimeIndex,
    `period`:
        The period over which to aggregate values by time of day. Default 'W' for
        weekly aggregation. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        for more values of this argument.
    
    Returns
    -------
    Tuple
        List of dataframes, and list of string time labels at which each dataframe ends.
    """
    periods = df.resample(period) # iterator over (date week ends, dataframe for week)
    plabels = [ts.isoformat() for ts in periods.groups.keys()]
    means_by_period = []
    for date, subframe in periods:
        grouped = subframe.groupby(subframe.index.time)
        period_mean = grouped.mean()
        period_mean.set_index(pd.to_datetime(period_mean.index, format='%H:%M:%S').time, inplace=True)
        means_by_period.append(period_mean)
    return means_by_period, plabels
