import os
import itertools
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import datetime as dt
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from lidarpy.preprocessing.lidar_preprocessing import preprocess
from lidarpy.preprocessing.lidar_preprocessing_tools import (
    apply_dead_time_correction,
)
from lidarpy.preprocessing.gluing_proportional import gluing
from lidarpy.general_utils.dates import numpy_to_datetime
from lidarpy.general_utils.numerics import residuals

plt.ion()


def plot_hist_num_measurements(year=None):
    """

    :return:
    """

    if year is not None:
        year_ini = year
        year_end = year
    else:
        year_ini = 2018
        year_end = 2020

    date_ini = dt.datetime(year_ini, 1, 1)
    date_end = dt.datetime(year_end, 12, 31)
    dates_ls = pd.date_range(date_ini, date_end)
    num_meas = np.zeros(len(dates_ls))
    for i, idate in enumerate(dates_ls):
        print(idate)
        i_fn = os.path.join(
            ".",
            "dead_time",
            "dead_time_%s_532p_ngt_scc_1.nc" % idate.strftime("%Y%m%d"),
        )
        if os.path.isfile(i_fn):
            try:
                ds = xr.open_dataset(i_fn)
                # corrijo un fallo de seleccion de medidas
                times_a = ds.times.values
                sun = solar.SUN(times_a, -3.61, 37.16, elev=680)
                csza = sun.get_csza()
                idx_ilu = np.where(csza < -0.01)
                times_ilu = times_a[idx_ilu]
                ds = ds.sel(times=times_ilu)
                num_meas[i] = ds.dims["times"]
                ds.close()
            except:
                print("%s: no data" % idate)
    df = pd.DataFrame.from_dict({"times": dates_ls, "num": num_meas})
    plt.close("all")
    f, ax = plt.subplots(figsize=(8.5, 3.75), constrained_layout=True)
    ax.bar(df.times, df.num)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.axes.set_xlim("2019-01-01", "2019-12-31")
    ax.axes.set_xlabel("Date")
    ax.axes.set_ylabel("num. of measurements")
    if year is not None:
        ax.set_title("MULHACEN: measurements along %s" % year)
        plt.savefig("lidar_measurements_%s.png" % (year,), dpi=300)
    else:
        ax.set_title("MULHACEN: measurements along %s-%s" % (year_ini, year_end))
        plt.savefig("lidar_measurements_%s_%s.png" % (year_ini, year_end), dpi=300)

    return df


def plot_J_lines(fn=None, ds=None):
    """ """
    if fn is not None:
        ds = xr.open_dataset(fn)
    dx = ds.isel(times=10)

    colors = mpl.cm.jet(np.linspace(0, 1, ds.dims["steps"]))
    custom_cycler = cycler(color=colors)
    cm = mpl.cm.get_cmap("jet", ds.dims["steps"])

    plt.close("all")
    f, ax = plt.subplots(constrained_layout=True)
    ax.set_prop_cycle(custom_cycler)
    lines = [
        np.column_stack([dx.tau.values, np.log(dx.J).values[:, i]])
        for i in range(dx.dims["steps"])
    ]
    lc = LineCollection(lines, cmap=cm, lw=3)
    lc.set_array(dx.range_glue_extended_thick[0, :].values)
    ax.add_collection(lc)
    axcb = f.colorbar(lc, ax=ax)

    x_label = r"dead time [ns]"
    y_label = r"residual, log(J)"
    hue_label = r"region thickness [m]"
    plot_title = r"J vs $\tau$ vs linear response region thickness"

    ax.axes.set_ylim(np.log(dx.J).min(skipna=True), np.log(dx.J).max(skipna=True))
    ax.axes.set_xlim(dx.tau.min(skipna=True) - 0.1, dx.tau.max(skipna=True) + 0.1)

    label_size = 12
    tick_label_size = 11

    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.set_title(plot_title)
    axcb.set_label(hue_label, fontsize=label_size)
    axcb.ax.tick_params(labelsize=tick_label_size)

    plt.savefig("J_lines.png", dpi=300)


def plot_J_contour(fn=None, ds=None):
    """ """
    if fn is not None:
        ds = xr.open_dataset(fn)
    dx = ds.isel(times=10)

    plt.close("all")
    f, ax = plt.subplots(constrained_layout=True)
    cs = ax.contourf(
        np.tile(dx.tau, [dx.dims["steps"], 1]).T,
        dx.range_glue_extended_thick,
        np.log(dx.J),
        cmap="jet",
    )
    axcb = f.colorbar(cs)

    x_label = r"dead time [ns]"
    y_label = r"region thickness [m]"
    hue_label = r"residual, log(J)"
    plot_title = r"J vs $\tau$ vs linear response region thickness"

    ax.axes.set_xlim(dx.tau.min(skipna=True) - 0.1, dx.tau.max(skipna=True) + 0.1)

    label_size = 12
    tick_label_size = 11

    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.set_title(plot_title)
    axcb.set_label(hue_label, fontsize=label_size)
    axcb.ax.tick_params(labelsize=tick_label_size)

    plt.savefig("J_contour.png", dpi=300)


def plot_scatter_dead_time(fn=None, ds=None):
    """
    SCATTER ESTIMATED DEAD TIME AS DEPENDENT ON REGION THICKNESS
    """

    if fn is not None:
        ds = xr.open_dataset(fn)

    date_ini = ds.times[0].values
    date_end = ds.times[-1].values
    date_ini = numpy_to_datetime(date_ini)
    date_end = numpy_to_datetime(date_end)
    delta_t = (date_end - date_ini).seconds / 3600
    plot_title = "%s: %.1f hours. %i profiles" % (
        date_ini.strftime("%Y%m%d"),
        delta_t,
        ds.dims["times"],
    )
    x_label = r"dead time [ns]"
    y_label = r"region thickness [m]"
    hue_label = r"r Pearson (Glued - Photon-counting)"

    label_size = 12
    tick_label_size = 11

    # height limits
    # h_limits = np.arange(0, 7500, 500)
    color_map = "jet"
    hue_limits = (0.95, 1)
    hue_step = 0.005
    num_colors = round((hue_limits[1] - hue_limits[0]) / hue_step)

    f, ax = plt.subplots(constrained_layout=True)
    # plot
    for i in range(ds.dims["times"]):
        cs = ds.isel(times=i).plot.scatter(
            ax=ax,
            x="tau_min",
            y="range_glue_extended_thick",
            hue="rvalue_gl_pc",
            cmap=mpl.cm.get_cmap(color_map, num_colors),
            vmin=hue_limits[0],
            vmax=hue_limits[1],
            extend="min",
            add_guide=False,
        )  # type: ignore
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.set_title(plot_title)
    # colorbar
    cb = plt.colorbar(cs)
    # cb.ax.locator_params(nbins=num_colors)
    # cb.ax.set_title('hola')
    cb.set_label(hue_label, fontsize=label_size)
    cb.ax.tick_params(labelsize=tick_label_size)

    plt.savefig("dead_time_distribution_region_thickness.png", dpi=300)

    ds.close()


def plot_dead_time_hist(fn=None, ds=None, label=None):
    """ """
    if fn is not None:
        ds = xr.open_dataset(fn)

    date_ini = ds.times[0].values
    date_end = ds.times[-1].values
    date_ini = numpy_to_datetime(date_ini)
    date_end = numpy_to_datetime(date_end)
    delta_t = (date_end - date_ini).seconds / 3600
    plot_title = "%s: %.1f hours. %i profiles" % (
        date_ini.strftime("%Y%m%d"),
        delta_t,
        ds.dims["times"],
    )
    label_size = 12
    tick_label_size = 11

    f, ax = plt.subplots(constrained_layout=True)
    ds.tau_best_avg.plot.hist(ax=ax, bins=ds.tau, color="royalblue")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.axes.set_xlabel("Dead Time [ns]", fontsize=label_size)
    ax.axes.set_ylabel("Counts", fontsize=label_size)
    ax.set_title(plot_title)  # , fontsize=16)
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    # l1 = mpatches.Patch(color='royalblue') #, label="day")
    # l1_t = mpatches.Patch(color='w', label="\n".join((r"$\mu = %.1f$" % (ds_day.tau_min.mean(), ), r"$\sigma = %.1f$" % (ds_day.tau_min.std(),))))
    l1_t = mpatches.Patch(
        color="royalblue",
        label="\n".join(
            (
                r"$\mu = %.1f$" % (ds.tau_best_avg.mean(),),
                r"$\sigma = %.1f$" % (ds.tau_best_avg.std(),),
            )
        ),
    )
    ax.legend(handles=[l1_t], fontsize=label_size)

    if label is not None:
        plt_fn = "dead_time_hist_%s.png" % label
    else:
        plt_fn = "dead_time_hist.png"
    plt.savefig(plt_fn, dpi=300)

    ds.close()


def plot_dead_time_ts(year=None, df=None):
    """ """

    if df is None:
        if year is not None:
            year_ini = year
            year_end = year
        else:
            year_ini = 2018
            year_end = 2020

        date_ini = dt.datetime(year_ini, 1, 1)
        date_end = dt.datetime(year_end, 12, 31)
        dates_ls = pd.date_range(date_ini, date_end)
        tau_532p_avg = np.zeros(len(dates_ls)) * np.nan
        tau_532p_std = np.zeros(len(dates_ls)) * np.nan
        tau_532c_avg = np.zeros(len(dates_ls)) * np.nan
        tau_532c_std = np.zeros(len(dates_ls)) * np.nan
        tau_355_avg = np.zeros(len(dates_ls)) * np.nan
        tau_355_std = np.zeros(len(dates_ls)) * np.nan
        for i, idate in enumerate(dates_ls):
            print(idate)
            i_fn = os.path.join(
                ".",
                "dead_time",
                "dead_time_%s_532p_ngt_scc_1.nc" % idate.strftime("%Y%m%d"),
            )
            # if "20191010" in i_fn:
            #    pdb.set_trace()
            if os.path.isfile(i_fn):
                try:
                    ds = xr.open_dataset(i_fn)
                    tau_532p_avg[i] = ds.tau_best_avg.mean(skipna=True)
                    tau_532p_std[i] = ds.tau_best_avg.std(skipna=True)
                    ds.close()
                except:
                    print("%s: no data for 532p" % idate)
            i_fn = os.path.join(
                ".",
                "dead_time",
                "dead_time_%s_532c_ngt_scc_1.nc" % idate.strftime("%Y%m%d"),
            )
            if os.path.isfile(i_fn):
                try:
                    ds = xr.open_dataset(i_fn)
                    tau_532c_avg[i] = ds.tau_best_avg.mean(skipna=True)
                    tau_532c_std[i] = ds.tau_best_avg.std(skipna=True)
                    ds.close()
                except:
                    print("%s: no data for 532c" % idate)
            i_fn = os.path.join(
                ".",
                "dead_time",
                "dead_time_%s_355_ngt_scc_1.nc" % idate.strftime("%Y%m%d"),
            )
            if os.path.isfile(i_fn):
                try:
                    ds = xr.open_dataset(i_fn)
                    tau_355_avg[i] = ds.tau_best_avg.mean(skipna=True)
                    tau_355_std[i] = ds.tau_best_avg.std(skipna=True)
                    ds.close()
                except:
                    print("%s: no data for 355" % idate)

        df = pd.DataFrame.from_dict(
            {
                "times": dates_ls,
                "tau_532p_avg": tau_532p_avg,
                "tau_532p_std": tau_532p_std,
                "tau_532c_avg": tau_532c_avg,
                "tau_532c_std": tau_532c_std,
                "tau_355_avg": tau_355_avg,
                "tau_355_std": tau_355_std,
            }
        )
    # plot
    plt.close("all")
    f, ax = plt.subplots(figsize=(8.5, 3.75), constrained_layout=True)
    ax.plot(df.times, df.tau_532p_avg, lw=0, label="532p", color="b", marker="o")
    # ax.fill_between(df.times, df.tau_532p_avg + df.tau_532p_std, df.tau_532p_avg - df.tau_532p_std, facecolor='b', alpha=0.5)
    ax.plot(df.times, df.tau_532c_avg, lw=0, label="532c", color="r", marker="o")
    # ax.fill_between(df.times, df.tau_532c_avg + df.tau_532c_std, df.tau_532c_avg - df.tau_532c_std, facecolor='r', alpha=0.5)
    ax.plot(df.times, df.tau_355_avg, lw=0, label="355", color="k", marker="o")
    # ax.fill_between(df.times, df.tau_355_avg + df.tau_355_std, df.tau_355_avg - df.tau_355_std, facecolor='k', alpha=0.5)
    label_size = 12
    tick_label_size = 11

    ax.legend(fontsize=label_size)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.axes.set_xlim("2019-01-01", "2019-12-31")
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    ax.set_xlabel("Date", fontsize=label_size)
    ax.set_ylabel("dead time [ns]", fontsize=label_size)
    if year is not None:
        ax.set_title("MULHACEN: dead time estimates. Year %s" % year)
        plt.savefig("dead_time_ts_%s.png" % (year,), dpi=300)
    else:
        ax.set_title(
            "MULHACEN: dead time estimates. Period %s-%s" % (year_ini, year_end)
        )
        plt.savefig("dead_time_ts_%s_%s.png" % (year_ini, year_end), dpi=300)

    return df


#        f, ax = plt.subplots(constrained_layout=True)
#        cs = ax.contourf(np.tile(tau, [n_steps, 1]).T, range_glue_ext_thick[0, :, :],
#                         J[0, :, :], levels=np.arange(0, 0.1, 0.005), cmap='jet')
#        plt.colorbar(cs)
#
#        # Plot rvalue vs thickness vs tau
#        f, ax = plt.subplots(constrained_layout=True)
#        cs = ax.contourf(np.tile(tau, [n_steps, 1]).T, range_glue_ext_thick[0, :, :],
#                         rvalue[0, :, :], levels=np.arange(0.8, 1.01, 0.01), cmap='jet')
#        plt.colorbar(cs)
#
#        # Plot J vs tau for every step
#        colors = mpl.cm.jet(np.linspace(0, 1, n_steps))
#        custom_cycler = cycler(color=colors)
#        f, ax = plt.subplots(constrained_layout=True)
#        ax.set_prop_cycle(custom_cycler)
#        ax.plot(tau, J[0, :, :])
#
#        # Plot tau_min vs thickness with color (min_range, rvalue of tau_min, num of profile)
#        x = []
#        y = []
#        c_thick = []
#        c_min_rvalue = []
#        c_profile = []
#        for i in range(n_t):
#            for j in range(n_tau):
#                if len(x) == 0:
#                    x = tau_min[i, :]
#                    y = range_glue_ext_min[i, j, :]
#                    c_thick = range_glue_ext_thick[i, j, :]
#                    c_min_rvalue = r_min[i, :]
#                    c_profile = np.tile(i, n_steps)
#                else:
#                    x = np.append(x, tau_min[i, :])
#                    y = np.append(y, range_glue_ext_min[i, j, :])
#                    c_thick = np.append(c_thick, range_glue_ext_thick[i, j, :])
#                    c_min_rvalue = np.append(c_min_rvalue, r_min[i, :])
#                    c_profile = np.append(c_profile, np.tile(i, n_steps))
#        colors_min_range = mpl.cm.get_cmap("jet", round((2000-0)/200))
#        f, ax = plt.subplots(constrained_layout=True)
#        cs = ax.scatter(x, y, marker='+', c=c_thick, cmap=colors_min_range, vmin=0, vmax=2000)
#        cb = plt.colorbar(cs)
#        cb.ax.locator_params(nbins=round((2000-0)/200))
#
#        colors_min_rvalue = mpl.cm.get_cmap("jet", round((1-0.9)/0.01))
#        f, ax = plt.subplots(constrained_layout=True)
#        cs = ax.scatter(x, y, marker='+', c=c_min_rvalue, cmap=colors_min_rvalue, vmin=0.9, vmax=1)
#        plt.colorbar(cs)
#
#        colors_profiles = mpl.cm.get_cmap("jet", n_t)
#        f, ax = plt.subplots(constrained_layout=True)
#        cs = ax.scatter(x, y, marker='+', c=c_profile, cmap=colors_profiles, vmin=-0.5, vmax=n_t-0.5)
#        plt.colorbar(cs)


def estimate_dead_time_old(
    date_str, channel, don, time_range=None, gluing_type="scc", expand_bins=True
):
    """ """

    # Group results
    if don == 1:
        don_str = "day"
    elif don == 2:
        don_str = "ngt"
    elif don == 0:
        don_str = "all"
    else:
        don_str = "_".join([str(t) for t in time_range])

    try:
        t_start = time.time()
        print(
            "Estimate Dead Time for %s. channel=%s. illum=%i" % (date_str, channel, don)
        )
        # get raw data
        (
            an,
            pc,
            an_dc,
            ranges,
            times,
            bg_range,
            rs_range,
            idx_bg_range,
            idx_rs_range,
            rs_ds,
            dc_ds,
            an_ch,
            pc_ch,
            wv,
            pol,
        ) = lidar_read.get_prepared_data(
            date_str, channel, don, 15000, time_range=time_range
        )
        # analog channel additional info
        channel_id = np.logical_and.reduce(
            (
                rs_ds.wavelength == wv,
                rs_ds.polarization == pol,
                rs_ds.detection_mode == 0,
            )
        )
        adc_range = 1000 * rs_ds.adc_range.values[channel_id][0]
        adc_bits = rs_ds.adc_bits.values[channel_id][0]
        n_res = 1

        # test 10 profiles
        # n_p_test = 5
        # times = times[:n_p_test]
        # an = an[:n_p_test, :]
        # pc = pc[:n_p_test, :]

        # taus [in nanoseconds]
        tau_step = 0.1  # 0.05
        tau = np.arange(0, 10 + tau_step, tau_step)

        # expand glue region
        bin_step = 10
        bin_max = 150
        exp_glue = np.arange(0, bin_max + bin_step, bin_step) * 7.5

        # Previous info for preprocessing signals
        bz_an = lidar.get_bin_zero(wv, pol, 0, ref_time=dt.datetime(2020, 5, 28))
        bz_pc = lidar.get_bin_zero(wv, pol, 1, ref_time=dt.datetime(2020, 5, 28))
        bg_an = lidar.estimate_raw_background(an, ranges, bg_range=bg_range)
        bg_pc = lidar.estimate_raw_background(pc, ranges, bg_range=bg_range)
        dc, dc_bg = lidar.preprocessing_dc_signal(an_dc, ranges, bz_an, bg_range)

        # re-dimension signals with range limits
        an = an[:, idx_rs_range]
        pc = pc[:, idx_rs_range]
        dc = dc[idx_rs_range]
        ranges = ranges[idx_rs_range]

        # dimensions
        n_t = len(times)
        n_ranges = len(ranges)
        n_tau = len(tau)
        n_steps = len(exp_glue)

        # Define variables for output
        tau_best_avg = np.zeros(n_t)
        tau_best_std = np.zeros(n_t)
        thickness_best_avg = np.zeros(n_t)
        thickness_best_std = np.zeros(n_t)
        tau_min = np.zeros((n_t, n_steps)) * np.nan
        r_min = np.zeros((n_t, n_steps)) * np.nan
        r_min2 = np.zeros((n_t, n_steps)) * np.nan
        thickness_minJ = np.zeros((n_t, n_steps)) * np.nan
        J = np.zeros((n_t, n_tau, n_steps)) * np.nan
        range_glue_first_min = np.zeros((n_t, n_tau)) * np.nan
        range_glue_first_max = np.zeros((n_t, n_tau)) * np.nan
        range_glue_min = np.zeros((n_t, n_tau)) * np.nan
        range_glue_max = np.zeros((n_t, n_tau)) * np.nan
        an_pp = np.zeros((n_t, n_ranges)) * np.nan
        pc_pp = np.zeros((n_t, n_ranges, n_tau)) * np.nan
        gl_pp = np.zeros((n_t, n_ranges, n_tau)) * np.nan
        rvalue = np.zeros((n_t, n_tau, n_steps)) * np.nan
        rvalue2 = np.zeros((n_t, n_tau, n_steps)) * np.nan
        range_glue_ext_min = np.zeros((n_t, n_tau, n_steps)) * np.nan
        range_glue_ext_max = np.zeros((n_t, n_tau, n_steps)) * np.nan
        range_glue_ext_thick = np.zeros((n_t, n_tau, n_steps)) * np.nan

        # Loop over signals
        for i in range(n_t):
            # preprocess analog
            an_i = lidar.preprocessing_analog_signal(
                an[i, :], bz_an, bg_an[i], dc, dc_bg
            )
            an_pp[i, :] = an_i

            # Loop over taus
            for j in range(n_tau):
                # preprocess photoncounting
                pc_ij = lidar.preprocessing_photoncounting_signal(
                    pc[i, :], tau[j], bz_pc, bg_pc[i]
                )
                pc_pp[i, :, j] = pc_ij

                # gluing
                if gluing_type == "scc":  # D'Amico
                    gl_ij, c_an, c_ph, idxs, glued = lidar.gluing(
                        an_i,
                        pc_ij,
                        ranges,
                        adc_range,
                        adc_bits,
                        n_res=n_res,
                        pc_threshold=20,
                        correlation_threshold=0.9,
                        range_threshold=(1000, 5000),
                        min_points=15,
                        slope_threshold=2,
                        stability_threshold=1,
                        step=5,
                        use_photon_as_reference=True,
                    )
                    if glued:
                        # ranges of gluing region
                        range_glue_first_min[i, j] = ranges[idxs[0]]
                        range_glue_first_max[i, j] = ranges[idxs[1]]
                        range_glue_min[i, j] = ranges[idxs[4]]
                        range_glue_max[i, j] = ranges[idxs[5]]
                        gl_pp[i, :, j] = gl_ij
                else:  # method dbp
                    print("Method dbp to be implemented")

            # Cost Function: para cada perfil uso el mismo rango para todos los taus
            min_glue_i = np.nanmax(range_glue_min[i, :])
            max_glue_i = np.nanmin(range_glue_max[i, :])
            for j in range(n_tau):
                for k in range(n_steps):
                    # anchura de region
                    min_glue = min_glue_i - exp_glue[k]
                    max_glue = max_glue_i
                    idx_glue = np.logical_and(ranges >= min_glue, ranges <= max_glue)
                    # rangos
                    range_glue_ext_min[i, j, k] = min_glue
                    range_glue_ext_max[i, j, k] = max_glue
                    range_glue_ext_thick[i, j, k] = max_glue - min_glue

                    # correlación AN vs PC en la region
                    rvalue[i, j, k] = np.corrcoef(
                        an_pp[i, idx_glue], pc_pp[i, idx_glue, j]
                    )[0, 1]
                    rvalue2[i, j, k] = np.corrcoef(
                        gl_pp[i, idx_glue, j], pc_pp[i, idx_glue, j]
                    )[0, 1]
                    if np.logical_and(
                        ~np.isnan(rvalue2[i, j, k]), rvalue2[i, j, k] >= 0.9
                    ):
                        if rvalue[i, j, k] >= 0.9:
                            J[i, j, k] = utils.residuals(
                                pc_pp[i, idx_glue, j], gl_pp[i, idx_glue, j]
                            )
            # Tau that Minimizes Cost Function and associated rvalues for an_pc and gl_pc
            for k in range(n_steps):
                jj = J[i, :, k]
                rr = rvalue[i, :, k]
                rr2 = rvalue2[i, :, k]
                th = range_glue_ext_thick[i, :, k]
                try:
                    if ~(np.isnan(jj).any()):
                        idx_Jmin = np.nanargmin(jj)
                        tau_min[i, k] = tau[idx_Jmin]
                        r_min[i, k] = rr[idx_Jmin]
                        r_min2[i, k] = rr2[idx_Jmin]
                        thickness_minJ[i, k] = th[idx_Jmin]
                except Exception as e:
                    print(str(e))
                    print(
                        "Tau min not estimated for profile %i-th, step %i-th" % (i, k)
                    )

            # CRITERIUM TO ESTIMATE DEAD TIME
            # for every thickness, a best tau is estimated above.
            # So, we have a list of best taus. from them, we derive the best
            # of the best, considering the correlation index gl_pc
            try:
                idx_best_tau = r_min2[i, :] >= (
                    np.nanmax(r_min2[i, :]) - np.nanstd(r_min2[i, :])
                )
                tau_best_avg[i] = np.nanmean(tau_min[i, idx_best_tau])
                tau_best_std[i] = np.nanstd(tau_min[i, idx_best_tau])
                thickness_best_avg[i] = np.nanmean(thickness_minJ[i, idx_best_tau])
                thickness_best_std[i] = np.nanstd(thickness_minJ[i, idx_best_tau])
            except Exception as e:
                print(str(e))
                print("Best Tau not estimated for profile %i-th" % i)

        # save results in netcdf
        ds = xr.Dataset(
            {
                "J": (["times", "tau", "steps"], J),
                "tau_min": (["times", "steps"], tau_min),
                "tau_best_avg": (["times"], tau_best_avg),
                "tau_best_std": (["times"], tau_best_std),
                "thickness_best_avg": (["times"], thickness_best_avg),
                "thickness_best_std": (["times"], thickness_best_std),
                "range_glue_extended_min": (
                    ["times", "tau", "steps"],
                    range_glue_ext_min,
                ),
                "range_glue_extended_max": (
                    ["times", "tau", "steps"],
                    range_glue_ext_max,
                ),
                "range_glue_extended_thick": (
                    ["times", "tau", "steps"],
                    range_glue_ext_thick,
                ),
                "rvalue_an_pc": (["times", "tau", "steps"], rvalue),
                "rvalue_gl_pc": (["times", "tau", "steps"], rvalue2),
            },
            coords={"times": times, "tau": tau, "steps": exp_glue, "ranges": ranges},
        )
        fn = os.path.join(
            config.dead_time_dn,
            "dead_time_%s_%s_%s_%s_%i.nc"
            % (date_str, channel, don_str, gluing_type, expand_bins),
        )
        ds.to_netcdf(fn)
        time_elapsed = time.time() - t_start
        print(
            "%.1f s of time elapsed for %s:%s (%i profiles). channel=%s"
            % (time_elapsed, date_str, don_str, ds.dims["times"], channel)
        )

    except Exception as e:
        print("ERROR: In study_dead_time. %s" % (str(e)))
        ds = None
        fn = ""

    return ds, fn


def run_period(date_ini, date_end, expand_step=True):
    """
    Perform Dead Time Estimation for a given period

    Params:
    ------
    date_ini (str, yyyymmdd):
    date_end (str, yyyymmdd):
    expand_step (bool):
        estimate for several thicknesses of gluing region
    """

    channel = ["532p", "532c", "355"]
    don = [1, 2]
    dates_str = [x.strftime("%Y%m%d") for x in pd.date_range(date_ini, date_end)]
    inp = [
        x
        for x in itertools.product(
            dates_str, channel, don, [None], ["scc"], [expand_step]
        )
    ]
    for x in inp:
        print(x)
        estimate_dead_time(*x)


if __name__ == "__main__":

    #    # Study of Impact of Gluing Range
    #    date_str = "20190108" #"20190516" #"20200702"
    #    channel = "532p"  # "532c" "355" "532p"
    #    don = 1
    #    time_range = None
    #    #time_range = (21, 22) #None# None (10, 11)
    #    ds, fn = estimate_dead_time(date_str, channel, don, time_range=time_range)
    #    #fn = "./dead_time/dead_time_20190516_20190516_532p_21_22_scc.nc"
    #    #fn = "./dead_time/dead_time_20200702_20200702_532p_21_22_scc.nc"
    #    #fn = "./dead_time/dead_time_20200702_20200702_532p_21_22_scc_1.nc"
    #    #fn = "./dead_time/dead_time_20200702_20200702_355_21_22_scc_1.nc"
    # fn = "./dead_time/dead_time_20200702_20200702_532c_21_22_scc_1.nc"
    # ds = study_impact_of_range_selection_on_dead_time_estimation(fn)

    # Run Period
    date_ini = "20190101"
    date_end = "20201031"
    run_period(date_ini, date_end, expand_step=True)

#    # Test Time Interval
#    date_ini = "20200702" #"20190516" #"20200702"
#    date_end = "20200702" #"20190516" #"20200702"
#    channel = "532p"  # "532c" "355" "532p"
#    don = -1
#    time_range = (21, 22) #None# None (10, 11)
#    x = estimate_dead_time(date_ini, date_end, channel, don, time_range=time_range, expand_step=True)

#    # Test Estimate Dead Time Period
#    date_ini = "20200528"
#    date_end = "20200528"
#    result = run_study_dead_time(date_ini, date_end)

#    # Test Gluing
#    date_str = "20200528"
#    an, pc, an_dc, ranges, times, bg_range, rs_range, idx_bg_range, idx_rs_range, \
#    rs_ds, dc_ds, an_ch, pc_ch, wv, pol = lidar_test.get_prepared_data(
#        date_str, "532p", -1, 15000, time_range=(10, 11))
#    # adc_bits, adc_range
#    channel_id = np.logical_and.reduce((rs_ds.wavelength == wv,
#                                        rs_ds.polarization == pol,
#                                        rs_ds.detection_mode == 0))
#    adc_range = 1000*rs_ds.adc_range[channel_id].values[0]
#    adc_bits = rs_ds.adc_bits[channel_id].values[0]
#    n_res = (2**adc_bits - 1)/5000.0
#
#    # preprocess with default dead time
#    bz_an = lidar.get_bin_zero(wv, pol, 0)
#    bz_pc = lidar.get_bin_zero(wv, pol, 1)
#    bg_an = lidar.estimate_raw_background(an, ranges, bg_range=bg_range)
#    dc, dc_bg = lidar.preprocessing_dc_signal(an_dc, ranges, bz_an, bg_range)
#    an = lidar.preprocessing_analog_signal(an, bz_an, bg_an, dc, dc_bg)
#    pc = lidar.preprocessing_photoncounting_signal(pc, 3.703, bz_pc, ranges, bg_range)
#
#    gl, c_an, c_ph, idxs = lidar.gluing(an[0, :], pc[0, :], ranges, adc_range, adc_bits, n_res=n_res,
#                 pc_threshold=20, r2_threshold=0.8, range_threshold=(1000, 5000),
#                 min_points=5, slope_threshold=2, stability_threshold=1, step=5,
#                 use_photon_as_reference=True)


def estimate_dead_time(
    filepath,
    channel,
    time_range=None,
    tau_range=None,
    gluing_range=(3000.0, 4000.0),
    gluing_type="scc",
    expand_bins=True,
):

    channel_an = channel + "a"
    channel_pc = channel + "p"

    channels = [channel_an, channel_pc]

    lidar = preprocess(
        filepath,
        channels=channels,
        apply_dt=False,
        save_bg=True,
        save_dc=True,
        apply_bz=False,
    )
    if time_range is not None:
        lidar = lidar.sel(time=slice(*time_range))

    if not channel_an in lidar.channel:
        raise ValueError("channel not found.")

    if not channel_pc in lidar.channel:
        raise ValueError("channel not found.")

    # select signal
    an = lidar[f"signal_{channel_an}"].sel(range=slice(*gluing_range))
    pc = lidar[f"signal_{channel_pc}"].sel(range=slice(*gluing_range))

    # select background
    lidar[f"bg_{channel_an}"].values
    lidar[f"bg_{channel_pc}"].values

    # select dark current
    dc_an = lidar[f"dc_{channel_an}"].sel(range=slice(*gluing_range)).values

    # select range
    ranges = lidar.range.sel(range=slice(*gluing_range)).values
    times = lidar.time.values

    # analog channel additional info
    adc_range = 1000 * lidar.adc_range.sel(channel=channel_an).values.item()
    adc_bits = lidar.adc_bits.sel(channel=channel_an).values.item()

    # taus [in nanoseconds]
    if tau_range is None:
        tau_step = 1.0  # 0.05
        tau = np.arange(3, 5 + tau_step, tau_step)
    else:
        tau = np.arange(*tau_range)

    # expand glue region
    bin_step, bin_min, bin_max = 10, 75, 100
    exp_glue = np.arange(bin_min, bin_max + bin_step, bin_step) * 7.5

    # dimensions
    n_t = len(times)
    n_ranges = len(ranges)
    n_tau = len(tau)
    n_steps = len(exp_glue)

    # Define variables for output
    tau_best_avg = np.zeros(n_t)
    tau_best_std = np.zeros(n_t)
    thickness_best_avg = np.zeros(n_t)
    thickness_best_std = np.zeros(n_t)
    tau_min = np.zeros((n_t, n_steps)) * np.nan
    r_min = np.zeros((n_t, n_steps)) * np.nan
    r_min2 = np.zeros((n_t, n_steps)) * np.nan
    thickness_minJ = np.zeros((n_t, n_steps)) * np.nan
    cost_function = np.zeros((n_t, n_tau, n_steps)) * np.nan
    np.zeros((n_t, n_tau)) * np.nan
    np.zeros((n_t, n_tau)) * np.nan
    np.zeros((n_t, n_tau)) * np.nan
    np.zeros((n_t, n_tau)) * np.nan
    np.zeros((n_t, n_ranges)) * np.nan
    pc_pp = {}
    gl_pp = np.zeros((n_t, n_ranges, n_tau)) * np.nan
    gl_pp = {}
    rvalue = np.zeros((n_t, n_tau, n_steps)) * np.nan
    rvalue2 = np.zeros((n_t, n_tau, n_steps)) * np.nan
    range_glue_ext_min = np.zeros((n_t, n_tau, n_steps)) * np.nan
    range_glue_ext_max = np.zeros((n_t, n_tau, n_steps)) * np.nan
    range_glue_ext_thick = np.zeros((n_t, n_tau, n_steps)) * np.nan
    array_gluing_height = np.zeros((n_t, n_tau)) * np.nan

    # Loop over signals
    for j, tau_ in enumerate(tau):
        # Dead time correction photoncounting
        pc_ij = apply_dead_time_correction(pc, tau_)
        pc_pp[j] = pc_ij.copy()

        # Gluing
        gl_pp[j], _, gluing_height = gluing(
            an, pc, range_min=gluing_range[0], range_max=gluing_range[1]
        )

        array_gluing_height[:, j] = gluing_height

    # correlación AN vs PC en la region
    for k in range(n_steps):
        for idx_time, _ in enumerate(times):
            # set_trace()
            for j, tau_ in enumerate(tau):
                # anchura de region
                # Cost Function: para cada perfil uso el mismo rango para todos los taus
                try:
                    min_glue = array_gluing_height[idx_time, j] - exp_glue[k] // 2.0
                    max_glue = array_gluing_height[idx_time, j] + exp_glue[k] // 2.0
                except:
                    raise ValueError("Gluing range out of bounds.")

                idx_glue = np.logical_and(ranges >= min_glue, ranges <= max_glue)
                rvalue[idx_time, j, k] = np.corrcoef(
                    an[idx_time, idx_glue], pc[idx_time, idx_glue]
                )[0, 1]
                if rvalue[idx_time, j, k] >= 0.9:
                    # rangos
                    range_glue_ext_min[idx_time, j, k] = min_glue
                    range_glue_ext_max[idx_time, j, k] = max_glue
                    range_glue_ext_thick[idx_time, j, k] = max_glue - min_glue
                    rvalue2[idx_time, j, k] = np.corrcoef(
                        gl_pp[j][idx_time, idx_glue], pc_pp[j][idx_time, idx_glue]
                    )[0, 1]
                    if rvalue2[idx_time, j, k] >= 0.5:
                        cost_function[idx_time, j, k] = residuals(
                            pc_pp[j][idx_time, idx_glue], gl_pp[j][idx_time, idx_glue]
                        )
                        print(
                            rvalue[idx_time, j, k],
                            rvalue2[idx_time, j, k],
                            cost_function[idx_time, j, k],
                        )

    # Tau that Minimizes Cost Function and associated rvalues for an_pc and gl_pc
    for k in range(n_steps):
        for idx_time, _ in enumerate(times):
            jj = cost_function[idx_time, :, k]
            rr = rvalue[idx_time, :, k]
            rr2 = rvalue2[idx_time, :, k]
            th = range_glue_ext_thick[idx_time, :, k]
            try:
                if not np.isnan(jj).any():
                    idx_Jmin = np.nanargmin(jj)
                    tau_min[idx_time, k] = tau[idx_Jmin]
                    r_min[idx_time, k] = rr[idx_Jmin]
                    r_min2[idx_time, k] = rr2[idx_Jmin]
                    thickness_minJ[idx_time, k] = th[idx_Jmin]
            except Exception as e:
                print(str(e))
                print(
                    "Tau min not estimated for profile %i-th, step %i-th"
                    % (idx_time, k)
                )

            # CRITERIUM TO ESTIMATE DEAD TIME
            # for every thickness, a best tau is estimated above.
            # So, we have a list of best taus. from them, we derive the best
            # of the best, considering the correlation index gl_pc
            # if np.isnan(jj).any() == True:
            #     set_trace()

            try:
                try:
                    idx_best_tau = r_min2[idx_time, :] >= (
                        np.nanmax(r_min2[idx_time, :]) - np.nanstd(r_min2[idx_time, :])
                    )
                except Exception as e:
                    raise ValueError(str(e))

                tau_best_avg[idx_time] = np.nanmean(tau_min[idx_time, idx_best_tau])
                tau_best_std[idx_time] = np.nanstd(tau_min[idx_time, idx_best_tau])
                thickness_best_avg[idx_time] = np.nanmean(
                    thickness_minJ[idx_time, idx_best_tau]
                )
                thickness_best_std[idx_time] = np.nanstd(
                    thickness_minJ[idx_time, idx_best_tau]
                )
            except Exception as e:
                print(str(e))
                print("Best Tau not estimated for profile %i-th" % i)

    # save results in netcdf
    ds = xr.Dataset(
        {
            "J": (["times", "tau", "steps"], cost_function),
            "tau_min": (["times", "steps"], tau_min),
            "tau_best_avg": (["times"], tau_best_avg),
            "tau_best_std": (["times"], tau_best_std),
            "thickness_best_avg": (["times"], thickness_best_avg),
            "thickness_best_std": (["times"], thickness_best_std),
            "range_glue_extended_min": (["times", "tau", "steps"], range_glue_ext_min),
            "range_glue_extended_max": (["times", "tau", "steps"], range_glue_ext_max),
            "range_glue_extended_thick": (
                ["times", "tau", "steps"],
                range_glue_ext_thick,
            ),
            "rvalue_an_pc": (["times", "tau", "steps"], rvalue),
            "rvalue_gl_pc": (["times", "tau", "steps"], rvalue2),
        },
        coords={"times": times, "tau": tau, "steps": exp_glue, "ranges": ranges},
    )
    fn = None
    # fn = os.path.join(config.dead_time_dn, "dead_time_%s_%s_%s_%s_%i.nc"
    #   % (date_str, channel, don_str, gluing_type, expand_bins))
    # ds.to_netcdf(fn)
    # time_elapsed = time.time() - t_start
    # print("%.1f s of time elapsed for %s:%s (%i profiles). channel=%s" % (
    # time_elapsed, date_str, don_str, ds.dims["times"], channel))

    return ds, fn
