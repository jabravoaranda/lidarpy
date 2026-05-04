"""
PLOTTING FUNCTIONS
"""

#Create a function with the following instructions:
#1. The input is a zip filepath with with netcdf files inside.
#2. Check filepath exists. If not return a warning.
#3. Open the zip file and read the netcdf files inside.
#4. Plot the netcdf files.


 

def plot_scc_input(filelist):
    """
    Plot SCC input files.
    Input: 
    filelist: File string format (e.g., /c/*.nc') [str]    
    Output: 
    png figure files. Figures are saved in the same file directory.
    """
    MAX_ALT = 1000 #Points [bins]

    if len(filelist) > 0:
        for file_ in filelist:
            if os.path.isfile(file_):
                lxarray = xr.open_dataset(file_)
                output_directory = os.path.dirname(file_)
                filename = os.path.basename(file_)    

                colorbar = matplotlib.cm.get_cmap('jet', lxarray.time.size)  #Cool        
                colors = colorbar(np.linspace(0, 1, lxarray.time.size))
                try:
                    for idx_channel in np.arange(lxarray.channel_ID.size):                  
                        channel = lxarray.channel_ID.values[idx_channel]
                        fig = plt.figure(figsize=(9,5))
                        try:
                            for idx_time in np.arange(lxarray.time.size):
                                lxarray['Raw_Lidar_Data'][idx_time,idx_channel,0:MAX_ALT].plot(c=colors[idx_time], label='%d' % idx_time)
                        except:
                            logger.warning("Error: in plot_scc_input for %s. Plotting Raw Lidar Data. Try to continue plotting" % file_)
                        try:
                            for idx_time in np.arange(lxarray.time_bck.size):
                                lxarray['Background_Profile'][idx_time,idx_channel,0:MAX_ALT].plot(c=colors[idx_time], label='%d' % idx_time)
                        except:
                            logger.warning("Error: in plot_scc_input for %s. Plotting Background. Try to continue plotting" % file_)
                        plt.title('SCC-channel %d | %s' % (channel, filename))
                        plt.ylabel('Raw_Lidar_Data channel %d [counts/mV]' % channel)
                        plt.xlim(0,MAX_ALT)
                        ymin_value = 0.9*lxarray['Raw_Lidar_Data'][:,idx_channel,0:MAX_ALT].min()
                        ymax_value = 1.1*lxarray['Raw_Lidar_Data'][:,idx_channel,0:MAX_ALT].max()
                        plt.ylim(ymin_value,ymax_value)        
                        plt.savefig(os.path.join(output_directory, '%s_Raw_Lidar_Data_%d.png' % (filename.split('.')[0], channel)), dpi=200)
                        plt.close(fig)
                        logger.info("Plot SCC Raw Data in file %s" % file_)
                except:
                    logger.warning("Error: No plot for %s" % file_)
    else:
        logger.warning('Plot SCC INPUT not Done. Slot File List Empty')


def plot_scc_output(output_dir, scc_code=None):
    """
    Plot SCC optical output products, if available:
    backstter, extinction, angstrom exponent, lidar ratio, particle and volume depolarization

    Parameters:
    ----------
    output_dir: str
        output directory where scc processing products are downloaded
    scc_code: int
        system id

    """

    # config
    font = {'size': 12}
    matplotlib.rc('font', **font)

    # TODO: products id associated to beta, alfa, depo will be read from config file
    #beta_products = [1203, 838, 845, 1199, 760, 669, 839, 863]
    #alfa_products = [850, 1200, 838, 669, 839, 863]
    #depo_products = [760, 1203, 838, 669, 839, 863]
    #inversion_id = {760: 'K', 838: 'K', 845: 'R', 850: 'R', 1199: 'K',
    #                1200: 'R', 1203: 'R', 669: 'K', 839: 'K', 863: 'K'}

    # Plot x,y ranges
    y_lim = (0.68, 10)
    x_lim = {'beta': (-1e-2, 10), 
             'alfa': (1e-2, 500), 
             'angstrom': (-1, 3),
             'depo': (0, 0.40), 
             'lr': (10, 100)}
    # Colors according to Wavelength and Type of retrieval (Klett, K, or Iterative, R)
    colors = {'355': 'tab:blue', '355K': 'dodgerblue', '355I': 'dodgerblue', '355R': 'darkblue', '355U': 'tab:blue',
              '532': 'tab:green', '532K': 'limegreen', '532I': 'limegreen', '532R': 'darkgreen', '532U': 'tab:green',
              '1064': 'tab:red', '1064K': 'red', '1064I': 'red', '1064U': 'tab:red'}

    # necessary variables to be found in scc_optical/pid*nc
    vars_ = ["backscatter", "extinction", "particledepolarization", "volumedepolarization"]

    # scc
    if scc_code is None:
        scc_code = int(
            re.search(r"scc\d{3}", output_dir).group().split("scc")[1])

    logger.info('Plot SCC output. Processing folder: %s' % output_dir)

    # we need all nc files in the directory scc_optical
    nc_files = glob.glob(os.path.join(output_dir, 'scc_optical', '*.nc'))
    if nc_files:
        scc_slot_date_dt = dt.datetime.strptime(re.search(r"/\d{4}/\d{2}/\d{2}/", nc_files[0]).group(), '/%Y/%m/%d/') 
        # Take unique dates in the folder to know the number of different inversions
        def find_dates_in_file(fn, STATION_ID):
            if "pid" in os.path.basename(fn):
                date_pattern = r"\d{2}\d{2}\d{2}\d{2}\d{2}"
                pid_type = 0
            else:  # >= 2021
                date_pattern = r"\d{4}\d{2}\d{2}" + STATION_ID + "\d{2}\d{2}"
                pid_type = 1
            try:
                if pid_type == 0:
                    dates_str = re.search(date_pattern, os.path.basename(fn).split('_')[1].split('.')[0]).group()
                elif pid_type == 1:
                    dates_str = re.search(date_pattern, os.path.basename(fn).split('_')[6]).group()
                    #dates_str = dt.datetime.strptime(dates_str, "%Y%m%d"+STATION_ID+"%H%M").strftime("%Y%m%d%H%M")
                else:
                    dates_str = None
            except Exception as e:
                logger.error("Something went wrong finding date pattern in pid files")
                dates_str = None
            return dates_str
        dates_str = [find_dates_in_file(fn, STATION_ID) for fn in nc_files]
        if dates_str is not None:
            dates_str = np.unique(dates_str).tolist()
        # Loop over dates
        for date_ in dates_str:
            # pid files for the date
            pid_files = glob.glob(os.path.join(output_dir, 'scc_optical', '*%s*.nc' % date_))
            if len(pid_files) > 0:
                # define fig
                fig = plt.figure(figsize=(15, 10))  # , constrained_layout=True)

                """ Store Products in a dictionary """
                profile = {}
                for pid_fn in pid_files:
                    # product id
                    if "pid" in pid_fn:
                        product_id = int(os.path.basename(pid_fn).split('_')[0].replace('pid', ''))
                    else:  # >= scc version 2021
                        product_id = int(os.path.basename(pid_fn).split('_')[3])

                    with xr.open_dataset(pid_fn) as aux_ds:
                        pid_ds = None
                        for i_var in vars_:
                            try:
                                var_ds = aux_ds[i_var].squeeze()  # remove dimensions of length=1
                                # change dimensions for backscatter and extinction
                                if np.logical_or(i_var == "backscatter",
                                                 i_var == "extinction"):
                                    var_ds *= 1e6
                                    var_ds.attrs['units'] = "1/(Mm*sr)"
                                # add to pid dataset
                                if pid_ds is None:
                                    pid_ds = var_ds.to_dataset(name=i_var)
                                else:
                                    pid_ds[i_var] = var_ds
                                # add inversion method as attribute
                                # TODO: particularize for rest of variables: extinction, particledepolarization, volumedepolarization
                                try:
                                    if i_var == "backscatter":
                                        method = aux_ds["backscatter_evaluation_method"].values
                                        if method == 0:
                                            algorithm = aux_ds["raman_backscatter_algorithm"].values
                                        elif method == 1:
                                            algorithm = aux_ds["elastic_backscatter_algorithm"].values
                                        else:
                                            algorithm = 0
                                    else: #if i_var == "extinction":
                                        method = -1 # aux_ds["backscatter_evaluation_method"].values
                                        algorithm = -1 # aux_ds["elastic_backscatter_algorithm"].values
                                    if method == 0:  # Raman
                                        inversion_method = "R"
                                    elif method == 1:  # Elastic Backscatter
                                        if algorithm == 0:
                                            inversion_method = "K"   # Klett-Fernald
                                        elif algorithm == 1:
                                            inversion_method = "I"  # Iterative
                                        else:
                                            inversion_method = ""  # Unknown
                                    else:
                                        inversion_method = ""  # Unknown
                                except Exception as e:
                                    logger.error(str(e))
                                    logger.error("Inversion Method not found")
                                    inversion_method = None
                                pid_ds[i_var].attrs["inversion_method"] = inversion_method
                            except Exception as e:
                                logger.warning("%i does not have %s profile" % (product_id, i_var))
                    # altitude in km
                    pid_ds["altitude"] = pid_ds["altitude"] * 1e-3
                    pid_ds["altitude"].attrs['units'] = "km"

                    # save in profile dictionary
                    profile[product_id] = pid_ds.squeeze()  # remove dimensions of length=1

                """ Plot Backscatter """
                plot_code = 'beta'
                ax = fig.add_subplot(151)
                # loop over pids
                for pid in profile.keys():
                    #if pid in beta_products:  # realmente, ¿hace falta saber si pid está en beta_products?
                    if 'backscatter' in profile[pid].keys():
                        try:
                            wave = int(profile[pid]["wavelength"])
                            inv_met = profile[pid]['backscatter'].attrs["inversion_method"]
                            beta_id = '%d%s' % (wave, inv_met)
                            try:
                                color_beta = colors[beta_id]
                            except:
                                color_beta = colors["%s" % wave]
                            profile[pid]['backscatter'].plot(y='altitude',
                                                             ax=ax, linewidth=2,
                                                             c=color_beta,
                                                             label=beta_id)
                        except Exception as e:
                            logger.warning("Backscatter Not Plotted for %i" % wave)
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.xaxis.grid(b=True, which='minor', linestyle='--')
                ax.set_ylabel(r'Altitude, km asl', fontsize='large')
                ax.set_xlabel(r'$\beta_{a}, Mm^{-1} sr^{-1}$', fontsize='large')
                ax.set_xlim(x_lim[plot_code])
                #ax.set_xscale('log')
                if len(ax.get_lines()) > 0:
                    plt.legend(loc=1, fontsize='medium')

                """ Plot Extinction """
                plot_code = 'alfa'
                ax = fig.add_subplot(152)
                # loop over pids
                for pid in profile.keys():
                    if 'extinction' in profile[pid].keys():
                        try:  # if pid in alfa_products:
                            wave = int(profile[pid]["wavelength"])
                            #inv_met = profile[pid]['extinction'].attrs["inversion_method"]
                            inv_met = ""
                            alfa_id = '%d%s' % (wave, inv_met)
                            try:
                                color_alfa = colors[alfa_id]
                            except:
                                color_alfa = colors["%i" % wave]
                            profile[pid]['extinction'].plot(y='altitude', ax=ax,
                                                            linewidth=2,
                                                            c=color_alfa,
                                                            label=alfa_id)
                        except Exception as e:
                            logger.warning("Extinction Not Plotted for %i" % wave)
                ax.xaxis.set_minor_locator(MultipleLocator(50))
                ax.xaxis.grid(b=True, which='minor', linestyle='--')
                ax.set_xlabel(r'$ \alpha _{a}, Mm^{-1} $', fontsize='large')
                ax.set_xlim(x_lim[plot_code])
                if len(ax.get_lines()) > 0:
                    plt.legend(loc=1, fontsize='medium')

                """ Plot Angstrom """
                ax = fig.add_subplot(153)
                plot_code = 'angstrom'
                angstrom = {}
                if profile != {}:
                    coefficients = ['backscatter', 'extinction']
                    coef_id = {'backscatter': 'beta', 'extinction': 'alpha'}
                    for coef_ in coefficients:
                        for pid_1 in profile.keys():
                            for pid_2 in profile.keys():
                                wave_1 = int(profile[pid_1]['wavelength'])
                                wave_2 = int(profile[pid_2]['wavelength'])
                                #inv_met_1 = profile[pid_1][coef_].attrs["inversion_method"]
                                #inv_met_2 = profile[pid_2][coef_].attrs["inversion_method"]
                                inv_met_1 = ""
                                inv_met_2 = ""
                                if wave_1 < wave_2:
                                    if np.logical_and(
                                            coef_ in profile[pid_1].keys(),
                                            coef_ in profile[pid_2].keys()):
                                        try:
                                            profile[pid_1][coef_][profile[pid_1][coef_] <= 0] = np.nan
                                            profile[pid_2][coef_][profile[pid_2][coef_] <= 0] = np.nan
                                            angstrom_id = r'$\%s$ (%d%s-%d%s)' % (coef_id[coef_], 
                                                    wave_1, inv_met_1, wave_2, inv_met_2)
                                            angstrom[angstrom_id] = (-1) * np.log(
                                                profile[pid_1][coef_] / profile[pid_2][coef_]) / np.log(wave_1 / wave_2)
                                            angstrom[angstrom_id].name = angstrom_id
                                            angstrom[angstrom_id].plot(y='altitude', ax=ax,
                                                                       linewidth=2, label=angstrom_id)
                                        except Exception as e:
                                            logger.warning("Angstrom for %s, %s, %s not plotted. %s" 
                                                    % (coef_, pid_1, pid_2,str(e)))
                    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
                    ax.xaxis.grid(b=True, which='minor', linestyle='--')
                    ax.set_ylim(y_lim)
                    ax.set_xlim(x_lim[plot_code])
                    ax.set_xlabel('Angstrom exponent, #', fontsize='large')
                    if len(ax.get_lines()) > 0:
                        plt.legend(loc=1, fontsize='medium')

                """ Lidar ratio """
                ax = fig.add_subplot(154)
                plot_code = 'lr'
                lr = {}
                if profile != {}:
                    for pid in profile.keys():
                        if np.logical_and('backscatter' in profile[pid].keys(),
                                          'extinction' in profile[pid].keys()):
                            try:
                                wave = int(profile[pid]["wavelength"])
                                #lr_id = '%d%s' % (wave, inversion_id[pid])
                                lr_id = '%d' % (wave,)
                                profile[pid]['extinction'][profile[pid]['extinction'] <= 0] = np.nan
                                profile[pid]['backscatter'][profile[pid]['backscatter'] <= 0] = np.nan
                                lr[lr_id] = profile[pid]['extinction'] / profile[pid]['backscatter']
                                lr[lr_id].name = lr_id
                                lr[lr_id].plot(y='altitude', ax=ax, linewidth=2,
                                               c=colors[lr_id], label=lr_id)
                            except Exception as e:
                                logger.warning("Lidar Ratio for %s not plotted. %s" % (pid, str(e)))
                    ax.xaxis.set_minor_locator(MultipleLocator(10))
                    ax.xaxis.grid(b=True, which='minor', linestyle='--')
                    ax.set_ylim(y_lim)
                    ax.set_xlim(x_lim[plot_code])
                    ax.set_xlabel('Lidar ratio, sr', fontsize='large')
                    if len(ax.get_lines()) > 0:
                        plt.legend(loc=1, fontsize='medium')

                """ Depolarization """
                plot_code = 'depo'
                ax = fig.add_subplot(155)
                for pid in profile.keys():
                    if np.logical_and(
                            "particledepolarization" in profile[pid].keys(),
                            "volumedepolarization" in profile[pid].keys()):# pid in depo_products:
                        wave = int(profile[pid]["wavelength"])
                        #depo_id = '%d%s' % (wave, inversion_id[pid])
                        depo_id = '%d' % (wave,)
                        try:
                            profile[pid]['particledepolarization'].plot(
                                y='altitude', ax=ax, linewidth=2,
                                c=colors[depo_id], label=depo_id)
                            profile[pid]['volumedepolarization'].plot(
                                y='altitude', ax=ax, linewidth=2,
                                linestyle='dashed', c=colors[depo_id], label=depo_id)
                        except Exception as e:
                            logger.warning("Not Plotted Depolarization for %i" % wave)
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                ax.xaxis.grid(b=True, which='minor', linestyle='--')
                ax.set_xlabel(r'$\delta_{a}$,#', fontsize='large')
                ax.set_xlim(x_lim[plot_code])
                if len(ax.get_lines()) > 0:
                    plt.legend(loc=1, fontsize='medium')

                # Fig Format and details
                fig_title = '%s' % np.datetime_as_string(profile[pid].time.values, unit='s')
                plt.suptitle(fig_title.replace('T', ' '),verticalalignment='baseline')
                plt.subplots_adjust(bottom=0.25, top=0.95)
                for ax in fig.get_axes():
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_ylim(y_lim)
                    ax.set_title('')
                    ax.grid()
                    ax.label_outer()

                # Save Fig
                if profile != {}:
                    fig_dir = os.path.dirname(output_dir)
                    fig_fn = os.path.join(fig_dir, 'scc%d_%s.png' %
                                          (scc_code, fig_title.replace('-', '').replace(':', '').replace('T', '_')))
                    plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    if os.path.isfile(fig_fn):
                        logger.info('Figure successfully created: %s' % fig_fn)
                        #os.path.basename(fig_fn))
    else:
        logger.warning('ERROR. No NC Files/scc_optical in Folder: %s ' % os.path.basename(output_dir))

