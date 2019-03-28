# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:23:39 2019

@author: L1656
"""

import pandas as pd
import numpy as np
import datetime
from pyAPES_utilities.plotting import plot_columns
import matplotlib.pyplot as plt
from pyAPES_utilities.dataprocessing_scripts import save_df_to_csv

def read_WTD_data():

    fpaths = [r"H:\Lettosuo\WTD_paavolta\Lettosuo_WTD_EC.csv",
               r"H:\Lettosuo\WTD_paavolta\Lettosuo_WTD_trans.csv"]

    index=pd.date_range('01-01-2010','01-01-2019',freq='1H')
    data=pd.DataFrame(index=index, columns=[])

    for fp in fpaths:
        dat = pd.read_csv(fp, sep=';', header='infer')
        dat.index = pd.to_datetime(dat.ix[:,0], dayfirst=True)
        dat.index = dat.index + datetime.timedelta(hours=0.5)
        dat.index = dat.index.map(lambda x: x.replace(minute=0, second=0))
        dat.ix[:,0]=dat.index
        dat = dat.drop_duplicates(subset=dat.columns[0])
        dat = dat.drop([dat.columns[0]], axis=1)
        if len(np.setdiff1d(dat.index, index)) > 0:
            print(fp, np.setdiff1d(dat.index, index))
            raise ValueError("Error")
        data=data.merge(dat, how='outer', left_index=True, right_index=True)

    return data

def process_WTD():
    # read data
    WTD=read_WTD_data()

    # putken pohja tulee vastaan
    WTD['ctrl8m'][WTD['ctrl8m'] < -83] = np.nan
    WTD['ctrl22.5m'][WTD['ctrl22.5m'] < -81] = np.nan

    # pitkäaikaiset loggerit
    plot_columns(WTD[['WT_E','WT_N','WT_S','WT_W']])  # -> korrelaatio 'WT_E','WT_S','WT_W' välillä > 0.95
    # keskiarvo ja keskihajonta pitkäaikaisista
    WTD['WT_ESW'] = np.nanmean([WTD['WT_E'].values,
                                WTD['WT_S'].values,
                                WTD['WT_W'].values], axis=0)
    WTD['WT_ESW_std'] = np.nanstd([WTD['WT_E'].values,
                                   WTD['WT_S'].values,
                                   WTD['WT_W'].values], axis=0)

    # ennen hakkuuta control, hakkuun jälkeen partial
    WTD['WT_ESW_part'] = np.where(WTD.index <= '03-15-2016', np.nan, WTD['WT_ESW'])
    WTD['WT_ESW_ctrl'] = np.where(WTD.index <= '03-15-2016', WTD['WT_ESW'], np.nan)

    # epämääräinen alkujakso avohakkuulla
    WTD['clear4m'][WTD.index < '12-01-2015'] = np.nan
    WTD['clear8m'][WTD.index < '12-01-2015'] = np.nan
    WTD['clear12m'][WTD.index < '12-01-2015'] = np.nan
    WTD['clear22.5m'][WTD.index < '12-01-2015'] = np.nan

    # kalibrointikausi hakkuun ajankohtaan asti
    WTD_calib = WTD[(WTD.index <= '03-15-2016')]

    # Tasokorjaus (slope=1) kalibrointi jakson perusteella, ehtona R2 > 0.7

    #plot_columns(WTD_calib[['ctrl4m','ctrl8m','ctrl12m','ctrl22.5m','WT_ESW']])
    #WTD['pred_ctrl8m'] = 0.99*WTD['ctrl8m'] + 11.68
    #WTD['pred_ctrl12m'] = 0.98*WTD['ctrl12m'] + 2.55
    #WTD['pred_ctrl22.5m'] = 0.80*WTD['ctrl22.5m'] + 1.53
    plot_columns(WTD_calib[['ctrl4m','ctrl8m','ctrl12m','ctrl22.5m','WT_ESW']],slope=1.0)
    WTD['pred_ctrl8m'] = WTD['ctrl8m'] + 12.33
    WTD['pred_ctrl12m'] = WTD['ctrl12m'] + 3.2
    WTD['pred_ctrl22.5m'] = WTD['ctrl22.5m'] + 11.27
    #WTD[['pred_ctrl8m','pred_ctrl12m','pred_ctrl22.5m','WT_ESW']].plot()

    #plot_columns(WTD_calib[['part4m','part8m','part12m','part22.5m','WT_ESW']],slope=1.0)
    #WTD['pred_part12m'] = 1.0*WTD['part12m'] + 10.84
    #WTD['pred_part22.5m'] = 0.85*WTD['part22.5m'] - 3.46
    plot_columns(WTD_calib[['part4m','part8m','part12m','part22.5m','WT_ESW']],slope=1.0)
    WTD['pred_part12m'] = WTD['part12m'] + 11.02
    WTD['pred_part22.5m'] = WTD['part22.5m'] + 2.51
    #WTD[['pred_part12m','pred_part22.5m','WT_ESW']].plot()

    #plot_columns(WTD_calib[['clear4m','clear8m','clear12m','clear22.5m','WT_ESW']])
    #WTD['pred_clear4m'] = 1.5*WTD['clear4m'] + 29.89
    #WTD['pred_clear12m'] = 1.32*WTD['clear12m'] + 15.85
    plot_columns(WTD_calib[['clear4m','clear8m','clear12m','clear22.5m','WT_ESW']],slope=1.0)
    WTD['pred_clear4m'] = WTD['clear4m'] + 8.69
    WTD['pred_clear12m'] = WTD['clear12m'] + 3.92
    #WTD[['pred_clear4m','pred_clear12m','WT_ESW']].plot()

    # keskiarvot ja vaihteluväli (keskihajonnan ja aikasarjojen vaihtelusta) käsittelyille

    WTD['control'] = 0.01 * np.nanmean([WTD['WT_ESW_ctrl'].values,
                                 WTD['pred_ctrl8m'].values,
                                 WTD['pred_ctrl12m'].values,
                                 WTD['pred_ctrl22.5m'].values], axis=0)
    WTD['control_min'] = 0.01 * np.nanmin([WTD['WT_ESW_ctrl'].values - WTD['WT_ESW_std'].values,
                                    WTD['pred_ctrl8m'].values - WTD['WT_ESW_std'].values,
                                    WTD['pred_ctrl12m'].values - WTD['WT_ESW_std'].values,
                                    WTD['pred_ctrl22.5m'].values - WTD['WT_ESW_std'].values], axis=0)
    WTD['control_max'] = 0.01 * np.nanmax([WTD['WT_ESW_ctrl'].values + WTD['WT_ESW_std'].values,
                                    WTD['pred_ctrl8m'].values + WTD['WT_ESW_std'].values,
                                    WTD['pred_ctrl12m'].values + WTD['WT_ESW_std'].values,
                                    WTD['pred_ctrl22.5m'].values + WTD['WT_ESW_std'].values], axis=0)

    WTD['partial'] = 0.01 * np.nanmean([WTD['WT_ESW_part'].values,
                                 WTD['pred_part12m'].values,
                                 WTD['pred_part22.5m'].values], axis=0)
    WTD['partial_min'] = 0.01 * np.nanmin([WTD['WT_ESW_part'].values - WTD['WT_ESW_std'].values,
                                    WTD['pred_part12m'].values - WTD['WT_ESW_std'].values,
                                    WTD['pred_part22.5m'].values - WTD['WT_ESW_std'].values], axis=0)
    WTD['partial_max'] = 0.01 * np.nanmax([WTD['WT_ESW_part'].values + WTD['WT_ESW_std'].values,
                                    WTD['pred_part12m'].values + WTD['WT_ESW_std'].values,
                                    WTD['pred_part22.5m'].values + WTD['WT_ESW_std'].values], axis=0)

    WTD['clearcut'] = 0.01 * np.nanmean([WTD['pred_clear4m'].values,
                                 WTD['pred_clear12m'].values], axis=0)
    WTD['clearcut_min'] = 0.01 * np.nanmin([WTD['pred_clear4m'].values - WTD['WT_ESW_std'].values,
                                    WTD['pred_clear12m'].values - WTD['WT_ESW_std'].values], axis=0)
    WTD['clearcut_max'] = 0.01 * np.nanmax([WTD['pred_clear4m'].values + WTD['WT_ESW_std'].values,
                                    WTD['pred_clear12m'].values + WTD['WT_ESW_std'].values], axis=0)

    # lopulliset kuvaan
    plt.figure()
    plt.fill_between(WTD.index, WTD['control_max'].values, WTD['control_min'].values,
                     facecolor='k', alpha=0.3)
    plt.plot(WTD.index, WTD['control'].values,'-k', linewidth=1.0)

    plt.fill_between(WTD.index, WTD['partial_max'].values, WTD['partial_min'].values,
                     facecolor='b', alpha=0.3)
    plt.plot(WTD.index, WTD['partial'].values,'-b', linewidth=1.0)

    plt.fill_between(WTD.index, WTD['clearcut_max'].values, WTD['clearcut_min'].values,
                     facecolor='r', alpha=0.3)
    plt.plot(WTD.index, WTD['clearcut'].values,'-r', linewidth=1.0)

    # ja tiedostoon
    save_df_to_csv(WTD[['control','control_max','control_min','partial','partial_max','partial_min','clearcut','clearcut_max','clearcut_min']],
                       'lettosuo_WTD_pred', readme=' - Check timezone!! \nSee Lettosuo_dataprocessing.process_WTD()')

def fit_pf_Laiho():
    from pyAPES_utilities.parameter_utilities import fit_pF
    # heads [kPa]
    head = [0.01, 0.3, 0.981, 4.905, 9.81, 33.0, 98.1]

    # volumetric water content measurements corresponding to heads for different peat types [%]
    watcont = [[94.69, 49.42, 29.61, 21.56, 20.05, 17.83, 16.54],
               [91.41, 66.26, 56.98, 45.58, 41.44, 39.32, 37.89],
               [89.12, -999, 72.83, 63.97, 54.40, 50.15, 48.80],
               [89.46, -999, 82.46, 76.79, 66.93, 63.61, 62.53],
               [92.22, -999, 87.06, 78.02, 74.76, 72.77, 71.70],
               [91.98, 66.70, 57.49, 39.95, 34.41, 29.83, 28.39],
               [88.75, 78.98, 78.77, 75.83, 72.37, 61.35, 45.66],
               [91.93, -999, 83.65, 78.39, 75.56, 74.08, 73.10],
               [93.45, 87.44, 87.33, 77.95, 76.46, 75.01, 73.01],
               [93.32, 87.15, 86.73, 82.90, 81.84, 80.51, 79.55],
               [93.05, 54.55, 42.88, 33.98, 29.80, 26.90, 25.87],
               [92.90, 78.15, 72.19, 54.47, 51.05, 49.66, 48.35],
               [90.11, -999, 80.86, 77.98, 69.66, 60.70, 53.40],
               [93.14, -999, 83.81, 78.69, 74.86, 73.26, 72.25],
               [93.17, -999, 89.76, 80.05, 76.67, 74.21, 72.84]]

    fit_pF(head, watcont[0:5], fig=True,percentage=True, kPa=True)
    fit_pF(head, watcont[5:10], fig=True,percentage=True, kPa=True)
    fit_pF(head, watcont[10:15], fig=True,percentage=True, kPa=True)