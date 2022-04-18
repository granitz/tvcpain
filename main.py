import os
import glob
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import nibabel as nib

import sys
sys.path.insert(1, '~/analysis_030321/code/')
from utility import *

import bids
import teneto
from teneto import TenetoBIDS
from bids import BIDSLayout, BIDSValidator

from nltools.file_reader import onsets_to_dm
from nltools.data import Brain_Data, Design_Matrix

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size':12})

subject_to_remove = [3,9,12,15,16,25,27,32] #  sub-04 and sub-10. Could remove [12,15,16,25,32] # correspond to sub-04,10,13,16,17,26,33.
num_subs_to_remove = len(subject_to_remove)
remove_subjects_bool = np.array([1 if x not in subject_to_remove else 0 for x in range(33) ],dtype=bool)

data_dir = '~/ds000140/'
layout = BIDSLayout(data_dir, derivatives=True)

def extract_timeseries_from_rois(tnet=None,runs='all',fmriprep_data=None):

    bids = '~/ds000140/'
    # Step 1:
    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]

    if runs=='all':
        bids_filter = {'subject':subjects,'run': ['1','2','3','4','5','6','7','8','9'],'task':['heatpainwithregulationandratings']}
    elif runs=='longrun':
        bids_filter = {'run': ['1'],'subject':subjects,'task':['longrun']}

    if tnet is None:
        tnet = TenetoBIDS(bids,
                        selected_pipeline='fMRIPrep',
                        bids_filter=bids_filter,
                        exist_ok=True)

    # Step 2: create a parcellation
    nilearn_params = {'standardize':True,'low_pass':0.1,'high_pass':0.008,'t_r':2.0}
    parcellation_params = {'atlas': 'Schaefer2018',
                            'resolution': '1',
                            'atlas_desc': '400Parcels7Networks',
                            'parc_params': nilearn_params}

    tnet.run('make_parcellation', input_params=parcellation_params)

    return tnet

def concatenate_parcellated_runs():
    """There are 9 runs in the fmri session"""
    datapath = '~/ds000140/derivatives/teneto-make-parcellation/'
    savepath = '~/ds000140/derivatives/teneto-make-parcellation_concatenated/'
    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]

    for sub_name in subjects:
        savename = savepath + 'sub-' + sub_name

        timeseries_dfs = []

        for run in ['1','2','3','4','5','6','7','8','9']:
            timeseries = pd.read_csv(datapath + 'sub-' + sub_name + '/func/sub-' + sub_name + '_run-' + run + '_task-heatpainwithregulationandratings_timeseries.tsv','\t',index_col=0)
            timeseries_dfs.append(timeseries)

        if not os.path.exists(savename):
            os.makedirs(savename + '/func/')

        df = pd.concat(timeseries_dfs,axis=1)
        df.to_csv(savename + '/func/sub-' + sub_name + '_run-1_task-longrun_timeseries.tsv','\t')

        tnet.create_output_pipeline('make_parcellation_concatenated', None, exist_ok=True)

def add_hrf_to_confounds(savepath=None):
    """
    Uses nltools to create design matrix

    note: for 'task-longrun', nifti for each run 1:9 was concatenated as well as
            their event files.
            Check /code/concat_nii.py
            Also, check function plot_events.
    """
    data_dir = '~/ds000140/'
    layout = BIDSLayout(data_dir, derivatives=True)
    fmriprep = '~/ds000140/derivatives/fmriprep/'

    nsubs = 33

    for subidx, sub in enumerate(range(1,nsubs+1)):
        subidx = sub-1
        if subidx < 9:
            subject = '0' + str(sub)
        else:
            subject = str(sub)

        dm = load_bids_events(layout,subject=subject) 

        if not (dm.any()).all():
            print('Check subject' + subject)

        conv = dm.convolve().values.sum(axis=1)

        # load confounds
        conf = pd.read_csv(fmriprep + '/sub-' + subject + '/func/sub-' + subject + '_task-longrun_run-01_desc-confounds_regressors.tsv', '\t', index_col=0)

        conf['hrf'] = conv

        conf.to_csv(fmriprep + '/sub-' + subject + '/func/sub-' + subject + '_task-longrun_run-01_desc-confounds_regressors.tsv','\t')

def clean_data():
    """
    loop through all participants and clean data and save results
    """
    from teneto.timeseries import remove_confounds
    from teneto.neuroimagingtools import load_tabular_file

    # no global signal and with hrf
    confounds_list = np.array(['trans_x','trans_x_power2','trans_x_derivative1','trans_x_derivative1_power2','trans_y','trans_y_power2',
    'trans_y_derivative1','trans_y_derivative1_power2','trans_z','trans_z_power2','trans_z_derivative1',
    'trans_z_derivative1_power2','rot_x','rot_x_power2','rot_x_derivative1','rot_x_derivative1_power2','rot_y',
    'rot_y_power2','rot_y_derivative1','rot_y_derivative1_power2','rot_z','rot_z_power2','rot_z_derivative1',
    'rot_z_derivative1','framewise_displacement','a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03',
    'a_comp_cor_04','a_comp_cor_05','hrf']) # Add global signal regression 

    bids_dir = '~/ds000140/'
    subjects = ['0' + str(x) if x < 10 else str(x) for x in range(1,34)]
    bids_filter = {'subject':subjects,'run': ['1'],'task':'longrun'}
    tnet = TenetoBIDS(bids_dir,
                    selected_pipeline='teneto-make-parcellation-concatenated',
                    bids_filter=bids_filter,
                    exist_ok=True)

    savedir = '~/ds000140/derivatives/cleaned/'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    for x in tnet.get_selected_files():
        sub = x.filename.split('_')[0]
        loaddir = x.dirname
        filename = x.filename
        run = filename.split('-')[2].split('_')[0]
        save_subject_name = savedir + '/' + sub + '/'

        if not os.path.exists(save_subject_name):
            os.makedirs(save_subject_name + '/func/')

        timeseries = pd.read_csv(loaddir + '/' + filename, '\t',index_col=0)
        confounds = pd.read_csv('~/ds000140/derivatives/fmriprep/' + sub + '/func/' + sub +  '_task-longrun_run-0' + run + '_desc-confounds_regressors.tsv', sep = '\t', header = 0)
        cleaned_ts = remove_confounds(timeseries, confounds, confounds_list)
        cleaned_ts.to_csv(save_subject_name + '/func/' + sub + '_run-1_task-longrun_timeseries.tsv','\t')

def split_cleaned_timeseries():
    """
    Splits concatenated timeseries and then estimates interpolation on each run.
    After scrubbing, concatenate censored timeseries again.
    """
    datapath = '~/ds000140/derivatives/teneto-remove-confounds'
    savepath = '~/ds000140/derivatives/teneto-remove-confounds-separate_runs'

    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for sub_name in subjects:
        savename = savepath + 'sub-' + sub_name

        if not os.path.exists(savename):
            os.makedirs(savename + '/func/')

        timeseries_dfs = []

        timeseries = pd.read_csv(datapath + 'sub-' + sub_name + '/func/sub-' + sub_name + '_run-1_task-longrun_timeseries.tsv','\t',index_col=0)

        run1 = timeseries.iloc[:,0:209]
        run2 = timeseries.iloc[:,209:418]
        run3 = timeseries.iloc[:,418:418+191]
        run4 = timeseries.iloc[:,609:818]
        run5 = timeseries.iloc[:,818:1027]
        run6 = timeseries.iloc[:,1027:1236]
        run7 = timeseries.iloc[:,1236:1236+191]
        run8 = timeseries.iloc[:,1427:1636]
        run9 = timeseries.iloc[:,1636:1845]

        run_timeseries = [run1,run2,run3,run4,run5,run6,run7,run8,run9]

        for run_i, run in enumerate(run_timeseries):
            run.to_csv(savename + '/func/sub-' + sub_name + '_run-0' + str(run_i+1) + '_task-longrun_timeseries.tsv','\t')

def scrubbing(tnet, savepath,replace_with=None):
    """
        tnet: get_selected_files must include cleaned data
        savepath = '~/ds000140/derivatives/teneto-censor-timepoints/'
    """
    from scipy.interpolate import interp1d
    import json

    confound_name = 'framewise_displacement'
    replace_with = replace_with
    relex = np.greater
    crit = 0.5
    tol=0.20

    if not os.path.exists(savepath):
        os.mkdir(savepath)


    for x in tnet.get_selected_files():
        # if not 'longrun' in x.filename:
        #     continue
        #
        # if 'timeseries' in x.filename:
        #     data = x.dirname + '/' + x.filename
        #     df = pd.read_csv(data,'\t')
        #     labels = df['Unnamed: 0']

        metadata = {}

        sub = x.filename.split('_')[0]
        loaddir = x.dirname
        filename = x.filename
        run = filename.split('-')[2].split('_')[0]

        if not os.path.exists(savepath + sub):
            os.mkdir(savepath + sub)
            os.mkdir(savepath + sub + '/func/')

        # load data
        timeseries = pd.read_csv(loaddir + '/' + filename, '\t',index_col=0)

        # load confounds
        confounds = pd.read_csv('~/ds000140/derivatives/fmriprep/' + sub + '/func/' + sub +  '_task-heatpainwithregulationandratings_run-' + run + '_desc-confounds_regressors.tsv', sep = '\t', header = 0)
        Z = timeseries.copy()
        ### clean data ###############################################################################################
        ci = confounds[confound_name]
        bad_timepoints = list(ci[relex(ci, crit)].index)
        bad_timepoints = list(map(str, bad_timepoints))

        # Column names in a run may not begin with zero. While confounds assumes 0:N.
        # Get the name of the timepoint given bad_timepoints
        bad_timepoints_name = [timeseries.columns[int(item)] for item in bad_timepoints]
        # get index of all timepoints
        all_timepoint_names = timeseries.columns
        all_timepoints_index = [np.where(all_timepoint_names==item)[0][0] for item in all_timepoint_names]

        timeseries[bad_timepoints_name] = np.nan

        if replace_with == 'cubicspline' and len(bad_timepoints) > 0:
            #good_timepoints = sorted(
            #    np.array(list(map(int, set(timeseries.columns).difference(bad_timepoints)))))
            good_timepoints = sorted(
                np.array(list(map(int, set(all_timepoints_index).difference(list(map(int, bad_timepoints)))))))

            bad_timepoints = np.array(list(map(int, bad_timepoints)))
            ts_index = timeseries.index
            timeseries = timeseries.values
            bt_interp = bad_timepoints[bad_timepoints > np.min(good_timepoints)]
            for n in range(timeseries.shape[0]):
                interp = interp1d(
                    good_timepoints, timeseries[n, good_timepoints], kind='cubic',fill_value='extrapolate')
                timeseries[n, bt_interp] = interp(bt_interp)
            timeseries = pd.DataFrame(timeseries, index=ts_index)
            bad_timepoints = list(map(str, bad_timepoints))
        ###############################################################################################

        Z = Z.values - timeseries.values
        print(Z.any())
        # save
        #if os.path.exists(savedir + sub + '/func/' + sub + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv'):
        #    os.remove(savedir + sub + '/func/' + sub + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv')

        timeseries.to_csv(savepath + sub + '/func/' + filename, '\t')
        # df.to_csv(savepath +  subs[ii] + '/func/conn-denoised_conn-name-' + str(ii+1) + '.tsv','\t') # both files and subs are sorted
        metadata[sub + ' num of censored volumes'] = len(bad_timepoints)
        metadata['bad_point_ratio'] = len(bad_timepoints)/1845
        metadata['badpoints'] = bad_timepoints

        if len(bad_timepoints)>(timeseries.shape[1]*0.25):
            print('Number of bad timepoints for subject ' + sub + '=' + str(len(bad_timepoints)))

        string = '_'

        with open(savepath + sub + '/func/' + string.join(filename.split('_')[:-1]) + '.json','w') as f:
            json.dump(metadata, f)


def concatenate_scrubbed_runs():
    datapath = '~/ds000140/derivatives/teneto-censor-timepoints'
    savepath = '~/ds000140/derivatives/teneto-censor-timepoints_longrun'
    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]

    for sub_name in subjects:
        savename = savepath + 'sub-' + sub_name

        timeseries_dfs = []

        for run in ['1','2','3','4','5','6','7','8','9']:
            timeseries = pd.read_csv(datapath + 'sub-' + sub_name + '/func/sub-' + sub_name + '_run-0' + run + '_task-longrun_timeseries.tsv','\t',index_col=0)
            timeseries_dfs.append(timeseries)

        if not os.path.exists(savename):
            os.makedirs(savename + '/func/')

        df = pd.concat(timeseries_dfs,axis=1)
        df.to_csv(savename + '/func/sub-' + sub_name + '_run-1_task-longrun_timeseries.tsv','\t')

def retrieve_scrubbing_info(long=False):
    """
    Get bad timepoints ratio per subject.
    Returns list with bad timepoints. Can be used with
    test_bad_timepoints_per_condition for statistical test on number of scrubbed
    timepoints for each condition.

    return bad point ratio for each participant and binary matrix with
        1==interpolated frame.
    """
    base = '~/ds000140/derivatives/teneto-censor-timepoints'

    subjects = ['sub-0' + str(x) if x<10 else 'sub-' + str(x) for x in range(1,34)]

    bad_point_ratio = []
    bad_points = []
    bad_point_mat = np.zeros((33,5,209))

    runs = list(range(1,10))

    if long:
        C = np.zeros((33,1845))
        G = [] # subject
        discard = np.zeros((191))

        for ii, subject in enumerate(subjects):
            A = []

            for run_i, run in enumerate(list(range(1,10))):
                bad_point_mat = np.zeros(209)

                if run in [3,7]:
                    A.append(discard)
                    continue
                if run in [5,6]:
                    A.append(bad_point_mat)
                    continue
                file = base + subject + '/func/' + subject + '_run-0' + str(run) + '_task-longrun.json'
                info = json.load(open(file,'rb'))
                bad_point_ratio.append(info['bad_point_ratio'])
                badpoints_as_int = np.array([int(x) for x in info['badpoints']])

                if badpoints_as_int.any():
                    bad_point_mat[badpoints_as_int] = badpoints_as_int

                A.append(bad_point_mat)
            G.append(A)
        for i,sub in enumerate(G):
            C[i,:] = np.concatenate(sub)

        bad_point_mat = C
    else:
        for ii, subject in enumerate(subjects):
            for run_i, run in enumerate([1,2,4,8,9]):
                file = base + subject + '/func/' + subject + '_run-0' + str(run) + '_task-longrun.json'
                info = json.load(open(file,'rb'))
                bad_point_ratio.append(info['bad_point_ratio'])
                badpoints_as_int = np.array([int(x) for x in info['badpoints']])

                if badpoints_as_int.any():
                    bad_point_mat[ii,run_i,badpoints_as_int] = badpoints_as_int

    return bad_point_ratio, bad_point_mat>0

def events_diagnostics(layout=None):
    """
    Since events were concatenated do a sanity check.
    For example, compares onsets from the concatenated events with events
    for each run. Also compares to nltools design matrix. Plots dm heatmap,
    checks onset for run with onset for longrun. Plots onset over heatmap.
    """

    trial_types = [44.3,0,43.3,42.3,45.3,46.3,47.3]

    if layout is None:
        data_dir = '~/ds000140/'
        layout = BIDSLayout(data_dir, derivatives=True)

    # random subject
    sub = np.random.randint(1,34)
    random_run = '0' + str(np.random.randint(1,10))

    if sub<10:
        sub = '0' + str(sub)
    else:
        sub = str(sub)

    # Load event file (longrun)
    onsets = pd.read_csv(layout.get(subject=sub,\
            task='longrun',\
            run='1',suffix='events',scope='raw')[0].path,\
            sep='\t')

    # Load event file (per run)
    onsets_run = pd.read_csv(layout.get(subject=sub,\
            task='heatpainwithregulationandratings',\
            run=random_run,suffix='events',scope='raw')[0].path,\
            sep='\t')

    # retreive design matrix
    dm = load_bids_events(layout,subject=sub,return_onsets=False,ratings=True) # this can be input into fir_design_matrix.m

    # visualize sections of the design matrix.
    # Sections correspond to the length of onsets_run.
    begin = round(onsets_run.iloc[0].onset/2.0) # divide by TR
    end = round(onsets_run.iloc[-1].onset/2.0)
    timeseries_range = range(begin,end)

    ax = sns.heatmap(dm.values[timeseries_range,:],cmap='gray_r')
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ytick_labels = ax.get_yticklabels()
    ytick_labels = [int(x.get_text()) for x in ytick_labels]

    onsets_for_run = []
    onsets_for_longrun = []
    earliest_onset = onsets_run.onset.values[0]
    for row in range(onsets_run.shape[0]):
        # onset time in seconds from a single run
        onset_time = onsets_run.iloc[row].onset
        # get the index of where this onset is found in the longrun
        cond_idx = np.where(np.round(onsets.onset.values,3)==round(onset_time,3))[0][0]
        # get row info of longrun at this idx
        row_info = onsets.iloc[cond_idx]
        # which condition/trial is it
        trial = onsets_run.iloc[row].trial_type

        if trial not in trial_types:
            continue

        # to which condition does this trial correspond?
        trial_idx = np.where(trial_types==trial)[0][0]
        # get the y point, used to plot horizontal line
        timepoint = (onset_time-earliest_onset)/2.0
        # todo. plot hlines from onsets on top of dm heatmap
        # plt.hlines(timepoint,trial_idx, trial_idx+1,'r')

        onsets_for_run.append(onset_time)
        onsets_for_longrun.append(row_info.onset)

    r = sts.pearsonr(onsets_for_run,onsets_for_longrun)[0]
    if r < 1:
        print('Mismatch between events')

def test_bad_timepoints_per_condition(layout=None):
    """
    Iterates each subject and imports design matrix, retrieves the index
    of timepoints that were scrubbed. Computes statistics for each pair
    of conditions and visualizes the results.
    """
    bad_timepoints_per_condition = np.zeros((5,33))

    if layout is None:
        data_dir = '~/ds000140/'
        layout = BIDSLayout(data_dir, derivatives=True)

    nsubs = 33

    _, bad_timepoint_mat = retrieve_scrubbing_info(suffix='/',style='_cubic',long=True)

    for subidx, sub in enumerate(range(1,nsubs+1)):
        subidx = sub-1
        print(sub)
        if subidx < 9:
            subject = '0' + str(sub)
        else:
            subject = str(sub)

        if subject in ['04','10']:
            continue
        # import design matrix
        dm = load_bids_events(layout,subject=subject,return_onsets=False,ratings=True) # this can be input into fir_design_matrix.m

        # retrieve timepoints associated with task for each condition
        idx1 = np.where(dm['42.3']>0)[0]
        idx2 = np.where(dm['43.3']>0)[0]
        idx3 = np.where(dm['44.3']>0)[0]
        idx4 = np.where(dm['45.3']>0)[0]
        idx5 = np.where(dm['46.3']>0)[0]


        # get scrubbed timepoints for this subject
        bad_timepoints_for_sub = bad_timepoint_mat[subidx]

        # collect bad timepoints for each condition
        bad_timepoints_per_condition[0, subidx] = bad_timepoints_for_sub[idx1].sum()
        bad_timepoints_per_condition[1, subidx] = bad_timepoints_for_sub[idx2].sum()
        bad_timepoints_per_condition[2, subidx] = bad_timepoints_for_sub[idx3].sum()
        bad_timepoints_per_condition[3, subidx] = bad_timepoints_for_sub[idx4].sum()
        bad_timepoints_per_condition[4, subidx] = bad_timepoints_for_sub[idx5].sum()
        # bad_timepoints_per_condition[5, subidx] = bad_timepoints_for_sub[idx6].sum()

    X = bad_timepoints_per_condition.copy()[:,remove_subjects_bool]
    # normality check
    stats_l, pvals_norm = [],[]
    for x in X:
        tmp_stat, tmp_p = scipy.stats.normaltest(x)
        stats_l.append(tmp_stat); pvals_norm.append(tmp_p)

    if any(np.array(pvals_norm)<0.05):
        print('H0: was rejected')
        center='median'
    else:
        print('H0: cannot be rejected')
        center='mean'
    # check equal variance with median
    W,p = scipy.stats.levene(X[0],X[1],X[2],X[3],X[4],center=center)
    if p<0.05:
        print('H0 of equal variance can be rejected')

    # statistics
    from netneurotools import stats
    stat_mat = np.zeros((5,5))
    pmat = np.zeros((5,5))
    for ii,x in enumerate(X):
        for jj, y in enumerate(X):
            average_diff_between_x_y, pval = stats.permtest_rel(x,y)
            stat_mat[ii,jj] = average_diff_between_x_y
            pmat[ii,jj] = pval

    # visualization
    fig, axes = plt.subplots(1,2)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cbar_kws={'shrink':.3,'use_gridspec':False,'location':'top'}
    mask = np.tril(np.ones_like(stat_mat, dtype=bool))
    labels = [r'$44.3^o$',r'$45.3^o$',r'$46.3^o$',r'$47.3^o$',r'$48.3^o$']

    sns.heatmap(stat_mat,mask=mask,center=0,cmap=cmap,ax=axes[0],cbar_kws=cbar_kws)
    sns.heatmap(-np.log10(pmat),mask=mask==0,vmin=1.2,vmax=1.3,cmap='Purples',ax=axes[1],cbar_kws=cbar_kws)

    axes[0].collections[0].colorbar.set_label('Average difference',fontsize='large')
    axes[0].collections[0].colorbar.set_ticks([-0.4,0,1])
    axes[0].collections[0].colorbar.set_ticklabels(['-0.4','0','1'])
    # ax.collections[0].colorbar.ax.tick_params(labelsize=14)
    # ax.collections[0].colorbar.ax.get_xaxis().set_label_coords(0.5,2)
    axes[1].collections[0].colorbar.set_label('-log10(p-value)',fontsize='large')
    axes[1].collections[0].colorbar.set_ticks([1.2,1.3])

    axes[0].set_yticks(np.arange(0.5,5.5,1))
    axes[0].set_xticks(np.arange(0.5,5.5,1))
    axes[1].set_xticks(np.arange(0.5,5.5,1))
    axes[1].set_yticks(np.arange(0.5,5.5,1))

    axes[0].set_yticklabels(labels)
    axes[0].set_xticklabels(labels)
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)
    axes[0].tick_params(left=False,bottom=False)
    axes[1].tick_params(left=False,bottom=False)

def retrieve_preprocessed_timeseries(sub='01',all_subjects=False, subset=None):
    """
    Retrieve cleaned and scrubbed timeseries from either single subject or all 

    return dataframe from one subject or a list of dataframes for >1 subject.
    """
    base = '~/ds000140/derivatives/teneto-censor-timepoints_longrun'

    datapath = base

    timeseries_dataframe_list = []

    if all_subjects: #
        subject_names = [subject for subject in os.listdir(datapath) if 'sub' in subject]
        for sub in subject_names: # sub-01, sub-02 ...
            timeseries_filename = sub + '_run-1_task-longrun_timeseries.tsv'
            timeseries_dataframe = pd.read_csv(datapath + sub + '/func/' + timeseries_filename, '\t',index_col=0)
            timeseries_dataframe_list.append(timeseries_dataframe)

        return timeseries_dataframe_list

    elif subset is not None: # subset of subjects
        pass

    else: # retrieve one subject
        sub = 'sub-' + sub # sub-01
        timeseries_filename = sub + '_run-1_task-longrun_timeseries.tsv'
        timeseries_dataframe = pd.read_csv(datapath + sub + '/func/' + timeseries_filename, '\t',index_col=0)

        return timeseries_dataframe

def extract_timeseries(layout=None, sub='01',n_timepoints='7'):
    """
    Leaves out inter-trial timepoints; capturing event-related action.
    """
    base = '~/ds000140/derivatives/teneto-censor-timepoints_longrun'
    savedir = '~/ds000140/derivatives/concatenated_timeseries' 

    if layout is None:
        data_dir = '~/ds000140/'
        layout = BIDSLayout(data_dir, derivatives=True)

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    ordered_keys = ['42.3','43.3','44.3','45.3','46.3']
    shortest_number_of_timepoints = 77 # , 73, 72, 77, 72] # # the shortest number of timepoints found among subjects is 72.
    all_timepoints = np.zeros((33,5))

    # There are 7 runs of length 209 timepoints, and 2 runs of 191 timepoints. Need to remove runs and keep only runs: 1, 2, 4, 8, and 9. The other four are either 'cognitive regulate up/down' or increases in temperature by a step
    # index_correct_runs = np.concatenate([np.repeat(1,209*2),np.repeat(0,191),np.repeat(1,209),np.repeat(0,209*2),np.repeat(0,191),np.repeat(1,209*2)])

    # load design matrix/matrices
    subject_names = [subject.split('-')[1] for subject in os.listdir(base) if 'sub' in subject]

    for sub_i,sub in enumerate(subject_names): # 01, 02 ...
        design_matrix__ = load_bids_events(layout, sub, task='longrun',run='01', return_onsets=False, ratings=False)
        dm_cols = design_matrix__.columns

        if n_timepoints=='7':
            design_matrix = remove_eighth_volume(design_matrix__)

        design_matrix = pd.DataFrame(design_matrix, columns=dm_cols)

        timeseries_node_times = retrieve_preprocessed_timeseries(suffix=suffix, style=style,sub=sub)
        timeseries_node_times = timeseries_node_times.T # make time, node to match design matrix

        save_subject_name = savedir + '/sub-' + sub

        if not os.path.exists(save_subject_name):
            os.makedirs(save_subject_name + '/func/')

        dfs = []

        dm_sum = np.sum(design_matrix,axis=1)==1
        concatenated_timeseries = timeseries_node_times.iloc[dm_sum.values,:] # keep only timepoints of interest
        concatenated_timeseries = concatenated_timeseries.T # save with shape node,time
        # Collect number of timepoints for all subjects to check if equal
        num_timepoints = concatenated_timeseries.shape[-1]
        all_timepoints[sub_i] = num_timepoints
        # concatenated_timeseries = concatenated_timeseries.iloc[:,:]

        if num_timepoints!=385:
            print('Mismatch in shape of dataframe: Sub: ' + sub + ' Length: ' + str(concatenated_timeseries.shape[-1]))

        concatenated_timeseries.to_csv(save_subject_name + '/func/' + 'sub-' + sub + '_run-1_task-longrun.tsv','\t')

def run_jackknife(weights=True):
    """
    For trial-time scale, a diagonal matrix is created with points covering a whole trial.
    I call this "leave-n-out", and is specifically "leave-7-out". (Even though the single time point scale is leave-1-out, I don't give it a particular name when saving the file)  
    """
    import os
    import numpy as np
    import pandas as pd
    from teneto.timeseries.derive import derive_temporalnetwork
    from teneto.utils import set_diagonal

    savedir = '~/ds000140/derivatives/teneto-derive-temporalnetwork/' 
    datadir = '~/ds000140/derivatives/concatenated_timeseries/'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if weights:
        derive_params = {'method': 'jackknife','postpro': 'fisher+standardize','user_weights':create_diagonal_matrix()}
    else:
        derive_params = {'method': 'jackknife','postpro': 'fisher+standardize'}

    subject_names = [subject.split('-')[1] for subject in os.listdir(datadir) if 'sub' in subject]

    for sub in subject_names: # 01, 02 ...
        save_subject_name = savedir + '/sub-' + sub + '/'

        if not os.path.exists(save_subject_name):
            os.makedirs(save_subject_name + '/func/')

        data = pd.read_csv(datadir + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun.tsv', '\t', index_col=0)

        # get timevarying connectivity
        network = derive_temporalnetwork(data, derive_params)
        network = set_diagonal(network, 0)

        shape = network.shape
        result = network.reshape([shape[0] * shape[1], shape[2]])
        result = pd.DataFrame(result)

        if weights:
            result.to_csv(save_subject_name + '/func/sub-' + sub + '_run-1_task-longrun_method-leave-n-out_tvcconn.tsv', sep='\t', header=True)
        else:
            result.to_csv(save_subject_name + '/func/sub-' + sub + '_run-1_task-longrun_tvcconn.tsv', sep='\t', header=True)

def run_SID(leaveNout='_method-leave-n-out'):
    """
    leaveNout : '_method-leave-n-out' or ''
    """
    import teneto.utils as utils
    from teneto.networkmeasures import temporal_degree_centrality

    communities = np.load('/home/granit/community_id.npy') # https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal 
    n_nodes = 400

    n_time = 385 # nvolumes x trials x conditions

    datadir = '~/ds000140/derivatives/teneto-derive-temporalnetwork/'
    saveTDC = '~/ds000140/derivatives/teneto-calc_networkmeasure-temporal_degree_centrality/'
    saveSID = '~/ds000140/derivatives/teneto-calc_networkmeasure-sid/' 

    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]
    conditions = ['42.3','43.3','44.3','45.3','46.3']

    calc = 'community_pairs_norm'
    axis = 0
    if not os.path.exists(saveTDC):
        os.mkdir(saveTDC)
    if not os.path.exists(saveSID):
        os.mkdir(saveSID)

    for sub in subjects:

        save_subject_name_TDC = saveTDC + '/sub-' + sub + '/'
        save_subject_name_SID = saveSID + '/sub-' + sub + '/'

        if not os.path.exists(save_subject_name_TDC):
            os.makedirs(save_subject_name_TDC + '/func/')

        if not os.path.exists(save_subject_name_SID):
            os.makedirs(save_subject_name_SID + '/func/')

        df = pd.read_csv(datadir + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun' + leaveNout + '_tvcconn.tsv','\t',index_col=0)

        if df.shape[-1]!=n_time:
            print('Shape is : ' + str(df.shape[-1] + '. Skipping.'))
            continue

        temp_net_arr = df.values.reshape(n_nodes,n_nodes,n_time)
        tnet = teneto.classes.TemporalNetwork(from_array=temp_net_arr)

        # Compute temp degree centrality
        tnet, netinfo = utils.process_input(tnet, ['C', 'G', 'TN'])
        D = temporal_degree_centrality(
            tnet, calc='pertime', communities=communities, decay=0)

        # Check network output (order of communitiesworks)
        network_ids = np.unique(communities)
        communities_size = np.array([sum(communities == n) for n in network_ids])

        # Compute SID
        sid = np.zeros([network_ids.max()+1, network_ids.max()+1, tnet.shape[-1]])
        for n in network_ids:
            for m in network_ids:
                betweenmodulescaling = 1/(communities_size[n]*communities_size[m])
                if netinfo['nettype'][1] == 'd':
                    withinmodulescaling = 1 / \
                        (communities_size[n]*communities_size[n])
                    withinmodulescaling_m = 1 / (communities_size[m]*communities_size[m])
                elif netinfo['nettype'][1] == 'u':
                    withinmodulescaling = 2 / \
                        (communities_size[n]*(communities_size[n]-1))
                    withinmodulescaling_m = 2 / (communities_size[m]*(communities_size[m]-1))
                    if n == m:
                        betweenmodulescaling = withinmodulescaling
                if calc == 'community_pairs_norm':
                    # Here normalize by avg of n and m
                    sid[n, m, :] = ((withinmodulescaling * D[n, n, :]) + (withinmodulescaling_m * D[m, m, :])) / 2 - betweenmodulescaling * D[n, m, :]
                else:
                    sid[n, m, :] = withinmodulescaling * \
                        D[n, n, :] - betweenmodulescaling * D[n, m, :]
        # If nans emerge than there is no connection between networks at time point, so make these 0.
        sid[np.isnan(sid)] = 0

        sid_global = np.sum(sid, axis=axis)

        # Save temporal degree centrality
        D_2D = pd.DataFrame(D.reshape(D.shape[0]*D.shape[0], n_time))
        D_2D.to_csv(save_subject_name_TDC + '/func/sub-' + sub + '_run-1_task-longrun' + leaveNout + '_tnet.tsv', sep='\t', header=True)

        # Save SID
        SID = pd.DataFrame(sid_global)
        SID.to_csv(save_subject_name_SID + '/func/sub-' + sub + '_run-1_task-longrun' + leaveNout + '_tnet.tsv', sep='\t', header=True)

def extract_event_tvc_sid(layout=None, method='sid',leave='method-leave-n-out_'):
    """leave can be set to '' if jackknife was computed with leave-1-out or be set to 'method-leave-n-out' if computed with leave-7-out"""

    datapath = '~/ds000140/derivatives/'
    if method=='sid':
        datapath = datapath + 'teneto-calc_networkmeasure-sid/'
    if method=='tnet':
        datapath = datapath + 'teneto-calc_networkmeasure-temporal_degree_centrality/'

    savedirbase = datapath

    if layout is None:
        data_dir = '~/ds000140/'
        layout = BIDSLayout(data_dir, derivatives=True)

    if not os.path.exists(savedirbase):
        os.mkdir(savedirbase)

    ordered_keys = ['42.3','43.3','44.3','45.3','46.3']
    shortest_number_of_timepoints = 77 # , 73, 72, 77, 72] # # the shortest number of timepoints found among subjects is 72.
    all_timepoints = np.zeros((33,5))

    # There are 7 runs of length 209 timepoints, and 2 runs of 191 timepoints. We need to remove runs and keep only runs: 1, 2, 4, 8, and 9. The other four are either 'cognitive regulate up/down' or increases in temperature by a step
    # index_correct_runs = np.concatenate([np.repeat(1,209*2),np.repeat(0,191),np.repeat(1,209),np.repeat(0,209*2),np.repeat(0,191),np.repeat(1,209*2)])

    # load design matrix/matrices
    subject_names = [subject.split('-')[1] for subject in os.listdir(datapath) if 'sub' in subject]

    for sub_i,sub in enumerate(subject_names): # 01, 02 ...
        design_matrix__ = load_bids_events(layout, sub, task='longrun',run='01', return_onsets=False, ratings=False)
        dm_cols = design_matrix__.columns
        design_matrix = remove_eighth_volume(design_matrix__)
        design_matrix = pd.DataFrame(design_matrix, columns=dm_cols)

        # Remove zeros from design matrix (removing inter-trial timepoints)
        design_matrix = design_matrix[(design_matrix == 1).values]
        timeseries_node_times = pd.read_csv(datapath + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_' + leave + 'tnet.tsv','\t',index_col=0)

        # transpose to match design matrix
        timeseries_node_times = timeseries_node_times.T

        save_subject_name = savedirbase + '/sub-' + sub

        if not os.path.exists(save_subject_name):
            os.makedirs(save_subject_name + '/func/')

        # Collect all timeseries to later be concatenated along timedimension
        dfs = []
        for key_idx, key in enumerate(ordered_keys):
            condition_idx = design_matrix[key].values == 1
            # correct_trial_and_correct_runs = np.logical_and(index_correct_runs, condition_idx)

            concatenated_timeseries = timeseries_node_times.iloc[condition_idx,:]
            concatenated_timeseries = concatenated_timeseries.T # save with shape node,time
            num_timepoints = concatenated_timeseries.shape[-1]

            all_timepoints[sub_i,key_idx] = num_timepoints

            if num_timepoints<77:
                print(sub_i)

            all_timepoints[sub_i,key_idx] = num_timepoints

            concatenated_timeseries = concatenated_timeseries.iloc[:,:]

            dfs.append(concatenated_timeseries)

        dfs = pd.concat(dfs,axis=1)
        dfs.to_csv(save_subject_name + '/func/' + 'sub-' + sub + '_run-1_task-longrun_desc-sorted_' + leave + 'tnet.tsv','\t')


def visualize_averaged_low_high():
    """
    Figure 2.
    """
    sorted = ''
    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]
    conditions = ['42.3','43.3','44.3','45.3','46.3']

    n_time = 55
    colors = get_network_colors()

    n_nets = 7
    palette = ['Purple','blue','green','violet','palegoldenrod','orange','crimson']

    SID = np.zeros((25,n_nets,11,5)) # subject, network, time, condition
    TDC = np.zeros((25,n_nets,n_nets,11,5)) # subject, network, network, time, condition

    base_dir = '~/ds000140/derivatives/'
    load_SID_path = base_dir + '/teneto-calc_networkmeasure-sid' 
    load_TDC_path = base_dir + '/teneto-calc_networkmeasure-temporal_degree_centrality' 

    dfs = []
    dfs_w = []
    dfs_b = []
    for sub_i, sub in enumerate(np.array(subjects)[remove_subjects_bool]):

        sid = pd.read_csv(load_SID_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_' + sorted + 'method-leave-n-out_tnet.tsv','\t',index_col=0)
        sid = get_one_val_per_trial(sid)

        tdc = pd.read_csv(load_TDC_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_' + sorted + 'method-leave-n-out_tnet.tsv','\t',index_col=0)
        tdc = get_one_val_per_trial(tdc)
        tdc_arr = tdc.values.reshape(n_nets,n_nets,n_time,order='F')

        # load design matrix and change ones to intensity names
        idx = np.zeros(385) # These three lines are from get_one_val_per_trial()
        idx = np.array([1 if x%7==0 else 0 for x in range(385)])
        idx = idx.astype(bool)

        design_matrix__ = load_bids_events(layout, sub, task='longrun',run='01', return_onsets=False, ratings=False)
        dm_cols = design_matrix__.columns
        design_matrix = remove_eighth_volume(design_matrix__)
        design_matrix = pd.DataFrame(design_matrix, columns=dm_cols)

        keys = design_matrix.keys()
        design_matrix.iloc[:,0].replace({1.0:float(keys[0])},inplace=True)
        design_matrix.iloc[:,1].replace({1.0:float(keys[1])},inplace=True)
        design_matrix.iloc[:,2].replace({1.0:float(keys[2])},inplace=True)
        design_matrix.iloc[:,3].replace({1.0:float(keys[3])},inplace=True)
        design_matrix.iloc[:,4].replace({1.0:float(keys[4])},inplace=True)

        design_matrix = design_matrix[(design_matrix != 0).values]

        new_idx = design_matrix[idx].sum(axis=1) # To obtain one column with event name (without interstimulusinterval)

        sidT = sid.T.copy()
        sidT.index = range(55)
        sidT['thermal_intensity'] = new_idx.values

        print(sidT[sidT.thermal_intensity==1.0].shape == (11,8))

        # Load ratings unsorted :
        rating = pd.read_csv('~/ratings/sub-' + str(sub) + '.tsv','\t')
        sidT['Ratings'] = rating.ratings
        sidT['therm'] = rating.thermal_intensity
        sidT['Subject'] = np.repeat(sub_i, 55)
        sidT['Trial'] = np.array(list(range(55)))
        dfs.append(sidT)

        # Within
        temporal_diag = np.diagonal(tdc_arr)
        temporal_diag_df = pd.DataFrame(temporal_diag)
        temporal_diag_df['thermal_intensity'] = new_idx.values
        temporal_diag_df['Ratings'] = rating.ratings
        temporal_diag_df['therm'] = rating.thermal_intensity
        temporal_diag_df['Subject'] = np.repeat(sub_i, 55)
        temporal_diag_df['Trial'] = np.array(list(range(55)))

        dfs_w.append(temporal_diag_df)

        # between
        F = np.zeros([7,55])
        for time_t in range(55):
            # Set diagonal (within network) to zero for condition i.
            diag_cond_i = tdc_arr[:,:,time_t].copy()
            np.fill_diagonal(diag_cond_i,0)
            F[:,time_t] = np.mean(diag_cond_i, axis=1)/2 # for each network, average over all other networks. # Divide since matrix is symmetric.

        temporal_b_df = pd.DataFrame(F.T)
        temporal_b_df['thermal_intensity'] = new_idx.values
        temporal_b_df['Ratings'] = rating.ratings
        temporal_b_df['therm'] = rating.thermal_intensity
        temporal_b_df['Subject'] = np.repeat(sub_i, 55)
        temporal_b_df['Trial'] = np.array(list(range(55)))

        dfs_b.append(temporal_b_df)

    dfsid = pd.concat(dfs)
    dfw = pd.concat(dfs_w)
    dfb = pd.concat(dfs_b)
    dfsid.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Intensity','Ratings','Therm','Subject','Trial']
    dfw.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Intensity','Ratings','Therm','Subject','Trial']
    dfb.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Intensity','Ratings','Therm','Subject','Trial']

    ### Demean and standardize
    mean = dfsid.groupby(['Intensity']).Ratings.transform('mean')
    std = dfsid.groupby(['Intensity']).Ratings.transform('std')
    dfsid['Rating (z)'] = (dfsid.Ratings-mean)/std
    dfw['Rating (z)'] = (dfsid.Ratings-mean)/std
    dfb['Rating (z)'] = (dfsid.Ratings-mean)/std

    ##############
    ### Run this piece for each measure # 'SID','\u03C9','\u03B2'
    ##############
    measure = '\u03B2'

    # change df to correspond to the desired measure 
    df_melt = dfb.melt(['Ratings','Intensity','Therm','Subject', 'Trial', 'Rating (z)'])

    df_melt.drop('Therm',1,inplace=True)
    df_melt.columns = ['Ratings', 'Therm', 'Subject', 'Trial', 'Rating (z)', 'Networks',measure]

    ### Added this to do the concatenate analysis according to reviewer comments.
    ### This part is the same style as in the code used to generate Figure 5.
    idx1 = df_melt['Therm']==42.3
    idx2 = df_melt['Therm']==43.3
    low = np.logical_or(idx1,idx2)
    idx3 = df_melt['Therm']==45.3
    idx4 = df_melt['Therm']==46.3
    high = np.logical_or(idx3,idx4)
    low_df = df_melt[low]
    low_df['Intensity'] = np.repeat('Low',3850)
    high_df = df_melt[high]
    high_df['Intensity'] = np.repeat('High',3850)
    new_df = pd.concat([low_df,high_df])

    new_df.columns = ['Ratings','Therm','Subject', 'Time', 'Rating (z)', 'Networks', measure,'Intensity']
    ##############

    ### Using the new_df produced above, paste it below
    # Collapse along time (trial n=55)
    dfsid_average = new_df.groupby(['Subject','Networks','Intensity']).mean().reset_index(level=[0,1,2])
    dfw_average = new_df.groupby(['Subject','Networks','Intensity']).mean().reset_index(level=[0,1,2])
    dfb_average = new_df.groupby(['Subject','Networks','Intensity']).mean().reset_index(level=[0,1,2])
    ###

    #### Rainplot instead of barplot
    dfw_average.columns = ['Subject', 'Networks', 'Intensity', 'Ratings', 'Therm', 'Time','Rating (z)', 'Within-network degree centrality']
    dfb_average.columns = ['Subject', 'Networks', 'Intensity', 'Ratings', 'Therm', 'Time','Rating (z)', 'Between-network degree centrality']

    dfsid_average.sort_values(['Intensity','Networks'],ascending=False,inplace=True)
    dfw_average.sort_values(['Intensity','Networks'],ascending=False,inplace=True)
    dfb_average.sort_values(['Intensity','Networks'],ascending=False,inplace=True)

    #same thing with a single command: now x **must** be the categorical value
    dx = "Networks"; ort = "h"; pal = "pastel"; sigma = .2
    # f, ax = plt.subplots(figsize=(7, 5))

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1,rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1,rowspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1,rowspan=1)

    dy = "SID"
    pt.RainCloud(x=dx, y=dy, hue='Intensity', data=dfsid_average, palette=pal, bw=sigma,
                 width_viol=0.6, ax=ax1, orient=ort, alpha=0.65, dodge=True,
                 pointplot=True)

    dy = "Within-network degree centrality"
    pt.RainCloud(x=dx, y=dy, hue='Intensity', data=dfw_average, palette=pal, bw=sigma,
                 width_viol=0.6, ax=ax2, orient=ort, alpha=0.65, dodge=True,
                 pointplot=True)

    dy = "Between-network degree centrality"
    pt.RainCloud(x=dx, y=dy, hue='Intensity', data=dfb_average, palette=pal, bw=sigma,
                 width_viol=0.6, ax=ax3, orient=ort, alpha=0.65, dodge=True,
                 pointplot=True)

    ax1.legend_.remove()
    ax2.legend_.remove()
    ax3.legend_.remove()

    sns.despine()

    ax1.set_yticks(range(7))
    # ax1.set_yticklabels(['Vis','SM','DA','SA','Limbic','FP','DMN'])
    ax1.set_ylabel('Networks')
    ax1.set_xlabel('SID',labelpad=18)

    ax2.set_yticks(range(7))
    # ax2.set_yticklabels(['Vis','SM','DA','SA','Limbic','FP','DMN'])
    ax2.set_ylabel('Networks')
    ax2.set_xlabel('Within-network degree',labelpad=18)

    ax3.set_yticks(range(7))
    # ax3.set_yticklabels(['Vis','SM','DA','SA','Limbic','FP','DMN'])
    ax3.set_ylabel('Networks')
    ax3.set_xlabel('Between-network degree',labelpad=18)

    import matplotlib.patches as mpatches
    c = sns.color_palette('pastel')
    blue = mpatches.Patch(color=c[0],fill=True,label='Low',alpha=0.65)
    orange = mpatches.Patch(color=c[1],fill=True,label='High',alpha=0.65)
    ax1.legend(handles=[orange,blue],bbox_to_anchor=(1.15,1.0),title='Intensity',frameon=False)

    # ax1.text(x=0.5,y=5,s='*',fontsize=19)

    ### Statistical test
    # test temporal network measure with t-test related
    from netneurotools.stats import permtest_rel
    # For each brain network, test high/low
    stats = []
    pvals = []
    measure='Between-network degree centrality'
    df = dfb_average.copy()
    for x in ['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN']:
         netdf = df[df.Networks==x]
         a = netdf[netdf.Intensity=='Low'][measure].values
         b = netdf[netdf.Intensity=='High'][measure].values
         stat, pval = permtest_rel(a, b)
         stats.append(stat)
         pvals.append(pval)

    corr_res = multipletests(pvals, method='fdr_bh')
    print('Stats:')
    print(stats)
    print('Pvals:')
    print(pvals)
    print('Corr pvals:')
    print(corr_res)

def visualise_trialdata_low_high(suffix='_with_hrf/', style='_cubic', measure='sid',sorted='desc-sorted_'):
    """
    Figure 3.
    measure: 'sid','within','between'
    """
    sorted = ''
    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]
    conditions = ['42.3','43.3','44.3','45.3','46.3']

    n_time = 55
    colors = get_network_colors(atlas='schaefer')

    n_nets = 7
    palette = ['Purple','blue','green','violet','palegoldenrod','orange','crimson']

    SID = np.zeros((25,n_nets,11,5)) # subject, network, time, condition
    TDC = np.zeros((25,n_nets,n_nets,11,5)) # subject, network, network, time, condition

    base_dir = '~/ds000140/derivatives/'
    load_SID_path = base_dir + '/teneto-calc_networkmeasure-sid'
    load_TDC_path = base_dir + '/teneto-calc_networkmeasure-temporal_degree_centrality'

    dfs = []
    dfs_w = []
    dfs_b = []
    for sub_i, sub in enumerate(np.array(subjects)[remove_subjects_bool]):

        sid = pd.read_csv(load_SID_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_' + sorted + 'method-leave-n-out_tnet.tsv','\t',index_col=0)
        sid = get_one_val_per_trial(sid)

        tdc = pd.read_csv(load_TDC_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_' + sorted + 'method-leave-n-out_tnet.tsv','\t',index_col=0)
        tdc = get_one_val_per_trial(tdc)
        tdc_arr = tdc.values.reshape(n_nets,n_nets,n_time,order='F')

        # load design matrix and change ones to intensity names
        idx = np.zeros(385) # These three lines are from get_one_val_per_trial()
        idx = np.array([1 if x%7==0 else 0 for x in range(385)])
        idx = idx.astype(bool)

        design_matrix__ = load_bids_events(layout, sub, task='longrun',run='01', return_onsets=False, ratings=False)
        dm_cols = design_matrix__.columns
        design_matrix = remove_eighth_volume(design_matrix__)
        design_matrix = pd.DataFrame(design_matrix, columns=dm_cols)

        keys = design_matrix.keys()
        design_matrix.iloc[:,0].replace({1.0:float(keys[0])},inplace=True)
        design_matrix.iloc[:,1].replace({1.0:float(keys[1])},inplace=True)
        design_matrix.iloc[:,2].replace({1.0:float(keys[2])},inplace=True)
        design_matrix.iloc[:,3].replace({1.0:float(keys[3])},inplace=True)
        design_matrix.iloc[:,4].replace({1.0:float(keys[4])},inplace=True)

        design_matrix = design_matrix[(design_matrix != 0).values]

        new_idx = design_matrix[idx].sum(axis=1) # To obtain one column with event name (without ITI)

        sidT = sid.T.copy()
        sidT.index = range(55)
        sidT['thermal_intensity'] = new_idx.values

        print(sidT[sidT.thermal_intensity==1.0].shape == (11,8))

        # Load ratings unsorted :
        rating = pd.read_csv('~/ratings/sub-' + str(sub) + '.tsv','\t')
        sidT['Ratings'] = rating.ratings
        sidT['therm'] = rating.thermal_intensity
        sidT['Subject'] = np.repeat(sub_i, 55)
        sidT['Trial'] = np.array(list(range(55)))
        dfs.append(sidT)


        # Within
        temporal_diag = np.diagonal(tdc_arr)
        temporal_diag_df = pd.DataFrame(temporal_diag)
        temporal_diag_df['thermal_intensity'] = new_idx.values
        temporal_diag_df['Ratings'] = rating.ratings
        temporal_diag_df['therm'] = rating.thermal_intensity
        temporal_diag_df['Subject'] = np.repeat(sub_i, 55)
        temporal_diag_df['Trial'] = np.array(list(range(55)))

        dfs_w.append(temporal_diag_df)

        # between
        F = np.zeros([7,55])
        for time_t in range(55):
            # Set diagonal (within network) to zero for condition i.
            diag_cond_i = tdc_arr[:,:,time_t].copy()
            np.fill_diagonal(diag_cond_i,0)
            F[:,time_t] = np.mean(diag_cond_i, axis=1)/2 # for each network, average over all other networks. # Divide since matrix is symmetric.

        temporal_b_df = pd.DataFrame(F.T)
        temporal_b_df['thermal_intensity'] = new_idx.values
        temporal_b_df['Ratings'] = rating.ratings
        temporal_b_df['therm'] = rating.thermal_intensity
        temporal_b_df['Subject'] = np.repeat(sub_i, 55)
        temporal_b_df['Trial'] = np.array(list(range(55)))

        dfs_b.append(temporal_b_df)

    dfsid = pd.concat(dfs)
    dfw = pd.concat(dfs_w)
    dfb = pd.concat(dfs_b)
    dfsid.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Intensity','Ratings','Therm','Subject','Trial']
    dfw.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Intensity','Ratings','Therm','Subject','Trial']
    dfb.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Intensity','Ratings','Therm','Subject','Trial']

    ### Demean and standardize
    mean = dfsid.groupby(['Intensity']).Ratings.transform('mean')
    std = dfsid.groupby(['Intensity']).Ratings.transform('std')
    dfsid['Rating (z)'] = (dfsid.Ratings-mean)/std
    dfw['Rating (z)'] = (dfsid.Ratings-mean)/std
    dfb['Rating (z)'] = (dfsid.Ratings-mean)/std

    measure = '\u03B2'
    df_melt = dfb.melt(['Ratings','Intensity','Therm','Subject', 'Trial', 'Rating (z)'])
    df_melt.drop('Therm',1,inplace=True)
    df_melt.columns = ['Ratings', 'Therm', 'Subject', 'Trial', 'Rating (z)', 'Networks',measure]

    ##########
    ### This part is the same style as in the code used to generate Figure 5.
    idx1 = df_melt['Therm']==42.3
    idx2 = df_melt['Therm']==43.3
    low = np.logical_or(idx1,idx2)
    idx3 = df_melt['Therm']==45.3
    idx4 = df_melt['Therm']==46.3
    high = np.logical_or(idx3,idx4)
    low_df = df_melt[low]
    low_df['Intensity'] = np.repeat('Low',3850)
    high_df = df_melt[high]
    high_df['Intensity'] = np.repeat('High',3850)
    new_df = pd.concat([low_df,high_df])

    new_df.columns = ['Ratings','Therm','Subject', 'Time', 'Rating (z)', 'Networks', measure,'Intensity']

    HL_df_TMP = new_df.copy()
    tmp_dfs = []
    high_low_df = pd.DataFrame()
    for net in ['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN']:
        tmp_df = HL_df_TMP[HL_df_TMP['Networks']==net]
        high_low_df[net] = tmp_df[measure].values.copy()

    # high_low_df.columns = ['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN']
    high_low_df[['Ratings', 'Therm', 'Subject', 'Time', 'Rating (z)','Intensity']] = tmp_df[['Ratings', 'Therm', 'Subject', 'Time', 'Rating (z)','Intensity']].values
    res = pg.pairwise_corr(high_low_df[['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN', 'Rating (z)']],columns='Rating (z)',method='skipped',padjust='fdr_bh')

    ###############

    # map lower and upper CI to their own columns
    CI95 = np.zeros((7,2))
    res = res_b.copy()
    for x_i in range(7):
        low = res.iloc[x_i]['CI95%'][0]
        high = res.iloc[x_i]['CI95%'][1]
        CI95[x_i, 0] = low
        CI95[x_i, 1] = high

    # Plot spearman's r barplot and mark significant
    barwidth = 0.3
    r1 = list(range(7))
    r2 = np.array(r1) + barwidth
    r3 = r2 + barwidth

    yerr_b = np.c_[res['r'].values - CI95[:,0], CI95[:,1] - res['r'].values].T # https://stackoverflow.com/questions/42919936/hard-coding-confidence-interval-as-whiskers-in-bar-plot

    # sid
    plt.figure()
    plt.bar(x=r1, height=res_sid['r'], width=barwidth, yerr=yerr_sid, alpha=0.65,color='grey') # https://www.python-graph-gallery.com/8-add-confidence-interval-on-barplot
    plt.bar(x=r2, height=res_w['r'], width=barwidth, yerr=yerr_w, alpha=0.65, hatch='x',color='grey')
    plt.bar(x=r3, height=res_b['r'], width=barwidth, yerr=yerr_b, alpha=0.65, hatch='/',color='grey')

    plt.xticks(list(range(7)),['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN'])
    plt.ylabel('Spearman\'s $r$')
    plt.legend(['SID','Within-network degree','Between-network degree'],frameon=False,loc='lower center',fontsize=16)
    sns.despine()

def get_timeseries_single_timepoint(measure='sid'):
    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]
    conditions = ['42.3','43.3','44.3','45.3','46.3']

    colors = get_network_colors()

    n_time = 385

    n_nets = 7
    palette = ['Purple','blue','green','violet','palegoldenrod','orange','crimson']

    SID = np.zeros((33,n_nets,n_time)) # subject, network, time, condition
    TDC = np.zeros((33,n_nets,n_nets,n_time)) # subject, network, network, time, condition

    base_dir = '~/ds000140/derivatives/'
    load_SID_path = base_dir + '/teneto-calc_networkmeasure-sid' 
    load_TDC_path = base_dir + '/teneto-calc_networkmeasure-temporal_degree_centrality'

    for sub_i, sub in enumerate(subjects):
        try:
            sid = pd.read_csv(load_SID_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_desc-sorted_tnet.tsv','\t',index_col=0)

            tdc = pd.read_csv(load_TDC_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_desc-sorted_tnet.tsv','\t',index_col=0)
            tdc_arr = tdc.values.reshape(n_nets,n_nets,n_time)

        except FileNotFoundError:
            print(FileNotFoundError)

        SID[sub_i,:,:] = sid.values
        TDC[sub_i,:,:,:] = tdc_arr

    print('SID.shape == ' + str(SID.shape)) # (33, 7, 77, 5) # nsub, network, volume x trial, temps,   
    print('TDC.shape == ' + str(TDC.shape)) # (33, 7, 7, 77, 5) # nsub, network, volume x trial, temps,   

def visualize_volumedata_low_high(measure='sid'):
    """
    Figure 5.

    uses select_timepoint to extract the first, second, ... last volume
    """
    subjects = ['0' + str(x) if x<10 else str(x) for x in range(1,34)]
    conditions = ['42.3','43.3','44.3','45.3','46.3']

    colors = get_network_colors()

    n_time = 385 # n_volumes(=7) x n_trials(=11) x n_temperatures(=5) 
    n_vols = 7 
    n_nets = 7
    n_trials=55 
    palette = ['Purple','blue','green','violet','palegoldenrod','orange','crimson']

    base_dir = '~/ds000140/derivatives/'
    load_SID_path = base_dir + '/teneto-calc_networkmeasure-sid'
    load_TDC_path = base_dir + '/teneto-calc_networkmeasure-temporal_degree_centrality'

    measure = '\u03C9' # SID, '\u03B2','\u03C9'
    HL_df_s = []

    for timepoint in range(n_vols):  # used for the idx variable in the for loop.
        dfs = []
        dfs_w = []
        dfs_b = []
        for sub_i, sub in enumerate(np.array(subjects)[remove_subjects_bool]):
            try:
                sid = pd.read_csv(load_SID_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_tnet.tsv','\t',index_col=0)
                sid = sid.values # [:,select].copy()

                tdc = pd.read_csv(load_TDC_path + '/sub-' + sub + '/func/sub-' + sub + '_run-1_task-longrun_tnet.tsv','\t',index_col=0)
                tdc_arr = tdc.values.reshape(n_nets,n_nets,n_time)
                tdc_arr = tdc_arr # [:,:,select].copy()

            except FileNotFoundError:
                print(FileNotFoundError)

            # SID[sub_i,:,:] = sid.values
            # TDC[sub_i,:,:,:] = tdc_arr

            # load design matrix and change 1's to intensity names
            # This is used for design matrix as well as the data arrays.
            # For the design matrix, a column for stimuli condition x_i contains similar labels within a trial.
            # For example, temp_1, temp_1, ... temp_1. Idx here just selects one value per trial (mapping the array of 385 timepoints to 55 (55 equals number of trials i.e. 11*5 for 5 temperatures and 11 trials each))
            # For data arrays, with lag = 0 (i.e. x%7==0) gives the data at the first volume, for each trial. Lag=1 (i.e. x%7==1) gives the data at the second volume. And so on.
            idx = np.zeros(n_time) # These three lines are from get_one_val_per_trial()
            idx = np.array([1 if x%7==timepoint else 0 for x in range(n_time)])
            idx = idx.astype(bool)

            design_matrix__ = load_bids_events(layout, sub, task='longrun',run='01', return_onsets=False, ratings=False)
            dm_cols = design_matrix__.columns
            design_matrix = remove_eighth_volume(design_matrix__)
            design_matrix = pd.DataFrame(design_matrix, columns=dm_cols)

            keys = design_matrix.keys()
            design_matrix.iloc[:,0].replace({1.0:float(keys[0])},inplace=True)
            design_matrix.iloc[:,1].replace({1.0:float(keys[1])},inplace=True)
            design_matrix.iloc[:,2].replace({1.0:float(keys[2])},inplace=True)
            design_matrix.iloc[:,3].replace({1.0:float(keys[3])},inplace=True)
            design_matrix.iloc[:,4].replace({1.0:float(keys[4])},inplace=True)

            design_matrix = design_matrix[(design_matrix != 0).values]

            new_idx = design_matrix[idx].sum(axis=1)

            sidT = sid[:,idx].T.copy()
            sidT = pd.DataFrame(sidT)
            sidT.index = range(n_trials)
            sidT['thermal_intensity'] = new_idx.values

            print(sidT[sidT.thermal_intensity==1.0].shape == (11,8))

            # Load ratings unsorted :
            rating = pd.read_csv('~/ratings/sub-' + str(sub) + '.tsv','\t')
            sidT['Ratings'] = rating.ratings
            sidT['therm'] = rating.thermal_intensity
            sidT['Subject'] = np.repeat(sub_i, n_trials)
            sidT['Trial'] = np.array(list(range(n_trials)))
            dfs.append(sidT)

            # Within
            temporal_diag = np.diagonal(tdc_arr[:,:,idx])
            temporal_diag_df = pd.DataFrame(temporal_diag)
            temporal_diag_df['Ratings'] = rating.ratings
            temporal_diag_df['therm'] = rating.thermal_intensity
            temporal_diag_df['Subject'] = np.repeat(sub_i, n_trials)
            temporal_diag_df['Trial'] = np.array(list(range(n_trials)))

            dfs_w.append(temporal_diag_df)

            # between
            F = np.zeros([7,idx.shape[0]])
            for time_t in range(idx.shape[0]):
                # Set diagonal (within network) to zero for condition i.
                diag_cond_i = tdc_arr[:,:,time_t].copy()
                np.fill_diagonal(diag_cond_i,0)
                F[:,time_t] = np.mean(diag_cond_i, axis=1)/2 # for each network, average over all other networks. # Divide since matrix is symmetric.

            F = F[:, idx]
            temporal_b_df = pd.DataFrame(F.T)
            temporal_b_df['Ratings'] = rating.ratings
            temporal_b_df['therm'] = rating.thermal_intensity
            temporal_b_df['Subject'] = np.repeat(sub_i, n_trials)
            temporal_b_df['Trial'] = np.array(list(range(n_trials)))

            dfs_b.append(temporal_b_df)

        df = pd.concat(dfs)
        df.drop('thermal_intensity',1,inplace=True)
        dfw = pd.concat(dfs_w)
        dfb = pd.concat(dfs_b)
        df.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Ratings','Therm','Subject','Trial']
        dfw.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Ratings','Therm','Subject','Trial']
        dfb.columns = ['Vis','SM','DA','SA','Limbic','FP','DMN','Ratings','Therm','Subject','Trial']

        ### Demean and standardize
        mean = df.groupby(['Therm']).Ratings.transform('mean')
        std = df.groupby(['Therm']).Ratings.transform('std')
        df['Rating (z)'] = (df.Ratings-mean)/std
        dfw['Rating (z)'] = (df.Ratings-mean)/std
        dfb['Rating (z)'] = (df.Ratings-mean)/std

        if measure == 'SID': # SID, '\u03B2','\u03C9'
            df_melt = df.melt(['Ratings','Therm', 'Subject', 'Trial', 'Rating (z)'])
            df_melt.columns = ['Ratings','Therm', 'Subject', 'Trial', 'Rating (z)','Networks',measure]

        elif measure == '\u03C9': # SID, '\u03B2','\u03C9'
            df_melt = dfw.melt(['Ratings','Therm', 'Subject', 'Trial', 'Rating (z)'])
            df_melt.columns = ['Ratings','Therm', 'Subject', 'Trial', 'Rating (z)','Networks',measure]

        elif measure == '\u03B2': # SID, '\u03B2','\u03C9'
            df_melt = dfb.melt(['Ratings','Therm', 'Subject', 'Trial', 'Rating (z)'])
            df_melt.columns = ['Ratings','Therm', 'Subject', 'Trial', 'Rating (z)','Networks',measure]

        df_melt['Pain'] = np.array(df_melt.Ratings>100).astype(int)
        df_melt['Ratings_Pain'] = df_melt[df_melt.Ratings>100].Ratings
        df_melt['Ratings_Pain'] = df_melt.Ratings_Pain.replace({np.nan:100})

        # Create low/high by combining first two temps into "low", and last 2 into "high".
        idx1 = df_melt['Therm']==42.3
        idx2 = df_melt['Therm']==43.3
        low = np.logical_or(idx1,idx2)
        idx3 = df_melt['Therm']==45.3
        idx4 = df_melt['Therm']==46.3
        high = np.logical_or(idx3,idx4)
        low_df = df_melt[low]
        low_df['Intensity'] = np.repeat('Low',3850)
        high_df = df_melt[high]
        high_df['Intensity'] = np.repeat('High',3850)
        new_df = pd.concat([low_df,high_df])

        new_df.columns = ['Ratings','Therm','Subject', 'Time', 'Rating (z)', 'Networks', measure, 'Pain','Ratings_Pain','Intensity']

        new_df['Timepoint'] = np.repeat(timepoint+1, 3850*2)
        HL_df_s.append(new_df)

    # Collect for each timepoint
    HL_df_SID = pd.concat(HL_df_s)
    HL_df_W = pd.concat(HL_df_s)
    HL_df_B = pd.concat(HL_df_s)

    ### Barplot
    tmp_dfs = []
    #HL_df_TMP = HL_df[HL_df.Timepoint==1].copy()
    # Restructrue df to long format.
    HL_df_TMP = HL_df_B.copy()
    for net in ['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN']:
        tmp_df = HL_df_TMP[HL_df_TMP['Networks']==net]
        for timepoint in range(1,8):
            tmp_df_t = pd.DataFrame(tmp_df[tmp_df.Timepoint==timepoint][[measure]].values)
            tmp_df_t.columns = [net + '_' + str(timepoint)]
            tmp_dfs.append(tmp_df_t)

    high_low_df = pd.concat(tmp_dfs,axis=1)
    # high_low_df.columns = ['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN']
    high_low_df['Ratings'] = tmp_df[tmp_df.Timepoint==timepoint]['Ratings'].values
    columns = high_low_df.columns

    res = pg.pairwise_corr(high_low_df,columns='Ratings',method='skipped',padjust='fdr_bh')

    # map lower and upper CI to their own columns
    CI95 = np.zeros((49,2))
    for x_i in range(49):
        low = res.iloc[x_i]['CI95%'][0]
        high = res.iloc[x_i]['CI95%'][1]
        CI95[x_i, 0] = low
        CI95[x_i, 1] = high

    # Plot spearman's r barplot and mark significant
    barwidth = 0.3
    r1 = list(range(7))
    r2 = np.array(r1) + barwidth
    r3 = r2 + barwidth

    yerr = np.c_[res['r'].values - CI95[:,0], CI95[:,1] - res['r'].values].T # https://stackoverflow.com/questions/42919936/hard-coding-confidence-interval-as-whiskers-in-bar-plot


    # Facetgrid barplot
    import matplotlib as mlp

    # res = res_b.copy()
    # yerr = yerr_w.copy()

    res['yerr_l'] = yerr[0]
    res['yerr_u'] = yerr[1]
    res['Time'] = [x.split('_')[1] for x in res.Y]
    res['Networks'] = [x.split('_')[0] for x in res.Y]

    grid = sns.FacetGrid(data=res,col='Networks',hue="Time",
                        legend_out=True,height=3.5,aspect=2,margin_titles=True)

    grid.map(plt.bar, 'Time','r',**{'color':'black','alpha':0.35,'yerr':None})

    axes = grid.axes
    for net_i, net_name in enumerate(['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN']):
        res_net = res[res.Networks==net_name]
        yerr_net = yerr[:, net_i*7:(net_i+1)*7]
        ax_i = axes[0][net_i]
        ax_i.errorbar(list(range(7)), res_net['r'],yerr_net)
        axes[0][net_i].errorbar(list(range(7)),res_net['r'],yerr_net,color='black')

        # Mark significant by coloring the bar crimson
        if any(res_net['p-corr']<0.05):
            where_sig = np.where(res_net['p-corr']<0.05)
            bars = [rect for rect in axes[0][net_i].get_children() if isinstance(rect, mlp.patches.Rectangle)]
            for timepoint in where_sig[0]:
                bars[timepoint].set_color('crimson')
                bars[0].set_alpha(0.65)

    # Fix some bars
    bars = [rect for rect in axes[0][0].get_children() if isinstance(rect, mlp.patches.Rectangle)]
    bars[0].set_color('grey')
    bars = [rect for rect in axes[0][2].get_children() if isinstance(rect, mlp.patches.Rectangle)]
    bars[0].set_color('grey')
    bars = [rect for rect in axes[0][2].get_children() if isinstance(rect, mlp.patches.Rectangle)]
    bars[0].set_color('grey')
    bars = [rect for rect in axes[0][3].get_children() if isinstance(rect, mlp.patches.Rectangle)]
    bars[0].set_color('grey')
    bars = [rect for rect in axes[0][4].get_children() if isinstance(rect, mlp.patches.Rectangle)]
    bars[0].set_color('grey')
    bars = [rect for rect in axes[0][5].get_children() if isinstance(rect, mlp.patches.Rectangle)]
    bars[0].set_color('grey')
    bars = [rect for rect in axes[0][6].get_children() if isinstance(rect, mlp.patches.Rectangle)]
    bars[0].set_color('grey')

    grid.set_titles(col_template='{col_name}')
    axes[0][0].set_ylabel('Spearman\'s $r$')

def permutation_test_volumes_low_high():
    """
    Figure 4.

    Get tensors for sid and temporal degree centrality from get_timeseries_single_timepoint.
    """
    from mne.stats import permutation_cluster_test, ttest_1samp_no_p
    from functools import partial
    from scipy import stats

    nsubs = 25
    sigma = 1e-3
    threshold = -stats.distributions.t.ppf(0.05, nsubs - 1)
    threshold_tfce = dict(start=0, step=0.2)
    nperm = 10000
    stat_fun = partial(ttest_1samp_no_p, sigma=sigma)

    network_name = ['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN']

    ###### SID #######
    x_axis = list(range(7)) # volumes

    network_name = ['Vis','SM','DA','SA','Limbic','FP','DMN']
    dfs = []
    # make dataframes. Each dataframe is a network. Each row is SID at a particular timepoint and thermal intensity
    for sub_i in range(nsubs):
        temp_net =  SID[remove_subjects_bool][sub_i].copy()
        temp_net = temp_net.reshape(7,7,11,5,order='F').mean(axis=2)
        for network_i in range(n_nets):
            network_df = pd.DataFrame(temp_net[network_i,:,:],columns=[1,2,3,4,5]).melt()
            network_df['time'] = x_axis*5
            network_df['network'] = np.repeat(network_name[network_i],network_df.shape[0])
            network_df['subject'] = np.repeat(str(sub_i), network_df.shape[0])
            dfs.append(network_df)

    df = pd.concat(dfs)
    df.columns = ['Intensity','SID','Volume','Network','Subject']
    # df['Intensity'] = df['Intensity'].replace({1:'44.3',2:'45.3',3:'46.3',4:'47.3',5:'48.3'})

    ###--------###
    ### within ###
    ###--------###
    x_axis = list(range(7)) # volumes
    network_name = ['Vis','SM','DA','SA','Limbic','FP','DMN']

    # Take diagnal for each node,node matrix. That is, iterate over volumes and conditions.
    dfs = []
    for sub_i in range(nsubs):
        sub_i_mat = TDC[remove_subjects_bool][sub_i]
        sub_i_mat = sub_i_mat.reshape(7,7,7,11,5,order='F').mean(axis=3)

        diag = np.diagonal(sub_i_mat) # returns vol, cond, net
        diag = diag.transpose([-1,0,1]) # -> net, vol, cond

        for network_i in range(n_nets):
            sub_i_df = pd.DataFrame(diag[network_i, :, :],columns=[1,2,3,4,5]).melt()
            sub_i_df['time'] = x_axis*5
            sub_i_df['network'] = np.repeat(network_name[network_i],sub_i_df.shape[0])
            sub_i_df['subject'] = np.repeat(str(sub_i), sub_i_df.shape[0])
            dfs.append(sub_i_df)

    df = pd.concat(dfs)
    df.columns = ['Intensity','\u03C9','Volume','Network','Subject']
    # df['Intensity'] = df['Intensity'].replace({1:'44.3',2:'45.3',3:'46.3',4:'47.3',5:'48.3'})

    ###---------###
    ### Between ###
    ###---------###
    # Take diagnal for each node,node matrix. That is, iterate over volumes and conditions.
    dfs = []
    for sub_i in range(nsubs):
        sub_i_mat = TDC[remove_subjects_bool][sub_i]
        sub_i_mat = sub_i_mat.reshape(7,7,7,11,5,order='F').mean(axis=3)

        test = np.zeros((7,7,5))
        for cond_i in range(5):
            for vol_j in range(7):
                diag = sub_i_mat[:,:,vol_j,cond_i].copy()
                np.fill_diagonal(diag,0)

                test[:,vol_j,cond_i] = np.mean(diag,axis=1)/2 # Divide since matrix is symmetric.

        for network_i in range(n_nets):
            sub_i_df = pd.DataFrame(test[network_i, :, :],columns=[1,2,3,4,5]).melt()
            sub_i_df['time'] = x_axis*5
            sub_i_df['network'] = np.repeat(network_name[network_i],sub_i_df.shape[0])
            sub_i_df['subject'] = np.repeat(str(sub_i), sub_i_df.shape[0])
            dfs.append(sub_i_df)

    df = pd.concat(dfs)
    df.columns = ['Intensity','\u03B2','Volume','Network','Subject']
    # df['Intensity'] = df['Intensity'].replace({1:'44.3',2:'45.3',3:'46.3',4:'47.3',5:'48.3'})

    #################################
    ### statistical test and plot ###
    ### Make the plot

    measure = '\u03C9' # SID, '\u03B2','\u03C9'

    # Basically what I do here is to remove the middle temperature and change name of "level" from 1-5 to low (1,2) and high (4,5)
    df = df.copy()
    # Create low/high by combining first two temps into "low", and last 2 into "high".
    low1 = df['Intensity']==1
    low2 = df['Intensity']==2
    low =  np.logical_or(low1,low2)
    high1 = df['Intensity']==4
    high2 = df['Intensity']==5
    high = np.logical_or(high1,high2)
    low_df = df[low]
    low_df['Intensity'] = np.repeat('Low',2450)
    high_df = df[high]
    high_df['Intensity'] = np.repeat('High',2450)
    new_df = pd.concat([low_df,high_df])

    # Statistical testing between low and high for each network.
    d = {}
    for network_i, net_name in enumerate(network_name):
        tmp_net = new_df[new_df['Network']==net_name]
        X1 = tmp_net[tmp_net.Intensity=='Low'][measure].values.reshape(2*nsubs,7) # shape is level(1,2),nsubjects,nvolumes
        X2 = tmp_net[tmp_net.Intensity=='High'][measure].values.reshape(2*nsubs,7) # shape is level(4,5)
        X = [X1,X2]
        f_obs, cluster, cluster_pvals, h0 = permutation_cluster_test(X, threshold=threshold_tfce,
                                                    n_permutations=nperm)

        d[str(network_i)] = [f_obs, cluster,cluster_pvals, h0]

    ## correct for multiple comparisons
    from statsmodels.stats.multitest import multipletests
    n_vols, n_nets = 7, 7
    res = np.zeros((n_nets,n_vols))
    corr_res = np.zeros((n_nets,n_vols))

    # collet p values for each network. Each p-val vector is 1 x num_vols
    for key_i, key in enumerate(d.keys()):
        res[key_i, :] = d[key][2]

    # correct for multiple comp.
    for timepoint in range(7):
        _,corr_pvals,_,_ = multipletests(res[:,timepoint],method='fdr_bh')
        corr_res[:,timepoint] = corr_pvals


    # Plot
    # Collect high/low for sid/w/b into one dataframe
    df_HL_SID_volume = new_df.copy() # Then rerun the above
    df_HL_W_volume = new_df.copy() # Then rerun the above
    df_HL_B_volume = new_df.copy() # Then rerun the above
    df_HL_SID_volume['Measure'] = np.repeat('SID',4900)
    df_HL_W_volume['Measure'] = np.repeat('\u03C9',4900)
    df_HL_B_volume['Measure'] = np.repeat('\u03B2',4900)
    df_HL_SID_volume.columns=['Intensity','Value','Time', 'Network', 'Subject', 'Measure']
    df_HL_W_volume.columns=['Intensity','Value','Time', 'Network', 'Subject', 'Measure']
    df_HL_B_volume.columns=['Intensity','Value','Time', 'Network', 'Subject', 'Measure']

    df = pd.concat([df_HL_SID_volume,df_HL_W_volume, df_HL_B_volume])
    ax = sns.relplot(data=df,x='Time',y='Value',row='Measure',col='Network',hue='Intensity',
                    palette=sns.color_palette('pastel')[0:2],
                    kind='line',marker='o',legend='full',facet_kws={'sharey':'row','sharex':False,'legend_out':True})
    ax.map(plt.axhline, y=0, ls=":", c=".5")

    leg = ax._legend
    leg.set_bbox_to_anchor([1,0.8])

    ax.set_titles(col_template='{col_name}',row_template='{row_name}')

    # Adjust the arrangement of the plots
    # grid.fig.tight_layout(w_pad=1)
    ax.set(xticks=range(7))
    ax.set_xticklabels(range(1,8))
    # ax.set(yticks=[-0.2,-0.1,0,0.1,0.20])

    plt.subplots_adjust(right=0.92)
    plt.tight_layout()

    axes = ax.axes


    # fill grey bands
    # Gray if sig. Crimson if FDR.
    DB = np.zeros((7,7))
    for x in d_sid.keys():
        DB[int(x),:] = d_sid[x][2]

    for x in range(7): # networks
        for y in range(7): # timepoints
            if DB[x,y]<0.05:
                ax_i = axes[0][x]
                if corr_res_sid[x,y]<=0.05:
                    ax_i.fill_betweenx((-0.18,0.18),x1=y-0.5, x2=y+0.5,color='crimson',alpha=0.15)
                else:
                    ax_i.fill_betweenx((-0.18,0.18),x1=y-0.5, x2=y+0.5,color='grey',alpha=0.15)

    # W
    DB = np.zeros((7,7))
    for x in d_w.keys():
        DB[int(x),:] = d_w[x][2]

    for x in range(7): # networks
        for y in range(7): # timepoints
            if DB[x,y]<0.05:
                ax_i = axes[1][x]
                if corr_res_w[x,y]<=0.05:
                    ax_i.fill_betweenx((-150,150),x1=y-0.5, x2=y+0.5,color='crimson',alpha=0.15)
                else:
                    ax_i.fill_betweenx((-150,150),x1=y-0.5, x2=y+0.5,color='grey',alpha=0.15)

    # B
    d = d_b.copy()
    DB = np.zeros((7,7))
    for x in d_b.keys():
        DB[int(x),:] = d_b[x][2]

    for x in range(7): # networks
        for y in range(7): # timepoints
            if DB[x,y]<0.05:
                ax_i = axes[2][x]
                if corr_res_b[x,y]<=0.05:
                    ax_i.fill_betweenx((-85,100),x1=y-0.5, x2=y+0.5,color='crimson',alpha=0.15)
                else:
                    ax_i.fill_betweenx((-85,100),x1=y-0.5, x2=y+0.5,color='grey',alpha=0.15)

    for i, ax_i in enumerate(axes):
        ax_i[0].set_ylabel(measures[i],rotation='horizontal',labelpad=10)

    for i, ax_i in enumerate(axes):
        for j, ax_j in enumerate(ax_i):
            if i==0:
                ax_j.set_title(['Vis', 'SM', 'DA', 'SA', 'Limbic', 'FP', 'DMN'][j],fontsize=16)
            else:
                ax_j.set_title('',fontsize=1)


def load_bids_events(layout, subject, task='longrun',run='1', return_onsets=False, ratings=False):
    ''' see utility.py'''
    pass

def change_filename():
    """ See utility.py """
    pass

def create_diagonal_matrix():
    """ See utility.py """
    pass

def get_network_colors(atlas='schaefer'):
    """ Check utility.py """
    pass
    
def remove_eighth_volume(design_matrix):
    """ See utility.py """
    pass

def get_one_val_per_trial(network):
    """ See utility.py"""
    pass

def save_ratings_unsorted():
    """ See utility.py """
    pass

def check_ratings():
    """ See utility.py """
    pass
