
import pandas as pd
import numpy as np
from nltools.file_reader import onsets_to_dm


def load_bids_events(layout, subject, task='longrun',run='1', return_onsets=False, ratings=False):
    '''
    For painregulate
    Create a design_matrix instance from BIDS event file

    or task = 'heatpainwithregulationandratings'
    '''
    tr = 2.0
    n_tr = 1845

    onsets = pd.read_csv(layout.get(subject=subject,
                            task=task,
                            run=run,
                            suffix='events',
                            scope='raw')[0].path,
                            sep='\t')

    onsets = onsets[['onset','duration','trial_type']]
    onsets.columns = ['Onset', 'Duration', 'Stim']

    if ratings:
        # For painregulate
        onsets['Stim'] = onsets['Stim'].replace({
                    42.3:'42.3',
                    43.3:'43.3',
                    44.3:'44.3',
                    45.3:'45.3',
                    46.3:'46.3',
                    47.3:'47.3',
                    48.3:'48.3',
                    np.nan:'ratings'}
        )

    else:
        onsets = onsets.dropna()
        # For painregulate
        onsets['Stim'] = onsets['Stim'].replace({
                    42.3:'42.3',
                    43.3:'43.3',
                    44.3:'44.3',
                    45.3:'45.3',
                    46.3:'46.3',
                    47.3:'47.3',
                    48.3:'48.3',
                    }
        )

    onsets['Stim'] = onsets['Stim'].replace({
                42.1:np.nan,
                40.8:np.nan,
                46.0:np.nan,
                43.4:np.nan,
                44.7:np.nan
                }
    )

    # for some reason I had to do this again for a few subs.
    if subject in ['13','16','17','21','33']:
        onsets['Stim'] = onsets['Stim'].replace({
                    42.3:'42.3',
                    43.3:'43.3',
                    44.3:'44.3',
                    45.3:'45.3',
                    46.3:'46.3',
                    47.3:'47.3',
                    48.3:'48.3',
                    }
        )

    if return_onsets:
        return onsets
    else:
        return onsets_to_dm(onsets.dropna(), sampling_freq=1/2.0, run_length=n_tr, keep_separate=False)



def get_one_val_per_trial(network):
    """
    from tvc or tnet data with leave-7-out approach. Take every eighth val.
    """
    if network is not None:
        idx = np.zeros(385)
        idx = np.array([1 if x%7==0 else 0 for x in range(385)])
        idx = idx.astype(bool)

        if len(network.shape)>2:
            newdata = network.iloc[:,:,idx]
        else:
            newdata = network.iloc[:,idx]

    return newdata


def get_network_colors(atlas='schaefer'):

    if atlas == 'schaefer':
        names = np.load('/home/granit/communities.npy', allow_pickle=True)
        colors = []
        for x in names:
            if 'Vis' in x:
                colors.append('Purple')
            if 'SomMot' in x:
                colors.append('blue')
            if 'DorsAttn' in x:
                colors.append('green')
            if 'SalVentAttn' in x:
                colors.append('violet')
            if 'Limbic' in x:
                colors.append('palegoldenrod')
            if 'Cont' in x:
                colors.append('orange')
            if 'Default' in x:
                colors.append('crimson')

    return colors


def remove_eighth_volume(design_matrix):
    """For a few trials for a few subjects, there are 8 instead of 7 volumes."""
    previous_value = 0
    new_lst = []

    shape = design_matrix.shape
    new_dm = np.zeros((shape[0],shape[1]))


    for col_i in range(shape[1]):
        for row_j in range(shape[0]):
            elem = int(design_matrix.iloc[row_j, col_i])

            if not elem: # set to zero when elem==0
                count = 0

            if elem:  # Found true
                if (count+1)==8:
                    continue

                new_dm[row_j, col_i] = elem
                count +=1

    return new_dm


def change_filename():
    subject_name = [x for x in os.listdir(d) if 'sub' in x]

    for sub_name in subject_name:
        source = d + sub_name + '/func/' + sub_name + '_run-1_task-longrun_timeseries.tsv'
        if not os.path.exists(source):
            continue
        destination = d + sub_name + '/func/' + sub_name + '_run-1_task-longrun_roi.tsv'
        os.rename(source,destination)


def permtest_spearmanr(a,b, n_perm=10000):
    from scipy import stats
    from sklearn.utils.validation import check_random_state
    rs = check_random_state(0)

    true_corr = stats.spearmanr(a,b)[0] / 1 # /1 forces coercion to float if ndim=0
    abs_true = np.abs(true_corr)

    permutations = np.ones(true_corr.shape)
    for perm in range(n_perm):
        ap = a[rs.permutation(len(a))]
        permutations += np.abs(stats.spearmanr(ap,b)[0]) >= abs_true

    pvals = permutations / (n_perm + 1) # +1 in denominator accounts for true_corr

    return true_corr, pvals

def create_diagonal_matrix():

    from scipy.sparse import coo_matrix, block_diag

    adjs = []
    for x in range(55):
        A = coo_matrix(np.ones((7,7)))
        adjs.append(A)

    adj = block_diag(adjs)

    weights = adj.toarray()

    return weights==0 # diagonal entries are zero and off-diagonal are ones


def save_ratings_unsorted():
    """
    Save these csv. Each trial is associated with a rating.
    Sort according to condition, then trial.
    """
    datadir = '~/ds000140/'
    subject_names = [subject for subject in os.listdir(datadir) if 'sub-' in subject]
    subject_names.sort()

    for subject in np.array(subject_names)[remove_subjects_bool]:
        df = pd.read_csv(datadir + subject + '/func/' + subject + '_task-longrun_run-01_events.tsv','\t',index_col=0)
        # remove regulation trials and context trials, leaving 55 timepoints.
        ratings = df.shift(-1)[~df.trial_type.isna()]
        thermal_intensity_names = df[~df.trial_type.isna()].trial_type
        df = pd.DataFrame(ratings.ratings)
        df['trial'] = df.index
        df['thermal_intensity'] = thermal_intensity_names.values
        df.to_csv('~/ratings_unsorted/' + subject + '.tsv','\t')

def check_ratings():
    datadir = '~/openneuro/ds000140/'
    subject_names = [subject for subject in os.listdir(datadir) if 'sub-' in subject]
    subject_names.sort()

    d = {}
    runs = ['0' + str(x) for x in [1,2,4,8,9]]
    for subject in subject_names:
        unique_temperatures = []
        dfs = []
        for run in runs:
            df = pd.read_csv(datadir + '/sub-' + subject + '/func/sub-' + subject + '_task-heatpainwithregulationandratings_run-' + run + '_events.tsv','\t')
            unique_temperatures.append(df.temperature.dropna().unique())
            dfs.append(df.temperature)

        df = pd.concat(dfs)
        d[subject] =  unique_temperatures
    onsets = onsets[['onset','duration','trial_type']]
    onsets.columns = ['Onset', 'Duration', 'Stim']
