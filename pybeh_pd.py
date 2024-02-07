import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from numpy import matlib
from scipy.spatial import distance, distance_matrix
# from pybeh.make_recalls_matrix import make_recalls_matrix
# from pybeh.crp import crp
# from pybeh.spc import spc
# from pybeh.sem_crp import sem_crp
# from pybeh.temp_fact import temp_fact
# from pybeh.dist_fact import dist_fact, dist_percentile_rank
# from pybeh.mask_maker import make_clean_recalls_mask2d
from pybeh_copy import crp, temp_fact, make_clean_recalls_mask2d, dist_percentile_rank, temp_percentile_rank

def get_itemno_matrices(evs, itemno_column='itemno', list_index=['subject', 'session', 'list']):
    """Expects as input a dataframe (df) for one subject"""
    evs.loc[:, itemno_column] = evs.loc[:, itemno_column].astype(int)
    evs['pos'] = evs.groupby(list_index).cumcount()
    itemnos_df = pd.pivot_table(evs, values=itemno_column, 
                                 index=list_index, 
                                 columns='pos', fill_value=0)
    itemnos = itemnos_df.values
    return itemnos

# def pd_spc(df, start_position=None, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type'):
#     """Expects as input a dataframe (df) for one subject"""
#     pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
#       pres_type=pres_type, rec_type=rec_type, type_column=type_column)
#     listLength = pres_itemnos.shape[1]
#     prob = spc(recalls=recalls,
#                subjects=['_'] * recalls.shape[0],
#                listLength=listLength,
#                start_position=start_position)[0]
#     sps = np.arange(listLength) + 1 if start_position is None else np.arange(start_position, listLength + 1)
#     d = {'prob': prob, 'serialpos': sps}
#     return pd.DataFrame(d, index=sps)

def make_recalls_matrix(pres_itemnos=None, rec_itemnos=None, max_n_reps=1):
    '''
    MAKE_RECALLS_MATRIX   Make a standard recalls matrix.
    Given presented and recalled item numbers, finds the position of
    recalled items in the presentation list. Creates a standard
    recalls matrix for use with many toolbox functions.
    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos)
    INPUTS:
    pres_itemnos:  [trials X items] matrix of item numbers of
                 presented items. Must be positive.
    rec_itemnos:  [trials X recalls] matrix of item numbers of recalled
                  items. Must match pres_itemnos. Items not in the
                  stimulus pool (extra-list intrusions) should be
                  labeled with -1. Rows may be padded with zeros or
                  NaNs.
    max_n_reps: maximum number of repetitions for repeated presentations
    OUTPUTS:
    recalls:  [trials X recalls] matrix. For recall(i,j), possible
             values are:
             >0   correct recall. Indicates the serial position(s) in
                  which the recalled item was presented.
              0   used for padding rows. Corresponds to no recall.
             <0   intrusion of an item not presented on the list.
    :param pres_itemnos:
    :param rec_itemnos:
    :return:
    '''
    n_trials = np.shape(pres_itemnos)[0]
    n_items = np.shape(pres_itemnos)[1]
    n_recalls = np.shape(rec_itemnos)[1]
    
    recalls = np.zeros([n_trials, n_recalls, max_n_reps], dtype=int)

    for trial in np.arange(n_trials):
        for recall in np.arange(n_recalls):
            if (rec_itemnos[trial, recall]) == 0 | (np.isnan(rec_itemnos[trial, recall])):
                continue

            elif rec_itemnos[trial, recall] > 0:

                serialpos = np.where(rec_itemnos[trial, recall] == pres_itemnos[trial,:])[0]+1
                
                if len(serialpos) > max_n_reps:
                    raise Exception('An item was presented more than max_n_reps.')
                if not any(serialpos):
                    recalls[trial, recall, :] = -1
                else:
                    recalls[trial, recall, :len(serialpos)] = serialpos
            else:
                recalls[trial, recall, :] = -1
    if max_n_reps == 1:
        recalls = np.squeeze(recalls, axis=2)
    return recalls

def make_poss_recalls_matrix(pres_itemnos=None, max_n_reps=1):
    n_trials = np.shape(pres_itemnos)[0]
    n_items = np.shape(pres_itemnos)[1]
    
    recalls = np.zeros([n_trials, n_items, max_n_reps], dtype=int)

    for trial in np.arange(n_trials):
        for item in np.arange(n_items):
            if (pres_itemnos[trial, item]) == 0 | (np.isnan(pres_itemnos[trial, item])):
                continue

            elif pres_itemnos[trial, item] > 0:

                serialpos = np.where(pres_itemnos[trial,item] == pres_itemnos[trial,:])[0]+1
                
                if len(serialpos) > max_n_reps:
                    raise Exception('An item was presented more than max_n_reps.')
                if not any(serialpos):
                    recalls[trial, item, :] = -1
                else:
                    recalls[trial, item, :len(serialpos)] = serialpos
            else:
                recalls[trial, item, :] = -1
    if max_n_reps == 1:
        recalls = np.squeeze(recalls, axis=2)
    return recalls

def get_min_trans(serialpos, rec):
    # positive values come first so argmin will select positive values
    # See Howard and Kahana 2005 for method -- always select positive in case of tie
    pt = [sp - r for sp in serialpos for r in rec]
    pt.sort(reverse=True)
    return pt[np.argmin(np.abs(pt))]

def get_all_matrices(df, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type', max_n_reps=1):
    types = [pres_type, rec_type]
    #only include lists if both presentations and recalls are present (i.e. ntypes == 2)
    df = df.query(type_column + ' in @types')
    ntypes_df = df[list_index + [type_column]].groupby(list_index).agg({type_column: 'nunique'}).reset_index().rename(columns={type_column: 'ntypes'})
    df = df.merge(ntypes_df).query('ntypes == 2')
    pres_itemnos = get_itemno_matrices(df.query(type_column + ' == @pres_type'), 
                                       itemno_column=itemno_column, 
                                       list_index=list_index)
    rec_itemnos = get_itemno_matrices(df.query(type_column + ' == @rec_type'), 
                                       itemno_column=itemno_column, 
                                       list_index=list_index)
    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos, max_n_reps=max_n_reps)
    return pres_itemnos, rec_itemnos, recalls

def pd_crp(df, lag_num=5, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    min_lag_num = min(pres_itemnos.shape[1], lag_num)
    if len(recalls) == 0:
        return pd.DataFrame()
    if min_lag_num != 0:
        prob = crp(recalls=recalls,
                    subjects=['_'] * recalls.shape[0],
                    listLength=pres_itemnos.shape[1],
                    lag_num=lag_num)[0]
    else:
        prob = np.full((lag_num*2)+1, np.nan)
    crp_dict = {'prob': prob, 
                'lag': np.arange(-lag_num, (lag_num+1))}
    return pd.DataFrame(crp_dict, index=np.arange(-lag_num, (lag_num+1)))

def pd_min_crp(df, lag_num=5, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type', max_n_reps=1):
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column, max_n_reps=max_n_reps)
    poss_recalls = make_poss_recalls_matrix(pres_itemnos=pres_itemnos, max_n_reps=max_n_reps)
    
    min_lag_num = min(pres_itemnos.shape[1], lag_num)
    if len(recalls) == 0:
        return pd.DataFrame()
    if min_lag_num != 0:
        prob = min_crp(recalls=recalls,
                       poss_recalls=poss_recalls,
                    subjects=['_'] * recalls.shape[0],
                    listLength=pres_itemnos.shape[1],
                    lag_num=lag_num)[0]
    else:
        prob = np.full((lag_num*2)+1, np.nan)
    crp_dict = {'prob': prob, 
                'lag': np.arange(-lag_num, (lag_num+1))}
    return pd.DataFrame(crp_dict, index=np.arange(-lag_num, (lag_num+1)))

def min_crp(recalls=None, poss_recalls=None, subjects=None, listLength=None, lag_num=None, skip_first_n=0):
    '''
    CRP   Conditional response probability as a function of lag (lag-CRP).
    
      lag_crps = min_crp(recalls_matrix, poss_recalls, subjects, list_length, lag_num)
    
      INPUTS:
             recalls:  A 2D or 3D iterable whose elements are serial positions of
                       recalled items.  The first dimension of this array should
                       represent recalls made by a single subject on a
                       single trial. The second two dimensions represent the serial
                       poisition(s) of potentially repeated presentations.
                       
        poss_recalls:  A 2D or 3D iterable whose elements are serial positions of
                       recalled items.  The first dimension of this array should
                       represent presentations for a single subject on a
                       single trial. The second two dimensions represent the serial
                       poisition(s) of potentially repeated presentations.
    
            subjects:  A column vector which indexes the rows of "recalls"
                       with a subject number (or other identifier).  The
                       subject identifiers should be repeated for each
                       row of "recalls" originating from the same subject.
    
         list_length:  A scalar indicating the number of serial positions in
                       the presented lists.  Serial positions are assumed to
                       run from 1:list_length.
    
             lag_num:  A scalar indicating the max number of lags to track.
    
        skip_first_n:  An integer indicating the number of recall
                       transitions to ignore from the start of the recall
                       period, for the purposes of calculating the CRP.
                       This can be useful to avoid biasing your results, as
                       the first 2-3 transitions are almost always
                       temporally clustered.  Note that the first n recalls
                       will still count as already recalled words for the
                       purposes of determining which transitions are
                       possible.  (DEFAULT=0)
    
    
      OUTPUTS:
            lag_crps:  A matrix of lag-CRP values.  Each row contains the
                       values for one subject.  It has as many columns as
                       there are possible transitions (i.e., the length of
                       (-list_length + 1) : (list_length - 1) ).  The center
                       column, corresponding to the "transition of length 0,"
                       is guaranteed to be filled with NaNs.  Any lag_crps
                       element which had no possible transitions for the
                       input data for that subject will also have a value of
                       NaN.
                       For example, if list_length == 4, a row in lag_crps
                       has 7 columns, corresponding to the transitions from
                       -3 to +3:
                       lag-CRPs:     [ 0.1  0.2  0.3  NaN  0.3  0.1  0.0 ]
                       transitions:    -3   -2    -1   0    +1   +2   +3
    '''
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if lag_num is None:
        lag_num = listLength - 1
    elif lag_num < 1 or lag_num >= listLength or not isinstance(lag_num, int):
        raise ValueError('Lag number needs to be a positive integer that is less than the list length.')
    if not isinstance(skip_first_n, int):
        raise ValueError('skip_first_n must be an integer.')
        
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects -- we will calculate a CRP for each
    usub = np.unique(subjects)
    # Number of possible lags = (listLength - 1) * 2 + 1; e.g. a length-24 list can have lags -23 through +23
    num_lags = 2 * listLength - 1
    # Initialize array to store the CRP for each subject (or other unique identifier)
    result = np.zeros((usub.size, num_lags))
    # Initialize arrays to store transition counts
    actual = np.empty(num_lags)
    poss = np.empty(num_lags)
    
     # For each subject/unique identifier
    for i, subj in enumerate(usub):
        # Reset counts for each participant
        actual.fill(0)
        poss.fill(0)
        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        mask_recalls = recalls[subjects == subj]
        if mask_recalls.ndim == 3:
            with_repeats = True
            mask_recalls = mask_recalls[:, :, 0]
        else:
            with_repeats = False
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(mask_recalls))
        # For each trial that matches that identifier
        for j, trial_recs in enumerate(recalls[subjects == subj]):
            seen = set()
            for k, rec in enumerate(trial_recs[:-1]):
                #remove 0s and add recalled sps to seen
                rec = rec[rec != 0]
                seen.update(rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j][k] and clean_recalls_mask[j][k + 1] and k >= skip_first_n:
                    next_rec = trial_recs[k + 1]
                    next_rec = next_rec[next_rec != 0]
                    if with_repeats:
                        pt = np.unique(np.array([get_min_trans(serialpos[serialpos != 0], rec) for serialpos in poss_recalls[j] if serialpos[0] not in seen], dtype=int))#don't increment more than once
                    else:
                        pt = np.unique(np.array([get_min_trans(serialpos[serialpos != 0], rec) for serialpos in poss_recalls[j] if serialpos not in seen], dtype=int))#don't increment more than once
                    # for min lag-crp, we get the minimum possible distances
                    poss[pt + listLength - 1] += 1
                    trans = get_min_trans(next_rec, rec)

                    # Record the actual transition that was made
                    actual[trans + listLength - 1] += 1

        result[i, :] = [a/p if p!=0 else np.nan for a,p in zip(actual, poss)]

    result[:, listLength - 1] = np.nan

    return result[:, (listLength - lag_num - 1):(listLength + lag_num)]

def get_sim_mat(df, sim_cols, itemno_col='itemno', word_val_type="WORD_VALS", p=2, type_column='type'):
    word_val_df = df.query(type_column + ' == @word_val_type').drop_duplicates().sort_values(itemno_col)
    sem_sims = distance_matrix(word_val_df[sim_cols].values, word_val_df[sim_cols].values, p=p)
    return sem_sims

def pd_sem_crp(df, itemno_column='itemno', 
                list_index=['subject', 'session', 'list'], sim_columns=None,
                sem_sims=None, n_bins=10, bins=None, pres_type="WORD", 
               rec_type="REC_WORD", word_val_type="WORD_VALS", type_column='type', ret_counts=False):
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    
    if sem_sims is None:
        if df.query('type == @word_val_type')[itemno_column].min() != 1:
            print('expects itemnos to start at 1')
            print('current word val itemnos start at ' + str(df.query('type == @word_val_type')[itemno_column].min()))
        sem_sims = get_sim_mat(df, sim_columns, itemno_col=itemno_column, word_val_type=word_val_type,
                               type_column=type_column)
    else:
        if df.query(itemno_column + ' == 0').shape[0] != 0:
            print('check itemnos column, includes 0 values')
    if bins is not None:
        n_bins = len(bins)
    
    if len(recalls) == 0:
        return pd.DataFrame()
    out = sem_crp(recalls=recalls, 
                 recalls_itemnos=rec_itemnos, 
                 pres_itemnos=pres_itemnos, 
                 subjects=['_'] * recalls.shape[0], 
                 sem_sims=sem_sims, 
                 n_bins=n_bins, 
                 bins=bins,
                 listLength=pres_itemnos.shape[1],
                 ret_counts=ret_counts)
    if ret_counts:
        bin_means, crp, actual, poss = out
    else:
        bin_means, crp = out
    crp_dict = {'prob': crp[0], 
                'sem_bin_mean': bin_means[0],
                'sem_bin': np.arange(n_bins)
               }
    if ret_counts:
        crp_dict['actual'] = actual
        crp_dict['poss'] = poss
        
    return pd.DataFrame(crp_dict).query('prob == prob') #remove bins with no data

def min_temp_fact(recalls=None, poss_recalls=None, subjects=None, listLength=None, skip_first_n=0):
    """
    Returns the lag-based temporal clustering factor for each subject (Polyn, Norman, & Kahana, 2009).
    :param recalls: A trials x recalls matrix containing the serial positions (between 1 and listLength) of words
        recalled on each trial. Intrusions should appear as -1, and the matrix should be padded with zeros if the number
        of recalls differs by trial.
    :param subjects: A list/array containing identifiers (e.g. subject number) indicating which subject completed each
        trial.
    :param listLength: A positive integer indicating the number of items presented on each trial.
    :param skip_first_n: An integer indicating the number of recall transitions to ignore from the start of each recall
        period, for the purposes of calculating the clustering factor. This can be useful to avoid biasing your results,
        as early transitions often differ from later transition in terms of their clustering. Note that the first n
        recalls will still count as already recalled words for the purposes of determining which transitions are
        possible. (DEFAULT=0)
    :return: An array containing the temporal clustering factor score for each subject (sorted by alphabetical order).
    """

    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    if subjects is None:
        raise Exception('You must pass a subjects vector.')
    if listLength is None:
        raise Exception('You must pass a list length.')
    if len(recalls) != len(subjects):
        raise Exception('The recalls matrix must have the same number of rows as the list of subjects.')
    if not isinstance(skip_first_n, int) or skip_first_n < 0:
        raise ValueError('skip_first_n must be a nonnegative integer.')

    # Convert recalls and subjects to numpy arrays if they are not arrays already
    recalls = np.array(recalls)
    subjects = np.array(subjects)

    # Initialize range for possible next recalls, based on list length
    possibles_range = range(1, listLength + 1)

    # Initialize arrays to store each participant's results
    usub = np.unique(subjects)
    total = np.zeros_like(usub, dtype=float)
    count = np.zeros_like(usub, dtype=float)
    
    mask_recalls = recalls
    if mask_recalls.ndim == 3:
        with_repeats = True
        mask_recalls = mask_recalls[:, :, 0]
    else:
        with_repeats = False

    # Identify locations of all correct recalls (not PLI, ELI, or repetition)
    clean_recalls_mask = np.array(make_clean_recalls_mask2d(mask_recalls))

    # Calculate temporal factor score for each trial
    for i, trial_data in enumerate(recalls):
        seen = set()
        # Identify the current subject's index in usub to determine their position in the total and count arrays
        subj_ind = np.where(usub == subjects[i])[0][0]
        # Loop over the recalls on the current trial
        for j, rec in enumerate(trial_data[:-1]):
            rec = rec[rec != 0]
            seen.update(rec)
            # Only count transition if both the current and next recalls are valid
            if clean_recalls_mask[i, j] and clean_recalls_mask[i, j+1] and j >= skip_first_n:
                # Identify possible transitions
                # possibles = np.array([abs(item - serialpos) for item in possibles_range if item not in seen])
                possibles = abs(np.unique(np.array(
                    [get_min_trans(serialpos[serialpos != 0], rec) for serialpos in poss_recalls[i] if serialpos[0] not in seen], 
                    dtype=int)))#don't increment more than once
                # Identify actual transition
                # next_serialpos = trial_data[j + 1]
                next_rec = trial_data[j + 1]
                next_rec = next_rec[next_rec != 0]
                # Record the actual transition that was made
                # actual = abs(next_serialpos - serialpos)
                actual = abs(get_min_trans(next_rec, rec))
                # Find the proportion of transition lags that were larger than the actual transition
                ptile_rank = temp_percentile_rank(actual, possibles)
                # Add transition to the appropriate participant's score
                if ptile_rank is not None:
                    total[subj_ind] += ptile_rank
                    count[subj_ind] += 1

    # Find temporal factor scores as the participants' average transition scores
    count[count == 0] = np.nan
    final_data = total / count

    return final_data

def pd_min_temp_fact(df, skip_first_n=0, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type', word_val_type="WORD_VALS", max_n_reps=1):
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column, max_n_reps=max_n_reps)
    poss_recalls = make_poss_recalls_matrix(pres_itemnos=pres_itemnos, max_n_reps=max_n_reps)
    
    #check if subject has any recalls
    if pres_itemnos.shape[1] == 0:
        return np.nan
    
    temp_fact_arr = min_temp_fact(recalls=recalls, 
                              subjects=['_']*recalls.shape[0],
                              listLength=pres_itemnos.shape[1],
                              skip_first_n=skip_first_n,
                             poss_recalls=poss_recalls)
    return temp_fact_arr[0]

def pd_temp_fact(df, skip_first_n=0, itemno_column='itemno', list_index=['subject', 'session', 'list'], pres_type="WORD", rec_type="REC_WORD", type_column='type', word_val_type="WORD_VALS"):
    """Expects as input a dataframe (df) for one subject"""
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    
    #check if subject has any recalls
    if pres_itemnos.shape[1] == 0:
        return np.nan
    
    temp_fact_arr = temp_fact(recalls=recalls, 
                  subjects=['_']*recalls.shape[0],
                  listLength=pres_itemnos.shape[1],
                  skip_first_n=skip_first_n)
    
    return temp_fact_arr[0]

def pd_dist_fact(df, rec_itemnos=None, itemno_column='itemno', 
                 list_index=['subject', 'session', 'list'], 
                 dist_mat=None, sim_columns=None, is_similarity=False, 
                 dist_columns=None,
                 skip_first_n=0,
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", type_column='type', word_val_type="WORD_VALS", ret_counts=False
                ):
    pres_itemnos, rec_itemnos, recalls = get_all_matrices(df, itemno_column=itemno_column, list_index=list_index, 
      pres_type=pres_type, rec_type=rec_type, type_column=type_column)
    
    #check if subject has any recalls
    if pres_itemnos.shape[1] == 0:
        return np.nan
    
    if dist_mat is None:
        if df.query('type == @word_val_type')[itemno_column].min() != 1:
            print('expects itemnos to start at 1')
            print('current word val itemnos start at ' + str(df.query('type == @word_val_type')[itemno_column].min()))
        dist_mat = get_sim_mat(df, sim_columns, itemno_col=itemno_column, 
                               type_column=type_column, word_val_type=word_val_type)
    
    dist_fact_arr = dist_fact(rec_itemnos=rec_itemnos, 
              pres_itemnos=pres_itemnos, 
              subjects=['_'] * recalls.shape[0],
              dist_mat=dist_mat, is_similarity=is_similarity, 
              skip_first_n=skip_first_n)
    return dist_fact_arr[0]

def sem_crp(recalls=None, recalls_itemnos=None, pres_itemnos=None, subjects=None, sem_sims=None, n_bins=10, bins=None, listLength=None, ret_counts=False):
    """bins should not include an upper bin"""
    if recalls_itemnos is None:
        raise Exception('You must pass a recalls-by-item-numbers matrix.')
    elif pres_itemnos is None:
        raise Exception('You must pass a presentations-by-item-numbers matrix.')
    elif sem_sims is None:
        raise Exception('You must pass a semantic similarity matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a listLength')
    elif len(recalls_itemnos) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    # Make sure that all input arrays and matrices are numpy arrays
    recalls = np.array(recalls, dtype=int)
    recalls_itemnos = np.array(recalls_itemnos, dtype=int)
    pres_itemnos = np.array(pres_itemnos, dtype=int)
    subjects = np.array(subjects)
    sem_sims = np.array(sem_sims, dtype=float)

    # Set diagonal of the similarity matrix to nan
    np.fill_diagonal(sem_sims, np.nan)
    # Sort and split all similarities into equally sized bins
    all_sim = sem_sims.flatten()
    all_sim = np.sort(all_sim[~np.isnan(all_sim)])
    if bins is None:
        bins = np.array_split(all_sim, n_bins)
        bins = [b[0] for b in bins]
    else:
        n_bins = len(bins)
    # Convert the similarity matrix to bin numbers for easy bin lookup later
    bin_sims = np.digitize(sem_sims, bins) - 1

    # Convert recalled item numbers to the corresponding indices of the similarity matrix by subtracting 1
    recalls_itemnos -= 1
    pres_itemnos -= 1

    usub = np.unique(subjects)
    bin_means = np.zeros((len(usub), n_bins))
    crp = np.zeros((len(usub), n_bins))
    # For each subject
    for i, subj in enumerate(usub):
        # Create a filter to select only the current subject's data
        subj_mask = subjects == subj
        subj_recalls = recalls[subj_mask]
        subj_rec_itemnos = recalls_itemnos[subj_mask]
        subj_pres_itemnos = pres_itemnos[subj_mask]

        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(subj_recalls))

        # Setup counts for number of possible and actual transitions, as well as the sim value of actual transitions
        actual = np.zeros(n_bins)
        poss = np.zeros(n_bins)
        val = np.zeros(n_bins)

        # For each of the current subject's trials
        for j, trial_recs in enumerate(subj_recalls):
            seen = set()
            # For each recall on the current trial
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j, k] and clean_recalls_mask[j, k+1]:
                    this_recno = subj_rec_itemnos[j, k]
                    next_recno = subj_rec_itemnos[j, k+1]
                    # Lookup semantic similarity and its bin between current recall and next recall
                    sim = sem_sims[this_recno, next_recno]
                    b = bin_sims[this_recno, next_recno]

                    # Get a list of not-yet-recalled word numbers
                    poss_rec = np.array([subj_pres_itemnos[j][x] for x in range(listLength) if x+1 not in seen])
                    # Lookup the similarity bins between the current recall and all possible correct recalls
                    poss_trans = np.unique([bin_sims[this_recno, itemno] for itemno in poss_rec])
                    actual[b] += 1
                    val[b] += sim
                    
                    for b in poss_trans:
                        poss[b] += 1

        crp[i, :] = actual / poss  # CRP is calculated as number of actual transitions / number of possible ones
        bin_means[i, :] = val / actual  # Bin means are defined as the average similarity of actual transitions per bin
    if ret_counts:
        return bin_means, crp, actual, poss
    else:
        return bin_means, crp
    
def dist_fact(rec_itemnos=None, pres_itemnos=None, subjects=None, dist_mat=None, is_similarity=False, skip_first_n=0, ret_counts=False):
    """
    Returns a clustering factor score for each subject, based on the provided distance metric (Polyn, Norman, & Kahana,
    2009). Can also be used with a similarity matrix (e.g. LSA, word2vec) if is_similarity is set to True.
    :param rec_itemnos: A trials x recalls matrix containing the ID numbers (between 1 and N) of the items recalled on
        each trial. Extra-list intrusions should appear as -1, and the matrix should be padded with zeros if the number
        of recalls differs by trial.
    :param pres_itemnos: A trials x items matrix containing the ID numbers (between 1 and N) of the items presented on
        each trial.
    :param subjects: A list/array containing identifiers (e.g. subject number) indicating which subject completed each
        trial.
    :param dist_mat: An NxN matrix (where N is the number of words in the wordpool) defining either the distance or
        similarity between every pair of words in the wordpool. Whether dist_mat defines distance or similarity can be
        specified with the is_similarity parameter.
    :param is_similarity: If False, dist_mat is assumed to be a distance matrix. If True, dist_mat is instead treated as
        a similarity matrix (i.e. larger values correspond to smaller distances). (DEFAULT = False)
    :param skip_first_n: An integer indicating the number of recall transitions to ignore from the start of each recall
        period, for the purposes of calculating the clustering factor. This can be useful to avoid biasing your results,
        as early transitions often differ from later transition in terms of their clustering. Note that the first n
        recalls will still count as already recalled words for the purposes of determining which transitions are
        possible. (DEFAULT = 0)
    :return: An array containing the clustering factor score for each subject (sorted by alphabetical order).
    """

    if rec_itemnos is None:
        raise Exception('You must pass a recall_itemnos matrix.')
    if pres_itemnos is None:
        raise Exception('You must pass a pres_itemnos matrix.')
    if subjects is None:
        raise Exception('You must pass a subjects vector.')
    if dist_mat is None:
        raise Exception('You must pass either a similarity matrix or a distance matrix.')
    if len(rec_itemnos) != len(subjects) or len(pres_itemnos) != len(subjects):
        raise Exception('The rec_itemnos and pres_itemnos matrices must have the same number of rows as the list of'
                        'subjects.')
    if not isinstance(skip_first_n, int) or skip_first_n < 0:
        raise ValueError('skip_first_n must be a nonnegative integer.')

    # Convert inputs to numpy arrays if they are not arrays already
    rec_itemnos = np.array(rec_itemnos).astype(int)
    pres_itemnos = np.array(pres_itemnos).astype(int)
    subjects = np.array(subjects)
    dist_mat = np.array(dist_mat)

    # Provide a warning if the user inputs a dist_mat that looks like a similarity matrix (scores on diagonal are
    # large), but has left is_similarity as False
    if (not is_similarity) and np.nanmean(np.diagonal(dist_mat)) > np.nanmean(dist_mat):
        warnings.warn('It looks like you might be using a similarity matrix (e.g. LSA, word2vec) instead of a distance'
                      ' matrix, but you currently have is_similarity set to False. If you are using a similarity'
                      ' matrix, make sure to set is_similarity to True when running dist_fact().')

    # Initialize arrays to store each participant's results
    usub = np.unique(subjects)
    total = np.zeros_like(usub, dtype=float)
    count = np.zeros_like(usub, dtype=float)

    # Identify locations of all correct recalls (not PLI, ELI, or repetition)
    clean_recalls_mask = np.array(make_clean_recalls_mask2d(make_recalls_matrix(pres_itemnos, rec_itemnos)))

    # Calculate distance factor score for each trial
    for i, trial_data in enumerate(rec_itemnos):
        seen = set()
        # Identify the current subject's index in usub to determine their position in the total and count arrays
        subj_ind = np.where(usub == subjects[i])[0][0]
        # Loop over the recalls on the current trial
        for j, rec in enumerate(trial_data[:-1]):
            seen.add(rec)
            # Only count transition if both the current and next recalls are valid
            if clean_recalls_mask[i, j] and clean_recalls_mask[i, j+1] and j >= skip_first_n:
                # Identify the distance between the current recall and all valid recalls that could follow it
                possibles = np.array([dist_mat[rec - 1, poss_rec - 1] for poss_rec in pres_itemnos[i] if poss_rec not in seen])
                # Identify the distance between the current recall and the next
                actual = dist_mat[rec - 1, trial_data[j + 1] - 1]
                # Find the proportion of possible transitions that were larger than the actual transition
                ptile_rank = dist_percentile_rank(actual, possibles, is_similarity)
                # Add transition to the appropriate participant's score
                if ptile_rank is not None:
                    total[subj_ind] += ptile_rank
                    count[subj_ind] += 1

    # Find temporal factor scores as the participants' average transition scores
    count[count == 0] = np.nan
    final_data = total / count
    
    if ret_counts:
        return final_data, total, count
    return final_data

def pd_sem_crp_list(df, sim_columns=None, bins=None, pres_type="WORD", 
               rec_type="REC_WORD", type_column='type', serialpos_col='serialpos', ret_counts=False, p=2):
    """Expects as input a dataframe (df) for one list.
    Doesn't require separate word_vals, expects them to be next to item. 
    Requires bins to be entered from elsewhere"""
    pres_df = df.query(type_column+' == @pres_type').sort_values(serialpos_col)
    pres_itemnos = pres_df[serialpos_col].values[np.newaxis, :]
    rec_itemnos = df.query(type_column+' == @rec_type')[serialpos_col].values[np.newaxis, :]
    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos)

    sem_sims = distance_matrix(pres_df[sim_columns].values, pres_df[sim_columns].values, p=p)
    n_bins = len(bins)

    out = sem_crp(recalls=recalls, 
                       recalls_itemnos=rec_itemnos, 
                       pres_itemnos=pres_itemnos, 
                       subjects=['_'] * recalls.shape[0], 
                       sem_sims=sem_sims, 
                       n_bins=n_bins, 
                       bins=bins,
                       listLength=pres_itemnos.shape[1],
                       ret_counts=ret_counts)
    if ret_counts:
        bin_means, crp, actual, poss = out
    else:
        bin_means, crp = out
    crp_dict = {'prob': crp[0], 
                'sem_bin_mean': bin_means[0],
                'sem_bin': np.arange(n_bins)
               }
    if ret_counts:
        crp_dict['actual'] = actual
        crp_dict['poss'] = poss

    return pd.DataFrame(crp_dict).query('prob == prob')

def pd_dist_fact_list(df, sim_columns=None,
                 skip_first_n=0, serialpos_col='serialpos',
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", type_column='type', ret_counts=False, p=2
                ):
    pres_df = df.query(type_column+' == @pres_type').sort_values(serialpos_col)
    pres_itemnos = pres_df[serialpos_col].values[np.newaxis, :]
 
    rec_itemnos = df.query(type_column+' == @rec_type')[serialpos_col].values[np.newaxis, :]
    #no recalls
#     if rec_itemnos.shape[1] == 0:
#         return np.nan  

    recalls = make_recalls_matrix(pres_itemnos, rec_itemnos)

    dist_mat = distance_matrix(pres_df[sim_columns].values, pres_df[sim_columns].values, p=p)
    
    final_data, total, count = dist_fact(rec_itemnos=rec_itemnos, 
              pres_itemnos=pres_itemnos, 
              subjects=['_'] * recalls.shape[0],
              dist_mat=dist_mat, is_similarity=False, 
              skip_first_n=skip_first_n, ret_counts=True)
#     print(final_data, total, count)
    if count == np.nan:
        print('nan')
        return np.nan
    dist_fact_dict = {'dist_fact': final_data, 
                'total': total,
                'count': count
               }
    return pd.DataFrame(dist_fact_dict)

def pd_dist_fact_list_sub(df, sim_columns=None,
                 skip_first_n=0, list_index=['subject', 'session', 'trial'],
                          sub_index=['subject'], serialpos_col='serialpos',
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    
    sub_dist_fact_df = df.groupby(list_index).apply(
        pd_dist_fact_list, sim_columns=sim_columns, serialpos_col=serialpos_col).reset_index().groupby(
        sub_index).agg({'total': 'sum', 'count': 'sum'}).reset_index()
    
    sub_dist_fact_df['dist_fact'] = sub_dist_fact_df['total'] / sub_dist_fact_df['count']
    sub_dist_fact_df.drop(columns=sub_index, inplace=True)
    return sub_dist_fact_df

def pd_sem_crp_list_sub(df, sim_columns=None,
                 skip_first_n=0, list_index=['subject', 'session', 'trial'],
                          sub_index=['subject'], bins=None, serialpos_col='serialpos',
#                  method=sp.spatial.distance.euclidean,
                 pres_type="WORD", rec_type="REC_WORD", type_column='type'):
    
    sub_sem_crp_df = df.groupby(list_index).apply(
        pd_sem_crp_list, sim_columns=sim_columns, bins=bins, ret_counts=True, serialpos_col=serialpos_col).reset_index().groupby(
        sub_index + ['sem_bin']).agg({'actual': 'sum', 'poss': 'sum'}).reset_index()
    
    sub_sem_crp_df['prob'] = sub_sem_crp_df['actual'] / sub_sem_crp_df['poss']
    sub_sem_crp_df.drop(columns=sub_index, inplace=True)
    return sub_sem_crp_df

def loftus_masson_analytic(df_long, sub_col, cond_col, value_col):
    #analytic version of loftus_masson SEs from long dataframe
    n_subs = df_long[sub_col].nunique()
    n_conds = df_long[cond_col].nunique()
    df_a = df_long.copy()
    df_a['M'] = df_a[value_col].mean()
    df_a['M_S'] = df_a.groupby([sub_col])[value_col].transform('mean')
    df_a['M_C'] = df_a.groupby([cond_col])[value_col].transform('mean')
    M_C = df_a.groupby([cond_col]).agg({value_col: 'mean'})[value_col].values

    #appendix A
    df_a['S_W'] = ((df_a[value_col] + df_a['M'] - df_a['M_S'] - df_a['M_C']) ** 2) 
    SS_W = df_a['S_W'].sum()
    df = (n_subs - 1) * (n_conds - 1)
    MS_SxC = SS_W / df
    SE = np.sqrt(MS_SxC / n_subs)
    CI_equal = SE * sp.stats.t.ppf(0.975, df) # Eq. 2, pg. 482
    
    #Unequal variances
    df_MS = df_a.groupby([cond_col, 'M_C'], as_index=False).agg({'S_W': 'sum'})
    df_MS['MS_W'] = df_MS['S_W'] / (n_subs - 1)
    df_MS['estimator'] = (n_conds / (n_conds - 1)) * (df_MS['MS_W'] - (MS_SxC / (n_conds)))
    df_MS['CI_unequal'] = np.sqrt(df_MS['estimator'] / n_subs) * sp.stats.t.ppf(0.975, (n_subs-1))
    df_MS['CI_equal'] = CI_equal
    return df_MS[[cond_col, 'M_C', 'CI_unequal', 'CI_equal']]

def coussineau(df_long, sub_cols, cond_col, value_col, within_cols=[]):
    if not isinstance(sub_cols, list):
        sub_cols = [sub_cols]
    if not isinstance(within_cols, list):
        within_cols = [within_cols]
    df_coussineau = df_long.copy()
    # sometimes want to calculate means/diffs within a condition rather than comparing conditions
    if len(within_cols) > 0:
        df_coussineau['M'] = df_long.groupby(within_cols)[value_col].transform('mean')
    else:
        df_coussineau['M'] = df_long[value_col].mean()
    df_coussineau['M_S'] = df_long.groupby(sub_cols + within_cols)[value_col].transform('mean')
    df_coussineau['adj_' + value_col] = df_coussineau[value_col] + df_coussineau['M'] - df_coussineau['M_S']
    
    #Cousineau-Morey-O'Brien adjustment https://link.springer.com/article/10.3758/s13428-013-0441-z
    n_conds = df_long[cond_col].nunique()
    df_coussineau['cmo_adj_' + value_col] = (np.sqrt(n_conds / (n_conds - 1)) * (df_coussineau[value_col] - df_coussineau['M_S'])) + df_coussineau['M']
    return df_coussineau

def loftus_masson_equal_variance_kahana(dat):
    # This script assumes that the variances for the different treatment groups
    # are equal, in other words, the sphericity assumption. If this is not the
    # case, then only errorbars can be computed for each contrast between treatments.
    numRows = dat.shape[0]
    numCols = dat.shape[1]
    D1data = np.reshape(dat, [1, numRows*numCols])
    grandMean = np.mean(D1data)
    grandTotal = np.sum(D1data)
    # total sum squares
    SS_T = np.sum((D1data-grandMean) ** 2)

    # sum squares for rows (subjects)
    Srow = np.sum(dat, 1)
    SSrow = np.sum((Srow ** 2) / numCols) - (grandTotal ** 2) / (numRows * numCols)
    # sum squares for columns (treatments)
    Scol = np.sum(dat, 0)
    SScol = np.sum((Scol ** 2) / numRows) - (grandTotal ** 2) / (numRows * numCols)

    # compute the mean sum squares for the interaction between rows and columns
    SSint = SS_T - SSrow - SScol
    df_int = (numRows * numCols - 1) - (numRows - 1) - (numCols - 1)
    MSint = SSint / df_int
    criterion = sp.stats.t.ppf(0.975, df_int)

    # implementation of Loftus-Masson (1994), equation (2)
    CI = np.sqrt(MSint / numRows) * criterion * np.ones(numCols)
    return CI

def loftus_masson_unequal_variance_kahana(dat):
    dat = mat
    # normalize the data
    grandMean = np.nanmean(dat)
    subjMean = np.nanmean(dat, axis=1)
    subjMean = np.matlib.repmat(subjMean, dat.shape[1], 1).T
    dat = dat - (subjMean - grandMean)

    # compute sums
    Tc = np.nansum(dat, axis=0)
    Nsubj = np.sum(~np.isnan(dat), axis=0)
    Ts = np.nansum(dat, axis=1)
    Ncond = np.sum(~np.isnan(dat), axis=1)
    T = np.nansum(dat)
    SS_T = np.nansum(dat ** 2)
    Nvalid = np.sum(~np.isnan(dat))

    SS_C = np.sum(Tc ** 2 / Nsubj)
    SS_S = np.sum(Ts ** 2 / Ncond)
    # compute average number of valid subjects
    NsubValid = np.sum(Nsubj) / dat.shape[1]
    NcondValid = np.sum(Ncond) / dat.shape[0]

    # compute final sums of squares
    SS_T = SS_T - (T ** 2) / Nvalid
    SS_S = SS_S - (T ** 2) / Nvalid
    SS_C = SS_C - (T ** 2) / Nvalid
    SS_SxC = SS_T - SS_S - SS_C
    # mean square of the interaction
    MS_SxC = SS_SxC / (Nvalid - (NsubValid + NcondValid - 1))
    # mean square w, i.e., variance between individuals (p.484)
    MS_w = (np.nansum(dat ** 2, axis=0) - ((Tc ** 2) / Nsubj)) / (Nsubj-1)
    # p.484
    estimator = (NcondValid / (NcondValid - 1)) * (MS_w - (MS_SxC / NcondValid))
    CI = np.sqrt(estimator / Nsubj) * sp.stats.t.ppf(0.975, dat.shape[0] - 1)
    return Tc, CI