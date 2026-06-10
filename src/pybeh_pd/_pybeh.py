import numpy as np
import copy

def make_clean_recalls_mask2d(data):
    """makes a clean mask without repetition and intrusion"""
    result = copy.deepcopy(data)
    for num, item in enumerate(data):
        seen = []
        for index, recall in enumerate(item):

            if recall > 0 and recall not in seen:
                result[num][index] = 1
                seen.append(recall)
            else:
                result[num][index] = 0
    return result

def crp(recalls=None, subjects=None, listLength=None, lag_num=None, skip_first_n=0):
    '''
    CRP   Conditional response probability as a function of lag (lag-CRP).
    
      lag_crps = crp(recalls_matrix, subjects, list_length, lag_num)
    
      INPUTS:
             recalls:  A 2D iterable whose elements are serial positions of
                       recalled items.  The rows of this array should
                       represent recalls made by a single subject on a
                       single trial.
    
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

    # Convert recalls and subjects to numpy arrays
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
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(recalls[subjects == subj]))
        # For each trial that matches that identifier
        for j, trial_recs in enumerate(recalls[subjects == subj]):
            seen = set()
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j][k] and clean_recalls_mask[j][k + 1] and k >= skip_first_n:
                    next_rec = trial_recs[k + 1]
                    pt = np.array([trans for trans in range(1 - rec, listLength + 1 - rec) if rec + trans not in seen], dtype=int)
                    poss[pt + listLength - 1] += 1
                    trans = next_rec - rec
                    # Record the actual transition that was made
                    actual[trans + listLength - 1] += 1

        result[i, :] = [a/p if p!=0 else np.nan for a,p in zip(actual, poss)]

    result[:, listLength - 1] = np.nan

    return result[:, listLength - lag_num - 1:listLength + lag_num]

def temp_fact(recalls=None, subjects=None, listLength=None, skip_first_n=0):
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

    # Identify locations of all correct recalls (not PLI, ELI, or repetition)
    clean_recalls_mask = np.array(make_clean_recalls_mask2d(recalls))

    # Calculate temporal factor score for each trial
    for i, trial_data in enumerate(recalls):
        seen = set()
        # Identify the current subject's index in usub to determine their position in the total and count arrays
        subj_ind = np.where(usub == subjects[i])[0][0]
        # Loop over the recalls on the current trial
        for j, serialpos in enumerate(trial_data[:-1]):
            seen.add(serialpos)
            # Only count transition if both the current and next recalls are valid
            if clean_recalls_mask[i, j] and clean_recalls_mask[i, j+1] and j >= skip_first_n:
                # Identify possible transitions
                possibles = np.array([abs(item - serialpos) for item in possibles_range if item not in seen])
                # Identify actual transition
                next_serialpos = trial_data[j + 1]
                actual = abs(next_serialpos - serialpos)
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


def temp_percentile_rank(actual, possible):
    """
    Helper function to return the percentile rank of the actual transition within the list of possible transitions.
    :param actual: The distance of the actual transition that was made.
    :param possible: The list of all possible transition distances that could have been made.
    :return: The proportion of possible transitions that were more distant than the actual transition.
    """
    # If there were fewer than 2 possible transitions, we can't compute a meaningful percentile rank
    if len(possible) < 2:
        return None

    # Sort possible transitions from largest to smallest
    possible = sorted(possible)[::-1]

    # Get indices of the one or more possible transitions with the same distance as the actual transition
    matches = np.where(possible == actual)[0]

    if len(matches) > 0:
        # Get the number of possible transitions that were more distant than the actual transition
        # If there were multiple transitions with the same distance as the actual one, average across their ranks
        rank = np.mean(matches)
        # Convert rank to the proportion of possible transitions that were more distant than the actual transition
        ptile_rank = rank / (len(possible) - 1.)
    else:
        ptile_rank = None

    return ptile_rank

def dist_percentile_rank(actual, possible, is_similarity=False):
    """
    Helper function to return the percentile rank of the actual transition within the list of possible transitions.
    :param actual: The distance of the actual transition that was made.
    :param possible: The list of all possible transition distances that could have been made.
    :is_similarity: If False, actual and possible values are assumed to be distances. If True, values are assumed to be
        similarity scores, where smaller values correspond to more distant transitions.
    :return: The proportion of possible transitions that were more distant than the actual transition.
    """
    # If there were fewer than 2 possible transitions, we can't compute a meaningful percentile rank
    if len(possible) < 2:
        return None

    # Sort possible transitions from largest to smallest distance (taking into account whether the values are
    # similarities or distances)
    possible = sorted(possible) if is_similarity else sorted(possible)[::-1]

    # Get indices of the one or more possible transitions with the same distance as the actual transition
    matches = np.where(possible == actual)[0]

    if len(matches) > 0:
        # Get the number of possible transitions that were more distant than the actual transition
        # If there were multiple transitions with the same distance as the actual one, average across their ranks
        rank = np.mean(matches)
        # Convert rank to the proportion of possible transitions that were more distant than the actual transition
        ptile_rank = rank / (len(possible) - 1.)
    else:
        ptile_rank = None

    return ptile_rank