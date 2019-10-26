import pandas as pd
import numpy as np


# helper functions
def potential_values(ra, k, sample_data):
    """
    ra = parameter (RADII parameter)
    k = index of row to find potential of in the data

    returns: the potential for that given row given no previous potential known

    """

    val_list = np.array([])

    for l in range(0, len(sample_data)):
        diff = sample_data[k] - sample_data[l]
        value = np.exp(-((diff ** 2).sum()) / (ra / 2) ** 2)
        val_list = np.append(val_list, value)

    return val_list.sum()


def min_dist(array_to_check, list_of_centers):
    '''

    :param array_to_check: array you wish to have compared
    :param list_of_centers: list of accepted centers to measure against
    :return: minimum distance for array_to_check to any of the centers

    Note: Should only be used with the normalised data in this question as it may not scale well
    '''
    dist_old = 10000
    for b in range(0, len(list_of_centers)):
        dist = array_to_check - list_of_centers[b]
        dist = np.sqrt((dist ** 2).sum())
        if dist < dist_old:
            dist_old = dist
        else:
            pass
    return dist_old


def update_array_with_potentials(old_list, accepted_center, accepted_centers_potential, rb, index):
    """

    :param old_list: last array of data points with their centers
    :param accepted_center: the just accepted center
    :param accepted_centers_potential: potential of accepted center
    :param rb: parameter
    :param index: number of accepted centers before this one

    :return: updated array of data points with their new potentials
    """
    for c in range(index + 1, len(old_list + 1)):
        diff = ((old_list[c][0:3] - accepted_center) ** 2).sum()
        Pnew = old_list[c][3] - accepted_centers_potential * np.exp(-(diff) / (rb / 2) ** 2)
        old_list[c][3] = Pnew
    old_list = np.append(old_list[:(index + 1), :],
                         old_list[(index + 1):, :][(-old_list[(index + 1):, :])[:, -1].argsort()], axis=0)

    return old_list


def step_6(E_up, E_down, ra, list_of_centers, sorted_data_with_potential2, index, P1, rb):
    """

    :param Eup: Parameter
    :param Edown: Parameter
    :param ra: Parameter
    :param list_of_centers: List of all accepted senters in array form
    :param sorted_data_with_potential2: all the points with their current potential
    :param index: number of accepted centers before this one
    :param P1: Potential of very first center that was accepted
    :param rb: Parameter
    :return: updated list of centers, the data with it's updated potential, the new index value
    """
    Pk = sorted_data_with_potential2[index][3]
    X = sorted_data_with_potential2[index][0:3]  # point we are considering
    if Pk / P1 > E_up:
        # accept it and look for more
        list_of_centers.append(X)  # add it to list of centers
        sorted_data_with_potential2 = update_array_with_potentials(sorted_data_with_potential2, X, Pk, rb, index)
        index = index + 1  # update sorted data with potential
        return list_of_centers, sorted_data_with_potential2, index

    elif Pk / P1 < E_down:
        # reject it and stop
        index = "stop"  # will use this index to stop the algorithm
        return list_of_centers, sorted_data_with_potential2, index

    else:
        # more ambiguous case
        dmin = min_dist(X, list_of_centers)
        value = dmin / ra + Pk / P1
        if value >= 1:
            list_of_centers.append(X)
            sorted_data_with_potential2 = update_array_with_potentials(sorted_data_with_potential2, X, Pk, rb, index)
            index = index + 1
            return list_of_centers, sorted_data_with_potential2, index
            # accept X and move to next point for next center
        else:
            index = index + 1
            return list_of_centers, sorted_data_with_potential2, index
            # reject X, set its potential to zero, and look for next center

def nearest_center(data, index, centers):
        dist_min = 1
        for i in range(0, len(centers)):
            dist_new = (np.sqrt(((data[index, :3] - centers[i]) ** 2).sum()))
            if dist_new < dist_min:
                dist_min = dist_new
                label = i
            else:
                pass

        return label

def de_normalize(original_data, new_data):
        if isinstance(original_data, pd.DataFrame) != True:
            original_data = pd.DataFrame(original_data)

        if isinstance(new_data, np.ndarray) != True:
            new_data = np.array(new_data)

        for i in range(0, 3):
            new_data[:, i] = new_data[:, i] * (original_data.max(axis=0)[i] - original_data.min(axis=0)[i]) + \
                             original_data.min(axis=0)[i]

        return new_data


# main logic
def subtractive_clustering_algorithm(ra, rb, E_up, E_down, data):
    """
    An implementation of the subtractive clustering algorithm.

    Further detail on the algorithm and how it works can be found in the following paper:
    "Support vector machines based on subtractive clustering"
    http://ieeexplore.ieee.org/abstract/document/1527702/


    :param ra: 0 < r ≤ 1 is the clustering RADII parameter, which defines the neighborhood radius of each point. The
    data outside the radius of xi have little influence on its potential. So the point having more data within the
    radius will get higher potential.

    :param rb: rb > 0 defines the neighborhood of a cluster center with which the existence of other cluster centers
    are discouraged. The points close to the selected cluster center will have significant reduction in their potential
    after computation. When Pi ≤ 0 , the point xi is rejected for the cluster center forever. In this way, some closer
    samples are replaced by their center. To avoid closely space centers, usually, rb = 1.5ra.

    :param E_up: ε_up is the accept ratio above which another data point will be accepted as a cluster center with no doubts.

    :param E_down: ε_down is the reject ratio below which another data point will be definitely rejected.

    :param data: 3-dimensional data points input as a pandas DataFrame.

    :return: list_of_centers: List containing the coordinates of all centers that were found

    :return: data: The data with a label indicating which center it belongs to

    """
    stored_data = data.copy()
    sample_data = pd.DataFrame(data)

    # step one: normalise into unit hyperbox
    for i in range(0,3):
        sample_data.iloc[:,i] = (sample_data.iloc[:,i] - sample_data.min(axis = 0)[i])/(sample_data.max(axis = 0)[i] - sample_data.min(axis = 0)[i])

    # step 2: compute the potential value for each xi
    sample_data = np.array(sample_data)

    potential = np.array([])
    for k in range(0,len(sample_data)):
        potential = np.append(potential, potential_values(ra, k, sample_data))

    data_with_potential = np.c_[sample_data, potential]

    # step 3: select point with highest potential as first center
    list_of_centers = []
    sorted_data_with_potential = data_with_potential[(-data_with_potential)[:,-1].argsort()]
    list_of_centers.append(sorted_data_with_potential[0][0:3])
    P1 = sorted_data_with_potential[0][3]  # and its associated potential

    # step 4: reduce the potential values of remaining data points
    x1_star = sorted_data_with_potential[0][0:3]

    for m in range(1,len(sorted_data_with_potential)):
        vector = sorted_data_with_potential[m][0:3] - x1_star
        vector1 = np.exp(-((vector**2).sum()) / (rb / 2)**2)
        sorted_data_with_potential[m][3] = sorted_data_with_potential[m][3] - P1*vector1

    # step 5: select the highest potential from the reduced potential
    sorted_data_with_potential2 = sorted_data_with_potential[(-sorted_data_with_potential)[:,-1].argsort()]

    # Pk = sorted_data_with_potential2[1][3]  # and its potential
    # step 6: determine next cluster center
    # step 7: Compute the potential for the remaining points
    updated_centers, new_data, index = step_6(E_up, E_down, ra, list_of_centers, sorted_data_with_potential2, 1, P1, rb)

    while index != 'stop':
        updated_centers, new_data, index = step_6(E_up, E_down, ra, updated_centers, new_data, index, P1, rb)

    for j in range(0, len(new_data)):
        new_data[j, 3] = nearest_center(new_data, j, updated_centers)

    new_data = de_normalize(stored_data, new_data)
    updated_centers = de_normalize(stored_data, updated_centers)

    return updated_centers, new_data

