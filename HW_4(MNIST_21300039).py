import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
# import scipy.misc
from PIL import Image
from numpy import linalg as LA
from matplotlib import pyplot as plt


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)

    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    return train_set, valid_set, test_set

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set



    #print(train_x.shape)
    #print(train_y.shape)
    # for i in range(100):
    #     tmp_img = train_x[i].reshape((28, 28)) * 255.9
    #     samp_img = Image.fromarray(tmp_img.astype(np.uint8))
    #     samp_img.save('test' + str(i) + '.jpg')
    #     print(train_y[i])

    #mean_img = train_x.mean(0)
    #print(mean_img.shape)
    #mean = mean_img.reshape((28, 28)) * 255.9
    #plt.imshow(mean, cmap='gray')
    #plt.show()

    #var_img = train_x.var(0)
    #var = var_img.reshape((28, 28)) * 255.9
    #plt.imshow(var, cmap='gray')
    #plt.show()

    #print(train_x)
    #print(np.sum(train_x <= 0.1))
    #print(np.sum(train_x >= 0.9))
    #print(mean_img.shape)
    #Implement K-
def display_digit(digit, labeled = True, title = ""):
    """
    graphically displays a 784x1 vector, representing a digit
    """
    if labeled:
        digit = digit[1]
    image = digit
    plt.figure()
    fig = plt.imshow(image.reshape(28,28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if title != "":
        plt.title("Inferred label: " + str(title))

# writing Lloyd's Algorithm for K-Means clustering.
# (This exists in various libraries, but it's good practice to write by hand.)
def init_centroids(labelled_data,k):
    """
    randomly pick some k centers from the data as starting values for centroids.
    Remove labels.
    """
    return map(lambda x: x[1], random.sample(labelled_data,k))

def sum_cluster(labelled_cluster):
    """
    from http://stackoverflow.com/questions/20640396/quickly-summing-numpy-arrays-element-wise
    element-wise sums a list of arrays. assumes all datapoints in labelled_cluster are labelled.
    """
    # assumes len(cluster) > 0
    sum_ = labelled_cluster[0][1].copy()
    for (label,vector) in labelled_cluster[1:]:
        sum_ += vector
    return sum_

def mean_cluster(labelled_cluster):
    """
    computes the mean (i.e. the centroid at the middle) of a list of vectors (a cluster).
    take the sum and then divide by the size of the cluster.
    assumes all datapoints in labelled_cluster are labelled.
    """
    sum_of_points = sum_cluster(labelled_cluster)
    mean_of_points = sum_of_points * (1.0 / len(labelled_cluster))
    return mean_of_points


def form_clusters(labelled_data, unlabelled_centroids):
    """
    given some data and centroids for the data, allocate each datapoint
    to its closest centroid. This forms clusters.
    """
    # enumerate because centroids are arrays which are unhashable,
    centroids_indices = range(len(unlabelled_centroids))

    # initialize an empty list for each centroid. The list will contain
    # all the datapoints that are closer to that centroid than to any other.
    # That list is the cluster of that centroid.
    clusters = {c: [] for c in centroids_indices}

    for (label, Xi) in labelled_data:
        # for each datapoint, pick the closest centroid.
        smallest_distance = float("inf")
        for cj_index in centroids_indices:
            cj = unlabelled_centroids[cj_index]
            distance = np.linalg.norm(Xi - cj)
            if distance < smallest_distance:
                closest_centroid_index = cj_index
                smallest_distance = distance
        # allocate that datapoint to the cluster of that centroid.
        clusters[closest_centroid_index].append((label, Xi))
    return clusters.values()


def move_centroids(labelled_clusters):
    """
    returns a list of centroids corresponding to the clusters.
    """
    new_centroids = []
    for cluster in labelled_clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids


def repeat_until_convergence(labelled_data, labelled_clusters, unlabelled_centroids):
    """
    form clusters around centroids, then keep moving the centroids
    until the moves are no longer significant, i.e. we've found
    the best-fitting centroids for the data.
    """
    previous_max_difference = 0
    while True:
        unlabelled_old_centroids = unlabelled_centroids
        unlabelled_centroids = move_centroids(labelled_clusters)
        labelled_clusters = form_clusters(labelled_data, unlabelled_centroids)
        # we keep old_clusters and clusters so we can get the maximum difference
        # between centroid positions every time. we say the centroids have converged
        # when the maximum difference between centroid positions is small.
        differences = map(lambda a, b: np.linalg.norm(a - b), unlabelled_old_centroids, unlabelled_centroids)
        max_difference = max(differences)
        difference_change = abs(
            (max_difference - previous_max_difference) / np.mean([previous_max_difference, max_difference])) * 100
        previous_max_difference = max_difference
        # difference change is nan once the list of differences is all zeroes.
        if np.isnan(difference_change):
            break
    return labelled_clusters, unlabelled_centroids

def cluster(labelled_data, k):
    """
    runs k-means clustering on the data. It is assumed that the data is labelled.
    """
    centroids = init_centroids(labelled_data, k)
    clusters = form_clusters(labelled_data, centroids)
    final_clusters, final_centroids = repeat_until_convergence(labelled_data, clusters, centroids)
    return final_clusters, final_centroids

def assign_labels_to_centroids(clusters, centroids):
    """
    Assigns a digit label to each cluster.
    Cluster is a list of clusters containing labelled datapoints.
    NOTE: this function depends on clusters and centroids being in the same order.
    """
    labelled_centroids = []
    for i in range(len(clusters)):
        labels = map(lambda x: x[0], clusters[i])
        # pick the most common label
        most_common = max(set(labels), key=labels.count)
        centroid = (most_common, centroids[i])
        labelled_centroids.append(centroid)
    return labelled_centroids
def classify_digit(digit, labelled_centroids):
    """
    given an unlabelled digit represented by a vector and a list of
    labelled centroids [(label,vector)], determine the closest centroid
    and thus classify the digit.
    """
    mindistance = float("inf")
    for (label, centroid) in labelled_centroids:
        distance = np.linalg.norm(centroid - digit)
        if distance < mindistance:
            mindistance = distance
            closest_centroid_label = label
    return closest_centroid_label

def get_error_rate(digits,labelled_centroids):
    """
    classifies a list of labelled digits. returns the error rate.
    """
    classified_incorrect = 0
    for (label,digit) in digits:
        classified_label = classify_digit(digit, labelled_centroids)
        if classified_label != label:
            classified_incorrect +=1
    error_rate = classified_incorrect / float(len(digits))
    return error_rate
# error_rates = {x:None for x in range(5,25)+[100]}
# for k in range(5,25):
#     trained_clusters, trained_centroids = cluster(training, k)
#     labelled_centroids = assign_labels_to_centroids(trained_clusters, trained_centroids)
#     error_rate = get_error_rate(validation, labelled_centroids)
#     error_rates[k] = error_rate

# Show the error rates
# x_axis = sorted(error_rates.keys())
# y_axis = [error_rates[key] for key in x_axis]
# plt.figure()
# plt.title("Error Rate by Number of Clusters")
# plt.scatter(x_axis, y_axis)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Error Rate")
# plt.show()

i=0
j=0
x=0
x2=0


for i in range(50000):
    if train_y[i] != 3 and 9:
        train_x = np.delete(train_x, (i), axis=0)
        train_y = np.delete(train_y, (i), axis=0)
for j in range(50000):
    if test_y[j] != 3 and 9:
        test_x = np.delete(test_x, (j), axis=0)
        test_y = np.delete(test_y, (j), axis=0)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

cov = np.cov(train_x.T)
cov2 = np.cov(test_x.T)
eig_val, eig_vec = np.linalg.eig(cov)
eig_val2, eig_vec2 = np.linalg.eig(cov2)
training=train_x
for k in range (2,3,5,10) :
    trained_clusters, trained_centroids = cluster(training, k)
    labelled_centroids = assign_labels_to_centroids(trained_clusters, trained_centroids)
    for x in labelled_centroids:
        display_digit(x, title=x[0])

dimension_2_train = np.matmul(eig_vec[:2].T, eig_vec[:2])
dimension_2_train = dimension_2_train.real
dimension_2_test = np.matmul(eig_vec2[:2].T, eig_vec2[:2])
dimension_2_test = dimension_2_test.real
training=dimension_2_train
for k in range (2,3,5,10) :
    trained_clusters, trained_centroids = cluster(training, k)
    labelled_centroids = assign_labels_to_centroids(trained_clusters, trained_centroids)
    for x in labelled_centroids:
        display_digit(x, title=x[0])

dimension_5_train = np.matmul(eig_vec[:5].T, eig_vec[:5])
dimension_5_train = dimension_5_train.real
dimension_5_test = np.matmul(eig_vec2[:5].T, eig_vec2[:5])
dimension_5_test = dimension_5_test.real
training=dimension_5_train
for k in range (2,3,5,10) :
    trained_clusters, trained_centroids = cluster(training, k)
    labelled_centroids = assign_labels_to_centroids(trained_clusters, trained_centroids)
    for x in labelled_centroids:
        display_digit(x, title=x[0])

dimension_10_train = np.matmul(eig_vec[:10].T, eig_vec[:10])
dimension_10_train = dimension_10_train.real
dimension_10_test = np.matmul(eig_vec2[:10].T, eig_vec2[:10])
dimension_10_test = dimension_10_test.real
training=dimension_10_train
for k in range (2,3,5,10) :
    trained_clusters, trained_centroids = cluster(training, k)
    labelled_centroids = assign_labels_to_centroids(trained_clusters, trained_centroids)
    for x in labelled_centroids:
        display_digit(x, title=x[0])


#x = np.linspace(0, 784, 784)
# plt.figure()
# plt.plot(x, eigen_val)
# plt.show()
#print(cov.shape)
# print(eig_vec.shape)
# for i in eig_vec.T[:10]:
#     i=i.real
#     plt.imshow(i.reshape((28, 28)) * 255.9, cmap='gray')
#     plt.show()
#print(eig_vec.T.shape)
#print(cov)
#print(cov.shape)
# for eigendecomposition
# check http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
