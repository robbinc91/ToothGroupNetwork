from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.svm import SVC # uncomment this line if you don't install thudersvm

class Upsampler(object):
    def __init__(self):
        pass
    def run(self, mesh_orig, mesh, predicted_labels, method = 'KNN'):
        barycenters = mesh.cellCenters() # don't need to copy
        fine_barycenters = mesh_orig.cellCenters() # don't need to copy

        if method == 'SVM':
            clf = SVC(kernel='rbf', gamma='auto')
            clf.fit(barycenters, np.ravel(predicted_labels))
            fine_labels = clf.predict(fine_barycenters)
            fine_labels = fine_labels.reshape(-1, 1)
        elif method == 'KNN':
            neigh = KNeighborsClassifier(n_neighbors=1)
            neigh.fit(barycenters, np.ravel(predicted_labels))
            fine_labels = neigh.predict(fine_barycenters)
            fine_labels = fine_labels.reshape(-1, 1)
        return fine_labels