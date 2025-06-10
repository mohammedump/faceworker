from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib

class FisherFacesModel:
    def __init__(self):
        self.pca = None
        self.lda = None
        self.clf = KNeighborsClassifier(n_neighbors=3)

    def train(self, X_train, y_train, n_components=None):
        self.pca = PCA(n_components=0.95, whiten=True)
        X_pca = self.pca.fit_transform(X_train)

        if n_components is None:
            n_components = min(len(set(y_train)) - 1, X_pca.shape[1])

        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = self.lda.fit_transform(X_pca, y_train)
        self.clf.fit(X_lda, y_train)

    def predict(self, X, threshold=8000):
        X_pca = self.pca.transform(X)
        X_lda = self.lda.transform(X_pca)
        distances, indices = self.clf.kneighbors(X_lda, n_neighbors=1)
        predictions = []
        for i in range(len(X)):
            if distances[i][0] > threshold:
                predictions.append(-1)
            else:
                predictions.append(self.clf.predict([X_lda[i]])[0])
        return predictions

    def save(self, path="models/fisherfaces_model.pkl"):
        joblib.dump((self.pca, self.lda, self.clf), path)

    def load(self, path="models/fisherfaces_model.pkl"):
        self.pca, self.lda, self.clf = joblib.load(path)
