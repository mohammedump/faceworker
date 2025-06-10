from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib

class EigenFacesModel:
    def __init__(self, n_components=50):
        self.pca = PCA(n_components=n_components, whiten=True)
        self.clf = KNeighborsClassifier(n_neighbors=3)

    def train(self, X_train, y_train):
        X_train_pca = self.pca.fit_transform(X_train)
        self.clf.fit(X_train_pca, y_train)

    def predict(self, X, threshold=8000):
        X_pca = self.pca.transform(X)
        distances, indices = self.clf.kneighbors(X_pca, n_neighbors=1)
        predictions = []

        for i in range(len(X)):
            if distances[i][0] > threshold:
                predictions.append(-1)  # Trop éloigné → Inconnu
            else:
                predictions.append(self.clf.predict([X_pca[i]])[0])
        return predictions

    def save(self, path="models/eigenfaces_model.pkl"):
        joblib.dump((self.pca, self.clf), path)

    def load(self, path="models/eigenfaces_model.pkl"):
        self.pca, self.clf = joblib.load(path)
