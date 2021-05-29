import numpy as np
from scipy.interpolate import interp2d
from sklearn.base import BaseEstimator, TransformerMixin
# from skimage.feature import hog



class Scaler(BaseEstimator, TransformerMixin):
    """This class will scale the data as: x = (x/a) - b"""

    def __init__(self, a=100, b=1):
        self.a = a
        self.b = b

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return ((X / self.a) - self.b)

class Flatten(BaseEstimator, TransformerMixin):
    """This class will scale the data as: x = (x/a) - b"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0],-1)

class Preprocessing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def resize(img, new_W=28, new_H=28):
            """
        Función para escalar imágenes
        Args:
           img (array):  array de 2 dimensiones que representa la imagen
        Kwargs:
            new_H (int): nueva altura
            new_W (int): nueva anchura
        Returns:
           array con la nueva imagen escalada
            """
            W, H = img.shape
            xrange = lambda x: np.linspace(0, 1, x)
            f = interp2d(xrange(H), xrange(W), img, kind="linear")
            print(f)
            new_img = f(xrange(new_W), xrange(new_H))
            return new_img
        X = np.mean(X, axis=2)
        img = -(resize(X)-255.0)
        # Creamos máscara sobre el dígito y hacemos que el resto sea 0 (fondo)
        mask = img > 115.0
        img[~mask] = 0.0
        return img[np.newaxis]

# class HogFeaturesExtraction(BaseEstimator, TransformerMixin):
#     """
#     This class will create a new dataset by extracting the hog features of
#     the input dataset
#
#     param orientations: the number of orientations in which the gradient will be calculated
#     param ppc: number of pixels per cell. It's recommended that the value of ppc is not lower than 4, as
#                 three scales will be concatenated for a better performance (ppc, ppc/2 and ppc/4)
#     param cpb: number of cells per block
#     """
#
#     def __init__(self, orientations=6, ppc=7, cpb=1):
#         self.orientations = orientations
#         self.ppc = ppc
#         self.cpb = cpb
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         X_hog = np.concatenate([np.concatenate([hog(xi, orientations=self.orientations,
#                                                     pixels_per_cell=(ppc, ppc),
#                                                     cells_per_block=(self.cpb, self.cpb),
#                                                     visualize=False,
#                                                     block_norm='L1')[np.newaxis, :] for xi in X],
#                                                axis=0) for ppc in [self.ppc, int(self.ppc / 2), int(self.ppc / 4)]],
#                                axis=1)
#         return X_hog