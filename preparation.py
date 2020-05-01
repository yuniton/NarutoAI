#データの準備

from keras.utils import np_utils
import numpy as np

categories = ["ナルト","サスケ","サクラ","カカシ先生"]
nb_classes = len(categories)

X_train, X_test, y_train, y_test = np.load("naruto_data.npy")

#データの正規化
X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float") / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)


