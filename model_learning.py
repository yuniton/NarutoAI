#モデルの学習
from src.naruto.preparation import X_train, y_train, X_test, y_test

from src.naruto.naruto_model import model

model = model.fit(X_train,
                  y_train,
                  epochs=10,
                  batch_size=6,
                  validation_data=(X_test,y_test))