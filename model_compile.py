#モデルのコンパイル

from keras import optimizers

from src.naruto.naruto_model import model

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])