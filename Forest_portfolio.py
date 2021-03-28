import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
data = pd.read_csv('cover_data.csv')
# print(Counter(data['class']))
column_list = []
for column in data.columns:
    if column == 'class':
        pass
    else:
        column_list.append(column)
features = data[column_list]
labels = data['class']
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
ct = ColumnTransformer([('scale', StandardScaler(), column_list)], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)
le = LabelEncoder()
labels_train = le.fit_transform(labels_train.astype(str))
labels_test = le.transform(labels_test.astype(str))
labels_train = to_categorical(labels_train, dtype='int64')
labels_test = to_categorical(labels_test, dtype='int64')
model = Sequential()
model.add(layers.InputLayer(input_shape=(features_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
# opt = Adam(learning_rate = 0.01)
opt = SGD(learning_rate = 0.1, momentum=0.9, decay=.1/60, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
es = EarlyStopping(monitor='accuracy', mode='max')
model.fit(features_train, labels_train, epochs=60, batch_size=150, verbose=1, callbacks=es)
ent, acc = model.evaluate(features_test, labels_test)
print(ent, acc)
