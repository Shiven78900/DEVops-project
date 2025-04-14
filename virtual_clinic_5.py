from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.src.layers.activations import LeakyReLU
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from keras.utils import to_categorical
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.keras.layers import LeakyReLU

symptom_severity = pd.read_csv("Symptom-severity.csv")
dataset = pd.read_csv("dataset.csv")

#label encoding (prolly not necessary, but used cause some models like decision tree classifier can only accept integer values
label_encoder = LabelEncoder()
symptom_severity['Symptom'] = label_encoder.fit_transform(symptom_severity['Symptom'])
dataset['Disease'] = label_encoder.fit_transform(dataset['Disease'])

#Feature engineering is going on. should be worked on MORE(assuming 'Symptom_1' to 'Symptom_17' are symptoms)
#we're conevrting symptom to syring type...
symptom_severity['Symptom'] = symptom_severity['Symptom'].astype(str)
data = dataset.merge(symptom_severity, how='left', left_on='Symptom_1', right_on='Symptom')
data.drop(columns=['Symptom'], inplace=True)

#we first identiy thecategorical columns and then one hto encode them
categorical_cols = data.select_dtypes(include=['object']).columns

column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

#one-hot encoding + impute
X_encoded = column_transformer.fit_transform(data.iloc[:, :-1])
y = data['Disease'].values

#IF THERE'S A MISSING VALUE, IMPUT IT WIT MEAN
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

#reshaping of the labels to one hot encode them
num_classes = len(np.unique(y))
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

#okay CNN expects an input with THREE dimensions, while we have only one. need to use expand_dims, but that only works on arrays and we have a sparse matrix, so we first conevrt into array
X_train = np.expand_dims(X_train.toarray(), axis=2)
X_test = np.expand_dims(X_test.toarray(), axis=2)

num_epochs = 10
batch_size = 64

#CNN architecture........
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=4, activation=LeakyReLU(alpha=0.01), input_shape=X_train.shape[1:]))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(num_classes, activation='softmax',kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

adam_optimizer = Adam(learning_rate=0.001)  # Adjust learning rate as needed
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['precision'])

#here's the training after compilign the model
model.fit(X_train, y_train_encoded, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test_encoded))
