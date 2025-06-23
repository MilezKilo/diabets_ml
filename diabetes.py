import sklearn
import tensorflow
import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)

dataset = sklearn.datasets.load_diabetes()

diabetes_val = pd.DataFrame(data=dataset.data, columns=dataset.feature_names) #Shape - 442, 10
diabetes_tar = pd.DataFrame(data=dataset.target, columns=['quantitative_measure']) #Shape - 442, 1

# print(diabetes_val.info())
# print(diabetes_tar.info())
# print(diabetes_val.describe())
# print(diabetes_tar.describe)

scaler = sklearn.preprocessing.StandardScaler()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(diabetes_val
    , diabetes_tar, test_size=0.2, random_state=1)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def sklearn_model():
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)

    print(f'Model accuracy: {score}')
    print(f'R2score {sklearn.metrics.r2_score(y_test, predictions)}')
    print(f'MSE {sklearn.metrics.mean_squared_error(y_test, predictions)}')
    print(f'RMSE {sqrt(sklearn.metrics.mean_squared_error(y_test, predictions))}')


def keras_model():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(10,)))
    model.add(keras.layers.Dense(units=16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.05)))
    # model.add(keras.layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.05)))
    model.add(keras.layers.Dense(units=1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
                  loss=keras.losses.mean_squared_error,
                  metrics=['mae'])

    history = model.fit(X_train_scaled, y_train, epochs=200, verbose=1, batch_size=16)
    predictions = model.predict(X_test_scaled)

    mse = keras.metrics.mean_squared_error(y_true=y_test, y_pred=predictions)
    mae = keras.metrics.mean_absolute_error(y_true=y_test, y_pred=predictions)

    mean_mse = np.mean(mse.numpy())
    mean_mae = np.mean(mae.numpy())

    print(f'Mean MSE: {mean_mse}, Mean MAE: {mean_mae}')
    # Mean MSE: 3097.82470703125, Mean MAE: 41.893619537353516

    # MSE LOSSES METRICS ON PLOT
    plt.plot(history.history['loss'], label='MSE')
    plt.legend(loc='upper right')
    plt.show()

    # DIFFERENCE BETWEEN TRUE AND PREDICTION VALUES ON SCATTER
    plt.scatter(y_test, predictions, s=10, alpha=0.8, linewidths=1)
    plt.xlabel('Actual values.')
    plt.ylabel('Prediction value.')
    plt.gray()
    plt.show()


def keras_model_KFold_cross():
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)

    k_fold = sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=1)

    f_num = 0
    metrics = []
    histories = []

    for train_index, val_index in k_fold.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(10,)))
        model.add(keras.layers.Dense(units=16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        # model.add(keras.layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(keras.layers.Dense(units=1))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
                      loss=keras.losses.mean_squared_error,
                      metrics=['mae'])

        early_stopper = keras.callbacks.EarlyStopping(patience=20, monitor='val_loss', min_delta=5, restore_best_weights=True)

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200,
                            batch_size=16, verbose=1, callbacks=[early_stopper])

        score = model.evaluate(X_val, y_val, verbose=0)
        print(f'Fold num: {f_num}, Val MSE: {score[0]:.2f}, Val MAE: {score[1]:.2f}')
        f_num += 1
        metrics.append(score)
        histories.append(history)

    MSE_metrics = [mse[0] for mse in metrics]
    MAE_metrics = [mae[1] for mae in metrics]
    predictions = model.predict(X_test)

    print(f'Mean MSE: {np.mean(MSE_metrics)}, Mean MAE: {np.mean(MAE_metrics)}')
    # Mean MSE: 2994.8065795898438, Mean MAE: 44.460309982299805

    # MSE LOSSES METRICS OF ALL K_FOLDS ON PLOT
    figure, ax = plt.subplots(1, 4, figsize=(17, 8))
    colors = ['red', 'blue', 'orange', 'green']
    for index in range(0, len(histories)):
        ax[index].plot(histories[index].history['loss'], c=colors[index])
        ax[index].set_title(f'MSE on epoch: {index}')
        ax[index].set_xlabel('Epochs')
        ax[index].set_ylabel('Losses')
    plt.show()

    # DIFFERENCE BETWEEN TRUE AND PREDICTION VALUES ON SCATTER
    plt.scatter(y_test, predictions, s=10, alpha=0.8, linewidths=1)
    plt.xlabel('Actual values.')
    plt.ylabel('Prediction value.')
    plt.gray()
    plt.show()