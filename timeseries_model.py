from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import  os
import joblib
import pandas as pd
import numpy as np

class time_series:
    def __init__(self,data,root_model):
        self.data = data
        self.root_model = root_model

    def train_test_order(self,X,y,n_test):
        return X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

    def RF_model_forecast(self,X_train,X_test,y_train,k,save_model):
        model = RandomForestRegressor(n_estimators=2000)
        model.fit(X_train,y_train)
        file_model = 'model_ts'+str(k)+'.pkl'
        if save_model==True:
            file_model = os.path.join(self.root_model, file_model)
            joblib.dump(model,file_model)
        y = model.predict([X_test])[0]
        return y,file_model

    def wf_validation(self,X_train,X_test,y_train,y_test,save_model):
        prediction = []
        file_model = []
        X_train = [x for x in X_train]
        y_train = [y for y in y_train]
        for i in range(len(X_test)):
            muestra_x,muestra_y = X_test[i,:],y_test[i]
            y_hat,name_model = self.RF_model_forecast(X_train=X_train,X_test=muestra_x,y_train=y_train,k=i,save_model=save_model)
            prediction.append(y_hat)
            file_model.append(name_model)
            X_train.append(muestra_x)
            y_train.append(muestra_y)
            # Crear y actualizar el gráfico
            plt.figure()  # Crea una nueva figura
            plt.plot(y_test[:i + 1], label='Expected', marker='o')  # Dibuja los valores reales hasta el momento
            plt.plot(prediction, label='Predicted', marker='x')  # Dibuja las predicciones hasta el momento
            plt.legend()
            plt.draw()  # Redibuja el gráfico
            plt.pause(0.1)  # Pausa breve para ver la actualización
            plt.show()
            print('>expected=%.1f, predicted=%.1f' % (muestra_y, y_hat))
        error = mean_absolute_error(y_test,prediction)
        if save_model==True:
            n_model= 'name_model_ts'+'.pkl'
            n_model = os.path.join(self.root_model, n_model)
            joblib.dump(file_model,n_model)
        return error,prediction


    def model_ts(self,target,EM, M,save_model):
        X = self.data.drop(target,axis=1)
        columns = X.columns
        y = self.data[target].ravel()
        X = X[:EM]
        y = y[:EM]
        X_train, X_test, y_train, y_test = self.train_test_order(X=X, y=y, n_test=M)
        scaler = MinMaxScaler()
        scaler.fit(X_train)

        if save_model==True:
            file_scaler= 'scaler_ts'+'.pkl'
            file_scaler = os.path.join(self.root_model  , file_scaler)
            joblib.dump(scaler,file_scaler)

            file_params= 'params_ts'+'.pkl'
            file_params = os.path.join(self.root_model, file_params)
            joblib.dump(columns,file_params)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        mae,y_pred = self.wf_validation(X_train=X_train_scaled,X_test=X_test_scaled,y_train=y_train,y_test=y_test,save_model=save_model)
        print('MAE: %.3f' % mae)
        print('R2: %.3f' % r2_score(y_test, y_pred))
        plt.plot(y_test, label='Expected')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        plt.show()


