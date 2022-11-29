import xlwings as xw
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from pmdarima import model_selection
import joblib  # for persistence
import os
import math
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from math import sqrt


def main():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]
    if sheet["A1"].value == "Hello xlwings!":
        sheet["A1"].value = "Bye xlwings!"
    else:
        sheet["A1"].value = "Hello xlwings!"


@xw.func
def hello(name):
    return f"Hello {name}!"

@xw.func
@xw.arg('X_train', np.array, ndim=2)
@xw.arg('Pilihan')
# @xw.arg('Index', np.array, ndim=2)
# @xw.arg('X_new', np.array, ndim=2)
def AutoArima(X_train, Pilihan):
    #wb = xw.Book.caller()
    model = pm.auto_arima(X_train, start_p=1, start_q=1,
                         test='adf',
                         max_p=12, max_q=12, m=12,
                         start_P=0, seasonal=False,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
    
    if Pilihan == 0 :
        model = model.fit(X_train)
        n_periods = 1
        X_pred = model.predict(n_periods=n_periods)
        return X_pred
    
    elif Pilihan == "p" :
        #n_periods = 1
        #X_pred = model.predict(n_periods=n_periods)
        ordera = model.order
        p,d,q = ordera
        return p
    
    elif Pilihan == "d" :
        #n_periods = 1
        #X_pred = model.predict(n_periods=n_periods)
        ordera = model.order
        p,d,q = ordera
        return d
            
    elif Pilihan == "q" :
        #n_periods = 1
        #X_pred = model.predict(n_periods=n_periods)
        ordera = model.order
        p,d,q = ordera
        return q
    
    elif Pilihan == "mape" :
        test = X_train[-10:]
        train = X_train[:-10]
        model = model.fit(train)
        Pred = model.predict(10)
        mape = mean_absolute_percentage_error(test, Pred)
        return mape
    
    elif Pilihan == "rmse" :
        test = X_train[-10:]
        train = X_train[:-10]
        model = model.fit(train)
        Pred = model.predict(10)
        rmse = sqrt(mean_squared_error(test, Pred))
        return rmse
    elif Pilihan == "r2" :
        test = X_train[-10:]
        train = X_train[:-10]
        model = model.fit(train)
        Pred = model.predict(10)
        #zx = (X_train-np.mean(X_train))/np.std(X_train, ddof=1)
        #zy = (Pred-np.mean(Pred))/np.std(Pred, ddof=1)
        #r = np.sum(zx*zy)/(len(X_train)-1)
        #return r**2
        #adj_r2 = 1 - ( 1-r2_score(X, y) ) * ( len(y) - 1 ) / ( len(y) - X.shape[1] - 1 )
        #adj_r2 = 1 - ( 1-r2_score(test, Pred) ) * ( len(Pred) - 1 ) / ( len(Pred) - test.shape[1] - 1 )
        #adj_r2 = r2_score(test,Pred)
        n = len(test)
        x_bar = sum(test)/n
        y_bar = sum(Pred)/n
        x_std = math.sqrt(sum([(xi-x_bar)**2 for xi in test])/(n-1))
        y_std = math.sqrt(sum([(yi-y_bar)**2 for yi in Pred])/(n-1))
        zx = [(xi-x_bar)/x_std for xi in test]
        zy = [(yi-y_bar)/y_std for yi in Pred]
        r = sum(zxi*zyi for zxi, zyi in zip(zx, zy))/(n-1)
        return r**2
        #return adj_r2
    elif Pilihan == "coeff" :
        Pred = model.predict()





@xw.func
@xw.arg('p')
@xw.arg('d')
@xw.arg('q')
@xw.arg('X_data', np.array, ndim=2)
@xw.arg('parameter')
def ManArima(p,d,q,X_data,parameter):
    arima = pm.ARIMA(order=(p, d, q))
    if parameter == 0 :
        arima.fit(X_data)
        pickle_tgt = "arima.pkl"
        try:
            # Pickle it
            joblib.dump(arima, pickle_tgt, compress=3)
        
            # Load the model up, create predictions
            arima_loaded = joblib.load(pickle_tgt)
            preds = arima_loaded.predict(n_periods=1)
            #print("Predictions: %r" % preds)
            return (preds)
        
        finally:
            # Remove the pickle file at the end of this example
            try:
                os.unlink(pickle_tgt)
            except OSError:
                pass
        #   print(model.summary())
    elif parameter == "mape" :
        test = X_data[-10:]
        train = X_data[:-10]
        arima.fit(train)
        pickle_tgt = "arima.pkl"
        try:
            # Pickle it
            joblib.dump(arima, pickle_tgt, compress=3)
        
            # Load the model up, create predictions
            arima_loaded = joblib.load(pickle_tgt)
            preds = arima_loaded.predict(10)
            mape = np.mean(np.abs(preds - test)/np.abs(test))
            return mape
        
        finally:
            try:
                os.unlink(pickle_tgt)
            except OSError:
                pass
    elif parameter == "rmse" :
        test = X_data[-10:]
        train = X_data[:-10]
        arima.fit(train)
        pickle_tgt = "arima.pkl"
        try:
            # Pickle it
            joblib.dump(arima, pickle_tgt, compress=3)
        
            # Load the model up, create predictions
            arima_loaded = joblib.load(pickle_tgt)
            preds = arima_loaded.predict(10)
            #rmse = np.mean((preds - test)**2)**.5
            rmse = sqrt(mean_squared_error(test, preds))
            return rmse
        
        finally:
            try:
                os.unlink(pickle_tgt)
            except OSError:
                pass
    elif parameter == "r2" :
        test = X_data[-10:]
        train = X_data[:-10]
        arima.fit(train)
        pickle_tgt = "arima.pkl"
        try:
            # Pickle it
            joblib.dump(arima, pickle_tgt, compress=3)
        
            # Load the model up, create predictions
            arima_loaded = joblib.load(pickle_tgt)
            preds = arima_loaded.predict(10)
            n = len(test)
            x_bar = sum(test)/n
            y_bar = sum(preds)/n
            x_std = math.sqrt(sum([(xi-x_bar)**2 for xi in test])/(n-1))
            y_std = math.sqrt(sum([(yi-y_bar)**2 for yi in preds])/(n-1))
            zx = [(xi-x_bar)/x_std for xi in test]
            zy = [(yi-y_bar)/y_std for yi in preds]
            r = sum(zxi*zyi for zxi, zyi in zip(zx, zy))/(n-1)
            return r**2
        
        finally:
            try:
                os.unlink(pickle_tgt)
            except OSError:
                pass

@xw.func
@xw.arg('X_data1', np.array, ndim=2)
@xw.arg('AR')
def ARone(X_data1, AR):
   # Load the data and split it into separate pieces
    
    train, test = model_selection.train_test_split(X_data1, train_size=.80)

    # Fit an ARIMA
    arima = pm.ARIMA(order=(1, 0, 0))
    fitted = arima.fit(X_data1)
    if AR == 0 :
        # Persist a model and create predictions after re-loading it
        pickle_tgt = "arima.pkl"
        try:
            # Pickle it
            joblib.dump(fitted, pickle_tgt, compress=3)
        
            # Load the model up, create predictions
            arima_loaded = joblib.load(pickle_tgt)
            preds = arima_loaded.predict(n_periods=1)
            #print("Predictions: %r" % preds)
            return (preds)
        
        finally:
            # Remove the pickle file at the end of this example
            try:
                os.unlink(pickle_tgt)
            except OSError:
                pass
        #   print(model.summary())
    elif AR == 1 :

            coef = fitted.arparams()
            #print("Predictions: %r" % preds)
            return (coef)

@xw.func
@xw.arg('X_all', np.array, ndim=2)
@xw.arg('Y_all', np.array, ndim=2)
@xw.arg('X_predx', np.array, ndim=2)
@xw.arg('Pilihanxg')
# @xw.arg('Index', np.array, ndim=2)
# @xw.arg('X_new', np.array, ndim=2)
def xg_boost(X_all, Y_all, X_predx, Pilihanxg):
    
    reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators=500,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
    
    if Pilihanxg == 0 :
        reg.fit(X_all, Y_all, eval_set=[(X_all, Y_all)], verbose=100)
        y_pred = reg.predict(X_predx)
        return(y_pred)
    
    elif Pilihanxg == "mape" :
       x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size = 0.2, random_state = 4)
       reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=100)
       y_pred = reg.predict(x_test)
       mape = np.mean(np.abs(y_pred - y_test)/np.abs(y_test))
       return(mape)
   
    elif Pilihanxg == "r2" :
       x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size = 0.2, random_state = 4)
       reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=100)
       y_pred = reg.predict(x_test)
       r2new = r2_score(y_test, y_pred)
       return(r2new)
   
    elif Pilihanxg == "rmse" :
       x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size = 0.2, random_state = 4)
       reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=100)
       y_pred = reg.predict(x_test)
       rmse = np.mean((y_pred - y_test)**2)**.5  # RMSE
       return(rmse) 
    
    
@xw.func
@xw.arg('X_all2', np.array, ndim=2)
@xw.arg('Y_all2', np.array, ndim=2)
@xw.arg('X_predx2', np.array, ndim=2)
@xw.arg('n_est')
@xw.arg('maxd')
@xw.arg('Pilihanxg2')
# @xw.arg('Index', np.array, ndim=2)
# @xw.arg('X_new', np.array, ndim=2)
def xg_boost_tuning(X_all2, Y_all2, X_predx2, n_est, maxd, Pilihanxg2):
    n_est = int(n_est)
    maxd = int(maxd)
    
    reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators= n_est,
                       objective='reg:linear',
                       max_depth= maxd,
                       learning_rate=0.01)
    
    if Pilihanxg2 == 0 :
        reg.fit(X_all2, Y_all2, eval_set=[(X_all2, Y_all2)], verbose=100)
        y_pred = reg.predict(X_predx2)
        return(y_pred)
    
    elif Pilihanxg2 == "mape" :
       x_train, x_test, y_train, y_test = train_test_split(X_all2, Y_all2, test_size = 0.2, random_state = 4)
       reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=100)
       y_pred = reg.predict(x_test)
       mape = np.mean(np.abs(y_pred - y_test)/np.abs(y_test))
       return(mape)
   
    elif Pilihanxg2 == "r2" :
       x_train, x_test, y_train, y_test = train_test_split(X_all2, Y_all2, test_size = 0.2, random_state = 4)
       reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=100)
       y_pred = reg.predict(x_test)
       r2new = r2_score(y_test, y_pred)
       return(r2new)
   
    elif Pilihanxg2 == "rmse" :
       x_train, x_test, y_train, y_test = train_test_split(X_all2, Y_all2, test_size = 0.2, random_state = 4)
       reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=100)
       y_pred = reg.predict(x_test)
       rmse = np.mean((y_pred - y_test)**2)**.5  # RMSE
       return(rmse) 

if __name__ == "__main__":
    xw.Book("AutoArima5.xlsm").set_mock_caller()
    xw.serve()
    #my_macro()
    main()

