import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def RMSLE(pred, true):
    """
    计算 RMSLE（均方对数误差）：
    RMSLE = sqrt( mean( (log(1 + y_pred) - log(1 + y_true))^2 ) )

    参数:
        pred: 预测值数组
        true: 真实值数组
    返回:
        RMSLE 值
    """
    return np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rmsle = RMSLE(pred, true)
    
    return mae,mse,rmse,mape,mspe