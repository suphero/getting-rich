"""
A Machine Learning Framework for Stock Selection
    
Authors:
XingYu Fu; JinHong Du; YiFeng Guo; MingWen Liu; Tao Dong; XiuWen Duan; 

Institutions:
AI&Fintech Lab of Likelihood Technology; 
Gradient Trading;
Sun Yat-sen University;

Contact:
fuxy28@mail2.sysu.edu.cn

All Rights Reserved.
"""


"""Import Modules"""
# Numerical Computation
import numpy as np
from math import sqrt
# Load Data
import pandas as pd
import os


"""Load Factor-Data from Disk"""
def load_tail( start, sample_num, F, Q, Sequential = False):
    """Specify the data path"""
    path_factor = r"./database/factor"
    path_price  = r"./database/price"
    
    """Loading Data"""
    Factor = []
    Price = []
    factor_begin = 0
    factor_count = 0
    flag = 1
    for data_name in os.listdir(path_factor):
        if data_name == (start+".csv"):
            factor_begin = 1

        if factor_begin == 1:
            if flag % (F+1) == 1: # Load Factor
                x = pd.read_csv( path_factor + '/'+ data_name, header = None)
                Factor.append(x)
                factor_count += 1
            else:
                y = pd.read_csv( path_price + '/'+ data_name, header = None)
                Price.append(y)
                
            flag += 1
    
        if factor_count == sample_num+1:
            Factor = Factor[:-1]
            break
    
    
    """Data Regularization"""
    X_return = []
    Y_return = []
    for t in range( len(Factor) ):
        X = np.array( Factor[t], dtype=object )
        P = [ np.array( Price[i], dtype=object ) for i in range(F*t,F*t+F) ]
      
        # Step1: Find the stocks that appear in factor matrix and price vectors at the same time.
        n_x = set( X[:,0] )
        name_p = [ set(pp[:,0]) for pp in P]
        for n_p in name_p:
            n_x = n_p & n_x
        XX = []
        YY = []
        for row in X:
            if row[0] in n_x:
                XX.append( row[1:] )
                y = []
                for pp in P:
                    n_p = pp[:,0]
                    index = np.where( n_p == row[0] )[0][0]
                    y.append( pp[index][1] )
                y = np.array(y)
                YY.append( y )
            else:
                continue
        XX = np.array(XX, dtype = np.float)
        YY = np.array(YY, dtype = np.float)
        
        # Step2: Replace nan with column average
        XX = np.where(np.isnan(XX), np.ma.array(XX, mask=np.isnan(XX)).mean(axis=0), XX)
        
        # Step3: Factor Normalization
        for j in range( len( XX[0] ) ):
            max_j = max( XX[:,j] )
            min_j = min( XX[:,j] )
            for i in range( len( XX ) ):
                XX[i][j] = (XX[i][j]-min_j)/(max_j-min_j)
        
        # Step4: Standard Deviation; Return_Rate; Anomaly Filtering;
        Filter = [True for y in YY]
        return_std = []
        for i in range( len(YY) ):
            y = YY[i]
            mean = sum( y )/np.float( len(y) )
            std = sqrt( sum([ (price-mean)**2 for price in y])/np.float( len(y) ) )
            
            if std == 0:
                Filter[i] = False
                
            if y[0]!=0:
                return_rate = y[-1]/y[0]
            else:
                return_rate = np.nan
                Filter[i] = False
                
            return_std.append( ( return_rate, std) )
        
        XX = XX[ Filter ]
        return_std = np.array( return_std )
        return_std = return_std[ Filter ]
        YY = np.array( [ pair[0]/pair[1] for pair in return_std] )
        
        # Step5: Tail Set Construction
        XX = XX[ (-YY).argsort() ]
        bd = np.int(np.round( len(XX)*Q ))
        Top = np.array( XX[0:bd,:] )
        Bottom = np.array( XX[-bd:,:] )
        XX = np.concatenate( (Top,Bottom) )
        YY = np.concatenate( (np.ones(bd),np.zeros(bd)) )
        permutation = np.random.permutation(2*bd)
        XX = XX[permutation]
        YY = YY[permutation]
        
        if t == 0:
            X_return = XX
            Y_return = YY
        else:
            X_return = np.concatenate((X_return, XX))
            Y_return = np.concatenate((Y_return, YY))
        
    # Shuffle
    if Sequential:
        pass
    else:
        permutation = np.random.permutation(X_return.shape[0])
        X_return = X_return[permutation]
        Y_return = Y_return[permutation]

    return X_return, Y_return


def load_whole( start, F):
    """Specify the data path"""
    path_factor = r"./database/factor"
    path_price  = r"./database/price"
    
    """Loading Data"""
    Factor = []
    Price = []
    begin = 0
    flag = 1
    for data_name in os.listdir(path_factor):
        if data_name == (start+".csv"):
            begin = 1

        if begin == 1:
            if flag == 1: # Load Factor
                Factor = pd.read_csv( path_factor + '/'+ data_name, header = None)
            elif flag <= F+1:
                y = pd.read_csv( path_price + '/'+ data_name, header = None)
                Price.append(y)
            else:
                break
            flag += 1
        else:
            continue
    
    """Data Regularization"""
    Factor = np.array( Factor, dtype = object)
    Price = [ np.array( Price[t], dtype=object ) for t in range(F) ]
      
    # Step1: Find the stocks that appear in factor matrix and price vectors at the same time.
    n_x = set( Factor[:,0] )
    n_p_list = [ set(p_t[:,0]) for p_t in Price]
    for n_p in n_p_list:
        n_x = n_p & n_x
    XX = []
    YY = []
    for row in Factor:
        if row[0] in n_x:
            XX.append( row[1:] )
            y = []
            for p_t in Price:
                n_p = p_t[:,0]
                index = np.where( n_p == row[0] )[0][0]
                y.append( p_t[index][1] )
            y = np.array(y)
            YY.append( y )
        else:
            continue
    XX = np.array(XX, dtype = np.float)
    YY = np.array(YY, dtype = np.float)
        
    # Step2: Replace nan with column average
    XX = np.where(np.isnan(XX), np.ma.array(XX, mask=np.isnan(XX)).mean(axis=0), XX)
        
    # Step3: Factor Normalization
    for j in range( len( XX[0] ) ):
        max_j = max( XX[:,j] )
        min_j = min( XX[:,j] )
        for i in range( len( XX ) ):
            XX[i][j] = (XX[i][j]-min_j)/(max_j-min_j)
        
    # Step4: Standard Deviation; Return_Rate; Anomaly Filtering;
    Filter = [True for y in YY]
    for i in range(len(YY)):
        price_vector = YY[i]
        if 0.0 in price_vector:
            Filter[i] = False
            continue
        mean = sum(price_vector)/len(price_vector)
        variance = sum( [ (p-mean)**2 for p in price_vector] )
        if variance == 0:
            Filter[i] = False
                
    XX = XX[ Filter ]
    YY = YY[ Filter ]
        
    
    # Step5: Shuffle
    permutation = np.random.permutation(XX.shape[0])
    XX = XX[permutation]
    YY = YY[permutation]

    return XX, YY