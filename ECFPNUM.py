import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from add_function_group.feature import molecules
# K.tensorflow_backend._get_available_gpus()
import argparse



def test(input_file, epochs, property, n_splits, normalize):
    '''
    the argument poperty is a list including the output property
    '''  
    data = pd.read_csv(input_file)
    if normalize:
        print("start to normalize")
        scaler_Y = StandardScaler()
        scaler_Y.fit(data[property])
        data[property] = scaler_Y.transform(data[property])
    
    log = np.zeros((n_splits,len(property),2))   # axis-0 : # of samples, axis-1 : # of property, axis-2 : MAE and RMSE
    
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        x_train = molecules(data['smiles'][train_index].tolist()).ECFP_num()
        y_train = data[property].iloc[train_index].values
      
        x_test = molecules(data['smiles'][test_index].tolist()).ECFP_num()
        y_test = data[property].iloc[test_index].values
        
        
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        # building model
        model = Sequential()
        model.add(Dense(output_dim=int(x_train.shape[1]/2), input_dim=x_train.shape[1],activation='relu'))
        model.add(Dense(output_dim=int(x_train.shape[1]/6),activation='relu'))
        model.add(Dense(output_dim=int(x_train.shape[1]/12),activation='relu'))
        model.add(Dense(output_dim=int(y_train.shape[1])))
        model.compile(loss='mae', optimizer='adam',metrics=['mae',rmse])
        print('Training -----------')
        model.fit(x_train, y_train, verbose=1, epochs=epochs)


        for j in range(len(property)):
            if normalize:
                log[i,j,0] = mean_absolute_error(scaler_Y.inverse_transform(y_test)[:,j],scaler_Y.inverse_transform(model.predict(x_test))[:,j])
                log[i,j,1] = mean_squared_error(scaler_Y.inverse_transform(y_test)[:,j],scaler_Y.inverse_transform(model.predict(x_test))[:,j],squared=False)
            else:
                log[i,j,0] = mean_absolute_error(y_test[:,j],model.predict(x_test)[:,j])
                log[i,j,1] = mean_squared_error(y_test[:,j],model.predict(x_test)[:,j],squared=False)


    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--data_file',
                        help='the file including smiles and corresponding IE and EA', default='MP_clean_canonize_cut.csv')
    parser.add_argument('-e', '--epochs',
                        help='how many epochs would be trained', default=100, type=int)
    parser.add_argument('-n', '--n_splits',
                        help='how many folds', default=10, type=int)
    parser.add_argument('-p', '--property', nargs='+',
                        help='which property do you want to train')
    
    parser.add_argument('--normalize',action='store_true')
    
    args = vars(parser.parse_args())
    
    if args["normalize"]:
        log = test(input_file=args["data_file"], epochs=args["epochs"], property=args["property"], 
                   n_splits=args["n_splits"], normalize=args["normalize"])
    else:
        log = test(input_file=args["data_file"], epochs=args["epochs"], property=args["property"], 
                   n_splits=args["n_splits"], normalize=False)

    
    # print('########################################################')
    
   
    # for i in range(len(args["property"])):
    #     print(args["property"][i]+' MAE mean : '+str(np.mean(log[:,i,0])))
    #     print(args["property"][i]+' MAE std : '+str(np.std(log[:,i,0])))
    #     print(args["property"][i]+' RMSE mean : '+str(np.mean(log[:,i,1])))
    #     print(args["property"][i]+' RMSE std : '+str(np.std(log[:,i,1])))
        

    with open("test",'a+') as f:
        
        f.write('########################################################\n')
        f.write(str(args['property'])+'\n')
        f.write('normalize :' + str(args['normalize']) + '\n')


        for i in range(len(args["property"])):
            f.write(args["property"][i]+' MAE mean : '+str(np.mean(log[:,i,0]))+'\n')
            f.write(args["property"][i]+' MAE std : '+str(np.std(log[:,i,0]))+'\n')
            f.write(args["property"][i]+' RMSE mean : '+str(np.mean(log[:,i,1]))+'\n')
            f.write(args["property"][i]+' RMSE std : '+str(np.std(log[:,i,1]))+'\n')
