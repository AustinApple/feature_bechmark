import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from add_function_group.feature import molecules
# K.tensorflow_backend._get_available_gpus()
import argparse



def train(input_file, epochs, random_seed, property):
    data = pd.read_csv(input_file)
    
    np.random.seed(random_seed)
    random_index = np.random.permutation(len(data))
    data = data.iloc[random_index]
    train, validation, test = np.split(data, [int(0.8*len(data)), int(0.9*len(data))])
    
    x_train = molecules(train['smiles'].tolist()).ECFP_num()
    y_train = train[[property]].values
    
    x_val = molecules(validation['smiles'].tolist()).ECFP_num()
    y_val = validation[[property]].values

    x_test = molecules(test['smiles'].tolist()).ECFP_num()
    y_test = test[[property]].values
    
    
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
    model.fit(x_train, y_train, verbose=1, epochs=epochs, validation_data=[x_val, y_val])
    
    print("test testing set")
    return model.evaluate(x_test, y_test)
    

# prediction = mlp.predict(X_test)
# print(mean_absolute_error(y_test, prediction))
# print(mean_squared_error(y_test, prediction))

    # save model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_file',
                        help='the file including smiles and corresponding IE and EA', default='MP_clean_canonize_cut.csv')
    parser.add_argument('-e', '--epochs',
                        help='how many epochs would be trained', default=100, type=int)
    parser.add_argument('-r', '--rounds',
                        help='how many rounds', default=10, type=int)
    parser.add_argument('-p', '--property',
                        help='which property do you want to train')
    args = vars(parser.parse_args())
    
    log = np.zeros((args["rounds"],2))

    for i in range(args["rounds"]):
        loss, mae, rmse = train(input_file=args["data_file"], epochs=args["epochs"],random_seed=i, property=args["property"])
        #print(train(input_file=args["data_file"], epochs=args["epochs"],random_seed=i))
        log[i,0] = mae
        log[i,1] = rmse
    
    print('########################################################')
    print('mean : '+str(np.mean(log, axis=0)))
    print('std : '+str(np.std(log, axis=0)))


