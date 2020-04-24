from __future__ import print_function
import numpy as np
import pandas as pd
import RNN_property_predictor
from add_function_group.feature import molecules
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse
try:
    import tensorflow.compat.v1 as tf 
    tf.compat.v1.disable_v2_behavior()
except:
    import tensorflow as tf


def test(input_file, epochs, property, n_splits, normalize):
    char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", "s", "O",
              "[", "Cl", "Br", "\\"]
    
    data = pd.read_csv(input_file)
    if normalize:
        print("start to normalize")
        scaler_Y = StandardScaler()
        scaler_Y.fit(data[property])
        data[property] = scaler_Y.transform(data[property])
    
    log = np.zeros((n_splits,len(property),2))   # axis-0 : # of samples, axis-1 : # of property, axis-2 : MAE and RMSE
    
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(data)):
        x_train, new_smi_train = molecules(data['smiles'][train_index].tolist()).one_hot(char_set=char_set)
        y_train = data[property].iloc[train_index].values
      
        x_test, new_smi_test = molecules(data['smiles'][test_index].tolist()).one_hot(char_set=char_set)
        y_test = data[property].iloc[test_index].values

        print('::: model training')

        seqlen_x = x_train.shape[1]
        print("the length of sequence"+ str(seqlen_x))
        dim_x = x_train.shape[2]
        dim_y = y_train.shape[1]
        dim_z = 100
        dim_h = 250

        n_hidden = 3
        batch_size = 32


        model = RNN_property_predictor.Model(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h,
                            n_hidden = n_hidden, batch_size = batch_size, char_set = char_set)
        with model.session:
            model.train(trnX_L=x_train, trnY_L=y_train, epochs=epochs)
            for j in range(len(property)):
                if normalize:
                    log[i,j,0] = mean_absolute_error(scaler_Y.inverse_transform(y_test)[:,j],scaler_Y.inverse_transform(model.predict(x_test))[:,j])
                    log[i,j,1] = mean_squared_error(scaler_Y.inverse_transform(y_test)[:,j],scaler_Y.inverse_transform(model.predict(x_test))[:,j],squared=False)
                else:
                    log[i,j,0] = mean_absolute_error(y_test[:,j],model.predict(x_test)[:,j])
                    log[i,j,1] = mean_squared_error(y_test[:,j],model.predict(x_test)[:,j],squared=False)
        
        tf.reset_default_graph()
    return log
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--data_file',
                        help='the file including smiles and corresponding IE and EA', default='MP_clean_canonize_cut.csv')
    parser.add_argument('-e', '--epochs',
                        help='how many epochs would be trained', default=200, type=int)
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

    
    with open("test_RNN",'a+') as f:
        
        f.write('########################################################\n')
        f.write(str(args['property'])+'\n')
        f.write('normalize :' + str(args['normalize']) + '\n')


        for i in range(len(args["property"])):
            f.write(args["property"][i]+' MAE mean : '+str(np.mean(log[:,i,0]))+'\n')
            f.write(args["property"][i]+' MAE std : '+str(np.std(log[:,i,0]))+'\n')
            f.write(args["property"][i]+' RMSE mean : '+str(np.mean(log[:,i,1]))+'\n')
            f.write(args["property"][i]+' RMSE std : '+str(np.std(log[:,i,1]))+'\n')