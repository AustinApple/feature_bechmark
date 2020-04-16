from __future__ import print_function
import numpy as np
import pandas as pd
import RNN_property_predictor
from add_function_group.feature import molecules
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
try:
    import tensorflow.compat.v1 as tf 
    tf.compat.v1.disable_v2_behavior()
except:
    import tensorflow as tf


def train(input_file, epochs, random_seed, property):
    char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", "s", "O",
              "[", "Cl", "Br", "\\"]
    
    data = pd.read_csv(input_file)
    np.random.seed(random_seed)
    random_index = np.random.permutation(len(data))
    data = data.iloc[random_index]
    train, validation, test = np.split(data, [int(0.8*len(data)), int(0.9*len(data))])

    x_train, x_train_g, new_smi_train = molecules(train['smiles'].tolist()).one_hot(char_set=char_set)
    y_train = train[[property]].values

    # this is for check
    print("the size of x_train"+str(x_train.shape))
    print("the size of y_train"+str(x_train.shape))

    
    x_val, x_val_g, new_smi_val = molecules(validation['smiles'].tolist()).one_hot(char_set=char_set)
    y_val = validation[[property]].values

    # this is for check
    print("the size of x_val"+str(x_val.shape))
    print("the size of y_val"+str(x_val.shape))


    x_test, x_test_g, new_smi_test = molecules(test['smiles'].tolist()).one_hot(char_set=char_set)
    y_test = test[[property]].values
    
    # this is for check
    print("the size of x_test"+str(x_test.shape))
    print("the size of y_test"+str(y_test.shape))



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
        model.train(trnX_L=x_train, trnY_L=y_train,valX_L=x_val, valY_L=y_val, epochs=epochs)
        #model.saver.save(model.session, "./SMILES(RNN)_model/SMILES(RNN)"+str(random_seed)+'.ckpt')
        y_test_hat=model.predict(x_test)
        
        mae = mean_absolute_error(y_test, y_test_hat)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_hat))

    tf.reset_default_graph()
    return mae, rmse 
    

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
        mae, rmse = train(input_file=args["data_file"], epochs=args["epochs"],random_seed=i, property=args["property"])
        #print(train(input_file=args["data_file"], epochs=args["epochs"],random_seed=i))
        log[i,0] = mae
        log[i,1] = rmse
        
    print('########################################################')
    print('mean : '+str(np.mean(log, axis=0)))
    print('std : '+str(np.std(log, axis=0)))

