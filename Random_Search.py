from sklearn.model_selection import KFold
import Bayesian_Neural_Network
import numpy as np

def K_Fold_Cross_Validation(cycles, learning_rate, l1_neurons, X, y, n_folds = 5,):
    performance = []          
   
    kf = KFold(n_splits =n_folds, shuffle = True, random_state = 42)
    test_index_set = 0
    for train_index, test_index in kf.split(X):
        test_index_set += 1
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
          
        N_train = x_train.shape[0]
        N_val = x_test.shape[0]
        # Make y 2d
        y_train = np.expand_dims(y_train, 1)
        y_test = np.expand_dims(y_test, 1)
        data_train = Bayesian_Neural_Network.tf.data.Dataset.from_tensors((x_train, y_train))
        data_test = Bayesian_Neural_Network.tf.data.Dataset.from_tensors((x_test, y_test))
        BNN_model = Bayesian_Neural_Network.BNN_Reg([np.shape(X)[1], l1_neurons, 1])
        optimizer = Bayesian_Neural_Network.tf.keras.optimizers.SGD(learning_rate= learning_rate) #gradient descent
        root_mse = Bayesian_Neural_Network.perform(BNN_model, optimizer, cycles, data_train, data_test, x_train.shape[0])
        performance.append(root_mse)
        print(f"Fold: {test_index_set}")
        print(f"Root MSE: {root_mse}")

    print("Mean Performance:", np.mean(performance))

    return np.mean(performance)

def random_search_BNN(hl_nodes,epochs,learning_rate, X, y):
    info = []
    for nodes in hl_nodes:
        for eps in epochs:
            for lr in learning_rate:
                rmse = K_Fold_Cross_Validation(eps, lr, nodes, X, y)
                info.append(rmse)
                print("Processed %0.3f out of %0.3f elements" % (len(info),len(epochs)*len(hl_nodes)*len(learning_rate)))

    opt = np.min(info,axis = 0)
    return print("Number of nodes: %0.3f, number of learning cycles: %0.3f, learning rate: %0.3f" %(opt[1], opt[2], opt[3]))
  
