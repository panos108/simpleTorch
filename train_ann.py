import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class Data(Dataset):

    def __init__(self, X, F):
        X_dtype = torch.FloatTensor
        F_dtype = torch.FloatTensor

        self.length = X.shape[0]

        self.X_data = torch.from_numpy(X).type(X_dtype)
        self.F_data = torch.from_numpy(F).type(F_dtype)

    def __getitem__(self, index):
        return self.X_data[index], self.F_data[index]

    def __len__(self):
        return self.length

class train_ann:

    def __init__(self, model, X, F, optimizer=None, loss_fn=None,
                 learning_rate=1e-3, print_val=True, epoch=200, batch_size=68,
                 validation_set=0.33, auto_normalize=True, normalize_x= (-1, 1),
                 normalize_y = (-1,1), l2_reg=0., plot=False):
        """
        Initialize the training of the ann
        :param model:            This is the ann import from outside
        :param X:                Available data of input F = model(X)
        :type X:                 np.array(N,n_in), n_in is the dimension of a single input
        :param F:                Labels for the training
        :type F:                 np.array(N,n_out), n_in is the dimension of a single label
        :param optimizer:        This is the optimizer for the training, if None is given, adam is used
        :param loss_fn:          Define a loss function if None is given, Mean squared error is employed
        :param learning_rate:    The learning rate for the training default = 1e-3
        :type learning_rate:     float
        :param print_val:        If True then it prints the progress of the training
        :type print_val:         Boolean
        :param epoch:            This is the number of epochs that the neural netowrk is trained with.
        :type epoch:             Integer
        :param batch_size:       This is the number of samples that are used in each epoch
        :type batch_size:        Integer
        :param validation_set:   This is the number of data that are used for validation
        :type validation_set:    Double
        :param normalize_x:      This is a tuple contains the min-max that the data inputs are normalized.
                                 default: min=-1, max=1
        :type normalize_x:       Tuple
        :param normalize_y:      This is a tuple contains the min-max that the data labels are normalized.
                                 default: min=-1, max=1
        :type normalize_y:       Tuple
        :param l2_reg:           This is the parameter for the L2 regularization
        :type l2_reg:            Positive float number (Default =0., no regularization)
        :param plot:             If this value is true then it plots the loss
        :type  plot:             Boolean
        """
        self.epoch          = epoch
        self.batch_size     = batch_size
        self.plot           = plot

        self.scale_x        = MinMaxScaler(feature_range=(normalize_x[0],normalize_x[1]))
        self.scale_f        = MinMaxScaler(feature_range=(normalize_y[0],normalize_y[1]))
        self.auto_normalize = auto_normalize
        self.l2_regulization= l2_reg
        if self.auto_normalize:
            X_scale = self.scale_x.fit_transform(X)
            F_scale = self.scale_f.fit_transform(F)
        else:
            X_scale = X
            F_scale = F

        #Split the set to train and test (validation)
        self.X_train, self.X_test, self.F_train, self.F_test = \
            train_test_split(X_scale, F_scale, test_size=validation_set, random_state=0)

        if optimizer == None:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        if loss_fn   == None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn   = loss_fn

        self.print_val     = print_val
        self.model         = model
        #Transform data for pytorch
        self.dataset_train = Data(self.X_train, self.F_train)
        self.dataset_test  = Data(self.X_train, self.F_train)

        # Run the training
        self.run()

    def train_batch(self, X, F):
        """
        This function preforms the batch training
        :param X: Input data of the batch as input
        :type X:  array of torch.float64
        :param F: Label data of the batch
        :type F:  array of torch.float64
        :return:  Loss of this batch
        :rtype:   float64
        """
        F_predict = self.model(minX)          # Forward propagation

        loss = self.loss_fn(F_predict, F)  # loss calculation
        # Add the L2 normalization
        lambdas = self.l2_regulization
        l2_reg = torch.tensor(0.)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        loss += lambdas *l2_reg

        self.optimizer.zero_grad()         # all grads of variables are set to 0 before backward calculation

        loss.backward()                    # Backward propagation

        self.optimizer.step()              # update parameters

        return loss.data.item()


    def train(self, loader):
        """
        This function takes the batch data loader and performs for training for all epochs
        :param loader: All the data for each batch for X, F
        :return: It returns all the losses
        :rtype:  list with the losses
        """
        losses = list()
        batch_index = 0
        for e in range(self.epoch):
            for X, F in loader:
                loss = self.train_batch(X, F)
                batch_index += 1
                if self.print_val:
                    print("Epoch: ", e + 1, " Batches: ", batch_index, " Loss: ", loss)
            losses.append(loss)

        self.losses_train = losses
        return losses

    def plot_loss(self, losses, show=True):
        ax = plt.axes()
        ax.set_ylabel("Loss")
        x_loss = list(range(len(losses)))
        plt.plot(x_loss, losses)
        plt.xlabel('Epoch')
        if show:
            plt.show()

        plt.close()

    def run(self):
        """
        This function does the wrapping of the training and validation.
        First structures the data ready for torch operations using DataLoader
        :return: Trained model
        """
        # Batch size is the number of training examples used to calculate each iteration's gradient
        batch_size_train = self.batch_size
        dataset_train = self.dataset_train
        dataset_test  = self.dataset_test
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
        data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=True)

        # Train and get the resulting loss per iteration
        loss = self.train(loader=data_loader_train)

        # Test and get the resulting predicted y values
        F_predict = self.perform_validation(loader=data_loader_test)
        if self.plot:
            self.plot_loss(self.losses_train)

        print('The loss of training is: ', self.losses_train[-1])
        print('The loss of validation is: ', self.losses_val[0])

        return self.model

    def validate_model(self, X, F):
        F_predict = self.model(X)

        return F, F_predict

    def perform_validation(self, loader):
        F_vectors = list()
        F_predict_vectors = list()
        losses = list()  # empty loss list to collect and track how loss changes with epoch

        batch_index = 0
        for X, F in loader:
            F, F_predict = self.validate_model(X=X, F=F)
            loss = self.loss_fn(F_predict, F).data.item() # loss calculation
            losses.append(loss)
            F_vectors.append(F.data.numpy())
            F_predict_vectors.append(F_predict.data.numpy())

            batch_index += 1
        self.losses_val = losses
        F_predict_vector = np.concatenate(F_predict_vectors)
        return F_predict_vector

    def predict(self, x):
        """
        This function performs the predictions after the training
        :param x: Input (feature) that we want to perform prediction with
        :type x:  numpy array vector
        :return:  Prediction of the value
        :rtype:   numpy array vector
        """
        #Perfom the prediciton for the trained ANN with normalization or without it
        if self.auto_normalize:
            x_scale = self.scale_x.transform(x)
        else:
            x_scale = x
        # Perform the prediction
        x_scale = torch.from_numpy(np.array(x_scale)).float()
        y_scale = self.model(x_scale).detach().numpy()

        if self.auto_normalize:
            y = self.scale_f.inverse_transform(y_scale)
        else:
            y = y_scale
        return y


