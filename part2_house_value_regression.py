import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

class Regressor(torch.nn.Module):

    def __init__(self, x, nb_epoch = 100):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Call the parent constructor
        super(Regressor, self).__init__()


        # Initialize preprocessing objects
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.label_binarizer = LabelBinarizer()
        self.scaler = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.output_size = 1  # Assuming regression task where output is a single value
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        
        # Neural network layers
        self.fc1 = torch.nn.Linear(self.input_size, 8)
        self.fc2 = torch.nn.Linear(8, self.output_size)
        # self.fc3 = torch.nn.Linear(5, 3)
        # self.fc4 = torch.nn.Linear(3, 2)
        # self.fc5 = torch.nn.Linear(2, 1)
        # self.fc6 = torch.nn.Linear(1, )
        
        # Initialize input and output size based on preprocessed data
        self.nb_epoch = nb_epoch

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Copy the input to avoid modifying the original DataFrame
        x_processed = x.copy()
        y_processed = y.copy() if y is not None else None

        # Handle Missing Values - Numeric Columns
        numeric_cols = x_processed.select_dtypes(include=['number']).columns
        if training:
            # If training, calculate and store values needed for preprocessing
            # For example, you might want to calculate mean values from the training data
            self.numeric_imputer.fit(x_processed[numeric_cols])

        # Apply the imputer to fill missing values in both training and testing data for numeric columns
        x_processed[numeric_cols] = self.numeric_imputer.transform(x_processed[numeric_cols])

        # Normalize Numerical Values
        if training:
            # If training, calculate and store values needed for normalization
            self.scaler.fit(x_processed[numeric_cols])
            if y is not None:
                self.scaler_y.fit(y_processed)

        # Apply normalization to numerical columns in both training and testing data
        x_processed[numeric_cols] = self.scaler.transform(x_processed[numeric_cols])
        if y is not None:
            y_processed[y_processed.columns] = self.scaler_y.transform(y_processed)
            y_processed = torch.tensor(y_processed.values, dtype=torch.float32)

        # Handle Textual Values - One-Hot Encoding
        categorical_cols = x_processed.select_dtypes(exclude=['number']).columns
        if training:
            # If training, fit the LabelBinarizer to learn the mapping
            self.label_binarizer.fit(x_processed[categorical_cols])

        # Apply one-hot encoding to categorical columns
        categorical_encoded = pd.DataFrame(self.label_binarizer.transform(x_processed[categorical_cols]),
                                           columns=self.label_binarizer.classes_)

        # Resetting the index of both DataFrames
        x_processed_reset = x_processed.reset_index(drop=True)
        categorical_encoded_reset = categorical_encoded.reset_index(drop=True)
        
        # Concatenate the encoded values to the processed DataFrame
        x_processed = pd.concat([x_processed_reset, categorical_encoded_reset], axis=1)

        # Drop the original categorical columns
        x_processed.drop(categorical_cols, axis=1, inplace=True)
        x_processed = torch.tensor(x_processed.values, dtype=torch.float32)

        # Return preprocessed x and y
        return x_processed, y_processed

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess the input data
        X, Y = self._preprocessor(x, y, training = True) # Do not forget

        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.12)  # adjustable learning rate

        # Assuming X and Y are tensors or numpy arrays
        dataset = TensorDataset(X, Y)

        # Training loop
        for epoch in range(self.nb_epoch):
            # Set a new random seed for each epoch
            torch.manual_seed(epoch)

            # Shuffle the dataset with the current epoch as the seed
            indices = torch.randperm(len(dataset))

            shuffled_dataset = TensorDataset(X[indices], Y[indices])

            # Iterate over batches
            for batch_X, batch_Y in DataLoader(shuffled_dataset, batch_size=128):
                
                # Forward pass
                outputs = self(batch_X)

                # Compute the loss
                loss = criterion(outputs, batch_Y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget

        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self(X).numpy()

        return predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
        
    def forward(self, x):
        # Define the forward pass with multiple layers
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.fc2(x)
        return x


    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        _, Y = self._preprocessor(x, y, training = False) # Do not forget

        Y_pred = self.predict(x)
        
        Y = self.scaler_y.inverse_transform(Y)
        Y_pred = self.scaler_y.inverse_transform(Y_pred)

        # Calculate evaluation metrics
        mse = mean_squared_error(Y, Y_pred, squared=False)
        r2 = r2_score(Y, Y_pred)
        
        # Return the evaluation metric of your choice
        return mse  # You can choose to return a different metric if needed

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:round(0.8 * len(data)), data.columns != output_label]
    y_train = data.loc[:round(0.8 * len(data)), [output_label]]
    x_test = data.loc[round(0.8 * len(data)):, data.columns != output_label]
    y_test = data.loc[round(0.8 * len(data)):, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

