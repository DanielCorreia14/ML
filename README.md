Time Series Analysis with MLPRegressor
This project aims to forecast time series values using a Multilayer Perceptron Regressor (MLPRegressor), an artificial neural network from scikit-learn, to predict future values of a stock market index (Ibovespa).

Code Description
The code follows these steps:

1. Data Reading and Cleaning
The ibovespa.csv file is read and processed.
The values in the "ibovespa" column are converted to numeric format using pd.to_numeric, invalid values are converted to NaN, and then removed with dropna().
2. Data Normalization
The data is normalized to a 0 to 1 range using MinMaxScaler from scikit-learn.
This step ensures that the input values are within a suitable range for model training.
3. Creating a Windowed Dataset with TimeseriesGenerator
The time series is divided into "windows" of observations, where each window contains 10 consecutive values.
TimeseriesGenerator creates input-output pairs (X, y), where inputs represent previous values and outputs represent the next predicted value.
4. Model Training
The MLPRegressor neural network model is configured with two hidden layers (100 and 50 neurons) and uses the ReLU activation function.
The model is trained using the windowed dataset from the previous step.
5. Model Evaluation
After training, the model is evaluated using three key metrics:

Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted values.
R² (Coefficient of Determination): Indicates how well the model explains the variability in the data.
6. Result Visualization
The code generates several plots to help analyze the model’s performance:

Loss Curve: Shows how the model's error decreases during training.
Training Data Comparison: A graph comparing actual vs. predicted values in the training set.
Residuals Plot: A scatter plot displaying the difference between actual and predicted values in the test set.
7. Saved Plots
All plots are saved as PNG files in the script’s execution directory:

first_window.png – Displays the first data window.
loss_curve.png – Shows the loss curve during training.
training_data.png – Compares actual vs. predicted values in training.
residuals.png – Displays model residuals.
Requirements
Python 3.x
Libraries:
pandas
numpy
scikit-learn
tensorflow
matplotlib
To install the required dependencies, run:


pip install -r requirements.txt
You can generate a requirements.txt file using:
 
pip freeze > requirements.txt
How to Run
To execute the code, run the following command:

python3 main.py
The script will generate the plots and display evaluation metrics in the terminal.
