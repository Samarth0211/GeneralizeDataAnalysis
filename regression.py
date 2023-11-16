# regression.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_session_state():
    if 'model_reg' not in st.session_state:
        st.session_state.model_reg = None
        st.session_state.X_train_reg = None
        st.session_state.y_train_reg = None
        st.session_state.X_test_reg = None
        st.session_state.y_test_reg = None


# Your existing regression analysis function
@st.cache(allow_output_mutation=True)
def train_regression_model(selected_model, data, y_column, x_columns=None):
    # If x_columns are not provided, use all columns except the target variable
    if x_columns is None:
        x_columns = [col for col in data.columns if col != y_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[x_columns], data[y_column], test_size=0.2, random_state=42)

    # Train the selected regression model
    if selected_model == 'Linear Regression':
        model = LinearRegression()
    elif selected_model == 'Random Forest Regression':
        model = RandomForestRegressor()
    elif selected_model == 'Support Vector Regression':
        model = SVR()

    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test

def display_regression_metrics(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Display metrics
    st.header('Regression Metrics')
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R-squared: {r_squared}')

def visualize_regression_results(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Create a scatter plot of actual vs. predicted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, label='Actual vs. Predicted')

    # Plot the regression line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '-', label='Regression Line', color='red')

    plt.title('Actual vs. Predicted Values with Regression Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(plt)



get_session_state()
