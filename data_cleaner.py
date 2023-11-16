import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_data(file_path):
    return pd.read_csv(file_path)


def get_total_rows(df):
    total_rows = df.shape[0]
    return total_rows


def get_num_column(df):
    # Get the data types of each column
    column_types = df.dtypes
    num_numeric_cols = column_types[column_types.apply(pd.api.types.is_numeric_dtype)].count()
    return num_numeric_cols


def get_cat_column(df):
    column_types = df.dtypes
    num_categorical_cols = column_types[column_types == 'object'].count()
    return num_categorical_cols


def remove_duplicates(df):
    # Remove duplicate rows if they exist in the DataFrame
    dups = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    duplicate_count = len(df.index)  # Get the count of removed duplicates
    return dups


def get_column_info(data):
    if data is not None:
        column_info = pd.DataFrame({
            'Data Type': data.dtypes
        })
        return column_info
    else:
        return pd.DataFrame()


def handle_missing_values(df):
    # Get the count of null values in each column
    null_count = df.isnull().sum()

    # Iterate through columns and handle missing values
    for column in df.select_dtypes(include=['number']).columns:
        # For numeric columns, fill missing values with the mean
        df[column].fillna(df[column].mean(), inplace=True)

    for column in df.select_dtypes(include=['object']).columns:
        # For non-numeric columns, fill missing values with the mode
        mode_value = df[column].mode().iloc[0]
        df[column].fillna(mode_value, inplace=True)

    return null_count


def check_unique_values(data, selected_column):
    if selected_column in data.columns:
        unique_values = data[selected_column].unique()
        return unique_values
    else:
        return None


def delete_selected_columns(df, selected_columns):
    df.drop(columns=selected_columns, inplace=True)
    return df


def handle_outliers_info(df):
    columns_with_outliers = []

    for column in df.select_dtypes(include=np.number).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        if outliers_count > 0:
            columns_with_outliers.append((column, outliers_count))

    return columns_with_outliers


def handle_outliers(df):
    columns_with_outliers = handle_outliers_info(df)

    for column, _ in columns_with_outliers:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), np.nan, df[column])

    return df, columns_with_outliers


def replace_values_in_column(data, selected_column, selected_value, replacement_value):
    if selected_column and selected_value is not None and replacement_value is not None:
        # Replace the values in the selected column
        data[selected_column] = data[selected_column].replace(selected_value, replacement_value)
    return data


def encode_categorical_columns(data, selected_columns, method='label'):
    encoded_data = data.copy()

    for column in selected_columns:
        if column in data.columns and data[column].dtype == 'O':  # Check if the column exists and is non-numeric
            if method == 'label':
                label_encoder = LabelEncoder()
                encoded_data[column] = label_encoder.fit_transform(data[column])
            elif method == 'onehot':
                onehot_encoder = OneHotEncoder(sparse=False, drop='first')
                encoded_column = onehot_encoder.fit_transform(data[[column]])
                encoded_data = pd.concat([encoded_data, pd.DataFrame(encoded_column, columns=[f'{column}_{i}' for i in
                                                                                              range(
                                                                                                  encoded_column.shape[
                                                                                                      1])])], axis=1)

    return encoded_data


def change_date_column_format(data, selected_date_column):
    if selected_date_column in data.columns and data[selected_date_column].dtype == 'O':
        try:
            # Attempt to convert the selected column to datetime format
            data[selected_date_column] = pd.to_datetime(data[selected_date_column])

            return True, data, data[selected_date_column].dtype
        except ValueError:
            return False, data, data[selected_date_column].dtype
    else:
        return False, data, data[selected_date_column].dtype


def summary_statistics(df):
    # Calculate and return summary statistics of the dataset
    summary_stats = df.describe()
    return summary_stats


def correlation_analysis(df):
    # Calculate and return correlations between variables
    correlation_matrix = df.corr()
    return correlation_matrix


def univariate_analysis(df, selected_column, plot_type):
    plt.figure(figsize=(10, 6))

    if plot_type == "Hist Plot":
        sns.histplot(df[selected_column], kde=True)
        plt.title(f"Univariate Analysis for {selected_column}")
        plt.xticks(rotation=45)

    elif plot_type == "Count Plot":
        sns.countplot(data=df, x=selected_column)
        plt.title(f"Univariate Analysis for {selected_column}")
        plt.xticks(rotation=45)

    elif plot_type == "Violin Plot":
        sns.violinplot(data=df, x=selected_column)
        plt.title(f"Univariate Analysis for {selected_column}")
        plt.xticks(rotation=45)

    elif plot_type == "Kernel Density Apply":
        sns.kdeplot(data=df, x=selected_column)
        plt.title(f"Univariate Analysis for {selected_column}")
        plt.xticks(rotation=45)

    return plt.gcf()


def save_plot_as_image(plot):
    # Save the plot as an image
    img_data = io.BytesIO()
    plt.savefig(img_data, format='jpeg')
    img_data.seek(0)
    return img_data


def bivariate_analysis(df, x_column, y_column, plot_type):
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "Scatter Plot":
        sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax)
        ax.set_title(f"Bivariate Analysis - Scatter Plot for {x_column} vs {y_column}")
    elif plot_type == "Box Plot":
        sns.boxplot(x=df[x_column], y=df[y_column], ax=ax)
        ax.set_title(f"Bivariate Analysis - Box Plot for {x_column} vs {y_column}")
        ax.tick_params(axis='x', rotation=45)
    elif plot_type == "Line Plot":
        sns.lineplot(x=df[x_column], y=df[y_column], ax=ax)
        ax.set_title(f"Bivariate Analysis - Line Plot for {x_column} vs {y_column}")
        ax.tick_params(axis='x', rotation=45)
    # Add more conditions for additional plot types if needed

    return fig

def perform_data_cleaning_and_eda(file_path):
    data = load_data(file_path)
    duplicate_count = remove_duplicates(data)
    null_count = handle_missing_values(data)
    num_cols = get_num_column(data)
    cat_cols = get_cat_column(data)
    total_rows = get_total_rows(data)
    return data, duplicate_count, null_count, num_cols, cat_cols, total_rows


