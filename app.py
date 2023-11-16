import streamlit as st
import pandas as pd
import data_cleaner as dc
import matplotlib.pyplot as plt
import seaborn as sns
import regression as r
import classification as cl
from data_cleaner import perform_data_cleaning_and_eda, handle_outliers, delete_selected_columns
from sklearn.model_selection import train_test_split

# Styling
st.markdown(
    """
    <style>
        body {
            background-color: #fffff;
            color: #32333;
            font-family: 'Arial', sans-serif;
        }
        .primary-button {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
            width : 200px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    st.title("Data Analysis Tool")
    st.markdown("---")

    # Upload a dataset
    uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded_file is not None:
        reload_button = st.sidebar.button("Reload Data", help="Click to reload data", use_container_width=True,
                                          type='primary')
        if reload_button:
            st.session_state.data = None
            st.success('Data Reloaded')

    columns_with_outliers = []

    if 'data' not in st.session_state:
        st.session_state.data = None

    if uploaded_file is not None:
        data, duplicate_count, null_count, num_cols, cat_cols, total_rows = perform_data_cleaning_and_eda(uploaded_file)
        if st.session_state.data is None:
            st.session_state.data = data
        else:
            data = st.session_state.data

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Rows")
            st.title(total_rows)
        with col2:
            st.header("Numerical Columns")
            st.title(num_cols)
        with col3:
            st.header("Categorical Columns")
            st.title(cat_cols)
        with col4:
            st.header("Duplicate Values")
            st.title(duplicate_count)
        st.subheader("Top Rows of the Data:")
        st.dataframe(data.head(50))

        st.header("Column Description")
        if st.button("Show Column Info", help="Click to display column information", key="column_info_button",
                     type='primary', use_container_width=True):
            column_info = dc.get_column_info(data)
            st.write("### Column Information:")
            st.dataframe(column_info)

        # st.markdown('---')

        st.sidebar.header("What do you want to begin with?")
        tabs = ["Data Cleaning", "Regression", "Classification"]
        selected_tab = st.sidebar.selectbox("Select Tab:", tabs)

        st.markdown('---')

        if selected_tab == "Data Cleaning":
            # Null Values
            st.header("Null Values")
            null_button = st.button("Show Null Count", help="Click to show null count", use_container_width=True,
                                    key="null_button", type='primary')
            if null_button:
                st.write("Number of Null Values:")
                st.write(dc.handle_missing_values(data))
            st.markdown('---')

            # Unique Values

            st.header("Check Unique Values")
            selected_check_column = st.selectbox("Select a column to check unique values:", list(data.columns))
            check_unique_button = st.button("Check Unique Values", help="Click to check unique values",
                                            use_container_width=True,
                                            key="check_unique_button", type='primary',
                                            disabled=not selected_check_column)
            if check_unique_button and selected_check_column != 'Choose an option':
                unique_values = dc.check_unique_values(data, selected_check_column)
                if unique_values is not None:
                    st.write(f"Unique values in {selected_check_column}:")
                    st.write(unique_values)
                else:
                    st.warning(f"Column '{selected_check_column}' not found.")

            st.markdown('---')

            # Delete Columns

            st.header("Delete Columns")
            selected_columns = st.multiselect("Select columns to delete:", data.columns)
            delete_columns_button = st.button("Delete Columns", disabled=not selected_columns, use_container_width=True,
                                              key="delete_button", type='primary')
            if delete_columns_button and selected_columns:
                data = delete_selected_columns(data, selected_columns)
                st.session_state.data = data  # Save the modified DataFrame
                st.success("Columns deleted successfully.")
                st.dataframe(data)

            st.markdown('---')

            # Treat Outliers

            st.header("Treat Outliers")

            treat_outliers_button = st.button("Treat Outliers", use_container_width=True, key="treat_outliers_button",
                                              type='primary')

            if treat_outliers_button:
                columns_with_outliers_info = dc.handle_outliers_info(data)
                if not columns_with_outliers_info:
                    st.warning("No columns have outliers.")
                else:
                    st.success("Columns with Outliers:")

                    # Create a list of tuples for the table
                    table_data = [(column, outliers_count) for column, outliers_count in columns_with_outliers_info]

                    # Display the table
                    st.table(table_data)

                data, _ = handle_outliers(data)
                st.success("Outliers treated successfully.")

            st.markdown('---')

            # Replace Columns

            st.header("Replace Values in Columns")
            selected_columns3 = st.selectbox("Select Categorical Column:", data.select_dtypes(include='O').columns)

            if selected_columns3:
                unique_values = data[selected_columns3].unique()
                selected_value = st.selectbox("Select Value to Replace:", unique_values, key="replace_value")
                replacement_value = st.text_input("Enter Replacement Value:")
                replace_button = st.button("Replace", help="Click to replace selected value", use_container_width=True,
                                           key="replace_button",
                                           type='primary', disabled=not selected_columns3)

                if replace_button and selected_value is not None and replacement_value is not None:
                    # Call the function to perform the actual replacement in the DataFrame
                    data = dc.replace_values_in_column(data, selected_columns3, selected_value, replacement_value)
                    st.success(
                        f"Value '{selected_value}' in column '{selected_columns3}' replaced with '{replacement_value}'.")

            st.markdown('---')

            # Encode Columns
            st.header("Encode Categorical Columns")
            cat_columns_list = [col for col in data.columns if data[col].dtype == 'O']
            selected_encode_columns = st.multiselect("Select non-numeric columns to encode:", cat_columns_list)
            encoding_method = st.radio("Select encoding method:", ('Label Encoding', 'One-Hot Encoding'))
            if st.button("Apply Encoding", help="Click to apply selected encoding method", type='primary',
                         use_container_width=True, disabled=not selected_encode_columns):
                if selected_encode_columns and encoding_method:
                    if encoding_method == 'Label Encoding':
                        encoded_data = dc.encode_categorical_columns(data, selected_encode_columns,
                                                                     method='label')
                    elif encoding_method == 'One-Hot Encoding':
                        encoded_data = dc.encode_categorical_columns(data, selected_encode_columns,
                                                                     method='onehot')
                    st.success(f"{encoding_method} applied to selected columns.")
                    st.session_state.data = encoded_data
                    st.dataframe(data)
                else:
                    st.warning("Please select at least one column and encoding method.")

            st.markdown('---')

            st.header("Change Date Column Format")

            selected_date_column = st.selectbox("Select a date column:",
                                                data.select_dtypes(include=['datetime', 'object']).columns)
            change_date_button = st.button("Change Date", key="change_date_button", disabled=not selected_date_column,
                                           use_container_width=True, type='primary')

            if change_date_button and selected_date_column:
                success, data, datatype = dc.change_date_column_format(data, selected_date_column)
                if success:
                    st.success(f"Date format changed successfully for column '{selected_date_column}'.")
                else:
                    st.warning(
                        f"Unable to convert column '{selected_date_column}' to datetime format as '{selected_date_column}' is of '{datatype}'.")

            st.markdown('---')

            # change datatype
            st.header("Change Datatype")
            all_data_types = ['int64', 'float64', 'object', 'bool', 'datetime64[ns]', 'category']
            selected_column = st.selectbox("Select Column:", data.columns)
            data_type_names = {
                'int64': 'Integer',
                'float64': 'Float',
                'object': 'String',
                'bool': 'Boolean',
                'datetime64[ns]': 'Datetime',
                'category': 'Categorical',
            }  # Add more data types as needed
            selected_data_type = st.selectbox("Select Data Type:", all_data_types, index=0,
                                              format_func=lambda x: data_type_names[x])

            # Allow the user to change the data type
            new_data_type = st.selectbox("Select New Data Type:", all_data_types,
                                         index=all_data_types.index(selected_data_type),
                                         format_func=lambda x: data_type_names[x])

            # Button to apply changes
            if st.button("Apply Changes"):
                # Change the data type of the selected column in the DataFrame
                data[selected_column] = data[selected_column].astype(new_data_type)

            st.markdown('---')

            # Correlation Matrix
            st.header("Correlation Matrix")

            if st.button('Show Correlation Matrix', use_container_width=True, type='primary'):
                st.header('Correlation Matrix')
                correlation_matrix = dc.correlation_analysis(data)

                # Create a heatmap using seaborn
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)

                # Display the figure using st.pyplot()
                st.pyplot(fig)

            st.markdown('---')

            # EDA

            st.header("Univariate Analysis")
            selected_column = st.selectbox("Select a column for univariate analysis: ", data.columns)
            plot_type = st.selectbox("Select the plot type:",
                                     ["Hist Plot", "Count Plot", "Violin Plot", "Kernel Density Apply"])
            show_plot = st.button("Show Plot", key="univariate_plot_button", use_container_width=True, type='primary')

            if show_plot:
                plot = dc.univariate_analysis(data, selected_column, plot_type)
                st.pyplot(plot)  # Pass the figure to st.pyplot
                download_plot_button = st.download_button(
                    label="Download Plot as JPEG",
                    data=dc.save_plot_as_image(plot),
                    file_name=f'univariate_analysis_{selected_column}.jpeg',
                    mime='image/jpeg',
                    key="download_univariate_plot_button",
                    type="primary",
                    help="Click to download the univariate analysis plot as a JPEG file.",
                    use_container_width=True
                )

            st.markdown('---')

            st.header("Bivariate Analysis")
            x_column = st.selectbox("Select the x-axis column:", data.columns)
            y_column_options = [col for col in data.columns if col != x_column]
            y_column = st.selectbox("Select the y-axis column:", y_column_options)
            if pd.api.types.is_numeric_dtype(data[x_column]) and pd.api.types.is_numeric_dtype(data[y_column]):
                plot_type = st.selectbox("Select the plot type:", ["Scatter Plot", "Line Plot"])
            else:
                plot_type = st.selectbox("Select the plot type:", ["Scatter Plot", "Box Plot", "Line Plot"])

            show_plot = st.button("Show Bivariate Plot", key="bivariate_plot_button", use_container_width=True,
                                  type='primary')
            if show_plot:
                plot = dc.bivariate_analysis(data, x_column, y_column, plot_type)
                st.pyplot(plot)
                download_bi_plot_button = st.download_button(
                    label="Download Plot as JPEG",
                    data=dc.save_plot_as_image(plot),
                    file_name=f'bivariate_analysis_{x_column},{y_column}_{plot_type}.jpeg',
                    mime='image/jpeg',
                    key="download_bivariate_plot_button",
                    type="primary",
                    help="Click to download the bivariate analysis plot as a JPEG file.",
                    use_container_width=True
                )
            apply_changes_button = st.sidebar.button("Apply Changes and Save to CSV", key="apply_changes_button",
                                                     type='primary',
                                                     use_container_width=True)
            if apply_changes_button:
                if st.session_state.data is not None:
                    st.session_state.data.to_csv(f'{uploaded_file.name}.csv', index=False)
                    st.success("Changes applied and saved to 'original_data.csv'.")
                else:
                    st.warning("No changes to apply. Please perform data analysis first.")

            st.markdown('---')

            download_button = st.download_button('Download Updated Data', data.to_csv(), file_name='updated_data.csv',
                                                 type="primary", use_container_width=True)

        elif selected_tab == "Regression":
            st.title('Regression Analysis')

            st.sidebar.header('Select Target Column for Regression')
            y_column_reg = st.sidebar.selectbox('Select the target variable:', data.columns)

            st.sidebar.header('Select Regression Model')
            selected_model_reg = st.sidebar.selectbox('Choose a regression model:',
                                                      ['Linear Regression', 'Random Forest Regression',
                                                       'Support Vector Regression'])

            # Add any additional parameters for the chosen model, if needed

            if st.sidebar.button('Run Regression Analysis'):
                # Train the regression model and get test set
                st.session_state.model_reg, st.session_state.X_train_reg, st.session_state.y_train_reg, st.session_state.X_test_reg, st.session_state.y_test_reg = r.train_regression_model(
                    selected_model_reg, data, y_column_reg)

                # Display regression metrics
                r.display_regression_metrics(st.session_state.model_reg, st.session_state.X_test_reg,
                                             st.session_state.y_test_reg)

                # Optionally, visualize regression results
                r.visualize_regression_results(st.session_state.model_reg, st.session_state.X_test_reg,
                                               st.session_state.y_test_reg)





        elif selected_tab == "Classification":
            st.title('Classification Analysis')
            st.sidebar.header('Select Target Column for Classification')
            y_column_clf = st.sidebar.selectbox('Select the target variable:', data.columns)
            x_columns_clf = [col for col in data.columns if col != y_column_clf]
            st.sidebar.header('Select Classification Model')
            selected_model_clf = st.sidebar.selectbox('Choose a classification model:',
                                                      ['Logistic Regression', 'Random Forest Classifier', 'SVM'])
            test_size_clf = st.sidebar.slider('Select Test Size:', 0.1, 0.5, value=0.2, step=0.05)
            random_state_clf = st.sidebar.number_input('Select Random State:', min_value=0, max_value=100, value=42,
                                                       step=1)
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data[x_columns_clf], data[y_column_clf],
                                                                                test_size=test_size_clf,
                                                                                random_state=random_state_clf)
            st.sidebar.info("Make sure to split your data into training and testing sets before running the analysis.")

            if st.sidebar.button('Run Classification Analysis'):
                st.session_state.model_clf = cl.train_classification_model(selected_model_clf, X_train_clf, y_train_clf)
                y_pred_clf = st.session_state.model_clf.predict(X_test_clf)
                cl.print_classification_metrics(y_test_clf, y_pred_clf)
                cl.plot_confusion_matrix(y_test_clf, y_pred_clf, class_names=data[y_column_clf].unique())

            # Prediction Section
            st.sidebar.header('Predict with the Trained Model')

            # Create input fields for each independent variable (X)
            input_values_clf = {}
            for col in x_columns_clf:
                if data[col].dtype == 'category':  # Check if the column is categorical
                    unique_values = data[col].unique()
                    selected_value = st.sidebar.selectbox(f'Select value for {col}:', unique_values)
                    input_values_clf[col] = selected_value
                else:
                    input_values_clf[col] = st.sidebar.number_input(f'Enter value for {col}:', min_value=0.0)

            # Add a button to trigger prediction
            if st.sidebar.button('Predict', key='predict_button_clf', type='primary'):
                if st.session_state.model_clf is not None:
                    # Create a DataFrame with the input values
                    input_data_clf = pd.DataFrame([input_values_clf])

                    # Use the trained model for prediction
                    prediction_clf = st.session_state.model_clf.predict(input_data_clf)

                    # Display the predicted result
                    st.success(f'The predicted {y_column_clf} is: {prediction_clf[0]}')
                else:
                    st.warning('Please run the classification analysis first.')


if __name__ == "__main__":
    main()
