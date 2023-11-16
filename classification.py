import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def classification_analysis(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.header("Classification Results")

    # Accuracy Score
    st.write("### Accuracy Score:")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(accuracy)

    # Classification Report
    st.write("### Classification Report:")
    st.write(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("### Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    return accuracy


def train_classification_model(selected_model_clf, X_train, y_train):
    if selected_model_clf == 'Logistic Regression':
        return LogisticRegression().fit(X_train, y_train)
    elif selected_model_clf == 'Random Forest Classifier':
        return RandomForestClassifier().fit(X_train, y_train)
    elif selected_model_clf == 'SVM':
        return SVC().fit(X_train, y_train)
    else:
        st.warning('Invalid model selection.')


def predict_classification(input_values_clf, y_test):
    if st.session_state.model_clf is not None:
        input_data_clf = pd.DataFrame([input_values_clf])
        prediction_clf = st.session_state.model_clf.predict(input_data_clf)
        st.success(f'The predicted {y_test} is: {prediction_clf[0]}')
    else:
        st.warning('Please run the classification analysis first.')


def print_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    st.success(f'Accuracy: {accuracy:.2f}')
    st.success(f'Precision: {precision:.2f}')
    st.success(f'Recall: {recall:.2f}')
    st.success(f'F1 Score: {f1:.2f}')


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)


