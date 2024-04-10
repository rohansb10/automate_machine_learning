import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC

def main():
    # Custom CSS to change background color to sky blue
    page_bg_img = '''
    <style>
    body {
    background-color: #87CEEB; /* Sky Blue */
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Auto Machine Learning")

    # Allow user to upload a CSV file for training
    uploaded_file = st.file_uploader("Upload CSV file for training", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file, encoding='latin1')  # Change encoding to 'latin1'

            # Display the shape of the DataFrame
            st.subheader("Shape of the DataFrame:")
            st.write(df.shape)

            # Display the count of null values
            st.subheader("Count of Null Values:")
            st.write(df.isnull().sum())

            # Display the count of duplicated rows
            st.subheader("Count of Duplicated Rows:")
            st.write(df.duplicated().sum())

            # Handling null values
            df.dropna(inplace=True)
            
            # Remove duplicated rows
            df.drop_duplicates(inplace=True)

            # Display the top 5 rows of the DataFrame after preprocessing
            st.subheader("Top 5 rows of the DataFrame (after preprocessing):")
            st.write(df.head())

            # Allow user to select columns to drop
            columns_to_drop = st.multiselect("Select columns to drop:", df.columns)

            # Drop selected columns
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)

            # Show the shape of the DataFrame after dropping columns
            st.subheader("Shape of the DataFrame (after dropping columns):")
            st.write(df.shape)

            # Select columns of type 'object' (categorical) for label encoding
            cat_columns = df.select_dtypes(include=['object']).columns.tolist()
            selected_columns = st.multiselect("Select columns for Label Encoding:", cat_columns)

            if selected_columns:
                # Perform label encoding
                label_encoders = {}
                for col in selected_columns:
                    label_encoder = LabelEncoder()
                    df[col] = label_encoder.fit_transform(df[col])
                    label_encoders[col] = label_encoder

                # Display encoded DataFrame
                st.subheader("DataFrame after Label Encoding:")
                st.write(df.head())

                # Check if user wants to use StandardScaler
                use_scaler = st.checkbox("Use StandardScaler?")
                if use_scaler:
                    # Apply StandardScaler to selected columns
                    columns_to_scale = st.multiselect("Select columns for StandardScaler:", df.columns)
                    if columns_to_scale:
                        scaler = StandardScaler()
                        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                        st.subheader("DataFrame after StandardScaler:")
                        st.write(df.head())

            # Allow user to specify the prediction (target) column
            prediction_column = st.selectbox("Select the prediction column:", df.columns)

            if prediction_column:
                # Perform train-test split
                X = df.drop(columns=[prediction_column])
                y = df[prediction_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Display predicted column
                st.subheader("Predicted Column:")
                st.write(y_train)

                # Allow user to choose task type (regression or classification)
                task_type = st.radio("Select Task Type:", ["Regression", "Classification"])

                if task_type:
                    if task_type == "Regression":
                        # Regression models
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Random Forest Regression": RandomForestRegressor(),
                            "Gradient Boosting Regression": GradientBoostingRegressor()
                        }
                    else:
                        # Classification models
                        models = {
                            "Logistic Regression": LogisticRegression(),
                            "Random Forest Classifier": RandomForestClassifier(),
                            "Gradient Boosting Classifier": GradientBoostingClassifier()
                        }

                    # Train and evaluate each model
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred_train = model.predict(X_train)  # Predictions on training data
                        y_pred_test = model.predict(X_test)  # Predictions on testing data
                        if task_type == "Regression":
                            st.subheader(f"Model: {name}")
                            st.write("Training R2 Score:", r2_score(y_train, y_pred_train))
                            st.write("Testing R2 Score:", r2_score(y_test, y_pred_test))
                        else:
                            st.subheader(f"Model: {name}")
                            st.write("Training Accuracy Score:", accuracy_score(y_train, y_pred_train))
                            st.write("Testing Accuracy Score:", accuracy_score(y_test, y_pred_test))
                            
                    st.subheader("NOTE : ")
                    st.markdown(" checking overfitting and underfitting models")
                    st.markdown(" check difference between testing and training data sets ")

                    # Decode predictions if label encoding was performed
                    if selected_columns:
                        st.subheader("Decoded Predictions on Testing Data:")
                        decoded_predictions_test = pd.DataFrame()
                        decoded_predictions_test[prediction_column] = y_test
                        for col in selected_columns:
                            decoded_predictions_test[col] = label_encoders[col].inverse_transform(X_test[col])
                        decoded_predictions_test['Predicted'] = y_pred_test
                        st.write(decoded_predictions_test.head())

                    # Allow user to input values for each column and generate predictions
                    st.header("Make Predictions")
                    input_data = {}
                    for col in X.columns:
                        input_data[col] = st.text_input(f"Enter value for {col}:", value="")
                    input_df = pd.DataFrame([input_data])

                    # Encode categorical columns
                    if selected_columns:
                        for col, encoder in label_encoders.items():
                            input_df[col] = encoder.transform([input_data[col]])

                    # Apply StandardScaler if selected
                    if use_scaler and columns_to_scale:
                        scaler = StandardScaler()
                        input_df[columns_to_scale] = scaler.fit_transform(input_df[columns_to_scale])

                    # Allow user to select a model
                    selected_model = st.selectbox("Select Model:", list(models.keys()))

                    # Make predictions for the input data using the selected model
                    if selected_model:
                        model = models[selected_model]
                        predictions = model.predict(input_df)
                        if task_type == "Regression":
                            st.subheader(f"Predictions for Input Data using {selected_model}:")
                            st.write(predictions)
                        else:
                            st.subheader(f"Predictions for Input Data using {selected_model} (Encoded):")
                            st.write(predictions)
                            # Decode predictions if label encoding was performed
                            if selected_columns:
                                decoded_predictions = label_encoders[prediction_column].inverse_transform(predictions)
                                st.subheader(f"Predictions for Input Data using {selected_model} (Decoded):")
                                st.write(decoded_predictions)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please make sure the file you uploaded is a valid CSV file.")

if __name__ == "__main__":
    main()
