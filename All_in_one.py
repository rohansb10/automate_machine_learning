import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Main Streamlit app
def main():
    st.title("Automate Machine Learning")

    # Create a dropdown to select the file
    selected_file = st.selectbox("Select a file", ["Machine learning", "Unsupervised Learning", "Data Analysis"])

    # Execute the selected file
    if selected_file == "Machine learning":
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

    elif selected_file == "Unsupervised Learning":
        st.title("Unsupervised Learning")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.subheader("Top 5 rows of the uploaded file:")
            st.write(df.head())

            preprocess_data = st.checkbox("Preprocess Data")

            if preprocess_data:
                remove_duplicates = st.checkbox("Remove Duplicates")
                if remove_duplicates:
                    df.drop_duplicates(inplace=True)

                handle_null_values = st.checkbox("Handle Null Values")
                if handle_null_values:
                    impute_numerical = st.checkbox("Impute Numerical Columns")
                    if impute_numerical:
                        numerical_cols = df.select_dtypes(include='number').columns
                        for col in numerical_cols:
                            df[col].fillna(df[col].mean(), inplace=True)

                    impute_categorical = st.checkbox("Impute Categorical Columns")
                    if impute_categorical:
                        categorical_cols = df.select_dtypes(include='object').columns
                        for col in categorical_cols:
                            df[col].fillna(df[col].mode()[0], inplace=True)

                check_outliers = st.checkbox("Check Outliers")
                if check_outliers:
                    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
                    selected_column = st.selectbox("Select a column to visualize outliers", numerical_cols)
                    st.subheader(f"Box plot for {selected_column}")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=df[selected_column], ax=ax)
                    st.pyplot(fig)

            st.subheader("Summary:")
            st.write(f"Total duplicate values: {df.duplicated().sum()}")
            st.write(f"Total null values: {df.isnull().sum().sum()}")

            st.subheader("Unsupervised Learning Steps:")

            # User selects two columns to create a new dataframe
            col1, col2 = st.columns(2)
            with col1:
                column1 = st.selectbox("Select first column", df.columns)
            with col2:
                column2 = st.selectbox("Select second column", df.columns)

            new_df = df[[column1, column2]]

            st.subheader("New DataFrame:")
            st.write(new_df)

            # Show scatterplot graph
            st.subheader("Scatterplot:")
            sns.scatterplot(data=new_df, x=column1, y=column2)
            st.pyplot()

            # Dimensionality Reduction
            st.subheader("Dimensionality Reduction:")

            dimensionality_reduction_method = st.radio("Select Dimensionality Reduction Method:", ("PCA", "t-SNE"))

            if dimensionality_reduction_method == "PCA":
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(new_df)
            else:
                tsne = TSNE(n_components=2)
                reduced_data = tsne.fit_transform(new_df)

            reduced_df = pd.DataFrame(reduced_data, columns=["Component 1", "Component 2"])

            st.subheader("Reduced DataFrame:")
            st.write(reduced_df)

            # Show scatterplot of reduced data
            st.subheader("Scatterplot of Reduced Data:")
            sns.scatterplot(data=reduced_df, x="Component 1", y="Component 2")
            st.pyplot()


            # Clustering
            st.subheader("Clustering:")

            clustering_algorithm = st.selectbox("Select Clustering Algorithm:",
                                                ("K-Means", "Hierarchical Clustering", "DBSCAN", "Gaussian Mixture Models"))

            if clustering_algorithm == "K-Means":
                num_clusters = st.number_input("Enter number of clusters:", min_value=2, max_value=10, value=2, step=1)
                kmeans = KMeans(n_clusters=num_clusters)
                clusters = kmeans.fit_predict(reduced_df)
            elif clustering_algorithm == "Hierarchical Clustering":
                num_clusters = st.number_input("Enter number of clusters:", min_value=2, max_value=10, value=2, step=1)
                hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
                clusters = hierarchical.fit_predict(reduced_df)
            elif clustering_algorithm == "DBSCAN":
                eps = st.number_input("Enter epsilon value for DBSCAN:", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
                min_samples = st.number_input("Enter min_samples value for DBSCAN:", min_value=1, max_value=10, value=5, step=1)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(reduced_df)
            else:  # Gaussian Mixture Models
                num_clusters = st.number_input("Enter number of clusters:", min_value=2, max_value=10, value=2, step=1)
                gmm = GaussianMixture(n_components=num_clusters)
                clusters = gmm.fit(reduced_df).predict(reduced_df)

            reduced_df['Cluster'] = clusters

            # Show scatterplot of reduced data with clusters
            st.subheader("Scatterplot of Reduced Data with Clusters:")
            sns.scatterplot(data=reduced_df, x="Component 1", y="Component 2", hue="Cluster", palette="viridis")
            st.pyplot()

    elif selected_file == "Data Analysis":
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Title for the Streamlit app
        st.title('Executing Data Analysis')

        # File uploader widget
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read the uploaded CSV file as a pandas DataFrame
            data = pd.read_csv(uploaded_file, encoding='latin1')

            # Display the DataFrame
            st.write(data)

            # Get column names
            columns = data.columns.tolist()

            # Select x-axis column
            x_axis_column = st.selectbox("Select x-axis column", columns)

            # Select y-axis column
            y_axis_column = st.selectbox("Select y-axis column", columns)

            # Select plot type
            plot_type = st.selectbox("Select plot type", [
                "Scatter Plot", "Line Plot", "Histogram", "Bar Plot",
                "Box Plot (Box-and-Whisker Plot)", "Violin Plot",
                "Pie Chart",  "Area Plot", "Density Plot",
                "Bubble Chart", "Parallel Coordinates Plot", "Treemap"
            ])

            if plot_type == "Scatter Plot":
                fig = px.scatter(data, x=x_axis_column, y=y_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Line Plot":
                fig = px.line(data, x=x_axis_column, y=y_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Histogram":
                fig = px.histogram(data, x=y_axis_column, nbins=20)
                st.plotly_chart(fig)

            elif plot_type == "Bar Plot":
                fig = px.bar(data, x=x_axis_column, y=y_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Box Plot (Box-and-Whisker Plot)":
                fig = px.box(data, x=x_axis_column, y=y_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Violin Plot":
                fig = px.violin(data, x=x_axis_column, y=y_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Pie Chart":
                fig = px.pie(data, values=y_axis_column, names=x_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Area Plot":
                fig = px.area(data, x=x_axis_column, y=y_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Density Plot":
                fig = ff.create_distplot([data[y_axis_column]], [y_axis_column], show_hist=False)
                st.plotly_chart(fig)

            elif plot_type == "Bubble Chart":
                fig = px.scatter(data, x=x_axis_column, y=y_axis_column, size=y_axis_column)
                st.plotly_chart(fig)


            elif plot_type == "Parallel Coordinates Plot":
                fig = px.parallel_coordinates(data, color=x_axis_column)
                st.plotly_chart(fig)

            elif plot_type == "Treemap":
                fig = px.treemap(data, path=[x_axis_column, y_axis_column], values=y_axis_column)
                st.plotly_chart(fig)

            else:
                st.write("Please select a plot type.")


if __name__ == "__main__":
    main()
