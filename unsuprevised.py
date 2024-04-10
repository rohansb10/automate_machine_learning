import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

def main():
    st.title("CSV File Analyzer")

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

if __name__ == "__main__":
    main()
