import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Disable the warning about calling st.pyplot() without any arguments
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title for the Streamlit app
st.title('CSV Data Analysis App')

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
