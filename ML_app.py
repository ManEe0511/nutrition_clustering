#######################
# Import libraries
import streamlit as st
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import pickle
from sklearn.manifold import TSNE

#######################
# Dataset configuration
datasetName = ["bfill",
               "ffill",
               "LInterpolation",
               "removeRow",
               "mean",
               "median",
               "mode",
               "zero"]

datasetOption = ["Backward fill method",
                 "Forward fill method",
                 "Linear Interpolation method",
                 "Removing rows with missing value(s)",
                 "Replace with Mean",
                 "Replace with Median",
                 "Replace with Mode",
                 "Replace with Zero value"]

Feature = ['Data display','EDA','Clustering']

EDA_method = ['HeatMap','Violin Plot','Scatter Plot','Word Cloud']

Clustering_method = ['Agglomerative',
                     'Gaussian Mixture Models', 
                     'K-Medoids',
                     'K-Means']

#######################
# Clustering model configuration
modelName = ['Agg_model',
             'Gmm_model',
             'Kmd_model',
             'Kmeans_model',]

#######################
# Page configuration

st.set_page_config(
    page_title="Nutrition Based Dashboard",
    page_icon='🍳',
    layout="wide",
    initial_sidebar_state="expanded")

#######################
# general function

def progress_bar():
    bar = st.progress(0)

    for i in range(100):
    # Update the progress bar with each iteration.
        bar.progress(i + 1)
        time.sleep(0.01)
    bar.empty()

#######################
# EDA function

def show_heatmap(data):
    width = st.sidebar.slider("plot width", 1, 25, 20)
    height = st.sidebar.slider("plot height", 1, 15, 10)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(data.corr(), ax=ax, annot=True)
    st.write(fig)

def violin_plot(data, feature_name):
    
    width = st.sidebar.slider("plot width", 500, 1400, 1000, 50)
    height = st.sidebar.slider("plot height", 300, 800, 500, 50)
    values = sorted(data[feature_name])
    violin = px.violin(x=values, box=True, title=f"Violin Plot with box of {feature_name}",width=width,height=height,
                      color_discrete_sequence=['lightseagreen'])
    violin.update_layout(
        xaxis_title=f"{feature_name}",
        yaxis_title="Frequency Distribution"
    )
    st.write(violin)

def scatter_plot(x_var, y_var, data, trendline= False):
    width = st.sidebar.slider("plot width", 500, 1300, 1000, 50)
    height = st.sidebar.slider("plot height", 500, 900, 700, 50)
    scatter_plot = px.scatter(
        data_frame=data, x=x_var, y=y_var,
        trendline="ols" if trendline else None,
        marginal_x="histogram",
        marginal_y="histogram",
        color_discrete_sequence=['lightseagreen'],
        width=width,
        height=height,
        title=f"{x_var} vs {y_var}"
    )
    scatter_plot.update_layout(
        xaxis_title=x_var,
        yaxis_title=y_var,
        title_font_size=16,
        font=dict(family="Arial", size=12)
    )
    st.write(scatter_plot)

def generate_wordcloud(data):
    wordcloud = WordCloud(background_color="white",
                          width = 1000,
                          height = 500).generate(data)
    fig, ax = plt.subplots(figsize=(8,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title('Food Description',fontsize=15)
    st.write(fig)

#######################
# Clustering function

def plot_clusters(X_reduced, cluster_labels, model_name):
    data = []
    width = st.sidebar.slider("plot width", 500, 1400, 1000, 50)
    height = st.sidebar.slider("plot height", 500, 900, 700, 50)
    for cluster in np.unique(cluster_labels):
        cluster_data = go.Scatter(
            x=X_reduced[cluster_labels == cluster, 0],
            y=X_reduced[cluster_labels == cluster, 1],
            mode='markers',
            name=f'Cluster {cluster}'
        )
        data.append(cluster_data)

    layout = go.Layout(
        title=f't-SNE with {model_name}',
        xaxis=dict(title='Dimension 1'),
        yaxis=dict(title='Dimension 2'),
        width=width,
        height=height,
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    st.write(fig)


#######################
# Sidebar
with st.sidebar:
    st.title("Nutrition Based Dashboard")

    selectedDataset = st.selectbox('Filling approach for missing value in dataset',datasetOption)
    index_selectItem = datasetOption.index(selectedDataset)

    macroNutrient_dataset =pd.read_csv(r"Dataset/Dataset_for_EDA/macroNutrient_{datasetName[index_selectItem]}.csv", encoding= 'unicode_escape')
    reducedDataset = pd.read_csv(r"Dataset\Dataset_for_reduced_data\{datasetName[index_selectItem]}_reduced.csv")
    reducedDataset = reducedDataset.iloc[:,[1,2]].to_numpy()

    selectedFeatures = st.selectbox('Feature to perform',
                                    Feature,
                                    index=None,
                                    placeholder='Choose a feature')
    
#######################
# Dashboard Main Panel
if selectedFeatures == Feature[0]:
    dropColumns = st.sidebar.multiselect('Drop Feature(s)',
                                         list(macroNutrient_dataset.columns),
                                         placeholder='Choose columns to drop')
    desired_column = [x for x in macroNutrient_dataset.columns if x not in dropColumns]
    st.dataframe(macroNutrient_dataset[desired_column],height=600)
    st.write(reducedDataset)

elif selectedFeatures == Feature[1]:
    edaMethod = st.sidebar.selectbox('Eda method to perform',
                                     EDA_method,
                                     index=None,
                                     placeholder='Choose an approach')
    desired_column = [x for x in macroNutrient_dataset.columns if x not in ["No.","Description","Category"]]
    if edaMethod == EDA_method[0]:
        progress_bar()
        show_heatmap(macroNutrient_dataset[desired_column])
    if edaMethod == EDA_method[1]:
        select_column = st.sidebar.selectbox('Feature name',
                                             desired_column,
                                             index=None,
                                             placeholder="Choose a column name")
        if select_column !=None:
            progress_bar()
            violin_plot(macroNutrient_dataset[desired_column],select_column)
    if edaMethod == EDA_method[2]:
        select_column = st.sidebar.selectbox('Feature name',
                                             desired_column,
                                             index=None,
                                             placeholder="Choose a column name")
        if select_column !=None:
            progress_bar()
            scatter_plot('Energy (Kcal)',
                         select_column,
                         macroNutrient_dataset[desired_column])
    if edaMethod == EDA_method[3]:
        progress_bar()
        text = " ".join(text for text in macroNutrient_dataset['Description'])

        generate_wordcloud(text)

elif selectedFeatures == 'Clustering':
    clusteringMethod = st.sidebar.selectbox("Apply t-SNE with: ",
                                            Clustering_method,
                                            index=None,
                                            placeholder='Choose a model')
    if clusteringMethod == Clustering_method[0]:
        with open(r'Model_fitted\{modelName[0]}_{datasetName[index_selectItem]}_pkl','rb') as file:
            loaded_model = pickle.load(file)

        plot_clusters(reducedDataset,loaded_model.fit_predict(reducedDataset),Clustering_method[0])
    
    elif clusteringMethod == Clustering_method[1]:
        with open(r'Model_fitted\{modelName[1]}_{datasetName[index_selectItem]}_pkl','rb') as file:
            loaded_model = pickle.load(file)

        plot_clusters(reducedDataset,loaded_model.fit_predict(reducedDataset),Clustering_method[1])

    elif clusteringMethod == Clustering_method[2]:
        with open(r'Model_fitted\{modelName[2]}_{datasetName[index_selectItem]}_pkl','rb') as file:
            loaded_model = pickle.load(file)

        plot_clusters(reducedDataset,loaded_model.fit_predict(reducedDataset),Clustering_method[2])

    elif clusteringMethod == Clustering_method[3]:
        with open(r'Model_fitted\{modelName[3]}_{datasetName[index_selectItem]}_pkl','rb') as file:
            loaded_model = pickle.load(file)

        plot_clusters(reducedDataset,loaded_model.fit_predict(reducedDataset),Clustering_method[3])
    
    # else:
    
else:
    st.markdown(r'$\textsf{\Huge Welcome to Nutrition Based DashBoard✨}$')
    st.markdown(r'$\textsf{\huge Have a start with the interactive sidebar📣}$')

