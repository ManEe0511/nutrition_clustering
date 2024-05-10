import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objs as go

# Dimensional Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Modeling
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn_extra.cluster import KMedoids

# Hypertuning Metrics and Parameters
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import ParameterGrid


def dimensionality_reduction(X, method, n_components):
    if method == 't-SNE':
        reducer = TSNE(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    elif method == 'PCA':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose 't-SNE' or 'PCA'.")

    x_data = [point[0] for point in X_reduced]
    y_data = [point[1] for point in X_reduced]

    return X_reduced, x_data, y_data, method

def scatter_plot_clustering(x_data,y_data,method):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, color='blue', s=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Plot of {method} Data in 2 Dimensions')
    plt.grid(True)
    plt.show()

def plot_silhouette_analysis(X_reduced, num_clusters_list):
      for n_clusters in num_clusters_list:
          # Create a subplot with 1 row and 2 columns
          fig, (ax1) = plt.subplots(1)
          fig.set_size_inches(18, 7)

          # The 1st subplot is the silhouette plot
          ax1.set_xlim([0, 1])

          # The (n_clusters+1)*10 is for inserting blank space between silhouette
          # plots of individual clusters, to demarcate them clearly.
          ax1.set_ylim([0, len(X_reduced) + (n_clusters + 1) * 10])

          # Initialize the clusterer with n_clusters value and a random generator
          # seed of 10 for reproducibility.
          clusterer = KMeans(n_clusters=n_clusters, random_state=10)
          cluster_labels = clusterer.fit_predict(X_reduced)

          # The silhouette_score gives the average value for all the samples.
          # This gives a perspective into the density and separation of the formed
          # clusters
          silhouette_avg = silhouette_score(X_reduced, cluster_labels)
          print(
              "For n_clusters =",
              n_clusters,
              "The average silhouette_score is :",
              silhouette_avg,
          )

          # Compute the silhouette scores for each sample
          sample_silhouette_values = silhouette_samples(X_reduced, cluster_labels)

          y_lower = 10
          for i in range(n_clusters):
              # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
              ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

              ith_cluster_silhouette_values.sort()

              size_cluster_i = ith_cluster_silhouette_values.shape[0]
              y_upper = y_lower + size_cluster_i

              color = cm.nipy_spectral(float(i) / n_clusters)
              ax1.fill_betweenx(
                  np.arange(y_lower, y_upper),
                  0,
                  ith_cluster_silhouette_values,
                  facecolor=color,
                  edgecolor=color,
                  alpha=0.7,
              )

              # Label the silhouette plots with their cluster numbers at the middle
              ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

              # Compute the new y_lower for next plot
              y_lower = y_upper + 10  # 10 for the 0 samples

          ax1.set_title("The silhouette plot for the various clusters.")
          ax1.set_xlabel("The silhouette coefficient values")
          ax1.set_ylabel("Cluster label")

          # The vertical line for average silhouette score of all the values
          ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

          ax1.set_yticks([])  # Clear the yaxis labels / ticks
          ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

          plt.suptitle(
              "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
              % n_clusters,
              fontsize=14,
              fontweight="bold",
          )

      plt.show()

def elbow_method_analysis(X_reduced):
    tsne_wcss = []
    for i in range(2,8):
        kmeans = KMeans(n_clusters=i,random_state=10)
        kmeans.fit(X_reduced)
        tsne_wcss.append(kmeans.inertia_)
        
    plt.plot(range(2,8),tsne_wcss)
    plt.title('The Elbow Method for t-SNE')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def plot_clusters(X_reduced, cluster_labels, model_name):
    data = []
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
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

def plot_cluster_distribution(value_counts, model_name):
    labels = value_counts.index
    values = value_counts.values

    trace = go.Pie(labels=labels, values=values, hole=0.3, textinfo='percent',
                   hoverinfo='label+percent')

    layout = go.Layout(title=f'{model_name} using t-SNE classes distribution')

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def get_food_groups(cluster_labels,food_group_name,food_groups):
    value_counts = cluster_labels.value_counts()
    food_groups[food_group_name] = value_counts.sort_values(ascending=False) # Added
    return food_groups, value_counts

def tuning_agglomerative(X_reduced, param_grid):

  best_score = -1
  best_params = {}
  silhouettescores = []

  # Perform grid search for hyperparameter tuning
  for params in ParameterGrid(param_grid):
      agglomerative = AgglomerativeClustering(**params)
      cluster_labels = agglomerative.fit_predict(X_reduced)
      silhouette = silhouette_score(X_reduced, cluster_labels)
      silhouettescores.append(silhouette)
      if silhouette > best_score:
          best_score = silhouette
          best_params = params

  print("Best silhouette score:", best_score)
  print("Best parameters:", best_params)

def tuning_kmedoids(X_reduced, param_grid):

  best_score = -1
  best_params = {}
  silhouettescores = []

  # Perform grid search for hyperparameter tuning
  for params in ParameterGrid(param_grid):
      medoid = KMedoids(**params)
      cluster_labels = medoid.fit_predict(X_reduced)
      silhouette = silhouette_score(X_reduced, cluster_labels)
      silhouettescores.append(silhouette)
      if silhouette > best_score:
          best_score = silhouette
          best_params = params

  print("Best silhouette score:", best_score)
  print("Best parameters:", best_params)


def perform_kmeans_clustering(X_reduced, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    model = model.fit(X_reduced)

    cluster_labels = model.labels_

    return cluster_labels, model

def perform_agg_clustering(X_reduced, n_clusters):
    model = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
    model = model.fit(X_reduced)
    cluster_labels = model.fit_predict(X_reduced)

    return cluster_labels,model

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def perform_gmm_clustering(X_reduced, n_components, covariance_type):
    model = GMM(n_components=n_components,covariance_type=covariance_type, random_state = 42).fit(X_reduced)
    cluster_labels = model.predict(X_reduced)

    return cluster_labels, model

def perform_kmd_clustering(X_reduced, n_clusters, init, method):
    model = KMedoids(n_clusters=n_clusters, random_state=0,init=init,method=method).fit(X_reduced)
    cluster_labels = model.predict(X_reduced)

    return cluster_labels, model