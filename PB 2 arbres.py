import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
from kneed import KneeLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from kneed import KneeLocator


# Étape 1: Chargement des données
file_path = 'KPIs-for-telecommunication.csv'
# Le fichier CSV utilise des points-virgules comme délimiteurs
telecom_data = pd.read_csv(file_path, delimiter=';')

#### question a 

# Étape 2: Analyse statistique de base des variables
# Calcule les statistiques descriptives pour chaque colonne (KPI)
statistical_summary = telecom_data.describe()
# Calcule la variance pour chaque KPI
variance = telecom_data.var()
statistical_summary.loc['variance'] = variance
# Calcule l'interquartile range (IQR) pour chaque KPI
iqr = telecom_data.quantile(0.75) - telecom_data.quantile(0.25)
statistical_summary.loc['IQR'] = iqr
# Affichage des statistiques descriptives
print(statistical_summary)


#### question c 

# Étape 3: Évaluation des données manquantes
# Calcule le pourcentage de valeurs manquantes pour chaque KPI
missing_values_percentage = telecom_data.isnull().mean() * 100
print(missing_values_percentage)

# Étape 4: Traitement des valeurs manquantes
# Crée un objet imputer qui remplit les valeurs manquantes avec la moyenne
imputer = SimpleImputer(strategy='mean')
# Applique l'imputation sur les données pour gérer les valeurs manquantes
telecom_data_imputed = pd.DataFrame(imputer.fit_transform(telecom_data), columns=telecom_data.columns)
# Vérifie s'il reste des valeurs manquantes
missing_after_imputation = telecom_data_imputed.isnull().sum()
print(missing_after_imputation, telecom_data_imputed.head())


#### question b 

# Étape 5: Division des données en ensembles d'apprentissage et de test
# Divise les données en 70% pour l'apprentissage et 30% pour le test
train_data, test_data = train_test_split(telecom_data_imputed, test_size=0.3, random_state=42)
print(train_data.shape, test_data.shape)


#### question d

# Étape 6: Construction et entraînement du modèle de forêt d'isolement

# Puisque nous n'avons pas d'étiquettes, nous ne pouvons pas utiliser roc_auc_score pour la recherche de grille.
# Nous allons procéder en entraînant une Forêt d'Isolation avec des paramètres par défaut raisonnables.

# Nous définirons un petit ensemble de paramètres à tester, car une recherche exhaustive n'est pas réalisable sans fonction de score.
params_to_try = [
    {'n_estimators': 100, 'max_samples': 'auto', 'contamination': 0.01},
    {'n_estimators': 100, 'max_samples': 'auto', 'contamination': 0.05},
    {'n_estimators': 200, 'max_samples': 'auto', 'contamination': 0.01},
    {'n_estimators': 200, 'max_samples': 'auto', 'contamination': 0.05},
]

# Nous allons stocker les modèles et leurs scores d'anomalie respectifs pour comparaison.
models = []
anomaly_scores = []

for params in params_to_try:
    # Initialisation de la Forêt d'Isolation avec l'ensemble actuel de paramètres
    iso_forest = IsolationForest(random_state=42, n_estimators=params['n_estimators'],
                                 max_samples=params['max_samples'], contamination=params['contamination'])
    # Entraînement du modèle
    iso_forest.fit(train_data)
    # Calcul des scores d'anomalie
    scores = -iso_forest.decision_function(test_data)
    # Stockage du modèle et des scores
    models.append(iso_forest)
    anomaly_scores.append(scores)

# Maintenant, visualisons la distribution des scores d'anomalie pour chaque modèle
for i, scores in enumerate(anomaly_scores):
    plt.figure(figsize=(10, 4))
    plt.hist(scores, bins=50)
    plt.title(f"Modèle {i+1} - n_estimators: {params_to_try[i]['n_estimators']}, contamination: {params_to_try[i]['contamination']}")
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Fréquence")
    plt.show()

# Étape 7: Calcul des scores d'anomalies sur l'ensemble de test
iso_forest_default = IsolationForest()
iso_forest_default.fit(train_data)
anomaly_scores = iso_forest_default.decision_function(test_data)
anomaly_scores_transformed = -anomaly_scores

# Étape 8: Identification des anomalies
# Convertit les scores d'anomalie en DataFrame pour une manipulation plus facile
anomaly_scores_df = pd.DataFrame(anomaly_scores_transformed, columns=['Anomaly_Score'])
# Ajoute les scores au jeu de données de test
test_data_with_scores = test_data.copy()
test_data_with_scores['Anomaly_Score'] = anomaly_scores_df
# Trie les données de test avec les scores pour trouver les scores d'anomalie les plus élevés et les plus bas
highest_anomaly_scores = test_data_with_scores.sort_values('Anomaly_Score', ascending=False).head(5)
lowest_anomaly_scores = test_data_with_scores.sort_values('Anomaly_Score').head(5)

# Étape 9: Analyse des résultats
# Affiche les observations avec les scores d'anomalie les plus élevés et les plus bas
print(highest_anomaly_scores, lowest_anomaly_scores)

# Analyse de Résultats:
# Les observations avec les scores d'anomalie les plus élevés sont considérées comme les plus atypiques par le modèle. 
# Cela peut indiquer des comportements ou des performances qui s'écartent significativement de la norme.
# À l'inverse, les observations avec les scores d'anomalie les plus bas sont considérées comme normales.

# Define the grid of parameters for the contamination levels and threshold for anomaly scores
contaminations = np.linspace(0.01, 0.1, 10)  # 10 levels from 1% to 10%
thresholds = np.linspace(-0.2, 0.2, 20)  # 20 levels from -0.2 to 0.2

# We will create two graphs, one for each value of n_estimators
n_estimators_values = [100, 200]

# Initialize a dictionary to store variances for each combination of parameters and n_estimators
variances_per_estimator = {n: [] for n in n_estimators_values}

# Iterate over each n_estimators value
for n_estimators in n_estimators_values:
    # Train the Isolation Forest model with the current n_estimators
    iso_forest = IsolationForest(n_estimators=n_estimators, random_state=42)
    iso_forest.fit(train_data)
    # Calculate anomaly scores
    anomaly_scores = -iso_forest.decision_function(test_data)
    
    # Iterate over each contamination level and threshold to calculate variances
    for contamination in contaminations:
        temp_variances = []
        for threshold in thresholds:
            # Apply the threshold to filter anomaly scores
            filtered_data = test_data[anomaly_scores < threshold]
            # Calculate variance of the remaining data
            if not filtered_data.empty:
                variance = filtered_data.var()
                # Sum the variances across all features to get a single value
                total_variance = variance.sum()
            else:
                # If all data is filtered out, set variance to NaN
                total_variance = np.nan
            temp_variances.append(total_variance)
        variances_per_estimator[n_estimators].append(temp_variances)

# Now plot the 3D graphs for each n_estimators value
for n_estimators, variances in variances_per_estimator.items():
    X, Y = np.meshgrid(thresholds, contaminations)
    Z = np.array(variances)

    # Creating the figure and a 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the surface
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, edgecolor='none', rstride=1, cstride=1, alpha=0.8)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=np.nanmin(Z), cmap=plt.cm.viridis, alpha=0.3)

    # Adding labels and title
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Contamination')
    ax.set_zlabel('Total Variance')
    ax.set_title(f'3D plot of Total Variance vs. Threshold vs. Contamination (n_estimators={n_estimators})')

    # Adding a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()
    
# Re-run the 3D plot code with the corrected meshgrid shapes

# Initialize a dictionary to store variances for each combination of parameters and n_estimators
variances_per_estimator = {n: [] for n in n_estimators_values}
percentages_remaining = []

# Total number of data points in the test set
total_data_points = test_data.shape[0]

# Iterate over each n_estimators value
for n_estimators in n_estimators_values:
    # Initialize lists for variances and percentages of data remaining
    variances = []
    percentages = []
    
    # Train the Isolation Forest model with the current n_estimators
    iso_forest = IsolationForest(n_estimators=n_estimators, random_state=42)
    iso_forest.fit(train_data)
    # Calculate anomaly scores
    anomaly_scores = -iso_forest.decision_function(test_data)
    
    # Iterate over each contamination level and threshold to calculate variances and data remaining
    for contamination in contaminations:
        temp_variances = []
        temp_percentages = []
        for threshold in thresholds:
            # Apply the threshold to filter anomaly scores
            filtered_data = test_data[anomaly_scores < threshold]
            # Calculate variance of the remaining data
            if not filtered_data.empty:
                variance = filtered_data.var()
                # Sum the variances across all features to get a single value
                total_variance = variance.sum()
                # Calculate the percentage of data remaining
                data_remaining_percentage = (filtered_data.shape[0] / total_data_points) * 100
            else:
                # If all data is filtered out, set variance and percentage to NaN
                total_variance = np.nan
                data_remaining_percentage = 0
            temp_variances.append(total_variance)
            temp_percentages.append(data_remaining_percentage)
        variances.append(temp_variances)
        percentages.append(temp_percentages)
    
    variances_per_estimator[n_estimators] = variances
    percentages_remaining = percentages

# Now plot the 3D graphs for each n_estimators value
for n_estimators, variances in variances_per_estimator.items():
    X, Y = np.meshgrid(thresholds, contaminations)
    Z = np.array(variances)
    
    # Percentages of data remaining
    P = np.array(percentages_remaining)

    # Creating the figure and a 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the surface
    surf = ax.plot_surface(P, Y, Z, cmap=plt.cm.viridis, edgecolor='none', rstride=1, cstride=1, alpha=0.8)
    cset = ax.contourf(P, Y, Z, zdir='z', offset=np.nanmin(Z), cmap=plt.cm.viridis, alpha=0.3)

    # Adding labels and title
    ax.set_xlabel('Percentage of Data Remaining')
    ax.set_ylabel('Contamination')
    ax.set_zlabel('Total Variance')
    ax.set_title(f'3D plot of Total Variance vs. Data Remaining vs. Contamination (n_estimators={n_estimators})')

    # Adding a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()



    
# Given parameter combinations
params_to_try = [
    {'n_estimators': 100, 'contamination': 0.01},
    {'n_estimators': 100, 'contamination': 0.05},
    {'n_estimators': 200, 'contamination': 0.01},
    {'n_estimators': 200, 'contamination': 0.05},
]

# Initialize a dictionary to store variances for each combination of parameters and thresholds
variances_per_combination = {}
percentages_remaining = {}

# Total number of data points in the test set
total_data_points = test_data.shape[0]

# Calculate the variances and percentages for each parameter combination
for params in params_to_try:
    n_estimators = params['n_estimators']
    contamination = params['contamination']
    
    # Train the Isolation Forest model with the current parameters
    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso_forest.fit(train_data)
    
    # Calculate anomaly scores
    anomaly_scores = -iso_forest.decision_function(test_data)
    
    # Initialize lists to collect variances and the percentage of data remaining for each threshold
    variances = []
    percentages = []
    
    # Calculate variances and percentages for each threshold
    for threshold in thresholds:
        # Filter the data by the threshold
        filtered_data = test_data[anomaly_scores < threshold]
        
        # Calculate the total variance of the remaining data
        if not filtered_data.empty:
            variance = filtered_data.var().sum()
        else:
            variance = np.nan  # Use NaN for cases where all data is filtered out
        
        # Calculate the percentage of remaining data
        percentage_remaining = (filtered_data.shape[0] / total_data_points) * 100
        
        variances.append(variance)
        percentages.append(percentage_remaining)
    
    variances_per_combination[(n_estimators, contamination)] = variances
    percentages_remaining[(n_estimators, contamination)] = percentages

# Plot the variance of the remaining data as a function of the percentage of remaining data for each parameter combination
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()  # Flatten the array to make it easier to index

for i, ((n_estimators, contamination), variances) in enumerate(variances_per_combination.items()):
    percentages = percentages_remaining[(n_estimators, contamination)]

    # Plot the variance against the percentage of data remaining
    axs[i].plot(percentages, variances, label=f'n_estimators={n_estimators}, contamination={contamination}')
    axs[i].set_title(f'Variance vs. Percentage of Data Remaining (n_estimators={n_estimators}, contamination={contamination})')
    axs[i].set_xlabel('Percentage of Data Remaining')
    axs[i].set_ylabel('Variance of Remaining Data')
    axs[i].legend()
    
plt.tight_layout()

# Initialize a dictionary to store elbow points for each combination
elbow_points = {}

# Plot the variance of the remaining data as a function of the percentage of remaining data for each parameter combination
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()  # Flatten the array to make it easier to index

# Calculate and plot elbow points for each parameter combination
for i, ((n_estimators, contamination), variances) in enumerate(variances_per_combination.items()):
    percentages = percentages_remaining[(n_estimators, contamination)]
    
    # Plot the variance against the percentage of data remaining
    axs[i].plot(percentages, variances, label=f'n_estimators={n_estimators}, contamination={contamination}')
    axs[i].set_title(f'Variance vs. Percentage of Data Remaining (n_estimators={n_estimators}, contamination={contamination})')
    axs[i].set_xlabel('Percentage of Data Remaining')
    axs[i].set_ylabel('Variance of Remaining Data')
    axs[i].legend()

    # Use KneeLocator to find the elbow point
    kn = KneeLocator(percentages, variances, curve='convex', direction='decreasing')
    elbow_point = (kn.knee, variances[percentages.index(kn.knee)] if kn.knee else None)
    
    elbow_points[(n_estimators, contamination)] = elbow_point
    
    # Plot the elbow point on the corresponding subplot if it exists
    if elbow_point[0] is not None:
        axs[i].plot(elbow_point[0], elbow_point[1], 'ro', label='Elbow Point')  # 'ro' for marking the point in red
        axs[i].legend()

plt.tight_layout()
plt.show()


#### question e

# Affichage des cinq observations avec les scores d'anomalie les plus élevés
print("Observations avec les plus hauts scores d'anomalies:")
highest_anomaly_scores = test_data_with_scores.sort_values(by='Anomaly_Score', ascending=False).head(5)
print(highest_anomaly_scores)

# Affichage des cinq observations avec les scores d'anomalie les plus bas
print("\nObservations avec les scores d'anomalie les plus bas:")
lowest_anomaly_scores = test_data_with_scores.sort_values(by='Anomaly_Score', ascending=True).head(5)
print(lowest_anomaly_scores)


