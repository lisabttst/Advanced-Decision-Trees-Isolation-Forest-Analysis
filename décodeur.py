import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Importation du classificateur d'arbre de décision
from sklearn.metrics import accuracy_score
import numpy as np

# Charger le fichier CSV avec le bon délimiteur
data = pd.read_csv('exemple_code.csv', delimiter=';')

# Séparer les caractéristiques et la variable cible
X = data.drop(columns=['y'])
y = data['y']

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le classificateur d'arbre de décision
clf = DecisionTreeClassifier(random_state=42)  # Utilisation de DecisionTreeClassifier

# Entraîner le classificateur
clf.fit(X_train, y_train)

# Prédire les étiquettes de l'ensemble de test
y_pred = clf.predict(X_test)

# Calculer l'exactitude (accuracy) du modèle
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Charger les nouvelles données
new_data = pd.read_csv('mot_code.csv', delimiter=';')

# Supprimer la colonne "Unnamed: 12"
new_data = new_data.drop(columns=['Unnamed: 12'])

# Utiliser le modèle pour prédire les étiquettes pour les nouvelles données
new_predictions = clf.predict(new_data)

# Réorganiser les prédictions en lignes de longueur 16
reshaped_predictions = new_predictions.reshape(-1, 16)

# Convertir le tableau numpy en DataFrame pour l'affichage
reshaped_df = pd.DataFrame(reshaped_predictions)

# Trouver la classe majoritaire pour chaque colonne
majority_class = reshaped_df.mode(axis=0).iloc[0]

# Regrouper les chiffres par groupes de deux et convertir en string
grouped_numbers = [''.join(map(str, majority_class[i:i+2])) for i in range(0, len(majority_class), 2)]

# Créer un mot en concaténant les groupes
word = ''.join(grouped_numbers)

# Générer le dictionnaire de traduction
keys = list(range(4, 100, 4))
letters = list('abcdefghijklmnopqrstuvwxyz')
translation_dict = {str(key).zfill(2): letter for key, letter in zip(keys, letters)}

# Traduire le mot
translated_word = ''.join([translation_dict[word[i:i+2]] for i in range(0, len(word), 2)])

print(translated_word)