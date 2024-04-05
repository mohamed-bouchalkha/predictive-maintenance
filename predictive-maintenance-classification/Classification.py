#!/usr/bin/env python
# coding: utf-8

# <h2 style = "text-align: center; font-size: 40px;">Predictive_maintenance<h2>

# <h2>Introduction</h2>

# <p>La maintenance prédictive est une stratégie proactive qui utilise des données en temps réel et des analyses avancées pour prédire les défaillances d'équipements avant qu'elles ne surviennent. En surveillant les paramètres de fonctionnement, elle permet d'anticiper les pannes, réduisant ainsi les temps d'arrêt non planifiés et les coûts de maintenance. Cette approche optimise la disponibilité des équipements, prolonge leur durée de vie utile et améliore la sécurité des opérations. En collectant des données à partir de capteurs et en utilisant des modèles prédictifs, elle permet une planification efficace des interventions de maintenance préventive. Globalement, la maintenance prédictive offre une meilleure gestion des ressources, des économies de coûts et une augmentation de la productivité des entreprises.</p>

# <h2>Objectifs</h2>

# <p>L'objectif est de traiter un ensemble de données de maintenance prédictive afin de le rendre utilisable pour développer différents modèles de machine learning de classification des produits en défaillance ou non. Cela implique l'utilisation de multiples caractéristiques répertoriées dans l'ensemble de données pour prédire avec précision si un produit subira une défaillance ou non</p>

# <h2>Index de la page</h2>
# <ol>
#     <li>Collecte de données</li>
#     <li>Traitement des données</li>
#     <li>Visualisation des données</li>
#     <li>Initialisation du modèle</li>
#     <li>Évaluation et optimisation du module de prédiction</li>
# </ol>

# <h1>1.Collecte de données</h1>

# <h3>Source</h3>

# Les données ont été collectées à partir du célèbre site web <b>Kaggle</b> qui contient de nombreux ensembles de données et informations utiles.<br>
# Lien : <a href="https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification">predictive-maintenance-classification</a>

# <h2>Importer les bibliothèque & dataset</h2>

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset =  r"C:\Users\hp\Desktop\Project Data_Mining\predictive-maintenance-classification\predictive_maintenance.csv"
data = pd.read_csv(dataset)
data.head(10)


# In[3]:


data.shape


# In[5]:


data.info()


# In[6]:


data['Type'].value_counts()


# In[7]:


data


# In[8]:


data.describe()


# <h3 style="text-align: center; color: #333;">Description de l'ensemble de données sur la maintenance prédictive</h3>
# </ol>
# <ul>
#     <li><span style="font-weight: bold; color: #555;">Product ID</span>: C'est un identifiant unique attribué à chaque produit dans l ensemble de données.</li>
#     <li><span style="font-weight: bold; color: #555;">Type</span>: Indique la catégorie à laquelle chaque produit appartient, classée en faible (L), moyen (M) ou élevé (H).</li>
#     <li><span style="font-weight: bold; color: #555;">Air temperature [K]</span>: Cela représente la température de l'air en Kelvin.</li>
#     <li><span style="font-weight: bold; color: #555;">Process temperature [K]</span>: Il s agit de la température du processus en Kelvin.</li>
#     <li><span style="font-weight: bold; color: #555;">Rotational speed [rpm]</span>: C'est la vitesse de rotation de la machine, mesurée en tours par minute (RPM).</li>
#     <li><span style="font-weight: bold; color: #555;">Torque [Nm]</span>: Le couple représente la force de rotation appliquée par la machine, mesurée en Newton-mètres (Nm).</li>
#     <li><span style="font-weight: bold; color: #555;">Tool wear [min]</span>: Il s'agit du temps d'utilisation de l'outil, mesuré en minutes.</li>
#     <li><span style="font-weight: bold; color: #555;">Target</span>: Indique si une défaillance s est produite ou non.</li>
#     <li><span style="font-weight: bold; color: #555;">Failure Type</span>: Si une défaillance s'est produite, cette colonne indique le type de défaillance.</li>
# </ul>

# <h1>2.Traitement des données</h1>

# <h2>Vérifier les valeur manquants</h2>

# In[9]:


data.isnull().sum()


# In[25]:


data.isna().sum()


# <h2>Vérifier les doublons</h2>

# In[11]:


data.duplicated().sum().any()


# In[75]:


data['Failure Type'].value_counts()


# <h2>Encoder les variables catégorique</h2>

# In[13]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
data['Product ID'] = ordinal_encoder.fit_transform(data[['Product ID']])
data['Type'] = ordinal_encoder.fit_transform(data[['Type']])
data['Failure Type'] = ordinal_encoder.fit_transform(data[['Failure Type']])
data


# In[69]:


# plt.figure(figsize=(15,10))
# sns.heatmap(data.corr(),annot = True,cmap = "viridis")
# plt.draw()


# In[15]:


data_selected=data.drop(['Product ID',"UDI","Failure Type","Type"], axis=1)


# In[74]:


data_selected.head(10)


# <h1>3.Visualisation des données</h1>

# In[29]:


# plt.figure(figsize=(15,10))
# for i,col in enumerate(data_selected.columns,1):
#     plt.subplot(3,3,i)
#     sns.histplot(data_selected[col],kde=True)


# In[19]:


data_selected.hist(figsize=(16, 16))


# In[20]:


data_selected.describe()


# In[30]:


# a = sns.relplot(x="Air temperature [K]", y="Process temperature [K]", hue="Target",size="Target", sizes=(120, 10),data=data_selected)


# In[24]:


# _ = sns.relplot(x="Torque [Nm]", y="Rotational speed [rpm]", hue="Target",size="Target", sizes=(100, 5),data=data_selected)


# In[35]:


data_selected


# In[50]:


# plt.figure(figsize = (12,6) )

# target_genre = data_selected['Tool wear [min]'].value_counts()
# barplot = sns.barplot(x = target_genre, y = target_genre.index)

# plt.title("distribution du temps d'utilisation de l'outil ")
# plt.show()


# In[52]:


# plt.figure(figsize = (12,6) )

# target_genre = data_selected['Torque [Nm]'].value_counts()
# barplot = sns.barplot(x = target_genre, y = target_genre.index)

# plt.title("distribution de la force de rotation appliquée par la machine ")
# plt.show()


# In[61]:


speed_target = data_selected[['Air temperature [K]', 'Target']]
avg_speed_by_target = speed_target.groupby('Target')['Air temperature [K]'].mean()
avg_speed_by_target = pd.DataFrame(avg_speed_by_target)
print(avg_speed_by_target)


# In[67]:


# correlation_matrix = data_selected.corr()
# plt.figure(figsize=(12, 5))
# heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
# heatmap.set_title('Correlation Heatmap')
# plt.show()


# In[73]:


# relationship_matrix = sns.pairplot(data_selected, hue='Target', corner=False, kind='scatter')


# <h1>3.Initialisation du modèle</h1>

# In[87]:


#séparer le variable cible et les variables d'entraînement
X = data_selected.drop(columns=['Target'])  
y = data_selected["Target"]  
X


# **- Séparer les variables d'entraînement (80%) des variables de test (20%)."**

# In[88]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40, test_size=0.33, stratify=y)


# <h1>**K-NN**<h1>

# ***Choisir le meilleur nombre de k à l'aide de la méthode du code.***

# In[86]:


# from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pyplot as plt
# vect_err = []
# k_values = range(1, 30)
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     error = 1 - knn.score(X_test, y_test)
#     vect_err.append(error)

# plt.plot(k_values, vect_err, marker='o', linestyle='-')
# plt.title("Courbe de l'erreur en fonction de k")
# plt.xlabel('Nombre de voisins (k)')
# plt.ylabel('Erreur de classification')
# plt.xticks(k_values)
# plt.show()


# In[90]:


# minimum=np.argmin(vect_err)
# print(" le meilleur nombre de k est :",minimum+1)


# In[119]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# matrice_confusion = confusion_matrix(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print("Matrice de confusion :\n", matrice_confusion)
# print("****************************************")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)


# In[122]:


# import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
# plot_confusion_matrix(knn, X_test, y_test)
# plt.title('Confusion Matrix')
# plt.show()


# In[124]:


# metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
# sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
# plt.title('Performance Metrics')
# plt.show()


# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <style>
#         table {
#             border-collapse: collapse;
#             width: 70%;
#             margin: auto;
#             font-size: 16px;
#         }
#         th, td {
#             border: 1px solid black;
#             padding: 12px;
#             text-align: left;
#         }
#         th {
#             background-color: #f2f2f2;
#         }
#     </style>
# </head>
# <body>
#     <h2>Résultats du modèle KNN</h2>
#     <table>
#         <tr>
#             <th>Mesure</th>
#             <th>Valeur</th>
#             <th>Interprétation</th>
#         </tr>
#         <tr>
#             <td>Accuracy</td>
#             <td>0.973</td>
#             <td>L'exactitude mesure la proportion d'exemples correctement classés par le modèle parmi tous les exemples. Dans ce cas, le modèle a correctement classé environ 97.3% des exemples.</td>
#         </tr>
#         <tr>
#             <td>Precision</td>
#             <td>0.897</td>
#             <td>La précision mesure la proportion d'exemples positifs correctement identifiés parmi tous les exemples classés comme positifs par le modèle. Ici, environ 89.7% des exemples classés comme positifs par le modèle étaient réellement positifs.</td>
#         </tr>
#         <tr>
#             <td>Recall</td>
#             <td>0.232</td>
#             <td>Le rappel mesure la proportion d'exemples positifs correctement identifiés parmi tous les exemples réellement positifs. Environ 23.2% des exemples réellement positifs ont été correctement identifiés par le modèle.</td>
#         </tr>
#         <tr>
#             <td>F1 Score</td>
#             <td>0.369</td>
#             <td>Le score F1 est une mesure combinée de la précision et du rappel. Il est utile lorsque les classes sont déséquilibrées. Le score F1 est de 0.369 dans ce cas, indiquant un équilibre entre la précision et le rappel.</td>
#         </tr>
#     </table>
#     <h2>Matrice de confusion</h2>
#     <p>La matrice de confusion montre comment le modèle a classé les exemples positifs et négatifs :</p>
#     <table>
#         <tr>
#             <th></th>
#             <th>Classe 0 (Négatif)</th>
#             <th>Classe 1 (Positif)</th>
#         </tr>
#         <tr>
#             <td>Classe 0 (Négatif)</td>
#             <td>3185 (Vrais Négatifs)</td>
#             <td>3 (Faux Positifs)</td>
#         </tr>
#         <tr>
#             <td>Classe 1 (Positif)</td>
#             <td>86 (Faux Négatifs)</td>
#             <td>26 (Vrais Positifs)</td>
#         </tr>
#     </table>
#     <p> le modèle a correctement classé la plupart des exemples (3185 vrais négatifs et 26 vrais positifs), mais a également fait quelques erreurs en classant à tort 3 exemples négatifs comme positifs et 86 exemples positifs comme négatifs.</p>
# </body>
# </html>
# 

# <h1>**Naive Bayes**<h1>

# In[117]:


# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred_nb = gnb.predict(X_test)
# matrice_confusion = confusion_matrix(y_test, y_pred_nb)
# accuracy = accuracy_score(y_test, y_pred_nb )
# precision = precision_score(y_test, y_pred_nb )
# recall = recall_score(y_test, y_pred_nb )
# f1 = f1_score(y_test, y_pred_nb )
# print("Matrice de confusion :\n", matrice_confusion)
# print("****************************************")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)


# <h1>**Arbre de décision**<h1>

# In[118]:


# from sklearn.tree import DecisionTreeClassifier
# tree_classifier = DecisionTreeClassifier(random_state=42)
# tree_classifier.fit(X_train, y_train)
# y_pred_tree = tree_classifier.predict(X_test)
# matrice_confusion = confusion_matrix(y_test, y_pred_tree)
# accuracy = accuracy_score(y_test, y_pred_tree )
# precision = precision_score(y_test, y_pred_tree )
# recall = recall_score(y_test, y_pred_tree )
# f1 = f1_score(y_test, y_pred_tree )
# print("Matrice de confusion :\n", matrice_confusion)
# print("****************************************")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)


# In[ ]:





from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Charger les données et entraîner le modèle KNN
# data_selected = pd.read_csv('chemin_vers_votre_fichier.csv')  # Assurez-vous de spécifier le chemin correct
X = data_selected.drop(columns=['Target'])
y = data_selected["Target"]
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X, y)

# Créer une instance de l'application Flask
app = Flask(__name__)

# Définir le contenu HTML de la page d'accueil avec le formulaire
content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Machine Fault Prediction</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 2rem;
    }

    h2 {
      text-align: center;
      margin-bottom: 1rem;
    }

    .prediction-form {
      display: flex;
      flex-direction: column;
      width: 400px;
      margin: 0 auto;
      padding: 1rem;
      border: 1px solid #ccc;
      border-radius: 5px;
    }

    .form-group {
      margin-bottom: 0.5rem;
    }

    label {
      display: block;
      margin-bottom: 0.2rem;
      font-weight: bold;
    }

    input[type="number"] {
      width: 100%;
      padding: 0.5rem;
      border: 1px solid #ccc;
      border-radius: 3px;
    }

    button[type="submit"] {
      background-color: #4CAF50; /* Green */
      color: white;
      padding: 0.7rem 1rem;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      margin-top: 1rem;
    }

    button[type="submit"]:hover {
      background-color: #45A049;
    }
  </style>
</head>
<body>
  <h2>Machine Fault Prediction</h2>
  <form action="/" method="post" class="prediction-form">
    <div class="form-group">
      <label for="airTemp">Air Temperature (K):</label>
      <input type="number" id="airTemp" name="airTemp" required>
    </div>
    <div class="form-group">
      <label for="processTemp">Process Temperature (K):</label>
      <input type="number" id="processTemp" name="processTemp" required>
    </div>
    <div class="form-group">
      <label for="rotationalSpeed">Rotational Speed (rpm):</label>
      <input type="number" id="rotationalSpeed" name="rotationalSpeed" required>
    </div>
    <div class="form-group">
      <label for="torque">Torque (Nm):</label>
      <input type="number" id="torque" name="torque" required>
    </div>
    <div class="form-group">
      <label for="toolWear">Tool Wear (min):</label>
      <input type="number" id="toolWear" name="toolWear" required>
    </div>
    <button type="submit">Predict</button>
  </form>
</body>
</html>
"""

# Définir la route pour la page d'accueil avec le formulaire HTML
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les données saisies dans le formulaire
        air_temp = float(request.form['airTemp'])
        process_temp = float(request.form['processTemp'])
        rotational_speed = float(request.form['rotationalSpeed'])
        torque = float(request.form['torque'])
        tool_wear = float(request.form['toolWear'])

        # Prédire avec votre modèle KNN
        prediction = knn.predict([[air_temp, process_temp, rotational_speed, torque, tool_wear]])

        # Afficher la prédiction sur une nouvelle page
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Résultat de la prédiction</title>
</head>
<body>
    <h2>Résultat de la prédiction :</h2>
    <p>La machine est en <strong>{}</strong>.</p>
</body>
</html>
""".format("n'est pas en défiance" if prediction[0] == 0 else "en défiance ")

    # Si la méthode est GET, afficher simplement le formulaire
    return content

# Exécuter l'application Flask
if __name__ == '__main__':
    app.run(debug=True)

