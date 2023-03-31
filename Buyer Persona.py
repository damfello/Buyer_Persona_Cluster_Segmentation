#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Libraries 

import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42) 




# In[2]:


data = pd.read_csv(r"C:\Users\Damfello\Desktop\Sexy Daticos\buyerpersona_supermarket.csv") ## , sep="\t")
print("Number of datapoints:", len(data))

# Show a Resume
data.head()


# In[3]:


# Limpieza
#In order to, get a full grasp of what steps should I be taking to clean the dataset. Let us have a look at the information in data.

data.info()
data.shape

## conlusion: Missing values in income. enrolmentDate no tiene formato de Fecha


# In[4]:


data.describe()


# In[5]:


data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))


# In[6]:


## Formato Fecha para enrolmentDate

data["enrolmentDate"] = pd.to_datetime(data["enrolmentDate"])
dates = [] 
for i in data["enrolmentDate"]:
    i = i.date()
    dates.append(i)
    
# Fechas de las últimas y primeras fechas de inscripción de los clientes 
print ("Fecha más reciente de inscripción de un cliente:", max(dates))
print("Fecha más antigua de inscripción de un cliente:", min(dates)) 



# In[7]:


## Crear "Customer_For"
### Número de días que pasaron para que empezaran a comprar con respecto a la fecha más reciente de inscripción (Max enrolment-Date)

    
    
days = []
d1 = max(dates) #Consideramos con el cliente más reciente registrado
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")


data.describe()







# In[8]:


## Explorar los valores únicos en las categorías Maritalstatus & EducationLevel 

print("Total categories in the feature Marital_Status:\n", data["maritalstatus"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["educationLevel"].value_counts())



# In[9]:


#Crear nuevas categorías con base en las existentes


#Edad actual
data["Age"] = 2023-data["yearBirth"]

#Gasto total en varios tipos de productos

data["Expenditure"] = data["wines"]+ data["fruits"]+ data["meat"]+ data["fish"]+ data["candy"]+ data["gold"]

#Derivar el grupo de vivienda de acuerdo a estado civil/Marital_status
data["Group_Living"]=data["maritalstatus"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Hijos Totales en Casa
data["Sons"]=data["kids"]+data["teenagers"]

# Miembros totales en casa
data["FamilySize"] = data["Group_Living"].replace({"Alone": 1, "Partner":2})+ data["Sons"]


# Padres
data["SonPadres"] = np.where(data.Sons> 0, 1, 0)

# Agrupar en tres nuveles de Educación
data["Education"]=data["educationLevel"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})


#Dropping some of the redundant features
to_drop = ["maritalstatus", "enrolmentDate", "Z_CostContact", "Z_Revenue", "yearBirth", "idClient", "educationLevel"]
data = data.drop(to_drop, axis=1)


# In[10]:


data.describe()


# In[11]:


## Visualizar algunas Variables

#Edtar Preferencias
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#e6e5e3"})
pallet = ["#2f57f7", "#92c1f7", "#c1d1f7", "#c8cede", "#ab7878", "#f36060"]
cmap = colors.ListedColormap(["#2f57f7", "#92c1f7", "#c1d1f7", "#c8cede", "#ab7878", "#f36060"])

## sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
## Antes pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
## cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

#Visualizando 
To_Plot = [ "income", "daysLastBuy", "Customer_For", "Age", "Expenditure", "SonPadres"]
   
print("Ser padres vs algunas variables: Ingreso, última compra, Edad, Gasto, Padres")
plt.figure()
sns.pairplot(data[To_Plot], hue= "SonPadres",palette= (["#2f57f7","#ff5252"]))
#Taking hue 
plt.show()


# In[12]:


# Eliminar outliers
data = data[(data["Age"]<90)]
data = data[(data["income"]<600000)]
print("El Total de Filas despúes de remover Ourliers en Edad e Ingresos:", len(data))


# In[13]:


#Matrix de Correlación 

corrmat= data.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)


# In[14]:


#Obtener la Lista de las Variables Categóricas 
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Variables de tipo Categórico en el dataset:", object_cols)


# In[15]:


# Formatear las variables a tipo númerico 

LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("Todas las variables son ahora númericas")


# In[16]:


# Crear una copia del dataset 

ds = data.copy()
# Crear un subconjunto de datos eliminando las variables con ofertas y promociones recibidas 

cols_del = ['camp1_Received', 'camp2_Received', 'camp3_Received', 'camp4_Received', 'camp5_Received', 'complain', 'response']
ds = ds.drop(cols_del, axis=1)

#Escalando los datos

scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("Todas las variables han sido escaladas")

## Se escalan los datos para qeu tengan una media igual a 0 y una desviación estándar igual a uno.
## Es útil para trabajar con modelos que asumen que las variables tienen una distribución normal


# In[17]:


# Datos Escalados que se utilizarán para reducir la dimensionalidad
print("Dataset para usar en el modelado:")
scaled_ds.head()


# In[18]:


# Iniciar Análisis de Componentes (PCA) para reducir las variables



pca = PCA(n_components=8)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["variable1","variable2", "variable3", "variable4",
                                                         "variable5", "variable6","variable7", "variable8"]))


## PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["variable1","variable2", "variable3", "variable4",
                                                        #   "variable5", "variable6","variable7", "variable8", 
                                                        # "variable9", "variable10"]))
PCA_ds.describe().T

## pca.explained_variance_ratio_
## pca.explained_variance_ratio_.sum()









# In[19]:


# Quick examination of elbow method to find numbers of clusters to make

print('El Método del Codo para determinar el número de Clusters que deberíamos conformar:')
Elbow_M = KElbowVisualizer(KMeans(), k=(8))
Elbow_M.fit(PCA_ds)
Elbow_M.show()


# In[20]:


#Iniciando la agrupación o Modelo Clustering


AC = AgglomerativeClustering(n_clusters=4)

# Ajustar el Model y Predecir los Clusters 
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
# Agregando los Clusters al dataframe original 

data["Clusters"]= yhat_AC


# In[21]:


#Visualizando los Clusters o Agrupaciones

#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["variable1"]
y =PCA_ds["variable2"]
z =PCA_ds["variable3"]



fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap )
ax.set_title("Las agrupaciones | Clusters")
plt.show()


# In[22]:


#Visualizar el Conteo por Agrupaciones

pal = ["#2f57f7","#edd768", "#5ff55f","#f74d4d"]

pl = sns.countplot(x=data["Clusters"], palette= pal)
pl.set_title("Distribución de los Clusters")
plt.show()


# In[23]:


### Income vs Expenditure por Clusters

pl = sns.scatterplot(data=data, x=data["income"], y=data["Expenditure"], hue=data["Clusters"], palette = pal)

pl.set_title("Ingreso vs Consumo agrupado en Clusters")
plt.legend()
plt.show()




# In[24]:


## distribución del consumo en el supermercado por Grupos/Clusters

plt.figure()
pl = sns.swarmplot(x=data["Clusters"], y=data["Expenditure"], color = "#e6f7f0", alpha = 0.5)
pl = sns.boxenplot(x=data["Clusters"], y=data["Expenditure"], palette = pal)
plt.show()



# In[25]:


## Visualizar la distribución del número de Deals 

plt.figure()
pl=sns.boxenplot(y=data["dealsPurchases"], x=data["Clusters"], palette = pal)
pl.set_title("Número de Prmociones Adquiridas")
plt.show()


# In[29]:


### Perfilando los grupos de Clusters


data["Campaigns"] = data["camp1_Received"] + data["camp2_Received"] + data["camp3_Received"] + data["camp4_Received"] + data["camp5_Received"]


Personal = ["SonPadres", "Sons", "kids","teenagers", "FamilySize", "Group_Living", "Campaigns", "dealsPurchases", "Age", "webPurchases", "catalogPurchases"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data[i], y=data["Expenditure"], hue=data["Clusters"], kind="kde", palette=pal)
    plt.show()


# In[ ]:


### Conclusiones finales


- Grupo Rojo (Cluster 3):
    - Son mayores de 40 años con Ingresos altos
    - Son Padres de Familia, la mayoría de adolescentes
    - Son un grupo familiar de 3 miembros
    - Reaccionan muy bien a las prmociones, y en menor medida a las campañas
    
- Grupo Verde (Cluster 2): 
    - Son un grupo de diversas edades, a partir de los 30 años
    - Son el grupo con ingresos más altos y con el mayor consumo en el supermercado
    - No son Padres
    - Es un grupo de Parejas o Solteros
    - Tienen una alta sensibilidad a las campañas, pero muy baja a a las promociones
- Grupo Amarillo (Cluster 1):
    - Tienen ingresos promedios y bajo consumo en supermercado
    - Son Padres de familia, con al menos un niño y un adolescente en casa
    - Tienen alta sensibilidad a las promociones, y moderada a las campañas
    - Son adultos a partir de los 40 años
    - Es un grupo familiar mayoritaríamente 4 miembros
- Grupo Azul (Cluster 0):
    - Son el grupo más jóven, a partir de los 20 años
    - Tienen ingresos bajos, y el menor consumo en el supermercado
    - Tienen una aceptación moderada a baja de campañas y promociones
    - Un porcentaje considerable son padres de por lo menos 1 niño


