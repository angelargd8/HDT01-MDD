#librerias 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix


#cargar los datos
df = pd.read_csv("movies.csv", encoding="latin1")


#exploracion de los datos
#mostrar las columanas
print("\n--Columnas:---")
# print(list(df.columns))

#resumen del set de datos
print("\n--Resumen del set de datos:--")
# print(df.describe())


#obtener el tipo de datos
print("\n----Tipo de datos:---")
# print(df.dtypes)
tipos_de_datos = df.dtypes.value_counts()
# print('resumen:')
# print(tipos_de_datos)


# revision de valores duplicados
# print(df.duplicated().sum())

# eliminar valores nulos o llenarlos con la media
df = df.dropna()

#llenar los valores nulos de video
df['video'].fillna('FALSE', inplace=True)

#calcular la media de solo las columnas numericas
mean_values = df.select_dtypes(include=[np.number]).mean()

#reemplazar valores nulos con la media
df.fillna(mean_values, inplace=True)


#convertir los tipos de datos
#convertir la columna video a numerica
df['video'] = df['video'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

#convertir la columna releaseDate a datetime
df['releaseDate'] = pd.to_datetime(df['releaseDate'])

#convertir las columnas popularity, revenue y voteAvg a enteros
df['popularity'] = df['popularity'].astype(int)
df['revenue'] = df['revenue'].astype(int)
df['voteAvg'] = df['voteAvg'].astype(int)

#asegurarse que todas las variables sean de tipo int
df['castWomenAmount'] = df['castWomenAmount'].astype(int)
df['castMenAmount'] = df['castMenAmount'].astype(int)


#-*---------------------------------------------------------------------
#normalizacion de datos
#no se deben de normalizar todos los dato, entonces los de 
#nobmres de las columans que no se van a normalizar, porque no es necesario
# exclude_columns = ['castWomenAmount', 'castMenAmount', 'id', 'video','budget','revenue', 'popularity','voteCount','voteAvg','productionCoAmount','genresAmount','productionCountriesAmount']
#columans que si se van a normalizar
numeric_columns = ['actorsAmount', 'runtime']
print(numeric_columns)
#normalizar los datos
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print(df.head())


#------------------------------------------------------------------------------------------------------

# #--------------------------------------------
# # b. ¿Cuáles son las 10 películas que más ingresos tuvieron?
# ingresos = df.sort_values(by='revenue', ascending=False)
# print('\nTop 10 películas con mas ingresos: ')
# print(ingresos[['title', 'revenue']].head(10))


# #--------------------------------------------
# # d. ¿Cuál es la peor película de acuerdo a los votos de todos los usuarios?
# peor = df.sort_values(by='voteAvg', ascending=True)
# print('\nPeor pelicula segun los votos de todos los usuarios: ')
# print(peor[['title', 'voteAvg']].head(1))

# #--------------------------------
# '''
# f.
# ¿Cuál es el género principal de las 20 películas más recientes? 
# ¿Cuál es el género principal que predomina en el conjunto de datos?
#  Represéntelo usando un gráfico. 
#  ¿A qué género principal pertenecen las películas más largas?
# '''
# #¿Cuál es el género principal de las 20 películas más recientes? 
# peliculas_recientes = df.sort_values(by='releaseDate', ascending=False)
# top_20 = peliculas_recientes.head(20)
# generos_top20= top_20['genres'].str.split('|').str[0]
# generos_top20= generos_top20.value_counts()
# print(generos_top20)
# nombres_generos= generos_top20.index
# print(nombres_generos)

# #grafica
# plt.figure(figsize=(10, 6))
# plt.pie(generos_top20, labels=nombres_generos, autopct='%1.1f%%')
# plt.title('Genero principal de las 20 peliculas mas recientes')
# plt.show()

# #¿A qué género principal pertenecen las películas más largas?
# peliculas_mas_largas= df.sort_values(by='runtime', ascending=False)
# top_20_pl = peliculas_mas_largas.head(20)
# generos_pl = top_20_pl['genres'].str.split('|').str[0]
# generos_pl = generos_pl.value_counts()
# nombres_generos_pl = generos_pl.index

# #grafica
# #grafica
# plt.figure(figsize=(10, 6))
# plt.pie(generos_pl, labels=nombres_generos_pl, autopct='%1.1f%%')
# plt.title('Genero principal de las peliculas mas largas')
# plt.show()


# #---------------------------------

# '''h. 
# ¿La cantidad de actores influye en los ingresos de las películas?
# ¿Se han hecho películas con más actores en los últimos años?
# '''
# #anio mas reciente
# anio_mas_reciente = df['releaseDate'].dt.year.max()
# #filtrar los ultimos 10 a;os porque son los mas recientes
# peliculas_5_ultimos_anios = df[df['releaseDate'].dt.year >= anio_mas_reciente - 10]

# print(peliculas_5_ultimos_anios)

# anios = []
# cantidad_actores_anio = []
# for anio, grupo in peliculas_5_ultimos_anios.groupby(peliculas_5_ultimos_anios['releaseDate'].dt.year):
#     # print(anio)
#     # print(grupo)
#     #canidad de actores por a;o 
#     cantidad_actores = grupo['castWomenAmount'] + grupo['castMenAmount']
#     anios.append(anio)
#     cantidad_actores_anio.append(cantidad_actores.mean())
#     #print("Año:" + str(anio)+ " Cantidad de actores: " + str(sum(cantidad_actores)))
# print(anios)
# print(cantidad_actores_anio)

# #grafica
# plt.figure(figsize=(10, 6))
# plt.plot(anios, cantidad_actores_anio, marker='o')
# plt.title('Cantidad de actores en las peliculas por año')
# plt.show()

# print("Como se puede ver en la grafica, definitivamente desde el 2019 bajo la cantidad de actores, por lo que se descarta que se han hecho peliculas con mas actores en los ultimos anios")


# #---------------------------------

'''j. 
¿Quiénes son los directores que hicieron las 20 películas mejor calificadas?

'''


