import csv
import requests
import pandas as pd
import random
import re
import numpy as np
from math import log


def main():

  USER = requestIdUser()

  restaurantsDf, ratingsDf = getDataset()

  preProcessDataset(restaurantsDf)

  # Seleccionamos las columnas 'idItem' y 'tags' de los restaurantes
  tagsDf = restaurantsDf.loc[:, ['idItem', 'tags']]
  tagsDf['tags'] = tagsDf['tags'].map(str)

  # Creamos la matriz TFIDF
  tfidf = createTfidf(tagsDf)

  # Obtenemos los id de los restaurantes valorados por el usuario
  itemsRated = getRatedItemsByUser(USER, ratingsDf)

  # Obtenemos la resta entre la valoración del usuario al item y la valoración media del usuario por cada item
  ratingAndMeanRatingDifference = getRatingAndMeanRatingDifference(USER, ratingsDf)

  # Obtenemos el perfil del usuario a partir de los items valroados, la matriz TF-IDF y la diferencia anterior
  activeUserProfile = getActiveUserProfile(itemsRated, tfidf, ratingAndMeanRatingDifference)

  # Obtenemos la matriz de vectores de los items que el usuario no ha valorado
  unratedItemsMatrix = getUnratedItemsMatrix(itemsRated, tfidf)

  # Obtenemos un dataframe con el idItem y la similitud del coseno respecto al perfil del usuario (de cada item)
  # Ordenado de mayor a menor similitud
  topCosineSimilarity = getCosineSimilarity(activeUserProfile, unratedItemsMatrix)

  # Creamos un dataframe que contiene el id y el nombre de cada item
  itemsNameDf= pd.merge(tagsDf['idItem'], restaurantsDf['name'], left_index=True, right_index=True, how='outer')
  
  # Formateamos el dataframe para añadir el nombre de los restaurantes según su respectivo idItem
  topCosineSimilarity['nameItem'] = topCosineSimilarity.index
  topCosineSimilarity['nameItem'] = topCosineSimilarity['nameItem'].apply(lambda id: getItemName(id, itemsNameDf))

  # Creamos una lista con las top 10 recomendaciones
  topRecommendations = formattedResult(topCosineSimilarity)

  # Y mostramos en la terminal las top 10 recomendaciones
  displayResult(USER, topRecommendations) 

def getDataset():
	datasets = []
	datasets.append(pd.read_csv('restaurants.csv', header=0, dtype={'phone': str}))
	datasets.append(pd.read_csv('ratings.csv', header=0))
	return datasets

def preProcessDataset(df):

	def modifyPriceCategory(row):
		if len(row['price']) == 1:
			return 'very_cheap'
		elif len(row['price']) == 2:
			return 'cheap'
		elif len(row['price']) == 3:
			return 'medium_price'
		elif len(row['price']) == 4:
			return 'expensive'
		elif len(row['price']) == 5:
			return 'very_expensive'
		else:
			return row['price']

	df['priceCategory'] = df.apply(modifyPriceCategory, axis=1)
	df['tags'] = df['tags'] + ', ' + df['priceCategory']

	maxNumRatings = df['numRatings'].max()

	def setCategory(num_ratings):
		if num_ratings > maxNumRatings * 0.8:
				return 'famous'
		elif num_ratings > maxNumRatings * 0.5:
				return 'well_known'
		elif num_ratings > maxNumRatings * 0.2:
				return 'moderately_known'
		elif num_ratings > maxNumRatings * 0.08:
				return 'less_known'
		else:
				return 'almost_known'

	df['fame'] = df['numRatings'].apply(setCategory)
	df['tags'] = df['tags'] + ', ' + df['fame']

def preprocessRestaurantsDf(restaurantsDf):
  restaurantsAuxDf = restaurantsDf.drop('idItem', axis=1)
  restaurantsDf['tags'] = restaurantsAuxDf.apply(lambda row: ', '.join([f'{col}_{row[col]}' for col in restaurantsAuxDf.columns if pd.notnull(row[col])]), axis=1)
  return restaurantsDf

def createTfidf(dataTagsDf):
  
  # Reemplazamos los espacos por guiones bajos para evitar separar un mismo tag por tener un espacio
  # Ej: "Tag de ejemplo" -> "Tag_de_ejemplo"
  dataTagsDf['tags'] = dataTagsDf['tags'].apply(lambda x: re.sub(r', +', ',', x))
  dataTagsDf['tags'] = dataTagsDf['tags'].apply(lambda x: x.replace(' ', '_'))
  dataTagsDf['tags'] = dataTagsDf['tags'].apply(lambda x: x.replace(',', ' '))

  # Creamos un diccionario de tags únicos y su frecuencia en todo el conjunto de datos
  tagFrequency = {}
    # Iteramos cada fila (cada item) 
  for row in dataTagsDf.itertuples():
    # Iteramos cada tag (row[2] = Posición de la columna tags en el dataframe) haciendo uso del metodo split para separar los tags por espacios
    for tag in row[2].split():
      if tag not in tagFrequency:
        # Si el tag no se encuentra en el diccionario, significa que es un tag nuevo y se almacena en el diccionario con valor 1
        tagFrequency[tag] = 1
      else:
        # Si el tag se encuentra en el diccionario, aumentamos en 1 el número de ocurrencias en el conjunto de items
        tagFrequency[tag] += 1 

  # Calcular la matriz TF
  # Creamos una matriz compuesta de 0 con tamaño (nºItems x nºTagsUnicos)
  # Almacenará el TF de cada tag en su respectivo idItem
  tfMatrix = np.zeros((len(dataTagsDf), len(tagFrequency)))
  # Iteramos cada fila del dataframe, cada fila contiene el idItem y los tags asociados respectivamente
  for i, row in enumerate(dataTagsDf.itertuples()):
    # Iteramos sobre cada tag del tagFrequency (contiene los tags unicos)
    for j, tag in enumerate(tagFrequency.keys()):
      # Contamos cuantas veces aparece el tag en los tags asociados al item
      # Ingresamos el resultado en su correspondiente posición de la matriz TF
      tfMatrix[i,j] = row[2].split().count(tag)

  # Calcular el IDF para cada tag única
  # Creamos un vector con ceros inicialmente, con tamaño = nºtags únicas, que almacenará el IDF de cada tag
  idfVector = np.zeros(len(tagFrequency))
  # Iteramos sobre cada tag
  for j, tag in enumerate(tagFrequency.keys()):
    # Calculamos según la formula del IDF
    #   log(N/N_t)
    #     -> N = Número de items
    #     -> N_t = Número de items que contienen el tag

    # 1. Filtramos el dataframe original para obtener solo las filas que contienen el tag
    itemsWithTag = dataTagsDf[dataTagsDf['tags'].apply(lambda x: tag in x.split())]
    # 2. Calculamos la cantidad de filas que contienen el tag
    numItemsWithTag = len(itemsWithTag)
    # 3. Calculamos la proporción inversa de la cantidad anterior respecto al número de items
    inverseProportion = len(dataTagsDf) / numItemsWithTag
    # 4. Calculamos el logaritmo de la proporción inversa para obtener el IDF de la etiqueta
    itemTdf = log(inverseProportion)
    # Añadimos el valor tdf del item al vector
    idfVector[j] = itemTdf
    
  # Calcular la matriz TF-IDF (TF-IDF = TF * IDF)
  tfidfMatrix = tfMatrix * idfVector

  # Creamos un dataframe con los valores de las matrices asociados a sus respecitov sidItem y tags 
  tfidf = pd.DataFrame(tfidfMatrix, index=dataTagsDf['idItem'], columns=list(tagFrequency.keys()))

  # Normalizamos la matriz para que todos tengan modulo 1
  # Para evitar que items con muchas etiquetas (Ej. populares) tengan mayor fuerza de ponderación

  # Normalización l2
  #   (La similitud del coseno entre dos vectores es su producto escalar)
  #   (divide cada vector por la norma Euclidiana, es decir, la raíz cuadrada de la suma de los cuadrados de los valores de sus elementos.)

  # Calculamos las normas de los vectores de cada item 
  norms = np.linalg.norm(tfidf.values, axis=1, ord=2)
  # Dividimos el vector de cada item por su respectiva norma para obtener el TF-IDF normalizado
  normalizeTfidf = tfidf.div(norms, axis=0)

  # Devolvemos el dataframe de la matriz TF-IDF normalizada
  return normalizeTfidf

def getRatedItemsByUser(idUser, ratingsDf):
  itemsRated = ratingsDf[ratingsDf['idUser'] == idUser][['idItem']]
  return np.array(itemsRated).flatten().tolist()

def getRatingAndMeanRatingDifference(idUser, ratingsDf):

  # Filtrar el DataFrame de ratings por el usuario con id = 1
  ratingsUsuario = ratingsDf[ratingsDf['idUser'] == idUser]

  # Calcular la media de los ratings del usuario
  mediaRatingsUsuario = ratingsUsuario['rating'].mean()

  # Creamos un dataframe que refleje la diferencia, junto con el item correspondiente
  ratingAndMeanRatingDifference = pd.DataFrame({
      'idItem': ratingsUsuario['idItem'],
      'difference': ratingsUsuario['rating'] - mediaRatingsUsuario
    })

  # Devolvemos el dataframe
  return ratingAndMeanRatingDifference

def getActiveUserProfile(itemsRated, tfidf, ratingAndMeanRatingDifference):
  
  # Recogemos las filas correspondientes a los items valorados por el usuario, de la matriz TF-IDF
  activeUserProfile = tfidf.loc[itemsRated,:]

  # Creamos un diccionario con los valores de diferencia entre la valoración y la valoración media para cada item
  diffDict = dict(zip(ratingAndMeanRatingDifference['idItem'], ratingAndMeanRatingDifference['difference']))

  if(all(value != 0 for value in diffDict.values())):
    # Agregamos una columna 'idItem' al perfil del usuario activo con los índices actuales y poder operar con ellos
    activeUserProfile['idItem'] = activeUserProfile.index

    # Realizamos la multiplicación de los valores en el perfil del usuario activo con los valores de diferencia correspondientes
    activeUserProfile.iloc[:, 0:] = activeUserProfile.iloc[:, 0:].multiply(activeUserProfile['idItem'].map(diffDict), axis=0)

    # Aliminamos la columna 'idItem' del perfil del usuario activo, ya no es necesaria
    activeUserProfile = activeUserProfile.drop('idItem', axis=1)

  # Calculamos la media de los valores en el perfil del usuario activo para obtener el vector que describirá los gustos del usuario
  activeUserProfile = activeUserProfile.mean()
  # También se puede utilizar la suma en lugar de la media
  #activeUserProfile = activeUserProfile.sum()

  # Devolvemos el perfil del usuario activo
  return activeUserProfile

def getUnratedItemsMatrix(itemsRated, tfidf):
  return tfidf.drop(itemsRated, axis=0)

def getCosineSimilarity(activeUserProfile, unratedItemsMatrix):

  # Similitud del coseno = producto escalar entre dos vectoes / producto de las normas de los dos vectores = (u · v) / (||u|| * ||v||)

  # Calculamos el producto escalar entre el perfil del usuario activo y los vectores de los items no valorados
  # Esto nos devolverá una matriz con el resultado del producto escalar de cada item
  dotProduct = np.dot(activeUserProfile, unratedItemsMatrix.T)

  # Calculamos el producto entre las normas del perfil de usuario activo y los vectores de los items no valorados
  # Esto nos devolverá una matriz con el resultado del producto por cada item
  normProduct = np.linalg.norm(activeUserProfile) * np.linalg.norm(unratedItemsMatrix, axis=1)

  # Calcular la similitud del coseno dividiendo el producto escalar por el producto de las normas, por cada item
  # Cuanto mayor es el resultado, mayor es la similitud del item con el usuario y por ende, más recomendado
  similarity = dotProduct / normProduct

  # Creamos un dataframe con los valores de similitud obtenidos y sus respectivos items
  similarityDf = pd.DataFrame(similarity, index=unratedItemsMatrix.index, columns=['cosineSimilarity'])

  # Ordenamos los items según la similitud con el usuario de mayor a menor
  topsimilarity = similarityDf.sort_values(by='cosineSimilarity', ascending=False)

  # Devolvemos el dataframe
  return topsimilarity

def getItemName(itemId, movieTitlesDf):
  return movieTitlesDf[movieTitlesDf['idItem'].isin([itemId])]['name'].values[0]

def displayResult(idUser, topRecommendations):
  print(f'\nTOP 10 RECOMMENDATIONS FOR USER {idUser}\n')
  for item in topRecommendations:
    print(item)
  print('\n')

def formattedResult(topCosineSimilarity):
  topRecommendations = []
  count = 0
  for _, row in topCosineSimilarity.iterrows():
      topRecommendations.append(f"{row['nameItem']}:  {round(row['cosineSimilarity'], 4)}")
      count += 1
      if count == 10:
        break
  return topRecommendations

def requestIdUser():
  idUser = input('\nEnter the user id: ')
  if(idUser == None or not idUser.isdigit() or int(idUser) < 1 or int(idUser) > 40):
    print('\nInvalid user id, please enter a valid user (1 - 40)')
    return requestIdUser()
  print(f'\nGetting top 10 recommendations for user {idUser}...')
  return int(idUser)

def createCsvRestaurants():
  api_key = 'API KEY DEL USUARIO YELP'
  url = 'https://api.yelp.com/v3/businesses/search'
  headers = {'Authorization': 'Bearer ' + api_key}
  params = {'location': 'Sevilla, Spain', 'categories': 'restaurants', 'limit': 50}
  page = 1
  idItem = 0
  with open('restaurants.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['idItem', 'name', 'tags', 'price', 'rating', 'numRatings', 'address', 'phone']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    while True:
      params['offset'] = (page - 1) * params['limit']
      response = requests.get(url, headers=headers, params=params)
      data = response.json()
      print(data)
      if "businesses" in data:
        pageResult = len(data['businesses'])
        if pageResult == 0:
          break
        for restaurante in data['businesses']:
          idItem += 1
          name = restaurante['name']
          tags = ', '.join([cat['title'] for cat in restaurante['categories']])
          price = restaurante['price']+'€' if 'price' in restaurante else '€'
          rating = restaurante['rating'] if 'rating' in restaurante else '-----'
          numRatings = restaurante['review_count'] if 'review_count' in restaurante else '0'
          location = ' '.join(restaurante['location']['display_address']) if 'location' in restaurante else '-----'
          phone = restaurante['phone'] if 'phone' in restaurante else '-----'
          writer.writerow({
            'idItem': idItem,
            'name': name,
            'tags': tags,
            'price': price,
            'rating': rating,
            'numRatings': numRatings,
            'address': location,
            'phone': phone
          })
            
        page += 1
      else:
        break
  return idItem

def createCsvRatings(NUM_RESTAURANTS):
	numUsers = 50
	items = NUM_RESTAURANTS
	minRatings = 15
	maxRatings = 55
	with open('ratings.csv', 'w', newline='') as csvfile:
		fieldnames = ['idUser', 'rating', 'idItem']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for user in range(1, numUsers + 1):
			numRatings = random.randint(minRatings, maxRatings)
			for _ in range(numRatings):
				idItem = random.randint(1, items)
				rating = random.randint(1, 5)
				writer.writerow({
						'idUser': user,
						'rating': rating,
						'idItem': idItem
				})




if __name__ == '__main__':
  #NUM_RESTAURANTS = createCsvRestaurants()
  NUM_RESTAURANTS = 1000
  createCsvRatings(NUM_RESTAURANTS)
  main()



