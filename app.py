


import numpy as np
import pandas as pd
import geocoder

from flask import Flask, render_template, request, flash, jsonify
from forms import *

app = Flask(__name__) #Initialize the flask App

import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
import pprint
#from IPython.display import clear_output
from sklearn.cluster import KMeans
import sklearn.neighbors

app.secret_key = 'development key'
@app.route('/', methods = ['GET'])
def hello():
    return 'This is my first API call!'
    # form = Registro()

    # if request.method == 'POST':
    #     if form.validate() == False:
    #         return render_template('index.html', form = form)
    #     else:
    #         name = form.name.data
    #         data_frame = return_df(name)
    #         # print(data_frame)
    #         return render_template('registro.html', name = data_frame)
    # elif request.method == 'GET':
    #     return render_template('index.html', form = form)

@app.route('/post', methods=['POST'])
def testpost():
     input_json = request.get_json(force=True)
     category = input_json['category']
     latitude = input_json['latitude']
     longitude = input_json['longitude']
     
    #  dictToReturn = {'text':input_json['text']}

     df = return_df(category, latitude, longitude)


     return jsonify(df)
    # input_json = request.get_json(force=True) 
    # dictToReturn = {'text':input_json['text']}
    # return jsonify(dictToReturn)

def return_df(profession, latitude, longitude):
    client = pymongo.MongoClient(
        "mongodb+srv://hizmat:KsUZ7B4Ehm4zBT4Q@app.p0zzj.mongodb.net/Hizmat?retryWrites=true&w=majority")
    db = client.test
    mydb = client["Hizmat"]
    peshawar_coords = mydb["gigs"]

    cursor = peshawar_coords.find()
    entries = list(cursor)
    df = pd.DataFrame(entries)
    df = df.drop(['_id'], 1)
    df["user"] = df["user"].astype("string")

    #df=pd.read_csv('peshawar.csv')
    coords = df[['longitude', 'latitude']]
    services_available = df.loc[:, 'category'].to_list()
    print(type(services_available))
    print(df.shape)
    print(profession not in services_available)
    if(profession not in services_available):
        data = {
                "service": [],
                "category": [],
                "imageUrl": [],
                "Distance in km": [],
                "user": []
        }
        return data

    if(df.shape[0]<=5):
        df["Distance in km"] = 0.0
        df = df.loc[:,['service',
        'category', 'user', 'imageUrl', 'Distance in km']]
        data = {
                "service": df["service"].to_list(),
                "category": df["category"].to_list(),
                "imageUrl": df["imageUrl"].to_list(),
                "Distance in km": df["Distance in km"].to_list(),
                "user": df["user"].to_list(),
        }

        return data
    distortions = []
    K = []
    if(df.shape[0] < 25):
        K = range(1, df.shape[0])
    else:
        K = range(1, 25)
    for k in K:
        kmeansModel = KMeans(n_clusters=k)
        kmeansModel = kmeansModel.fit(coords)
        distortions.append(kmeansModel.inertia_)

    kmax = k
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(coords)
        labels = kmeans.labels_

    df['cluster'] = kmeans.predict(df[['longitude', 'latitude']])

    #import geocoder

    # g = geocoder.ip('me')
    #print(g.latlng)


    def haversine_distance(lat1, lon1, lat2, lon2):
        r = 6371
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
        return np.round(res, 2)


    # longitude =lat
    # latitude =g.lat



    def recommend_worker(df, profession):
        df['profession'] = df['category'].str.contains(profession)

        df = df.loc[df.profession == True]

        # Predict the cluster for longitude and latitude provided

        cluster = kmeans.predict(np.array([longitude, latitude]).reshape(1, -1))[0]
        

        dfs = df[df['cluster'] == cluster].iloc[0:5][['service','latitude','longitude',
        'category', 'imageUrl', 'user']]
        cities = pd.DataFrame(dfs)
        distances_km = []

        for row in cities.itertuples(index=False):
            distances_km.append(
                haversine_distance(latitude, longitude, row.latitude, row.longitude)
            )

        dfs['Distance in km'] = distances_km

        round_to_tenths = [round(num, 2) for num in distances_km]
        dfs = dfs.sort_values(by=['Distance in km'], ascending=True)
        
        
        

        # print(cluster)
        # Get the best worker in this cluster

        lat1 = 33.96123
        lat2 = 34.11463
        long1 = 71.37623
        long2 = 71.67234

        if ((longitude <= long1 or longitude >= long2) or (latitude <= lat1 or latitude >= lat2)):
            print("you are out side from peshawar")
        else:
            print("you are here \n")
            dfs=dfs[['service', 'user', 'imageUrl',
        'category','Distance in km']]
            
            data = {
                "service": dfs["service"].to_list(),
                "category": dfs["category"].to_list(),
                "imageUrl": dfs["imageUrl"].to_list(),
                "Distance in km": dfs["Distance in km"].to_list(),
                "user": dfs["user"].to_list(),
            }


            return data




    data_frame = recommend_worker(df,profession)
    return data_frame






