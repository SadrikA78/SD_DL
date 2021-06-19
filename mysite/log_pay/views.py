from django.shortcuts import render
from django.contrib.admin.models import LogEntry
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from .models import *
from django.http import Http404
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import datetime
from django.core.serializers import serialize
from .forms import *
from django.contrib.auth.models import User
from django.contrib.auth import logout
import time
from django.views.generic import View
import plotly.offline as opy
import plotly.graph_objs as go
from plotly.offline import plot
import urllib.request
import re
import pandas as pd
import numpy as np
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import random
import sys
from nltk.cluster import euclidean_distance, cosine_distance
np.random
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import seaborn as sns; 
from sklearn.metrics import davies_bouldin_score
from scipy.cluster.hierarchy import complete, dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import average, dendrogram
#!pip install kneed
from kneed import KneeLocator
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import plotly.figure_factory as ff
import plotly.express as px
stemmer = SnowballStemmer("english")
nltk.download('punkt')

def get_distant_matrix(normal_matrix):
  cos_dist = 1 - cosine_similarity(normal_matrix)
  mat_dist = pairwise_distances(normal_matrix, metric='manhattan') 
  ucl_dist = pairwise_distances(normal_matrix, metric='euclidean') 
  return cos_dist, mat_dist, ucl_dist

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
now = datetime.date.today()
def custom_logout(request):
    print('Loggin out {}'.format(request.user))
    logout(request)
    print(request.user)
    return HttpResponseRedirect('')

    
@login_required
def faq(request):
    vessels = Vessel.objects.all()
    cats = Type_cat.objects.all()
    content = {'vessels':vessels, 'TYPE_VESSEL':TYPE_VESSEL, 'cats':cats}
    return render(request, 'log_pay/faq.html', content)

def registration(request):
    if request.POST:
        username = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']
        if password == password2:
            new_user = User.objects.create_user(username, email, password)
            new_user.save()
        return HttpResponseRedirect('/accounts/login/')
    content = {}
    return render(request, 'registration/registration.html', content)

def password_reset(request, is_admin_site=False):
    if request.POST:
        return HttpResponseRedirect('/accounts/login/')
    content = {}
    return render(request, 'registration/password_reset.html', content)

class LogoutView(View):
    template_name = 'registration/logged_out.html'

    def get(self, request):
        response = logout(request)

        return render(response, self.template_name)
    
@login_required
def index(request):
    papers = pd.read_csv('papers.csv', index_col=0)
    frontirs = papers['session'].unique()
    papers["keywords_abstract"]= papers["author_keywords"].str.cat(papers['abstract'], sep =" ") 

    encoder = preprocessing.LabelEncoder()
    encoder.fit(papers['session'])

    papers['session_encoded'] = encoder.transform(papers['session'])
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(papers['abstract'])

    reform_tfidf_matrix = tfidf_matrix.toarray()
    
    
    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(papers['abstract'])

    reform_tfidf_matrix = tfidf_matrix.toarray()


    print("Количество уникальных фронтиров:", papers['session'].nunique())
    print("\nКоличество уникальных слов: ", reform_tfidf_matrix.shape)
    print(tfidf_vectorizer.get_feature_names())
    cos_dist, mat_dist, ucl_dist = get_distant_matrix(tfidf_matrix)
    clustering1 = AgglomerativeClustering(affinity='euclidean', n_clusters=20, linkage='complete').fit(reform_tfidf_matrix)
    print(clustering1)

    print(len(clustering1.labels_))
    print(clustering1.labels_[:5])


    clustering2 = AgglomerativeClustering(affinity='euclidean', n_clusters=20, linkage='average').fit(reform_tfidf_matrix)
    print(clustering2)

    print(len(clustering2.labels_))
    print(clustering2.labels_[:5])


    clustering3 = AgglomerativeClustering(affinity='euclidean', n_clusters=20, linkage='single').fit(reform_tfidf_matrix)
    print(clustering3)
    linkage_matrix = average(cos_dist) 
    fig, ax = plt.subplots(figsize=(15, 20))
    ax = dendrogram(linkage_matrix, orientation="right", labels= list(papers['session']))

    plt.tick_params(\
        axis= 'x',        
        which='both',     
        bottom='off',      
        top='off',         
        labelbottom='off',
        labelsize=15
        )
    plt.tick_params(\
        axis= 'y',         
        labelsize=15
        )

    plt.tight_layout() 
    fig = go.Figure()
    geojson_oh = json.load(open("ohot_sea.geojson"))

    points_oh = []
    for  feature in geojson_oh['features']:
        if feature['geometry']['type'] == 'Polygon':
            points_oh.extend(feature['geometry']['coordinates'][0])    
            points_oh.append([None, None]) # mark the end of a polygon   
        elif feature['geometry']['type'] == 'MultiPolygon':
            for polyg in feature['geometry']['coordinates']:
                points_oh.extend(polyg[0])
                points_oh.append([None, None]) #end of polygon
        elif feature['geometry']['type'] == 'MultiLineString': 
            points_oh.extend(feature['geometry']['coordinates'])
            points_oh.append([None, None])
        else: pass   
    lons_oh, lats_oh = zip(*points_oh) 
    print (lons_oh)
    fig.add_trace(go.Scattermapbox(
        mode = "lines",
        fill="toself",
        fillcolor = 'yellow',
        lat = lats_oh,
        lon = lons_oh,
        name="США",
        hovertemplate="<b>США</b><br><br>" +
                                            "Фронтиров: 16<br>" +
                                            "Новых: 3<br>" +
                                            "<b>Степень определения:</b>"+ "<br>"+"высокая<br>" +
                                            "<extra></extra>",
        marker = {'color':"blue", 'size': 6}))
    fig.update_layout(
                    autosize=True,
                    height=1000,
                    hovermode='closest',
                    mapbox=dict(
                        style='outdoors',
                        accesstoken='pk.eyJ1Ijoic2FkcmlrIiwiYSI6ImNrZ3ZhNWhrZDBpbXgycXJ5Z2cwNmFzZTcifQ.cxbz_15TzOfpWeFafy3sUQ',
                        bearing=0,
                        center=go.layout.mapbox.Center(
                            lat=30.889537, 
                            lon=60.744884
                        ),
                        pitch=0,
                        zoom=3
                    )
                )
    fig.update_yaxes(automargin=True)
    plt_div = plot(fig, output_type='div', include_plotlyjs = False)
    content = {'frontirs':frontirs, 'plt_div':plt_div}
    return render(request, 'log_pay/index.html', content)
@login_required
def prognoz(request):

    papers = pd.read_csv('papers.csv', index_col=0)
    frontirs = papers['session'].unique()

    fig = go.Figure()
    dataset = pd.read_csv('2.csv')
 
    years = ["2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030",
            "2031"]
    # make list of continents
    continents = []
    for continent in dataset["continent"]:
        if continent not in continents:
            continents.append(continent)
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [30, 85], "title": "Охват"}
    fig_dict["layout"]["yaxis"] = {"title": "Количество исследований", "type": "log"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Вперед",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Пауза",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Год:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make data
    year = 1952
    for continent in continents:
        dataset_by_year = dataset[dataset["year"] == year]
        dataset_by_year_and_cont = dataset_by_year[
            dataset_by_year["continent"] == continent]

        data_dict = {
            "x": list(dataset_by_year_and_cont["lifeExp"]),
            "y": list(dataset_by_year_and_cont["gdpPercap"]),
            "mode": "markers",
            "text": list(dataset_by_year_and_cont["continent"]),
            "marker": {
                "sizemode": "area",
                "sizeref": 200000,
                "size": list(dataset_by_year_and_cont["pop"])
            },
            "name": continent
        }
        fig_dict["data"].append(data_dict)

    # make frames
    for year in years:
        frame = {"data": [], "name": str(year)}
        for continent in continents:
            dataset_by_year = dataset[dataset["year"] == int(year)]
            dataset_by_year_and_cont = dataset_by_year[
                dataset_by_year["continent"] == continent]

            data_dict = {
                "x": list(dataset_by_year_and_cont["lifeExp"]),
                "y": list(dataset_by_year_and_cont["gdpPercap"]),
                "mode": "markers",
                "text": list(dataset_by_year_and_cont["continent"]),
                "marker": {
                    "sizemode": "area",
                    "sizeref": 200000,
                    "size": list(dataset_by_year_and_cont["pop"])
                },
                "name": continent
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [year],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": year,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)


    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)
    plt_div = plot(fig, output_type='div', include_plotlyjs = False)
    content = {'frontirs':frontirs, 'plt_div':plt_div}
    return render(request, 'log_pay/prognoz.html', content)