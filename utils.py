# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator


def scrape(journal, vol_start, vol_end):

    result = requests.get("https://www.journals.elsevier.com/"+journal)
    soup = BeautifulSoup(result.content, features="lxml")
    metrics = soup.find_all("span", "tooltip")
    met_names = ['CiteScore','ImpactFactor','5-year Impact Factor','Norm. Impact per Paper','Journal Rank']
    data = [a.b.text for a in metrics if a.b != None]
    del data[2:4]
    met_dict = dict(zip(met_names,data))
    
    volumes = [str(i) for i in range(vol_start,vol_end)]
    titles_str = []
    for i in volumes:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
        url = "https://www.sciencedirect.com/journal/" + journal + "/vol/" + i +"/suppl/C"
        r = requests.get(url, headers=headers)
        site = BeautifulSoup(r.content)
        titles = site.find_all("span", class_="js-article-title")
        for j in titles:
            titles_str.append(j.text)
            
    data = pd.DataFrame(titles_str)
    data.to_csv('data/raw_data_'+journal+'.csv')
    return met_dict


def clean_data(journal):
    data = pd.read_csv('data/raw_data_'+journal+'.csv')
    titles_str = list(data.values[:,1])
    sws = stopwords.words('english')
    titles_list = [re.split(r'\W+',i) for i in titles_str]
    titles_set = [set(i).difference(sws)-{''} for i in titles_list]
    return titles_set

def wordcloud(clean_data, max_words, max_font_size, min_font_size, background_color):
    text = ' '.join([' '.join(list(i)) for i in clean_data])
    wc = WordCloud(max_words = max_words, max_font_size = max_font_size, min_font_size = min_font_size, background_color=background_color).generate(text)
    return wc


def word_freq(clean_data):
    total_list = [x for _list in clean_data for x in _list]
    total_set = list(set(total_list))
    count = [total_list.count(i) for i in total_set]
    total_dict = dict(zip(total_set,count))
    return {k: v for k, v in sorted(total_dict.items(), key=lambda item: item[1],reverse=True)}


def word_trends(clean_data, num_journals, num_words, factor):
    count = int(len(clean_data)/num_journals)*factor
    section_dict = {}
    section_keys = set()
    for i in range(0,int(num_journals/factor)):
        temp = clean_data[i*count:i*count+count]
        temp_dict = word_freq(temp)
        keys = list(temp_dict.keys())[0:num_words]
        values = [temp_dict[i] for i in keys]
        temp_dict_red = dict(zip(keys,values))
        section_dict["Section: " + str(i)] = temp_dict_red
        section_keys = set(section_keys | set(keys))

    section_graph = {}
    for i in list(section_keys):
        section_graph[i] = []

    for i in section_dict.values():
        for j in list(section_keys):
            if j not in i.keys():
                i[j] = 0
            section_graph[j].append(i[j])
      
    return section_graph



def word_connections(clean_data, words, connections):

    total_list = [x for _list in clean_data for x in _list]
    total_set = list(set(total_list))
    total_num = dict((j,i) for i,j in enumerate(total_set))
    word_mat = np.zeros((len(clean_data), len(total_set)))
    for i,j in enumerate(clean_data):
        for k in list(j):
            word_mat[i,total_num[k]] = 1

    words_ind = [total_set.index(i) for i in words]

    new_word_mat = np.zeros((len(words),len(total_set)))
    for i in word_mat:
        for j,k in enumerate(words_ind):
            if i[k] == 1:
                new_word_mat[j] = new_word_mat[j] + i

    conn_dict = {}
    for i in words:
        temp_max = list(new_word_mat[words.index(i)].argsort()[-connections:-1][::-1])
        temp_val = new_word_mat[words.index(i),temp_max]
        temp_word = np.array(total_set)[temp_max]
        conn_dict[i] = dict(zip(temp_word,temp_val))

    return conn_dict





