# Get top 100 movies of 2017

import urllib3

from bs4 import BeautifulSoup
import os
import sys

urllib3.disable_warnings()

storage_path = ''
file_name = 'movie_list.txt'

#Modify to put limit
num_movies = int(input("No of movies? (max 100) : "))

rt_url = "https://www.rottentomatoes.com/top/bestofrt/?year=2017"
http = urllib3.PoolManager()
f = open(storage_path+file_name , 'w')
page = http.request('GET',rt_url);

if page.status == 200:
    
    soup = BeautifulSoup(page.data,'lxml')
    movies_table = soup.find_all('table', class_='table')
    soup = BeautifulSoup(str(movies_table),'lxml')
    movie_links = soup.find_all('a', class_='articleLink')
    movie_links = movie_links[:num_movies]
    for movie in movie_links:
        f.write(str(movie['href']).strip())
        f.write('\n')

print ('Done: Got ' + str(num_movies) + ' movies')
f.close
