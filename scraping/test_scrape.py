import urllib3
from bs4 import BeautifulSoup
import os
import sys

urllib3.disable_warnings()

storage_path = '../training/data/test/'
movie_list = 'test_movie_list.txt'
review_page_limit = 30      #Modify as required

try:
    with open(movie_list) as f:
        movies = f.readlines()
    movies = [x.strip().replace('\n','') for x in movies]
except IOError:
    print ('\n' + 'No test movie list to iterate. Please create test_movie_list.txt!' + '\n')
    sys.exit(1)

for movie in movies:
    f = open(storage_path + movie[3:] + '.txt' , 'w+')
    print (movie)
    page_no = 1
    base_url = "https://www.rottentomatoes.com/"
    review_url = "/reviews/?type=user&page="
    http = urllib3.PoolManager()
    page = http.request('GET',base_url+ movie + review_url +str(page_no));

    review_list, star_list = [], []
    while(page.status == 200 and page_no<=review_page_limit):
        
        soup = BeautifulSoup(page.data,'lxml')
        review_rows = soup.find_all('div', class_='review_table_row')
        for review in review_rows:
            review_soup = BeautifulSoup(str(review),'lxml')
            user_review = review_soup.find_all('div','user_review')
            review_text = user_review[0].text
            f.write(str(review_text).strip())
            f.write('\n')
        
        page_no += 1
        page = http.request('GET',base_url+ movie + review_url +str(page_no));
    f.close
print ('Done')
