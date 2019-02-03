#Paruchuri, V., 2016. Python Web Scraping Tutorial using BeautifulSoup. [Online] 
#Available at: https://www.dataquest.io/blog/web-scraping-tutorial-python/
#[Accessed 06 12 2018].

#Create GitHub Project Page to host website
#Use website created personally
#Available at: https://pages.github.com
#[Accessed 14 01 2019].


import requests

page = requests.get("https://shannendowling.github.io/FYPSample-2/")
page



page.status_code



page.content


#Parsing a page
import bs4
from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')


print(soup.prettify())


list(soup.children)


[type(item) for item in list(soup.children)]


[bs4.element.Doctype, bs4.element.NavigableString, bs4.element.Tag]


html = list(soup.children)[2]



list(html.children)


body = list(html.children)[3]


list(body.children)


p = list(body.children)[1]


p.get_text()



#Find all instances of a tag

soup = BeautifulSoup(page.content, 'html.parser')
soup.find_all('p')


soup.find_all('p')[0].get_text()


soup.find('p')



#Search by class/id

page = requests.get("https://shannendowling.github.io/FYPSample-2/")
soup = BeautifulSoup(page.content, 'html.parser')
soup


soup.find_all('p', class_='outer-text')


soup.find_all(class_="outer-text")


soup.find_all(id="first")


#HTML classes/ids 
#Data for training
form = soup.find_all("form")


#CSS Selectors
#soup.select("div p")



#Export data to CSV file
#Leonard Mok, 2016. Learn to love web scraping with Python and BeautifulSoup. [Online]
#Available at: http://altitudelabs.com/blog/web-scraping-with-python-and-beautiful-soup/
#[Accessed 16 01 2019].

import csv
from datetime import datetime

# open a csv file with append, so old data will not be erased
with open('indexJS1.csv', 'a') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow([form, datetime.now()])

