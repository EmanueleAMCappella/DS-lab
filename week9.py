# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:05:14 2017

@author: EAMC
"""
  

# coding: utf-8


from bs4 import BeautifulSoup
import urllib.request as ur

urlToScrape = "http://www.city.ac.uk/courses?level=Undergraduate"
r = ur.urlopen(urlToScrape).read()
soup = BeautifulSoup(r, "lxml")

courseList = soup.find_all('div', attrs={'class': 'course-finder__results__item course-finder__results__item--undergraduate'})
for courseListItem in courseList:
    courseNameElement = courseListItem.find('div', attrs={'class': "col-sm-24 col-md-18 col-lg-20"})
    courseName = courseNameElement.find('a').text
    print (courseName)



from bs4 import BeautifulSoup
import urllib.request as ur

from xml.etree.ElementTree import Element, SubElement, Comment, tostring


from xml.dom import minidom


# since we want to generate a single XML file, we start with the root
# before we start running the loop here.
root = Element('Courses')

# Just get the first five comment pages from the course list at City
for i in range (0,5):
    
    # This is the address format we find out through observation
    # Find this out by clicking through the first couple of pages and looking at how the address line is changing.
    urlToScrape = "http://www.city.ac.uk/courses?level=Undergraduate&p=" + str(i * 10 + 1) 
    r = ur.urlopen(urlToScrape).read()
    soup = BeautifulSoup(r, "lxml")
    
    # get all the "div"s that ar of the class we are interested in.
    # note that the following line will return us a collection of "div" sections
    courseList = soup.find_all('div', attrs={'class': 'course-finder__results__item course-finder__results__item--undergraduate'})
    for courseListItem in courseList:
        
        # Here we cover everything with try/except constructs to ensure that we are not failing when an element is not there.    
        try:
            # first go inside the DIV to access the course name, note that it is under a <a> tag, so that's where we need to access here.
            courseNameElement = courseListItem.find('div', attrs={'class': "col-sm-24 col-md-18 col-lg-20"})
            courseName = courseNameElement.find('a').text
        except Exception as e:
            courseName = ""
                
        try:
            # here, we access the course description, you can find that it is under a DIV
            courseDescriptionElement = courseListItem.find('div', attrs={'class': "course-finder__results__item__description"})
            courseDescription = courseDescriptionElement.text
        except Exception as e:
            courseDescription = ""
        
        try:
            # now on to scraping the name of the school offering the course, this time under an <a> tag 
            courseSchoolElement = courseListItem.find('div', attrs={'class': "course-finder__results__item__md course-finder__results__item__md--school"})
            courseSchool = courseSchoolElement.find('a').text
        except Exception as e:
            courseSchool = ""

        try:
            # when we try to get the course code, we notice that there are two <span> sections under the section 
            # we are interested in, we want to get the second one, hence => .find_all('span')[1]

            courseCodeElement_1 = courseListItem.find('div', attrs={'class': "course-finder__results__item__md course-finder__results__item__md--code"})
            courseCodeElement_2 = courseCodeElement_1.find_all('span')[1]
            courseCode = courseCodeElement_2.text
        except Exception as e:
            courseCode = ""
        
        # And let's start populating the XML file by adding the scraped data one by one
        courseXML = SubElement(root, 'Course')
        
        courseNameXML = SubElement(courseXML, 'CourseName')
        courseNameXML.text = courseName
        
        courseDescriptionXML = SubElement(courseXML, 'CourseDescription')
        courseDescriptionXML.text = courseDescription
        
        courseSchoolXML = SubElement(courseXML, 'CourseSchool')
        courseSchoolXML.text = courseSchool
        
        courseSchoolXML = SubElement(courseXML, 'CourseCode')
        courseSchoolXML.text = courseCode
        
        # The following print lines are not needed since we will be saving to a file, 
        # but good to see the progress and for diagnostics
        print (courseName, i) 
        print (courseDescription)
        print (courseSchool)
        print (courseCode)
        print ("-------------")


f = open('CityCourses.xml', 'wb')
f.write(tostring(root, 'utf-8'))
f.close()


def mysum(list):
    somma= 0
    for i in list:
        somma+= i
    return somma

mysum([1, 2, 3, 4, 5, 6])

def mylenght(list):
    daje= 0
    for i in list:
        daje+= 1
    return daje

mylenght([1, 2, 3, 4, 5, 6])

def mymean (list):
    risultato= mysum(list)/mylenght(list)
    return risultato

mymean([1, 2, 3, 4, 5, 6])
    
    
    
    