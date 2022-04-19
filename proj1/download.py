#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import zipfile
import requests
from requests import NullHandler, get
from bs4 import BeautifulSoup
import os.path
import numpy as np
import csv
import zipfile
import time
import pickle
import gzip
# Kromě vestavěných knihoven (os, sys, re, requests …) byste si měli vystačit s: gzip, pickle, csv, zipfile, numpy, matplotlib, BeautifulSoup.
# Další knihovny je možné použít po schválení opravujícím (např ve fóru WIS).


class DataDownloader:
    """ TODO: dokumentacni retezce  

    Attributes:
        headers    Nazvy hlavicek jednotlivych CSV souboru, tyto nazvy nemente!  
        regions     Dictionary s nazvy kraju : nazev csv souboru
    """

    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a","region"]

    types = {"p1":int, "p36":int, "p37":int, "p2a":"str", "weekday(p2a)":int, "p2b":int, "p6":int, "p7":int, "p8":int, "p9":int, 
             "p10":int, "p11":int, "p12":int, "p13a":int,"p13b":int, "p13c":int, "p14":int, "p15":int, "p16":int, "p17":int, 
             "p18":int, "p19":int, "p20":int, "p21":int, "p22":int, "p23":int, "p24":int, "p27":int, "p28":int,"p34":int, "p35":int, 
             "p39":int, "p44":int, "p45a":str, "p47":int, "p48a":int, "p49":int, "p50a":int, "p50b":int, "p51":int, "p52":int, "p53":int, "p55a":int,
             "p57":int, "p58":int, "a":float, "b":float, "d":float, "e":float, "f":float, "g":float, "h":str, "i":str, "j":int, "k":str, "l":str, "n":str,
             "o":float, "p":str, "q":str, "r":int, "s":int, "t":str, "p5a":int, "region":str}

    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    tmp = set()
    dicts_in_memory = dict()

    def __init__(self, url="https://ehw.fit.vutbr.cz/izv/", folder="data", cache_filename="data_{}.pkl.gz"):
        self.url = url
        self.folder = folder
        self.cache_filename = cache_filename
        for i in range(0,65):
            self.dicts_in_memory[self.headers[i]] = np.array([],dtype=self.types[self.headers[i]])

    def make_folder_if_not_exist(self):
        folder_to_make = os.getcwd() + "/" + self.folder
    
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

    def download_data(self):
        html = requests.get(self.url).text
        for_parse = BeautifulSoup(html,"html.parser")

        self.make_folder_if_not_exist()
        
        for to_download in for_parse.find_all('tr'):
            path_to_file = to_download.find_all('button')[-1].get('onclick').split('\'')[1]
            filename = path_to_file.split("/")[1]
            with open (self.folder + "/" + filename, "wb") as file:    
                response = get(self.url + path_to_file)
                file.write(response.content)

    ## this function adds to class set and returns true if it is not duplicate 
    # false when it is duplicate (not added to set)
    def do_add(self, x):
        l = len(self.tmp)
        self.tmp.add(x)
        return len(self.tmp) != l 

    def parse_region_data(self, region):
        # find if data are downloaded if no data will be downloaded to specific folder
        folder_with_data = os.getcwd() + "/" + self.folder
        if not os.path.exists(folder_with_data):
            self.download_data() 
        elif not any(x.endswith(".zip") for x in os.listdir(folder_with_data)):
            self.download_data() 

        num_of_region = self.regions.get(region)
        dict_without_np = dict()
        dict_to_return = dict()
        self.tmp.clear()
        
        for i in range(0,64):
            dict_without_np[self.headers[i]] = list()

        for filename in os.listdir(self.folder):
            if filename.endswith("zip"): 
                #print(os.path.join('data', filename))
                with zipfile.ZipFile(os.path.join('data', filename), 'r') as zip_ref:
                    zip_ref.extract(num_of_region + ".csv",self.folder)
                    with open(self.folder+"/"+num_of_region + ".csv", 'r', 
                              encoding="cp1250") as csvfile:
                        reader = csv.reader(csvfile,delimiter=';', quotechar='"')
                        for row in reader:
                            counter = 0
                            # selecting duplicates -> adding id to set (if duplicate is added to set it is not added)
                            if self.do_add(row[0]):                              
                                for el in row:
                                    # if data are invalid and type should be int we substitute value with -1
                                    if self.types[self.headers[counter]] == int and (el == "" or el == "XX"):
                                        dict_without_np[self.headers[counter]].append(-1)
                                    # if float we need substitute coma with dot
                                    elif  self.types[self.headers[counter]] == float and el.find(',') != -1 :
                                        dict_without_np[self.headers[counter]].append(el.replace(",","."))
                                        
                                    else:
                                        if self.types[self.headers[counter]] == float:
                                            if el == "" or el.isdigit() == 0:
                                                dict_without_np[self.headers[counter]].append(np.NaN) 
                                            else:
                                                dict_without_np[self.headers[counter]].append(el)                                      
                                        else:
                                            dict_without_np[self.headers[counter]].append(el)                                  
                                    counter+=1      
                            else :
                                continue
        
        for key in dict_without_np:
            dict_to_return[key] = np.array(dict_without_np[key],dtype=self.types[key])  
        dict_to_return["region"] = np.array([region for _ in range(len(dict_to_return["p1"]))],dtype=self.types["region"])
        
        return dict_to_return

    ##this function mekes pickle file from given drectory and district -> just for name of file
    def make_pickle_file(self,dict_to_gzip,district):
        file = gzip.open("data/" + self.cache_filename.format(district),"wb")
        pickle.dump(dict_to_gzip,file)
        file.close()
        
    ## this function returns empty list of indexes if district is in memory 
    #  or returns list of indexes where district is in the list  
    def is_in_memory(self,district):
        return np.where(self.dicts_in_memory["region"] == district)

    #this function returns None if district is not in cache
    #if it is in cache it returns dictionary from cache
    def is_in_cache(self,district):
        folder_with_cache = os.getcwd() + "/" + self.folder

        tmp = None
        if  os.path.exists(folder_with_cache) and self.cache_filename.format(district) in  os.listdir(self.folder):
            infile = gzip.open(self.folder + "/" + self.cache_filename.format(district),'rb')
            tmp = pickle.load(infile,encoding='bytes')
            infile.close()
        return tmp
        
    ##this function adds to given dest dictionary data from source dictionary
    #if param indexes is Not none it add data from memory
    #if it is None it means that data are adding from cache and data are also in memory
    #if indexes are [] means that we should also add data to memory
    def append_to_dict(self, dest, source, indexes=[]):
        if indexes == None:
            for key in source:
                dest[key] = np.append(dest[key],source[key])
                #self.dicts_in_memory[key] = np.append(self.dicts_in_memory[key],source[key])
        elif indexes == []:          
            for key in source:
                dest[key] = np.append(dest[key],source[key])
                self.dicts_in_memory[key] = np.append(self.dicts_in_memory[key],source[key])
        else:
            for key in source:
                    dest[key] = np.append(dest[key],[source[key][i] for i in indexes])
        return dest

    def get_dict(self, regions=None):
        my_dict = dict()
        tmp = dict()
        indexes = np.array([])
  
        #initialize dictionary 
        for i in range(0,65):
            my_dict[self.headers[i]] = np.array([])
        
        
        
        if regions == None or regions == []:
            regions = self.regions
        
        
        for district in regions:

            tmp = self.is_in_cache(district)
            indexes = self.is_in_memory(district)
            
            if tmp != None:
                if district in self.dicts_in_memory["region"]:
                    my_dict = self.append_to_dict(my_dict,tmp,None)
                else:
                    my_dict = self.append_to_dict(my_dict,tmp)
            else:
                if np.size(indexes) > 0: 
                    my_dict = self.append_to_dict(my_dict,self.dicts_in_memory,indexes)
                else:
                    tmp = self.parse_region_data(district)
                    self.make_pickle_file(tmp,district)
                    my_dict = self.append_to_dict(my_dict,tmp)
                       
        return my_dict    
        #print(self.dicts_in_memory)
        
def print_info(data):
    print("colums in dataset are are",data.keys())
    print("in each list is: {} records".format(len(data["p1"])))
    print("in dataset are this regions:",np.array(np.unique(data["region"])))

# TODO vypsat zakladni informace pri spusteni python3 download.py (ne pri importu modulu)
if __name__ == '__main__':
    data = DataDownloader().get_dict(["KVK","LBK","VYS"])
    print_info(data)
