# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:56:51 2020

@author: cucusenok
"""
import glob


class File:
    def __init__(self, path):
        self._path = path
        
    def getContent(self):
        content = ''
        with open(self._path, 'r') as content_file:
            content = content_file.read()
        
        return content
