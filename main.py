#Project 1 for AI - Deven Damiano - dad152@zips.uakron.edu - Nicholas Horvath - nch16@zips.uakron.edu

import sys
import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle

def learn_tree():
    
    running = True
    
    # Additional testing is needed here, trying to cover multiple input methods, user can input a full data table, or individual attributes
    
    while(running):
        
        print("--------------SUBMENU 1--------------")
        print("1: Import a complete dataset\n2: Individual attributes or training sets\n3: Exit")
        choice = input("Enter 1,2, or 3: ")
    
        if(choice == "1"):
        
            file_name = input("Enter the filename of the dataset: ")
            data = pd.read_csv(file_name)
                
            attribute_list = data.columns
            
            ## EMPTY THIS BEFORE SUBMITION
            training_list = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness", "duration_ms"]

            train, test = train_test_split(data, test_size = .15)
            
            c = tree.DecisionTreeClassifier(min_samples_split=100)

            # UNCOMMENT BEFORE SUBMITION
            '''while(True):
                    
                attribute = input("Please input the names of the attributes you would like to train with, q to quit: ")

                if (attribute in attribute_list and attribute not in training_list):
                    training_list.append(attribute)
                            
                elif(attribute == "q"):
                    break
                                    
                else:
                    print("This is not an attribut of the inputed")
                    continue'''
            
            # FOR NOW, USE PRESET
                        
            X_train = train[training_list]
            y_train = train["target"]
                
            X_test = test[training_list]
            y_test = test["target"]
            
            dt = c.fit(X_train, y_train)
            
            #write tree to file
            
            joblib.dump(c, 'tree.joblib')
        
            print("Tree saved as 'tree.joblib'")
    
        elif(choice == "2"):
            
            # Input individual attributes to build a tree from
            
            while(True):

                attribute = input("Please input the names of the attributes you would like to train with, q to quit: ")

                if (attribute in attribute_list and attribute not in training_list):
                    training_list.append(attribute)
    
                elif(attribute == "q"):
                    break

                else:
                    print("This is not an attribut of the inputed")

        elif(choice == "3" or choice == "q"):
            break

        else:
            continue

def test_accuracy():
    
    tree = joblib.load('tree.joblib')
    
    print("-------------SUBMENU 2---------------")

    fileName = input("Please enter the file name of testing data: ")

    #validate and open file
    #test data with tree
    #output confusion matrix

def apply_tree():
    
    tree = joblib.load('tree.joblib')
    
    print("-------------SUBMENU 3---------------")
    
    while(True):

        attribute = input("Please enter values of new condition attributes, enter 'q' to quit: ")

        if(input == "q"):

            break

        else:
            
            #add new cases to dataset
            continue


def load_model():
    
    print("-------------SUBMENU 4---------------")

    fileName = input("Please enter the file name for the tree: ")

    tree = joblib.load(input)

    #test tree with new cases

def select_option(choice):
    
    if(choice == "1"):
        learn_tree()
    elif(choice == "2"):
        test_accuracy()
    elif(choice == "3"):
        apply_tree()
    elif(choice == "4"):
        load_model()
    elif(choice == "5" or choice == "q"):
        print("Aborting..")
        exit()
    else:
        print("invalid input")


running = True

while(running):

    print("----------------MAIN MENU------------------")
    print("1. Learn a decision tree and save the tree")
    print("2. Testing accuracy of the decision tree")
    print("3. Applying the decision tree to new cases")
    print("4. Load a tree model and apply to new cases interactively as in menu 3")
    print("5. Quit")

    choice = input("Please enter 1-5 to select operation: ")

    select_option(choice)


