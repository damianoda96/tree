#Project 1 for AI - Deven Damiano - dad152@zips.uakron.edu - Nicholas Horvath - nch16@zips.uakron.edu

import sys
import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

def learn_tree():
    
    #Here is an example based off of a small dataset
    
    features = np.array([
        [29,23,72],
        [31,25,77],
        [31,27,82],
        [29,29,89],
        [31,31,72],
        [29,33,77],
    ])
   
    labels = np.array([
        [0],
        [1],
        [1],
        [0],
        [1],
        [0],
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
    )

    clf = tree.DecisionTreeClassifier()
    clf.fit(X=X_train,y=y_train)
    clf.feature_importances_ # [1., 0., 0.]
    clf.score(X=X_test, y=y_test) #1.0


    dot_data = tree.export_graphviz(

        clf,
        out_file=None,
        feature_names=["minutes", "age", "height"],
        class_names=["score_equal_to_or_over_20", "scored_under_20"],
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)

    graph.render('tree.gv', view=True)
    

    #print("Learning a decision tree and saving the tree")
    
    while(True):

        fileName = input("Please enter file names of attributes and training examples, press q to quit: ")

        if(input == "q"):
            
            break
        
        else:

            #open file here
            #create tree
            #export tree to file

            continue

def test_accuracy():

    fileName = input("Please enter the file name of testing data: ")

    #validate and open file
    #test data with tree
    #output confusion matrix

def apply_tree():
    
    while(True):

        attribute = input("Please enter values of new condition attributes, enter 'q' to quit: ")

        if(input == "q"):

            break

        else:
            
            #add new cases to dataset
            continue


def load_model():

    fileName = input("Please enter the file name for the tree: ")

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
    elif(choice == "5"):
        print("Aborting..")
        exit()
    else:
        print("invalid input")


running = True

while(running):

    print("MENU:")
    print("1. Learn a decision tree and save the tree")
    print("2. Testing accuracy of the decision tree")
    print("3. Applying the decision tree to new cases")
    print("4. Load a tree model and apply to new cases interactively as in menu 3")
    print("5. Quit")

    choice = input("Please enter 1-5 to select operation: ")

    select_option(choice)


