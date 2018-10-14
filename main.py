#Project 1 for AI - Deven Damiano - dad152@zips.uakron.edu - Nicholas Horvath - nch16@zips.uakron.edu

import sys
import numpy as np
import pandas as pd
import graphviz
import random
from sklearn import tree
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#-------------------OPTION 1--------------------------

def learn_tree():
    
    running = True
    
    # Additional testing is needed here, trying to cover multiple input methods, user can input a full data table, or individual attributes
    
    while(running):
        
        print("--------------SUBMENU 1--------------")
        print("1: Import a complete dataset\n2: Individual attributes or training sets\n3: Exit")
        choice = input("Enter 1,2, or 3: ")
    
        if(choice == "1"):
        
            file_name = input("Enter the filename of the dataset: ")
            
            try:
                data = pd.read_csv(file_name)
            except:
                print("File not found..")
                return
                
            attribute_list = data.columns
            
            ## EMPTY THIS BEFORE SUBMITION
            #training_list = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness", "duration_ms"]
            
            training_list = []

            train, test = train_test_split(data, test_size = .15)
            
            c = tree.DecisionTreeClassifier(min_samples_split=100)

            # UNCOMMENT BEFORE SUBMITION
            while(True):
                    
                attribute = input("Please input the label names of the columns you would like to train with, q to quit: ")

                if (attribute in attribute_list and attribute not in training_list):
                    training_list.append(attribute)
                            
                elif(attribute == "q"):
                    break
                                    
                else:
                    print("This is not an attribute of the inputed file..")
                    continue
            
            # FOR NOW, USE PRESET
                        
            X_train = train[training_list]
            y_train = train["target"]
                
            X_test = test[training_list]
            y_test = test["target"]
            
            dt = c.fit(X_train, y_train)
            
            #write tree to file
            
            joblib.dump(c, 'tree.joblib')
        
            print("Tree saved as 'tree.joblib'")
        
            #running = False
        
            # Past here is for testing data function
            
            #accuracy
            
            #y_pred = c.predict(X_test)
    
            #score = accuracy_score(y_test, y_pred) * 100
    
            #print("Accuracy: ", score)
            
            #decision matrix
        
            #print(confusion_matrix(y_test, y_pred))
        
            return training_list
    
        elif(choice == "2"):
            
            # Input individual attributes to build a tree from
            
            datasets = []
            
            while(True):

                file_name = input("Please input the file names of the attributes you would like to train with, q to quit: ")
                
                if(file_name == "q"):
                    break
                else:
                    
                    try:
                        data = pd.read_csv(file_name, sep=",", index_col=0)
                        datasets.append(data)
                    except:
                        print("File not found...")
        
            if(datasets):
                
                result = pd.concat(datasets, axis = 1, sort = False)
            
                train, test = train_test_split(data, test_size = 0)
            
                c = tree.DecisionTreeClassifier(min_samples_split=100)
            
                training_list = data.columns
            
                X_train = train[training_list]
                y_train = train["target"]

                #X_test = test[training_list]
                #y_test = test["target"]

                dt = c.fit(X_train, y_train)
            
                joblib.dump(c, 'tree.joblib')
    
                print("Tree saved as 'tree.joblib'")
            
                return training_list
                    
            else:
            
                print("Nothing was entered")

        elif(choice == "3" or choice == "q"):
            break

        else:
            continue

#-----------------OPTION 2----------------------------

def test_accuracy(training_list):
    
    tree = joblib.load('tree.joblib')
    
    print("-------------SUBMENU 2---------------")

    file_name = input("Please enter the file name of testing data: ")

    try:
        data = pd.read_csv(file_name, sep=",", index_col=0)
    except:
        print("File not found...")
        return

    #tree = joblib.load('tree.joblib')

    train, test = train_test_split(data, test_size = .50)

    X_test = test[training_list]
    y_test = test["target"]

    #accuracy

    y_pred = tree.predict(X_test)
    
    score = accuracy_score(y_test, y_pred) * 100
    
    print("Accuracy: ", score)

    #confusion matrix
        
    print(confusion_matrix(y_test, y_pred))

#-------------------OPTION 3--------------------------

def apply_tree(training_list):
    
    #tree = joblib.load('tree.joblib')
    
    print("-------------SUBMENU 3---------------")
    
    training_list.append("target")
    
    df = pd.DataFrame(columns=training_list)
    
    while(True):
        
        #file_name = input("Please enter the filename of the data you would like to add new cases to, enter 'q' to quit: ")
        
        #df = pd.DataFrame(columns=training_list)
        
        choice = input("Please enter 1 to add new case, 2 to quit: ")
        
        if(choice == 'q' or choice == "2"):
            break
        
        else:
            '''try:
                data = pd.read_csv(file_name, sep=",", index_col=0)
            except:
                print("File not found")
                return'''
            
            print("You will now be directed to enter a value for each column..")
            
            row = []
                        
            for i in range(len(training_list)): #This will give us number of columns
                
                print("Please input your value for",training_list[i],": ")
                
                if(training_list[i] == "target"):
                    column_input = random.randint(0,1)
                else:
                    #column_input = input("")
                    column_input = random.random()
                        
                row.append(float(column_input))
                        
                        #print(row)

            #apply this to help with our problem
            
            #df.append(row, ignore_index=True)
            
            df2 = pd.DataFrame([row], columns=training_list)
            
            df = pd.concat([df, df2], axis=0, ignore_index=True)

            print(df)

    #once finished, save csv file with new additions
                
    df.to_csv('updated_data.csv')

#----------------------OPTION 4-----------------------

def load_model():
    
    print("-------------SUBMENU 4---------------")
    
    tree_file = input("Please enter the filename of the tree: ")
    
    try:
        tree = joblib.load(tree_file)
    except:
        print("File not found...")
        return

    #load updated csv
    
    data = pd.read_csv("updated_data.csv", sep=",", index_col=0)

    training_list.pop()

    print(training_list)

    #test tree with new cases

    train, test = train_test_split(data, test_size = .15)

    X_test = test[training_list]
    y_test = test["target"]

    #accuracy

    y_pred = tree.predict(X_test)
    
    score = accuracy_score(y_test, y_pred) * 100
    
    print("Accuracy: ", score)
    
    #confusion matrix
    
    print(confusion_matrix(y_test, y_pred))

#---------------------------------------------

running = True

training_list = []

tree_created = False
new_cases_created = False

while(running):

    print("----------------MAIN MENU------------------")
    print("1. Learn a decision tree and save the tree")
    print("2. Testing accuracy of the decision tree")
    print("3. Applying the decision tree to new cases")
    print("4. Load a tree model and apply to new cases interactively as in menu 3")
    print("5. Quit")

    choice = input("Please enter 1-5 to select operation: ")

    if(choice == "1"):
        training_list = learn_tree()
        tree_created = True
    elif(choice == "2"):
        if(tree_created):
            test_accuracy(training_list)
        else:
            print("You must execute option 1 first..")
    elif(choice == "3"):
        if(tree_created):
            apply_tree(training_list)
            new_cases_created = True
        else:
            print("You must execute option 1 first..")
    elif(choice == "4"):
        if(new_cases_created):
            load_model()
        else:
            print("You must execute option 3 first..")
    elif(choice == "5" or choice == "q"):
        print("Aborting..")
        exit()
    else:
        print("invalid input")


