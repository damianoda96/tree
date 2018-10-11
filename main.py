#Project 1 for AI - Deven Damiano - dad152@zips.uakron.edu - Nicholas Horvath - nch16@zips.uakron.edu

import sys
import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

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
                    
                attribute = input("Please input the label names of the columns you would like to train with, q to quit: ")

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
        
            #running = False
        
            # Past here is for testing data function
            
            #accuracy
            
            y_pred = c.predict(X_test)
    
            score = accuracy_score(y_test, y_pred) * 100
    
            print("Accuracy: ", score)
            
            #decision matrix
        
            print(confusion_matrix(y_test, y_pred))
        
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

def test_accuracy(training_list):
    
    tree = joblib.load('tree.joblib')
    
    print("-------------SUBMENU 2---------------")

    file_name = input("Please enter the file name of testing data: ")

    try:
        data = pd.read_csv(file_name, sep=",", index_col=0)
    except:
        print("File not found...")
        return

    tree = joblib.load('tree.joblib')

    train, test = train_test_split(data, test_size = .50)

    X_test = test[training_list]
    y_test = test["target"]

    #accuracy

    y_pred = tree.predict(X_test)
    
    score = accuracy_score(y_test, y_pred) * 100
    
    print("Accuracy: ", score)

    #confusion matrix
        
    print(confusion_matrix(y_test, y_pred))

def apply_tree():
    
    #tree = joblib.load('tree.joblib')
    
    print("-------------SUBMENU 3---------------")
    
    while(True):
        
        file_name = input("Please enter the filename of the data you would like to add new cases to, enter 'q' to quit: ")
        
        if(input == "q"):
            break
        
        else:
        
            try:
                data = pd.read_csv(file_name, sep=",", index_col=0)
                
                while(True):
                
                    if(input == "q"):
                    
                        break
            
                    else:
                        
                        row = []

                        data_columns = data.columns
                        
                        for i in data_columns: #This will give us number of columns
                            
                            column_input = input("Please input your value for " + str(data_columns[i]) + ": ")
                        
                            row.append(column_input)
                        
                        print(row)
                        
                        
                        #apply this to help with our problem
                        #df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
                        
                        #loop to input each column item using length
                            
                        #value = input("")
                    
                        #add new record here
                    
                        #input("Please enter new values for 'column name', q to quit: ")
                        
                
                    

                
            except:
                print("File not found...")

    #data.to_csv('updated_data.csv')

    #once finished, save csv file with new additions

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

    #test tree with new cases

    #output accuracy and confusion matrix

def select_option(choice):
    
    if(choice == "1"):
        training_list = learn_tree()
    elif(choice == "2"):
        test_accuracy(training_list)
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

training_list = []

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
    elif(choice == "2"):
        test_accuracy(training_list)
    elif(choice == "3"):
        apply_tree()
    elif(choice == "4"):
        load_model()
    elif(choice == "5" or choice == "q"):
        print("Aborting..")
        exit()
    else:
        print("invalid input")

#select_option(choice)


