#Project 1 for AI - Deven Damiano - dad152@zips.uakron.edu

import sys
from sklearn import tree

#X = [[0, 0], [1, 1]]
#Y = [0, 1]
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X, Y)
#clf.predict([[2., 2.]])

def learn_tree():

    print("Learning a decision tree and saving the tree")

def test_accuracy():

    print("Testing accuracy of the decision tree")

def apply_tree():

    print("Applying the decision tree to new cases")

def load_model():

    print("Loading a tree model and apply to new cases interactively as in menu 3")

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


