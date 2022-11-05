# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#df['col1'] = df['col1'].fillna(df['col1'].mode()[0])

#Load dataset
df = pd.read_csv(r"C:\Users\Anudnya\Desktop\Sem7\ROSPL\EndSem\Website\Final_Recipe.csv")
df.head()
#df.drop("ID",inplace=True,axis=1)

df['Vegies'] = df['Vegies'].str.replace(" ","")
vals_to_replace_1 = {'Potato':1,'Okra':2,'coconut':3,'blackbeans':4,'Jalepeno': 5,'chickpeas': 6,'cucumber':7,'cabbage':8,'cauliflower':9,'avocadoes':10,'butternutsquash':11,'shallot':12,'chinesebroccoli':13,'parsnips':14,'greenpeppers':15,'waterspinach':16,'chanadal':17,'redgrapefruit':18,'tomatillos':19,'uraddal':20,'chickpeas':21,'Coriander':22}
df['Vegies'] = df['Vegies'].map(vals_to_replace_1)
df['Vegies'].fillna(int(df['Vegies'].mean()), inplace=True)


df['Vegies2'] = df['Vegies2'].str.replace(" ","")
vals_to_replace_2 = {'eggplant' : 1, 'Onion' : 2, 'carrot' : 3, 'celery':4,'frozencranberrries':5}
df['Vegies2'] = df['Vegies2'].map(vals_to_replace_2)
df['Vegies2'].fillna(int(df['Vegies2'].mean()), inplace=True)


df['Vegies3'] = df['Vegies3'].str.replace(" ","")
vals_to_replace_3 = {'Peas':1,'habaneropeppers':2,'tomato':3,'lemon':4,'cilantro':5,'rasberries':6,'yellowtomato':7,'yellowandbrownmustardseeds':8}
df['Vegies3'] = df['Vegies3'].map(vals_to_replace_3)
df['Vegies3'].fillna(int(df['Vegies3'].mean()), inplace=True)

df['Flour/Bread'] = df['Flour/Bread'].str.replace(" ","")
vals_to_replace_4 = {'all-purposeflour':1,'flour':2,'cornstarch':3,'wheatflour':4}
df['Flour/Bread'] = df['Flour/Bread'].map(vals_to_replace_4)
df['Flour/Bread'].fillna(int(df['Flour/Bread'].mean()), inplace=True)


df['Flour/Bread2'] = df['Flour/Bread2'].str.replace(" ","")
vals_to_replace_5 = {'breadflour':1,'breadcrumbs':2,'bread':3,'breadcrumbs':4}
df['Flour/Bread2'] = df['Flour/Bread2'].map(vals_to_replace_5)
df['Flour/Bread2'].fillna(int(df['Flour/Bread2'].mean()), inplace=True)


df['Sauces'] = df['Sauces'].str.replace(" ","")
vals_to_replace_6 = {'tahini':1,'maltsyrup':2,'tomatosauce':3,'gochujang':4,'fishsauce':5,'maplesyrup':6,'thairedcurrypaste':7,'hotsauce':8,'whitevinegar':9,'oystersauce':10,'Mayo':11}
df['Sauces'] = df['Sauces'].map(vals_to_replace_6)
df['Sauces'].fillna(int(df['Sauces'].mean()), inplace=True)


df['Sauces2'] = df['Sauces2'].str.replace(" ","")
vals_to_replace_7 = {'liquidsmoke':1,'applecidervinegar':2,'soysauce':3}
df['Sauces2'] = df['Sauces2'].map(vals_to_replace_7)
df['Sauces2'].fillna(int(df['Sauces2'].mean()), inplace=True)


df['Sauces3'] = df['Sauces3'].str.replace(" ","")
vals_to_replace_8 = {'vinegar':1,'ricewinevinegar':2,'sauce':3}
df['Sauces3'] = df['Sauces3'].map(vals_to_replace_8)
df['Sauces3'].fillna(int(df['Sauces3'].mean()), inplace=True)


df['Meat'] = df['Meat'].str.replace(" ","")
vals_to_replace_9 = {'beef':1,'steak':2,'bonelessporkbutt':3,'chicken':4,'chickenbreasts':5,'pork':6,'lard':7,'ribs':8,'driedshrimp':9,'salami':10,'wholeturkey':11,'bonelesschickenthighs':12}
df['Meat'] = df['Meat'].map(vals_to_replace_9)
df['Meat'].fillna(int(df['Meat'].mean()), inplace=True)

df['Meat2'] = df['Meat2'].str.replace(" ","")
vals_to_replace_10 = {'lamb':1,'eggs':2,'bonelessporkshoulder':3,'egg':2}
df['Meat2'] = df['Meat2'].map(vals_to_replace_10)
df['Meat2'].fillna(int(df['Meat2'].mean()), inplace=True)


df['Rice/Noodles'] = df['Rice/Noodles'].str.replace(" ","")
vals_to_replace_11 = {'taco':1,'rice':2,'tortillas':3,'basmatirice':4,'stickyrice':5,'macaroni':6,'noodles':7,'ricenoodles':8,'longrice':9,'oldjasminerice':10}
df['Rice/Noodles'] = df['Rice/Noodles'].map(vals_to_replace_11)
df['Rice/Noodles'].fillna(int(df['Rice/Noodles'].mean()), inplace=True)


df['Spices'] = df['Spices'].str.replace(" ","")
vals_to_replace_12 = {'biryanimasala':1,'blackpeppercorns':2,'curryleaves':3,'gramsgroundcannabis':4,'poblanopeppers':5,'guagillochiles':6,'garlic':7,'Greekseasoning':8,'bayleaves':9,'freshthyme':10,'peppercons':11,'blackpepper':12,'greenchillies':13,'Ginger':14,}
df['Spices'] = df['Spices'].map(vals_to_replace_12)
df['Spices'].fillna(int(df['Spices'].mean()), inplace=True)


df['Spices2'] = df['Spices2'].str.replace(" ","")
vals_to_replace_13 = {'anchochiles':1,'groundcaraway':2,'NewMexicochiles':3,'serranochiles':4,'chilesdeárbol':5,'seasmeseeds':6,'driedchilles':7,'chilesmoritas':8,'Chillipepper':9}
df['Spices2'] = df['Spices2'].map(vals_to_replace_13)
df['Spices2'].fillna(int(df['Spices2'].mean()), inplace=True)


df['Spices3'] = df['Spices3'].str.replace(" ","")
vals_to_replace_14 = {'pasillachiles':1,'arbolchiles':2,'redchilies':3,'salicion':4}
df['Spices3'] = df['Spices3'].map(vals_to_replace_14)
df['Spices3'].fillna(int(df['Spices3'].mean()), inplace=True)


df['Other'] = df['Other'].str.replace(" ","")
vals_to_replace_15 = {'vegetablestock':1,'sugar':2,'yogurt':3,'Tomatillosalsa':4,'coconutmilk':5,'Cheesecloth':6,'ghee':7,'oil':8,'cheese':9,'darkbrownsugar':10,'mozarellacheese':11,'driedblackbeans':12,'vegetableoil':13,'tofu':14,'granulatedsugar':15,'oliveoil':16,'whitebeans':17,'pecan':18,'bakingsoda':19,'brownsugar':20,'chickenorvegetablestock':21,'honey':22,'whitesugar':23}
df['Other'] = df['Other'].map(vals_to_replace_15)
df['Other'].fillna(int(df['Other'].mean()), inplace=True)


df['Other2'] = df['Other2'].str.replace(" ","")
vals_to_replace_16 = {'chickenstock':1,'butter':2,'lemonjuice':3,'buttersquash':4,'mustardseeds':5,'yeast':6,'longbeans':7,'unsaltedbuttermelted':8,'coldbeer':9}
df['Other2'] = df['Other2'].map(vals_to_replace_16)
df['Other2'].fillna(int(df['Other2'].mean()), inplace=True)


df['Diabetic'] = df['Diabetic'].str.replace(" ","")
vals_to_replace_17 = {'Yes':1,'No':2}
df['Diabetic'] = df['Diabetic'].map(vals_to_replace_17)
df['Diabetic'].fillna(int(df['Diabetic'].mean()), inplace=True)
# df.head(91)

# # Import label encoder 
# from sklearn import preprocessing
# # label_encoder object knows how to understand word labels. 
print(df.head())

from sklearn.model_selection import train_test_split
trainLabel = np.asarray(df['Target'])
trainData = np.asarray(df.drop('Target',1))
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(trainData, trainLabel, test_size=0.3, random_state=7) # 70% training and 30% test

from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

print(y_pred)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


'''
from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 15)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=1)
#fit model to data
knn_gscv.fit(trainData, trainLabel)
#check top performing n_neighbors value
print(knn_gscv.best_params_)
#check mean score for the top performing value of n_neighbors
print(knn_gscv.best_score_)
'''
# import pickle
# # Saving model to current directory
# # Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
# pickle.dump(knn, open('model.pkl','wb'))

# #Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2,1,1,1,3,3]]))

import pickle
# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(knn, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,1,1,1,3,3,2,1,1,1,3,3,1,1,1,1,1]]))