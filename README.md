# Movie-Rating

This project aims to utilize Keras to build a model for predicting movie ratings, utilizing UserID and MovieID as features, with the goal of predicting ratings (Rating).

The workflow involves reading the data into the program, randomly splitting the data into a training set (80%) and a test set (20%). Subsequently, we will construct a matrix factorization model, which will be employed to predict ratings on the test set. Finally, we will compute the Mean Absolute Error (MAE) of the predicted results to evaluate the predictive capability of the model.

Dataset :The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data?select=ratings.csv
