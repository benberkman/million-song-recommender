# Million Song, Million User Recommendation System

NYU DS-GA 1004: Big Data term project.

Abstract: We build and evaluate a recommender system on a dataset of one million songs and one million user interactions (as well as metadata, audio features, and more). Given the nature of play count data, we utilize PySpark’s implicit feedback model to train the model and perform hyperparameter tuning on the rank and regularization of the latent factors, maximum iterations of the Alternating Least Squares model, scaling parameter of the counts, and whether to use nonnegative constraints for least squares. Once we establish an initial ALS model, we implement two extensions: a popularity-based baseline model and a visualization of the items and users using the UMAP algorithm. We conclude with a discussion on next steps.
