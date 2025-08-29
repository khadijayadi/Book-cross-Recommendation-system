# Overview

This project implements a Book Recommendation System using the popular Book-Crossing dataset (Kaggle).
The system explores Collaborative Filtering (user-based & item-based), Content-Based Filtering, and a Hybrid approach, comparing their performance using RMSE as an evaluation metric.

The complete analysis and results are documented in the full report:
📄 book cross - recommendation system.docx

# Project Objectives

- Help users discover books they are likely to enjoy.
- Compare different recommendation techniques on real-world data.
- Evaluate performance using metrics and visualization to identify the most accurate model.

# Dataset

The project uses the Book-Crossing Dataset consisting of three files:

- Books.csv – Book metadata (title, author, publisher, year).
- Users.csv – User demographics (age, location).
- Ratings.csv – User ratings (0–10 scale).

# Methodology

# Data Preprocessing

- Cleaned missing values and inconsistent data.
- Removed implicit (0) ratings.
- Merged datasets (Books + Users + Ratings).

# Exploratory Data Analysis (EDA)

- Distribution of ratings.
- Age-based rating behavior.
- Top active users and their influence.

# Models Implemented

- Item-Based Collaborative Filtering (IBCF) → Best performing model (RMSE ≈ 1.75).
- User-Based Collaborative Filtering (UBCF) → RMSE ≈ 1.92.
- Content-Based Filtering (CBF) → RMSE ≈ 1.95.

# Results

📊 Best Model: Item-Based Collaborative Filtering outperformed others with lowest RMSE.

> Users tend to rate generously (skewed distribution towards 7–10).
> Age and active “power users” significantly influence rating behavior.

# Tech Stack

- Python (pandas, numpy, scikit-learn, scipy, difflib)
- Visualization: matplotlib, seaborn
- Text Processing: TF-IDF Vectorization

# Key Takeaways

> Recommendation systems are crucial for personalizing user experience.
> Hybrid approaches combining CF & CBF can enhance performance.
> Data preprocessing and bias handling are critical for better predictions.
