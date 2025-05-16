📊 Predictive Movie Analysis – Binary Classification of Ratings 🎬
This repository contains a binary classification model to predict movie ratings (1 for "good" and 0 for "bad") based on an IMDB dataset. The analysis uses AdaBoost and Gradient Boosting algorithms. The code also includes data preprocessing, exploratory analysis, and model evaluation.

🚀 Code Structure
🔧 Library Imports
Libraries such as pandas, numpy, matplotlib, and seaborn are used for data manipulation, visualization, and predictive modeling.

scikit-learn is used for building and evaluating machine learning models.

📂 Data Source
The dataset used in this project is publicly available and can be accessed at:
https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data/data

📂 Data Loading and Cleaning
The IMDB dataset is loaded from a CSV file.

Missing values in the Revenue (Millions) column are filled with the median, and those in the Metascore column are filled with the mean.

Unnecessary columns are removed, and multi-genre movies are encoded using dummy variables.

📈 Exploratory Data Analysis
Correlation heatmaps are generated to visualize relationships between features and the target variable (Binary_Rating).

Detailed reports are created using the pandas_profiling and sweetviz libraries.

🤖 Model Building
The code implements AdaBoostClassifier and GradientBoostingClassifier to perform binary classification.

The data is split into training and testing sets in a 70%/30% ratio.

Model performance is evaluated using metrics such as Accuracy, AUC (AUROC), Precision, Recall, and F1-score.

📊 Performance Evaluation
Performance reports are generated for both models based on the metrics listed above.

Feature importance is analyzed for the Gradient Boosting model.

📝 How to Use
📋 Requirements
To run this code, you need a Python environment with the following libraries installed:

numpy

pandas

matplotlib

seaborn

scikit-learn

pandas_profiling

sweetviz

preditiva (Custom library for performance metrics and reporting)

Install dependencies with:


pip install -r requirements.txt
▶️ Execution
After installing the dependencies, simply run the Python code to load the dataset, perform exploratory analysis, train the models, and evaluate their performance.

Results will be shown through visualizations 📈 and metrics 📊.

⚙️ Customization
You can tweak model hyperparameters such as n_estimators and learning_rate to optimize results.

It's also possible to add or remove explanatory variables as needed.

🔍 Notes
The dataset includes movie information such as Rating, Votes, and Revenue (Millions). The target for binary classification is the Rating, transformed into Binary_Rating:

1 for movies with a rating ≥ 7

0 for movies with a rating < 7

This analysis can be easily adapted to other movie datasets or different binary classification problems.