# Twitter-Sentiment-Analysis
This project involves building a machine learning model to perform sentiment analysis on Twitter data. The goal of the project is to classify tweets into three categories: Positive, Neutral, and Negative sentiments. The dataset used for training and testing the model is sourced from publicly available Twitter data.

## Project Structure:

### Data Preprocessing:

Cleaned and preprocessed the text data, including tasks like removing special characters, stopwords, and tokenization.
Transformed the labels into categorical values using a mapping function.

### Model Development:

Employed a sequential neural network model built with TensorFlow and Keras to classify the sentiments.
The model consists of embedding layers, LSTM layers, and dense layers with a softmax activation function for multi-class classification.
Implemented early stopping to prevent overfitting during training.

### Training and Evaluation:

Trained the model on the preprocessed dataset, with an 80-20 split for training and validation.
Used various evaluation metrics such as accuracy, precision, recall, and F1-score to assess the model's performance.

### Results:

The model successfully classifies tweets into the respective sentiment categories with a high degree of accuracy.
A confusion matrix and classification report are generated to visualize and summarize the model's performance.

### How to Run the Project:
Clone the repository to your local machine.
Ensure you have the necessary dependencies installed, which can be found in the requirements.txt file.
Run the Jupyter Notebook or Python script to preprocess the data, train the model, and evaluate the results.

Technologies Used:
Programming Language: Python
Libraries: TensorFlow, Keras, Pandas, NumPy, Scikit-learn
Model: LSTM (Long Short-Term Memory) Neural Network
Tools: Jupyter Notebook
