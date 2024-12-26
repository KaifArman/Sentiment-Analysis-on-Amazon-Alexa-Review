**_Sentiment-Analysis-on-Amazon-Alexa-Review_**

**Table of Contents**
1. [Project Overview](#project-overview)
2. [Key Steps](#key-steps)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Engineering](#feature-engineering)
   - [Text Vectorization](#text-vectorization)
   - [Data Splitting](#data-splitting)
   - [Scaling](#scaling)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Results](#results)
6. [Model Saving](#model-saving)
7. [File Structure](#file-structure)
8. [Technologies Used](#technologies-used)
9. [How to Run](#how-to-run)
10. [Future Scope](#future-scope)
11. [Conclusions](#conclusions)

### **Project Overview:**

This project aims to predict the sentiment of Amazon Alexa reviews (positive or negative) based on customer feedback. The reviews are preprocessed, feature engineering is performed, and machine learning models are used to classify the sentiment. The project uses natural language processing (NLP) techniques such as text cleaning, tokenization, stemming, and vectorization to build a sentiment analysis pipeline.

### **Key Steps:**

#### **Data Preprocessing:**
- Removed non-alphabetical characters from reviews and converted them to lowercase.
- Tokenized reviews by splitting them into words.
- Stopped words were removed, and stemming was applied to each word to reduce them to their root form.

#### **Feature Engineering:**
- A new column, **'length,'** was added to store the length of each review.

#### **Text Vectorization:**
- The **`CountVectorizer`** was used to convert the text into a **Bag of Words** representation with a maximum of 2500 features.

#### **Data Splitting:**
- The dataset was split into a training set (70%) and a testing set (30%).

#### **Scaling:**
- The feature values were scaled between 0 and 1 using the **`MinMaxScaler`** to improve model performance.

### **Model Training:**

- **Random Forest Classifier:** Trained on the scaled data, achieving a high training accuracy (99.4%) and testing accuracy (94.2%).
- **XGBoost Classifier:** Trained with excellent accuracy (97.0% on training, 94.1% on testing).
- **Decision Tree Classifier:** Achieved a training accuracy of 99.4%, but a slightly lower testing accuracy of 91.0%.

### **Model Evaluation:**

- Accuracy was measured for each model on both the training and testing datasets.
- **Confusion matrices** were used to assess the performance of each model.
- **K-fold cross-validation** was applied, with the **Random Forest** model achieving a mean accuracy of 93.6% and a low standard deviation.
- A **Grid Search** was applied to fine-tune the **Random Forest** model's hyperparameters, achieving the best parameter combination for better accuracy.

### **Results:**
- The models show high performance, with **Random Forest** and **XGBoost** yielding excellent results. The accuracy for the test set is over 90% for both models.
- The **confusion matrices** confirm the models' ability to correctly classify the majority of reviews.

### **Model Saving:**
- The trained models (**Random Forest, XGBoost, Decision Tree**) and preprocessing objects (**CountVectorizer, Scaler**) were saved using **pickle** for future use.

### **File Structure:**
- **`amazon_alexa.tsv`**: Input dataset containing Amazon Alexa reviews.
- **`Models/`**: Directory containing saved models and vectorizers.
  - **`countVectorizer.pkl`**: Saved Count Vectorizer.
  - **`scaler.pkl`**: Saved Scaler.
  - **`model_xgb.pkl`**: Saved XGBoost Classifier model.
  - **`model_rf.pkl`**: Saved Random Forest Classifier model.
  - **`model_dt.pkl`**: Saved Decision Tree Classifier model.

### **Technologies Used:**
- **Python**
- **pandas**, **numpy**, **re** (for text preprocessing)
- **Scikit-learn** (for machine learning and model evaluation)
- **XGBoost** (for classification)
- **Matplotlib** (for plotting confusion matrices)
- **Pickle** (for saving models)

### **How to Run:**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/KaifArman/Sentiment-Analysis-on-Amazon-Alexa-Review.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd Sentiment-Analysis-on-Amazon-Alexa-Review
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook Sentiment_Analysis_on_Amazon_Alexa_Review.ipynb
   ```

This will align with your project folder name, **Sentiment-Analysis-on-Amazon-Alexa-Review**.

### **Future Scope:**
- Incorporate **deep learning models** like **LSTMs** and **transformers** for better accuracy.
- Expand the dataset to include reviews from other Amazon products.
- Develop a **web application** for real-time sentiment analysis.

### **Conclusions:**
The project demonstrates how natural language processing can be used to analyze customer feedback effectively. It provides actionable insights for businesses to improve their products and services based on customer sentiment.
