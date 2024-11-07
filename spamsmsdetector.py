import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    sms = pd.read_csv(file_path, encoding='latin-1')
    sms = sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    sms = sms.rename(columns={"v1": "label", "v2": "text"})
    sms['label'] = sms.label.map({'ham': 0, 'spam': 1})
    return sms

# Function to train and evaluate the model
def train_and_evaluate_model(sms_data):
    # Vectorize the text data
    count = CountVectorizer()
    text = count.fit_transform(sms_data['text'])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(text, sms_data['label'], test_size=0.20, random_state=1)

    # Train an MLP Classifier
    model = MLPClassifier()
    model.fit(x_train, y_train)

    # Make predictions on the test set
    prediction = model.predict(x_test)

    # Return the model and count vectorizer
    return model, count

# Function to make predictions and print the result
def predict_and_print_result(model, count, input_text):
    input_vector = count.transform(input_text)
    prediction = model.predict(input_vector)
    return prediction

# Main code
if __name__ == "__main__":
    file_path = 'C:/Users/acer/Desktop/CODSOFT/Spam Sms Detection/spam.csv'
    sms_data = load_and_preprocess_data(file_path)
    trained_model, count = train_and_evaluate_model(sms_data)

    input_texts = ["I HAVE A DATE ON SUNDAY WITH WILL!!"]

    for input_text in input_texts:
        prediction = predict_and_print_result(trained_model, count, [input_text])
        if prediction[0] == 1:
            print("Spam message:", input_text)
        else:
            print("Not a spam message:", input_text)
