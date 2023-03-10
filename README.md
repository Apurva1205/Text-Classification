# Text-Classification





Import necessary libraries: This code imports the required libraries such as pandas, numpy, torch, transformers, and scikit-learn.

Load dataset: This code loads the dataset from a CSV file using pandas and checks for missing values. Rows with missing values are dropped.

Preprocess the text data: The text data is preprocessed by performing cleaning, removing stopwords, punctuations, etc. This step is not shown in the code and is left to the user to implement.

Split the dataset into train, validation, and test sets: The dataset is split into three sets, train, validation, and test, using the train_test_split function from scikit-learn.

Load pre-trained model and tokenizer: The pre-trained BERT model and tokenizer are loaded from the Transformers library.

Tokenize the input text: The text data is tokenized using the BERT tokenizer. The train, validation, and test datasets are tokenized separately, and the new dataset containing a new text to classify is also tokenized.

Create PyTorch datasets: PyTorch datasets are created for the train, validation, and test datasets using the CustomDataset class defined in the code. This class converts the tokenized data into PyTorch tensors and creates a PyTorch dataset.

Fine-tune the pre-trained model for text classification: The pre-trained BERT model is fine-tuned for text classification using the train dataset and evaluated using the validation dataset. The Trainer class from Transformers library is used to train the model.

Evaluate the model using the test set: The fine-tuned model is evaluated using the test dataset, and the metrics such as accuracy, precision, recall, and F1 score are calculated using the compute_metrics function defined in the code.


