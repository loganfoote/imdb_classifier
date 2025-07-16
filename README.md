# IMDb Review Sentiment Classifier 
This project provides classical ML and deep-learning models for classifying IMDb reviews by negative or positive sentiment. We start with data loading and exploratory analysis, followed by text vectorization and training using several classical ML models. 

## Dataset 
The dataset used is the [IMDb Large Movie Review Dataset](http://www.aclweb.org/anthology/P11-1015), which contains 25,000 training and 25,000 testing reviews.

## Classical ML Results
Test accuracies obtained with classical ML models are:

| Model                        | Accuracy (%) |
|------------------------------|--------------|
| Logistic Regression           | ~88          |
| Random Forest                | ~85          |
| Naive Bayes                  | ~85          |
| Support Vector Machine (SVM) | ~88          |

These results align with expected baselines on this dataset.

## Deep-Learning Results 
An LSTM model is implemented using PyTorch. With fine-tuning of parameters, the model achieves 82% accuracy on the test dataset, and begins to overfit after a few epochs. These results align with expectations for an LSTM model. The next step is to implement a different model (BERT, etc) which should be able to surpass 90% accuracy for this type of problem.

## Getting Started 
### Install dependences:
```bash
pip install -r requirements.txt 
```

### Install package
```bash
pip install .
```

### Download dataset 
Download the [IMDb Large Movie Review Dataset](http://www.aclweb.org/anthology/P11-1015) and the [GloVe](https://nlp.stanford.edu/projects/glove/) embedding weights and place them in imdb_classifier/data/.

## Usage 
Run the Jupyter notebook `classical_ml.ipynb` to reproduce the classical ML workflow. Run the Jupyter notebook `dl.ipynb` for reproduce the deep-learning workflow.
