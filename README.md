# Naive Bayes Email Spam Classifier

## Project Overview

This repository contains the implementation of a Naive Bayes email spam classifier. The project focuses on exploring different configurations and their impact on the performance of the classifier.


## Design Explanation

#### Parsing
- The method parses a directory, opening each file and saving the email content as a string in the list 'X'.
- Emails are classified as Ham (0) or Spam (1) based on the file name, with the classifications stored in the list 'Y'.

#### Training
- Count the number of occurrences of each word according to its class (HAM or SPAM) and save it in a dictionary where each word is the key in lowercase.
- Words separated by punctuation or ending with it are saved without them to prevent spammers from manipulating the classification.
- For example: `ham_fd[The]` ⟷ `ham_fd[the]`, `spam_fd[World]` ⟷ `spam_fd[world]`.

#### Testing
- The basic idea of testing is to check how many times each word in an email appeared as HAM and SPAM in our model, then apply a formula to calculate its probability of being HAM or SPAM.
- The formula has two cases:
  - Log case: Apply Log to the probability of each word and then add it to other words. Otherwise, multiply probabilities.
  - Smoothing case: Add 1 to each word’s occurrences. Otherwise, add epsilon to avoid 0 probabilities, which is not acceptable by Logs.
- Also, apply lowercase and remove punctuations from words to match the training model.

#### F1-score
- The method calculates F1-score for our prediction to the test set using Precision and Recall.
  - Precision: Counting True Positive (how many we predicted to be true, and it is true) and dividing it by True Postive + False Positive, indicating how many did it catch.
  - Recall: Similar to Precision but divide it by True Postive + False Negative, indicating how many did it miss.
- Lastly, apply the F1-score formula.

## Experiment Results

| Configuration                    | F1-score | F1-score in % |
|-----------------------------------|----------|---------------|
| Log=False, Smoothing=False        | 0.645    | 64.5          |
| Log=True, Smoothing=False         | 0.655    | 65.5          |
| Log=False, Smoothing=True         | 0.964    | 96.4          |
| Log=True, Smoothing=True          | 0.975    | 97.5          |

