Project Overview
-------------------

This is my first Kaggle experience, I have learned a lot during this competition. The submissions are evaluated on the mean column-wise ROC AUC. Our team achieved a private score of 0.9857, ranking top 26% among the 4551 teams worldwide on leaderboard. This is a really tough competition, as admitted by most teams, including many Kaggle Masters.

## Kaggle Link

[Toxic Comment Classification Challenge - Identify and classify toxic online comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Background

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).

In this competition, we are challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. I will be using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

## Data Description

I am provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

    toxic

    severe_toxic

    obscene

    threat

    insult

    identity_hate

## Objective

Create a model which predicts a probability of each type of toxicity for each comment.

## Evaluation Metric

Submissions are evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.

## Data Installation

[Training set, test set, and sample submission](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

[Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

You can download the packages used in this project by typing 'pip install -r requirements.txt'

Note:

train.csv - the training set, contains comments with their binary labels

test.csv - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.

sample_submission.csv - a sample submission file in the correct format


## Data Understanding

The training set has 159571 rows and testing set has 153164 rows. The training set has 15294 'toxic' comments, 1595 'severe toxic' comments, 8449 'obscene' comments, 478 'threat' comments, 7877 'insult' comments, 1405 'identity_hate', and 143346 comments without any tags.

We can see that the labels are not evenly distributed across classes. The number of comments without any tags far outweigh any classes. So this is an severely unbalanced problem. Also, we can notice that the sum of all the comments with and without tags (i.e. 178444) is larger than the size of the training set (i.e. 159571). So there must be multi-tagging, namely, each comment can have more than one tag.

|---|---| severe_toxic | severe_toxic | obscene | obscene | threat | threat | insult | insult | identity_hate | identity_hate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Label | Value | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| severe_toxic | 0 | 144277 | 0	| 143754 | 523	| 144248 | 29 | 143744 | 533 | 144174	| 103 |
| toxic | 1 | 13699 | 1595 | 7368 | 7926 | 14845 | 449 | 7950 | 7344 | 13992 | 1302 |

Then, from the cross-tab of the classes, we can find that a 'severe_toxic' comment is always a 'toxic' comment. Other classes seem to be subclasses of 'toxic' comments with just a few exceptions.

## Feature Engineering

In this project, I used two approaches to prepare the data:

1) TF-IDF

The word unigram and character n-grams are used as features. 

2) Word embeddings (using Word2vec)

In this competition I used glove.840B.300d to obtain vector representation of words.

## Building Models

The major models used include Logistic Regression, lightGBM, bidirectional gru with convolution, LSTM, and Naive Bayes SVM. In addition, I used random search or grid search cross validation approach to tune parameters.

## Model Evaluation

Submissions are evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column. In the end, my team achieved a score of 0.9857, ranking top 26% on the private leaderboard.

## Lessons

In the future, for such Natural Language Processing kaggle competitions, we can try to use stacking to further improve model performance, which we did not have enough time and computing resources during this competition. Also, deep learning have a higher score ceiling generally, compared to non-neuralnets approaches. This indicates that deep learning can capture the complex patterns in unstructured data better.

## Reference

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation
