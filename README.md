# FindICI: Using Machine Learning to Detect Linguistic Inconsistencies between Code and NaturalLanguage Descriptions in Infrastructure as Code

This repository contains the code implementation for tha paper ***FindICI: Using Machine Learning to Detect Linguistic Inconsistencies between Code and NaturalLanguage Descriptions in Infrastructure as Code***.
The contents are explained below.

The folder **1 Find and merge repositories** presents data collection. This included Ansible repositories detection, bug-related commits extraction and repository selection based on the research criteria.

The folder **2 Find Ansible tasks** implements a heuristic mechanism in order to identify and extract Ansible tasks from a repository. 

The folder **3 Map tasks to ansible documentation** introduces a mechanism which identifies the used module within an Ansible task by scraping information from the Ansible documentation website. In addition, the mechanism identifies the used parameters of each module based in the same fashion.

The folder **4 Build ast and tokenize** implements an AST generator engine in order to create a token sequence from Ansible task bodies.  

The folder **5 Create inconsistent observations** contains the implementation of the applied transformations in order to generate the inconsistent observations in Ansible tasks

The folder **6 Detect linguistic inconsistencies** Contains the implementation and results for every classifier we used, namely:
1. Random Forest.
2. Support Vector Machines.
3. eXtreme Gradient Boosting (XGBoost).
4. Multi-Layer Perceptron.
5. Convolutional Neural Networks.
6. Long Short-Term Memory.

In addition, for each one of the classifiers we compared different word embedding techniques, namely:

1. Word2Vec
2. DocVec
3. fastText 

For more details regarding the linguistic inconsistency detection check the folder **6 Detect linguistic inconsistencies**. 