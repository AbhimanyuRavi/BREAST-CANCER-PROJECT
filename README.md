# Breast Cancer Classification using Machine Learning
**This project aims to develop and compare different machine learning classifiers for breast cancer diagnosis based on the features of the tumor. The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository. The dataset contains 569 samples of malignant and benign tumor cells. The first two columns in the dataset store the unique ID numbers of the samples and the corresponding diagnosis (M=malignant, B=benign), respectively. The columns 3-32 contain 30 real-valued features that are computed from digitized images of the cell nuclei, which can be used to characterize the shape, size, texture, and symmetry of the cells.**

### The project involves the following steps:

**Data exploration and preprocessing: Exploring the dataset to understand its characteristics and distribution, checking for missing values and outliers, and scaling the features to a common range.
Model development: Developing four different classifiers using Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest algorithms. Tuning the hyperparameters of each model using GridSearchCV to find the optimal values that maximize the model performance.
Model evaluation: Evaluating the performance of each model using accuracy, precision, recall, and F1-score metrics. Comparing the results of different models and selecting the best one. Plotting the confusion matrix to visualize the true positives, false positives, true negatives, and false negatives of each model.
Results**
### The results of this project are summarized below:

**Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.9123	0.9231	0.8889	0.9057
SVM	0.9123	0.9231	0.8889	0.9057
Decision Tree	0.9123	0.9231	0.8889	0.9057
Random Forest	0.9474	0.9615	0.9259	0.9434
The Random Forest classifier had the highest accuracy of 94.74% and also had good precision, recall, and F1-score values, indicating that it performed well in correctly identifying both malignant and benign tumors.**

The confusion matrix shows that the model correctly classified 162 out of 171 samples (94.74%). Out of the 108 actual malignant samples, the model correctly identified 104 samples, and it correctly classified 58 out of the 62 actual benign samples. There were only nine misclassifications in total.

## Conclusion
**The results of this project indicate that machine learning algorithms can accurately classify breast cancer tumors as malignant or benign based on the features of the tumor. These models have the potential to aid medical professionals in making more accurate diagnoses and improving patient outcomes. The Random Forest classifier performed well in this project and can be used as a reliable model for classifying breast cancer tumors.

#How to run this project
**To run this project, you need to have Python 3 installed on your system along with the following libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
You can install these libraries using pip or conda commands.

**You also need to download the dataset from this link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Save the dataset as data.csv in the same folder as the Jupyter notebook file.

You can then open the Jupyter notebook file and run each cell sequentially to reproduce the results.

## References
**Breast Cancer Wisconsin (Diagnostic) Dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php**
