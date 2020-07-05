# Required environment

All the program are implemented by python, and runs on  Tensorflow 2.0-GPU framework

the packages need to be install are list:

tensorflow-gpu==2.2

numpy

random

scipy

libtlda



# Function illustration of every program

1. data_Preprocessing.py :  the original data is filtered and augmented by this program.
2. withinSubjectModel_1_to_8_Training.py:  in this program, test data set is used for model training and validation, our model training process is divided into two stages: in the first stage, four sub-models are trained individually, and all sub-models have the same input data set and the output data set; in the second stage, MCNN model is generated on the basis of these four sub-models,  and MCNN model is trained.
3. withinSubjectModel_1_to_8_evaluation.py: in this program, test data is used for evaluating the accuracy of  within-subject.
4. withinSubjectModel_1_to_8_Predict.py:  this program is used to predict the label of P01~P08.
5. crossSubjectModel_1_to_8_training.py: in this program, test data set is used for model training and validation. To test the $i-th$ cross-subject accuracy, the i-th subject’s trail samples are regarded as the test data set, and the other seven subjects’ trial samples are regarded as the training data.
6. crossSubjectModel_1_to_8_evaluation.py: in this program,   test data is used for evaluating the accuracy of  cross-subject.
7. crossSubjectModel_9_10_training.py: in this program, the model trained  8 subjects of  test data.
8.  crossSubjectModel_9_and_10_predict.py: in this program, the model  is used for predict  the label of P09~P10.

