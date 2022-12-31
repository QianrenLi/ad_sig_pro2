# README
This classification based on the ECG signal processing tool box which is helpful to locate the point of P, QRS and T wave.
https://github.com/ElsevierSoftwareX/SOFTX-D-20-00010

This package should be under the same directory of workspace.
## The architecture of SVM classification:

+ ```wave_feature_decompose.m``` -- decompose loaction of P, QRS and T wave into feature vector.

+ ```fature_importance.m``` -- reduce the dimention of feature vector and obtain better generalization performance.

+ ```SVM_feature_extraction.m``` -- Feature extraction and SVM training

The ```SVM_feature_extraction.m``` requires a filterd ECG signal seqence loaded as ```data.mat``` consists of the training data, training labels and testing data and testing labels which need to be contruct from previous section.
