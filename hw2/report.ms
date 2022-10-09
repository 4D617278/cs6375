.TL 
CS 6375 HW2
.AU 
Brandon Luo

.NH 1
Collaborative Filtering

.LP
Mean Absolute Error: 0.302780007413595

Root Mean Squared Error: 0.7334442448790081

.NH 1
Neural Networks, K-nearest neighbors, and SVMs

.LP
Support Vector Machine
.TS
tab(&), allbox;
c c c c c .
C & Degree & Kernel & Gamma & Accuracy
2.4 & 3 & poly & scale & 0.979
2.0 & _ & rbf & scale & 0.9832
4.0 & _ & rbf & scale  & 0.9843
2.2 & _ & rbf & auto & 0.9506
1.6 & 5 & poly & scale & 0.9602
0.8 & _ & sigmoid & scale & 0.7777
1.0 & _ & linear & scale & 0.9404
1.2 & 6 & poly & scale & 0.9399
0.1 & _ & rbf & scale & 0.9595
2.0 & _ & poly & auto & 0.7414
.TE

Multilayer Perceptron
.TS
tab(&), allbox;
c c c c c .
Activation & Solver & Alpha & Hidden Layer Sizes & Accuracy
tanh & adam & 1e-5 & (100,) & 0.9804
logistic & adam & 1e-5 & (100,) & 0.9779
logistic & lbfgs & 1e-1 & (100,) & 0.9723
relu & adam & 1e-3 & (1200,) & 0.9845
tanh & adam & 1e-6 & (400,400) & 0.9824
tanh & adam & 1e-10 & (100,100,100) & 0.9797
relu & adam & 1e-6 & (400,400) & 0.9811
tanh & adam & 1e-8 & (800,) & 0.9834
relu & adam & 1e-4 & (1200,) & 0.9845
identity & adam & 1e-8 & (800,) & 0.9236
.TE

K-nearest Neighbors
.TS
tab(&), allbox;
c c c c c c c .
Algorithm & Leaf Size & Metric & Neighbors & Weights & P & Accuracy
KD Tree & 30 & l1 & 5 & Distance & 2 & 0.9629
KD Tree & 30 & l1 & 5 & Uniform & 2 & 0.9618
KD Tree & 30 & Minkowski & 7 & Distance & 2 & 0.9623
Ball Tree & 30 & Minkowski & 3 & Uniform & 4 & 0.9742
Ball Tree & 30 & Minkowski & 4 & Uniform & 3 & 0.9712
Ball Tree & 30 & Minkowski & 4 & Uniform & 4 & 0.9738
KD Tree & 30 & l1 & 7 & Uniform & 2 & 0.9615
Ball Tree & 30 & Minkowski & 3 & Uniform & 3 & 0.9718
Ball Tree & 20 & Minkowski & 6 & Distance & 2 & 0.9709
Ball Tree & 20 & Minkowski & 6 & Distance & 3 & 0.9737
.TE

.TS
tab(&), allbox;
c c .
Classifier & Best Error
Support Vector Machine & 0.9843
Multilayer Perceptron & 0.9845
K-nearest Neighbors & 0.9742
.TE
