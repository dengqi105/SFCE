
# SFCE: mining distinguishing correlative features for mechanical fault diagnosis towards time-varying conditions

## Results
In addition to the datasets in the paper, several additional datasets have been used to validate the effectiveness of SFCE
| dataset name  	|     File Name     	| Descriptions 	|                                                           Paper                                                           	|
|:------:	|:-----------------:	|:------------:	|:-------------------------------------------------------------------------------------------------------------------------:	|
|    1   	|      CCDG.py      	|     Model    	|                             Conditional Contrastive Domain Generalization for Fault Diagnosis                             	|
|    2   	|      CNN-C.py     	|     Model    	| Learn Generalization Feature via Convolutional Neural Network:A Fault Diagnosis Scheme Toward Unseen Operating Conditions 	|
|    3   	|      DANN.py      	|     Model    	|                                     Adversarial traning among mulitple source domains                                     	|
|    4   	|     DCORAL.py     	|     Model    	|                                         Reduce CORAL among mulitple source domains                                        	|
|    5   	|       DDC.py      	|     Model    	|                                          Reduce MMD among mulitple source domains                                         	|
|    6   	|      DGNIS.py     	|     Model    	|     A domain generalization   network combing invariance and specificity towards real-time intelligent fault diagnosis    	|
|    7   	|       ERM.py      	|     Model    	|                                                 Reduce Classfication loss                                                 	|
|    8   	|     IEDGNet.py    	|     Model    	|   A hybrid generalization network for intelligent fault diagnosis of rotating machinery under unseen working conditions   	|
|    9   	| data_loaded_1d.py 	| Data prepare 	|                                                             /                                                             	|
|   10   	|   resnet18_1d.py  	|    Network   	|                                                             /                                                             	|
|   11   	|      utlis.py     	| Some metrics 	|                                                             /                                                             	|


## Data availability


## how to use the code
## key parameters
```
fs:           the sampling frequency

K:            the number of FIMFs, 5~10 is suitable for most situations.

d_method:     the data augmentation method. FDM has a superior performance.

corr:         the correlation measure method. The default is the Correntropy.

clf:          the classifier. The default is the Ridge regression.

z_score:      the data normalization method.
```

## Examples

SFCE fuses the features from inner-sensor scale-varied and intra-sensor scale-aligned correlative features (inner_features and intra_features in the code). The number of inner_features and intra_features are S\*(K+1)\*(K+1) and (K+1)\*S\*S, respectively. S is the number of sensors.

Note that the format of input X should be (Examples, Sensors, Length).  fs (sampling frequency) is a must for "FDM", and kernel_size is indispensable for "Correntropy". 

```python
Extractor = SFCE (fs, K,kernel_size=45,d_method="FDM",corr="Correntropy",clf="RR",z_score=True)

features, inner_features, intra_features = Extractor.transform(X)
```

After feature extraction, the result can be predicted.  X_train and Y_train are the training set and corresponding labels.  X_test is the testing set.

```python

Y_pre = Extractor.fit_predict(X_train, Y_train, X_test)
```

