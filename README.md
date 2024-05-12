
# SFCE: mining distinguishing correlative features for mechanical fault diagnosis towards time-varying conditions

## Results under the time-varying condition
In addition to the datasets in the paper, several additional datasets have been used to further validate the effectiveness of SFCE
| dataset name  	|     number of sensors(S)     	| sample length(L) 	| required training sample number for 95% accuracy|  required training sample number for 99% accuracy|
|:------:	  |:-----------------:|:------------:	| :------------:	| :------------:	|
|    KAIST  |     4   	| 2700 	| 15	| 40	|
|    WT   	|     4   	| 6000 	| 14	| 35	|
|    HIT   	|     6   	| 512 	| 6	  | 13	|
|  MCC5-THU(speed) |     6   	| 2456 	| 10	| 33	|
|  MCC5-THU(load)  |     6   	| 2 	  | 2   | 30	|

## Results under constant speed condition
| dataset name  	|     number of sensors(S)     	| sample length(L) 	| required training sample number for 95% accuracy|  required training sample number for 99% accuracy|
|:------:	   |:-----------------:	|:------------:	| :------------:	| :------------:	|
|    KAIST   |     4   	| 2700 	| 1	| 2	|
|    WT   	 |     4   	| 6000 	| 3	| 5	|
| SEU-gearbox|     8   	| 512 	| 3	| 4	|
| SEU-bearing|     8   	| 512 	| 2	| 7	|

## Data availability


## how to use the code
## key parameters
```
fs:           the sampling frequency

K:            the number of FIMFs, 5~9 is suitable for most situations.

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

