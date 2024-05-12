
# SFCE: mining distinguishing correlative features for mechanical fault diagnosis towards time-varying conditions

## Results
In addition to the datasets in the paper, several additional datasets have been used to validate the effectiveness of SFCE
| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |


## Data availability


## how to use the code
## key parameters
```
**fs** is the sampling frequency

**K** is the number of FIMFs, 5~10 is suitable for most situations.

**d_method** is the data augmentation method. FDM has a superior performance.

**corr** is the correlation measure method. The default is the Correntropy.

**clf** is the classifier. The default is the Ridge regression.

**z_score** is the data normalization method.
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

