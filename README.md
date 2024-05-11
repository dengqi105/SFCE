
# SFCE: mining distinguishing correlative features for mechanical fault diagnosis towards time-varying conditions

## how to use the code
## key parameters
fs is the sampling frequency

K is the number of FIMFs, 5~10 is suitable for most situations.

## Examples

Extract the features from all measured signals, the format of input X should be (examples, sensors, length)
```python
Extractor = SFCE (fs, K)

features, inner_features, intra_features = Extractor.transform(X)
```

and predict the result

```python

Y_pre = Extractor.fit_predict(X_train, Y_train, X_test)
```
## Results
In addition to the dataset in the paper, several additional datasets have been used to validate the effectiveness of SFCE
| Dataset name  | Second Header | Second Header |
| ------------- | ------------- | Second Header |
| Content Cell  | Content Cell  | Second Header |
| Content Cell  | Content Cell  | Second Header |


## Data availability
