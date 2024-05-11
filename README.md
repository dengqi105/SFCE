
# SFCE: mining distinguishing correlative features for mechanical fault diagnosis towards time-varying conditions

## how to use the code
## Examples

extract the features between differnent measured sensors, the format of input X should be (examples, sensors, length)
```python
Extractor = SFCE (fs, K)

features, inner_features, intra_feature = Extractor.transform(X)
```

predict the result

```python
Extractor = SFCE (fs, K, kernel_size)

Y_pre = Extractor.fit_predict(X_train, Y_train, X_test)
```
