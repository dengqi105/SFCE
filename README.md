# MCE
## how to use the code

extract the features between differnent measured sensors, the format of input X should be (examples, sensors, length)
```python
clf = MCE (fs, K, kernel_size)

features, inner_features, intra_feature = clf.transform(X)
```

predict the result

```python
clf = MCE (fs, K, kernel_size)

Y_pre = clf.fit_predict(X_train, Y_train, X_test)
```
