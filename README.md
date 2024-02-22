# MCE
how to use the code

extract the features between differnent measured sensors,the format of input X should be (examples,sensors,length)

transformer = MCE(fs,K,kernel_size)

features, inner_features, intra_features=transformer.transform(X)

predict the result

clf = MCE(fs,K,kernel_size)

y_pre=clf.fit_predict(X_train,Y_train,X_test)
