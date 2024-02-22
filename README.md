# MCE

extract the features between differnent sensors,the format of input X should be (examples,sensors,length)
transformer = MCE(fs,K,kernel_size)
features, inner_features, intra_features=transformer.transform(X)
predict the result
