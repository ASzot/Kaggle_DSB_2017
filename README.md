# Kaggle Data Science Bowl 2017 Submission

This is my submsission to my first Kaggle competition. It employs a combination
of many techniques seen in the kernels for the competition. 

The goal of the competition is, given a patients lung scan, predict whether
they have cancer. The competition is further complicated by the lung scans
having different resolutions, being 3 dimensional, and the cancerous nodes
being very small. 

I employed the technique of first preprocessing the data to have normalized
scales of measurement on the images. Then I split the data up into batches of
image blocks with depth 3. This was necessary in my next step of passing the
data through a convolutional neural network. I also normalized and scaled the
data for the appropriate units used in the scans. 

Next I used the Keras pretrained ResNet-50 network to extract features from the
scans. This then extracted 2048 dimension feature vectors from each of the
scans. After extracting the features for each batch of 3 depth volumes I then
compressed the features into a vector with depth 1 by simply taking the average
of all of the volume depth slices. This then gave me a 2048 dimension vector
representing a patient's scan.

I then trained a Gradient Boosting Classifier on these feature vectors. In the
end I obtained 0.6 log-loss for the competition.
