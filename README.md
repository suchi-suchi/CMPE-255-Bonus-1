ML-MODEL FLASK DEPLOYMENT:

In this project we have used Tensor flows library keras and used resnet model. to draw the comparision i.e times taken to run in the original(i.e on Normal cloab run) vs the inffered one (which is done using FLask)

This is flask webapp deploying resnet50 model

We need to install all the required packages(flask,keras) on the local machine before executing out model.

need to run app.py which will load the deployed model in the localhost 5000 port.

Then we drew the comparision on both the approaches and obsered the following:
the time taken to predict using inference is 34 milliseconds while the time taken in cloab is 83 milliseconds, there’s a significant difference i.e the inference is 2.4 times faster than the one on colab.