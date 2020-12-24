# Self-Driving-Car
DISCLAIMER - This project is purely for knowledge purpose.It should not be used ofr any real-life trainingmodelf for autonomous vehicles.

**Description**
It is a supervised regression problem between the steering angles and the road images in front of a car. The network is based on The [NVIDIA model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

**Files Included**

* IMG folder - Training and testing data set images
* drivinglog.csv - comma seperated values of the dataset
* train_model.ipynb - Model build to train the the car referencing  NVIDIA model
* test.py - test the model ona new environment using Anaconda and Unity.
* model.h5 - saved trained model
* environment.yml - Specifying all the requirements
* Final Report


# **How to run the model**

You will need anaconda to run the model.

**Creating an environment in Anaconda**

conda env create -f environment.yml

**running the pre-trained model**

python test.py model.h5

**To train the model**

python train_model.ipynb

To understand the the model architecture in detail go through the **Final-Report**
