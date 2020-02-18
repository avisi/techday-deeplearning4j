# Hands-on 

## Prerequisites

### Java version
* DL4J inspects your CPU prior to running a network on it. Not all MacOS Java ports handle this correctly, causing
an error like:
````C  [libc++abi.dylib+0x30b4]  _ZNK10__cxxabiv120__si_class_type_info27has_unambiguous_public_baseEPNS_19__dynamic_cast_infoEPvi+0x4````

On a MacBook we used Java 11.0.3 with no such problem.

## Setup of the project

Execute the following steps to get the project up and running:

* Import the project as a Maven Project.

* Run ````mvn clean install````

## Starter

For the starter part we are going to run a fully trained network and we are going to experiment with it. \
The fully trained network is able to make a distinction between bananas and no bananas. 
 
Run the application with the profile name "fruit" (Set this profile in the Run Configuration). Now, you will be able to access the trained classifier at http://localhost:8081. You can upload images or use your laptop's camera.
Experiment with the classifier and find out what features of a banana are used by the neural network to determine if we have a banana.
For example: does the color matter or the curved shape? If we bent our hand, can we pretend that it is a banana?
Make some notes about your findings, so you can use them later on when we are going to discuss the results.

Tip: If you're lucky, you can grab a banana from the bar and use the camera of your laptop!

## Intermediate

In this second part we are going to train a network ourselves. 

### Setting up the dataset
You can use one of the predefined datasets from: https://console.cloud.google.com/storage/browser/techday_feb2020
Please choose a maximum of three different categories. For example, three types of dogs from the Dogs dataset: chihuahua, beagle and pug. For adequate training about 30 images for each
category will be sufficient.

Create a folder for your dataset inside [src/main/resources/datasets](src/main/resources/datasets) and place the different categories inside them as folders.

### Create a dataset iterator
Implement the missing functionality inside the [DefaultDataSetIterator](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/intermediate/DefaultDataSetIterator.java)
This iterator will retrieve the images we just placed inside our resources folder.

### Create a transfer learning trainer
Implement the missing functionality inside the [BasicTrainer](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/intermediate/BasicTrainer.java)
In this trainer we will we use the existing VGG16 network and modify it. We will remove the classifier from it and append our new classifier for our dataset.

You can use the [FruitTransferLearningTrainer](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/starter/FruitTransferLearningTrainer.java) as an example.

### Train your network
When you finished implementing the transfer learning trainer you can start training by running the main function inside the BasicTrainer.
While training you can keep track of the progress in the console and check the current scores of the network. 

### Test your classifier
Extend the [DataClassifier](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/DataClassifier.java) as a Spring Boot @Component and assign it a unique @Profile e.g. "helloWorld".
Next run the application with "helloWorld" or the other profile you have set. Experiment with it by uploading some new pictures related to your dataset.


## Advanced
If you are able to succeed in the first two levels you can start thinking of your own use case, where it is possible to apply image classification.
Search for images which you can use and execute the same steps as in the intermediate level for your new dataset. Instead of using the VGG-16 architecture as an alternative any of the other zoo models can be used
e.g.
- VGG-19
- 
