# Hands-on 

## Prerequisites

### Java version
* DL4J inspects your CPU prior to running a network on it. Not all MacOS Java ports handle this correctly, causing
an error like:
````C  [libc++abi.dylib+0x30b4]  _ZNK10__cxxabiv120__si_class_type_info27has_unambiguous_public_baseEPNS_19__dynamic_cast_infoEPvi+0x4````

On a MacBook we used Java 11.0.3 with no such problem.

## Setup

Execute the following steps to get the project up and running:

* Import the project as a Maven Project.

* Download the following to files (if you have not done this yet):
- https://console.cloud.google.com/storage/browser/_details/techday_feb2020/vgg16_dl4j_inference.zip
- https://console.cloud.google.com/storage/browser/_details/techday_feb2020/FruitModel.zip

Place vgg16_dl4j_inference.zip inside dl4j-models/models/vgg16 and FruitModel.zip in the root folder of this project.

For Gradle users:
* Import the project as a Gradle Project.
* Run ```./gradlew :clean :build``` 

For Maven users:

* Import the project as a Maven Project.
* Run ``mvn clean install``


## Starter


For the starter part we are going to run a fully trained network and we are going to experiment with it. \
The fully trained network is able to make a distinction between bananas and no bananas. 
 
Run the application with the profile name "fruit" (Set this profile in the Run Configuration). Now, you will be able to access the trained classifier at http://localhost:8081. You can upload images or use your laptop's camera.
Experiment with the classifier and find out what features of a banana are used by the neural network to determine if we have a banana.
For example: does the color matter or the curved shape? If we bent our hand, can we pretend that it is a banana?
Make some notes about your findings, so you can use them later on when we are going to discuss the results.

Tip: If you're lucky, you can grab a banana from the bar and use the camera of your laptop!

## Intermediate
In this second part we are going to train a network ourselves. During this part, you may want to use the docs from [here](https://deeplearning4j.org/docs/latest/) for further explanation about methods that were used.

### Setting up the dataset
You can use one of the predefined datasets from: https://console.cloud.google.com/storage/browser/techday_feb2020/datasets Download one of the zip files and extract it.
Please choose a maximum of three different categories from the extracted folders. For example, three types of dogs from the Dogs dataset: chihuahua, beagle and pug. For adequate training about 50 images for each
category will be sufficient. You can use more images, but this will cause the training time to be longer.

Create a folder for your dataset inside [src/main/resources/datasets](src/main/resources/datasets) and place the different categories inside them as folders.

### Create a dataset iterator
Implement the missing functionality inside the [DefaultDataSetIterator](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/intermediate/IntermediateDataSetIterator.java)
This iterator will retrieve the images we just placed inside our resources folder.

You can use the [FruitDataSetIterator](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/starter/FruitDataSetIterator.java) as an example.

### Create a transfer learning trainer
Implement the missing functionality inside the [BasicTrainer](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/intermediate/IntermediateTrainer.java)
In this trainer we will we use the existing VGG16 network and modify it. We will remove the classifier from it and append our new classifier for our dataset.

You can use the [FruitTransferLearningTrainer](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/starter/FruitTransferLearningTrainer.java) as an example.
Do not only copy the contents of the FruitTransferLearningTrainer. Try to find out what steps are taken. You may also have to modify one of the class variables.

### Train your network
When you finished implementing the transfer learning trainer you can start training by running the main function inside the BasicTrainer.
While training you can keep track of the progress in the console and check the current scores of the network. 

### Test your classifier
Extend the [DataClassifier](src/main/java/nl/avisi/labs/deeplearning/transferlearning/handson/DataClassifier.java) as a Spring Boot @Component and assign it a unique @Profile e.g. "helloWorld".
Next run the application with "helloWorld" or the other profile you have set. Run the application from the Run Menu or alternatively execute the following command:
```mvn spring-boot:run -Dspring.profiles.active=helloWorld```

Experiment with your trained model by uploading some new pictures related to your dataset. Just like you did at the starter part.
You may want to train your network again if it doesn't perform well. For example you can try to use more iterations for training.

## Advanced
If you are able to succeed in the first two levels you can start thinking of your own use case, where it is possible to apply image classification.
Search for images which you can use and execute the same steps as in the intermediate level for your new dataset. Instead of using the VGG-16 architecture as an alternative any of the [other zoo models](https://deeplearning4j.org/docs/latest/deeplearning4j-zoo-models) can be used
e.g.
- VGG-19
