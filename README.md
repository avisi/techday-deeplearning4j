
* DL4J inspects your CPU prior to running a network on it. Not all MacOS Java ports handle this correctly, causing
an error like 

````C  [libc++abi.dylib+0x30b4]  _ZNK10__cxxabiv120__si_class_type_info27has_unambiguous_public_baseEPNS_19__dynamic_cast_infoEPvi+0x4````

On a MacBook we used Java 11.0.3 with no such problem.

* Import the project as a Maven Project.

* Run 

````mvn clean install````

* Run the FruitTransferLearningTrainer

* Run the Application as a Spring Boot application

* Point your browser to http://localhost:8081

* Upload an image file or take a picture with your laptop's camera

* Handson

Three different applications

Three branches 

- Runnable jar met bijv. bananenidee, dat kun je runnen

- Implementatie van dataset iterator (obv de hints) en implementatie van de transfer learning config (laatste x layers vervangen)

- Dataset definieren (eigen dataset) of ingaan op de vraag waarom het model een bepaald resultaat oplevert



* Running the fruit recognition application
- Run the FruitTransferLearningTrainer application
- Run the FruitClassifier application by running the Application with Spring Boot profile "fruit"


* Creating your own classifier:
- Implement the TransferLearningIterator interface
- Extend the BaseTransferLearningTrainer class  
- Extend the DataClassifier class as a Spring Boot @Component and assign it a unique @Profile 
- 
