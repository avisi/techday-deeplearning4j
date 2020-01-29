
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


