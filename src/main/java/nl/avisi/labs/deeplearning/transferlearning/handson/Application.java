package nl.avisi.labs.deeplearning.transferlearning.handson;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import nl.avisi.labs.deeplearning.transferlearning.handson.datastore.DataStorageProperties;
import nl.avisi.labs.deeplearning.transferlearning.handson.datastore.DataStorageService;

@SpringBootApplication
@EnableConfigurationProperties(DataStorageProperties.class)
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @Bean
    CommandLineRunner init(DataStorageService dataStorageService) {
        return args -> {
            dataStorageService.deleteAll();
            dataStorageService.init();
        };

    }
}
