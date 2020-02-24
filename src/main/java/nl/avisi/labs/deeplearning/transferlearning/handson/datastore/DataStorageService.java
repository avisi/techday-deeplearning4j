package nl.avisi.labs.deeplearning.transferlearning.handson.datastore;
import java.nio.file.Path;
import java.util.stream.Stream;

import org.springframework.core.io.Resource;
import org.springframework.web.multipart.MultipartFile;

public interface DataStorageService {

    void init();

    void store(MultipartFile file);

    void store(String file);

    Stream<Path> loadAll();

    Path load(String filename);

    Resource loadAsResource(String filename);

    void deleteAll();

}
