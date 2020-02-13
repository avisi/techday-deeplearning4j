package nl.avisi.labs.deeplearning.transferlearning;

import java.io.IOException;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import nl.avisi.labs.deeplearning.transferlearning.model.Prediction;
import nl.avisi.labs.deeplearning.transferlearning.storage.StorageService;

@Controller
public class UploadController {

    private final StorageService storageService;

    private DataClassifier classifier;

    @Autowired
    public UploadController(StorageService storageService, FruitClassifier fruitClassifier) {
        this.storageService = storageService;
        this.classifier = fruitClassifier;
    }

    @GetMapping("/")
    public String listUploadedFiles(Model model) throws IOException {
        return "fruitCam";
    }

    @PostMapping("/upload/webcam")
    public String handleWebcamUpload(@RequestParam("imgBase64") String data, Model model) {

        String base64Image = data.split(",")[1];
        storageService.store(base64Image);

        Resource image = storageService.loadAsResource("image.jpg");
        List<Prediction> predictions = null;
        try {
            predictions = classifier.classify(image.getInputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }

        model.addAttribute("image", "/files/image.jpg");
        model.addAttribute("predictions", predictions);
        model.addAttribute("message", "Webcam image uploaded");

        return "fruitCam :: cameraPredictions";

    }

    @PostMapping("/upload/file")
    public String handleFileUpload(@RequestParam("image") MultipartFile file, Model model) {

        storageService.store(file);

        Resource image = storageService.loadAsResource(file.getOriginalFilename());
        List<Prediction> predictions = null;
        try {
            predictions = classifier.classify(image.getInputStream());
        } catch (Exception e) {
            e.printStackTrace();
        }

        model.addAttribute("image", "/files/" + file.getOriginalFilename());
        model.addAttribute("predictions", predictions);
        model.addAttribute("message", "Image uploaded");

        return "fruitCam :: filePredictions";

    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity handleStorageFileNotFound(Exception exc) {
        return ResponseEntity.notFound().build();
    }

}
