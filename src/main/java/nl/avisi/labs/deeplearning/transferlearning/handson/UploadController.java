package nl.avisi.labs.deeplearning.transferlearning.handson;

import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;
import java.util.Locale;

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

import nl.avisi.labs.deeplearning.transferlearning.handson.model.Prediction;
import nl.avisi.labs.deeplearning.transferlearning.handson.datastore.DataStorageService;

@Controller
public class UploadController {

    private final DataStorageService dataStorageService;

    private final DataClassifier classifier;

    private final DecimalFormat df;

    @Autowired
    public UploadController(DataStorageService dataStorageService, DataClassifier dataClassifier) {
        this.dataStorageService = dataStorageService;
        this.classifier = dataClassifier;
        NumberFormat nf = NumberFormat.getNumberInstance(Locale.US);
        df = (DecimalFormat) nf;
        df.applyPattern("#.##");
        df.setRoundingMode(RoundingMode.FLOOR);
    }

    @GetMapping("/")
    public String listUploadedFiles(Model model) throws IOException {
        return "fruitCam";
    }

    @PostMapping("/upload/webcam")
    public String handleWebcamUpload(@RequestParam("imgBase64") String data, Model model) {

        String base64Image = data.split(",")[1];
        dataStorageService.store(base64Image);

        Resource image = dataStorageService.loadAsResource("image.jpg");
        List<Prediction> predictions = null;
        try {
            predictions = classifier.classify(image.getInputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (Prediction prediction : predictions) {
            prediction.setScore(Double.valueOf(df.format(prediction.getScore())));
        }

        model.addAttribute("image", "/files/image.jpg");
        model.addAttribute("predictions", predictions);
        model.addAttribute("message", "Webcam image uploaded");

        return "fruitCam :: cameraPredictions";

    }

    @PostMapping("/upload/file")
    public String handleFileUpload(@RequestParam("image") MultipartFile file, Model model) {

        dataStorageService.store(file);

        Resource image = dataStorageService.loadAsResource(file.getOriginalFilename());
        List<Prediction> predictions = null;
        try {
            predictions = classifier.classify(image.getInputStream());
        } catch (Exception e) {
            e.printStackTrace();
        }
        for (Prediction prediction : predictions) {
            prediction.setScore(Double.valueOf(df.format(prediction.getScore())));
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
