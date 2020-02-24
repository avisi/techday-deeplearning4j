package nl.avisi.labs.deeplearning.transferlearning.handson.datastore;

import org.springframework.boot.context.properties.ConfigurationProperties;

public class StorageProperties {

    /**
     * Folder location for storing uploaded files
     */
    private String location = "upload-dir";

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

}
