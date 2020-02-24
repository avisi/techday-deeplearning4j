package nl.avisi.labs.deeplearning.transferlearning.handson.datastore;

class DataStorageFileNotFoundException extends DataStorageException {

    DataStorageFileNotFoundException(String message) {
        super(message);
    }

    DataStorageFileNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}
