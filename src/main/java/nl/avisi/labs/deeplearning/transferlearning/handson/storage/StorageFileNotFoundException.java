package nl.avisi.labs.deeplearning.transferlearning.handson.storage;

class StorageFileNotFoundException extends StorageException {

    StorageFileNotFoundException(String message) {
        super(message);
    }

    StorageFileNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}
