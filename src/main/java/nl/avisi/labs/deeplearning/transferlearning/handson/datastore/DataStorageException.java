package nl.avisi.labs.deeplearning.transferlearning.handson.datastore;

class DataStorageException extends RuntimeException {

    DataStorageException(String message) {
        super(message);
    }

    DataStorageException(String message, Throwable cause) {
        super(message, cause);
    }
}
