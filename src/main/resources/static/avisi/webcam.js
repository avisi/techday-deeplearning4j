// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var video = document.getElementById('video');
let videoCapture = new cv.VideoCapture(video);
// source image element matrix
let src = new cv.Mat(240, 320, cv.CV_8UC4);
// destination matrix, will act as a copy so we dont change the source
let dst = new cv.Mat(240, 320, cv.CV_8UC4);

/**
*	Request access to the webcam and show stream
*/
function getAccessToTheCamera() {
	if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
	    // Not adding `{ audio: true }` since we only want video now
	    navigator.mediaDevices.getUserMedia({video: true}).then(function (stream) {
	        try {
	            video.srcObject = stream;
	        } catch (error) {
	            video.src = window.URL.createObjectURL(stream);
	        }
	        video.play();
	    });
	}
}

/**
 *	Shut down the camera again
 */
function shutdownCamera() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({video: true}).then(function (stream) {
      stream.getTracks().forEach(track => track.stop());
    });
  }
}


function processVideo() {
    // start processing the current videoframe
    videoCapture.read(src);
    src.copyTo(dst);
   
    cv.imshow('canvas', src);
	
    postImage();
        
   // schedule the next frame
  if (handle != null) {
    setTimeout(processVideo, 2000);
  }
    
};

function postImage() {
    var params = canvas.toDataURL("image/jpeg", 0.85);
    $.ajax({
        type: "POST",
        url: "/upload/webcam",
        data: {
            contentType: "multipart/form-data",
            imgBase64: params,
            processData: false
        }
    }).done(function (predictions) {
        console.log('predictions' + predictions);
        $('#cameraPredictions').html(predictions);


    });
}




