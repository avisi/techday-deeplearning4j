// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var video = document.getElementById('video');
let videoCapture = new cv.VideoCapture(video);

var localMediaStream;
// source image element matrix
let src = new cv.Mat(240, 320, cv.CV_8UC4);
// destination matrix, will act as a copy so we dont change the source
let dst = new cv.Mat(240, 320, cv.CV_8UC4);

/**
 *  Request access to the webcam and show stream
 */
function getAccessToTheCamera() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({video : true}).then(function(stream) {
      try {
        video.srcObject = stream;
        localMediaStream = stream;
      } catch (error) {
        video.src = window.URL.createObjectURL(stream);
      }
      video.play();
    });
  }
}

/**
 *  Shut down the camera again
 */
function shutdownCamera() {
  video.pause();
  localMediaStream.getTracks().map(function(val) {
      val.stop();
    }
  );
  video.srcObject = null;
}


function processVideo() {
  // start processing the current videoframe
  videoCapture.read(src);
  src.copyTo(dst);

  cv.imshow('canvas', src);

  uploadImage();

  // schedule the next frame
  if (handle != null) {
    setTimeout(processVideo, 2000);
  }

};

function uploadImage() {
  var params = canvas.toDataURL("image/jpeg", 0.85);
  $.ajax({
    type : "POST",
    url : "/upload/webcam",
    data : {
      contentType : "multipart/form-data",
      imgBase64 : params,
      processData : false
    }
  }).done(function(predictions) {
    $('#cameraPredictions').html(predictions);
  });
}

var handle;
$('#tabs' ).tabs({
  activate: function( event, ui ) {
    console.log(ui);
    console.log(ui.newTab.index());
    if (ui.newTab.index() == 0) {
      shutdownCamera();
      clearTimeout(handle);
      handle = null;
    }
    if (ui.newTab.index() == 1) {
      getAccessToTheCamera();
      handle = setTimeout(processVideo, 200);
    }
  },

});
$('#upload').on('click', function(){
  $('#upload').prop('disabled', true);
  var data = new FormData();
  data.append('image', $('#file').prop('files')[0]);
  let imgUrl = URL.createObjectURL($('#file').prop('files')[0]);
  let utils = new Utils('errorContainer');
  utils.loadImageToCanvas(imgUrl, 'fileCanvas');



  $.ajax({
    type: 'POST',
    processData: false, // important
    contentType: false, // important
    data: data,
    url: '/upload/file',
    dataType : 'text',
    success: function(predictions){
      $('#filePredictions').html(predictions);
      $('#upload').prop('disabled', false);
    },
    error: function(xhr, error){
      console.log(xhr);
      console.log(error);
      $('#errorContainer').append(error);
      $('#upload').prop('disabled', false);
    }

  });
});






