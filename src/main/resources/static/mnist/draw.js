var canvas = document.getElementById('canvas');
var context = document.getElementById('canvas').getContext("2d");
var paint = false;
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();

$('#canvas').mousedown(function(event) {
    var mouseX = event.pageX - this.offsetLeft;
    var mouseY = event.pageY - this.offsetTop;

    paint = true;
    addClick(mouseX, mouseY);
    redraw();
});

$('#canvas').mousemove(function(event) {
    if(paint) {
        addClick(event.pageX - this.offsetLeft, event.pageY - this.offsetTop, true);
        redraw();
    }
});

$('#canvas').mouseup(e => paint=false);

$('#clear').click(e => reset(true));
$('#send').click(e => classify());

function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

function redraw() {
    reset(false);

    context.strokeStyle = "#ffffff";
    context.lineJoin = 'round';
    context.lineWidth = 10;

    for(var i = 0; i < clickX.length; i++) {
        context.beginPath();
        if(clickDrag[i] && i) {
            context.moveTo(clickX[i-1], clickY[i-1]);
        } else {
            context.moveTo(clickX[i] -1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i]);
        context.closePath();
        context.stroke();
    }
}

function reset(clearDrawing) {
    context.fillStyle="black";
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);
    if(clearDrawing) {
        clickX = new Array();
        clickY = new Array();
        clickDrag = new Array();
    }
}

function classify() {
    var tempCan = document.createElement('canvas');
    tempCan.width = 28;
    tempCan.height = 28;
    var canContext = tempCan.getContext("2d");
    canContext.drawImage(canvas, 0, 0, 28, 28);
    
    var params = tempCan.toDataURL("image/jpeg", 1);

    $.ajax({
        type: "POST",
        url: "/upload/drawing",
        data: {
            contextType: "multipart/form-data",
            imgBase64: params,
            processData: false
        }
    }).done(function(predictions) {
        $('#predictions').html(predictions);
    });
    tempCan.remove();
}
