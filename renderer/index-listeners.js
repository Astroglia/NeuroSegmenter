var $ = global.jQuery = require('jquery');
const {dialog} = require('electron').remote;
let { PythonShell } = require('python-shell');

var folderPath; //path to folder.

function getFolder() 
{
    folderPath = dialog.showOpenDialog({
        properties: ['openFile']
    });
    console.log(folderPath.toString())
    //PythonShell.end(function (err,code,signal) {
    //    if (err) throw err;
    //    console.log('The exit code was: ' + code);
    //    console.log('The exit signal was: ' + signal);
    //    console.log('finished');
    //    console.log('finished');
    //  });
}

function processImages(imageType="RIBBON")
{
    let folderOptions = {
        mode: 'text',
        pythonOptions: ['-u'], // get print results in real-time
       // scriptPath: 'path/to/my/scripts',
        args: [folderPath, imageType] // ### argv[1] = folderpath, argv[2] = imagetype, argv[0] is the python script itself, for some reason.
      };

    var pyshell = new PythonShell('pyfolder/processFile.py', folderOptions);
  

    var checkprocessing = false;

    pyshell.on('message', function (message) 
    {
    // received a message sent from the Python script (a simple "print" statement
    var div = document.getElementById('terminalExtraContentDiv');
    var processingDiv = document.getElementById('processingDiv');
    var blockCursor = document.getElementById('blockCursor')

    if(message == "PROCESSING") {
        checkprocessing = true
        message = "Processing first image..."
    }
    if(checkprocessing) {
        processingDiv.innerHTML = message;
        div.appendChild(processingDiv)
        processingDiv.append(blockCursor);
    } else {
        div.innerHTML += message;
        div.appendChild(document.getElementById('blockCursor'));
        div.innerHTML += "<br>";
    }
    console.log(message);
    });

    // end the input stream and allow the process to exit
    pyshell.end(function (err) 
    {
        if (err)
        {
            throw err;
    };

    console.log('finished');
});
}

function createListeners()
{
document.getElementById('selectFolderButton').addEventListener('click', () => {
    getFolder()
})
document.getElementById('segmentRibbonButton').addEventListener('click', () =>  { 
    processImages("RIBBON") 
})
document.getElementById('segmentONLButton').addEventListener('click', () => {
    processImages("STRING") 
})
} //end createListeners()

$( document ).ready(function()
 {
    createListeners()
});