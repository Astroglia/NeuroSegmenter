const {app, BrowserWindow} = require('electron')
let { PythonShell } = require('python-shell');
require('jquery');

function createWindow() {
    window = new BrowserWindow( {
        width:1200, height: 900, 
        webPreferences: {
            nodeIntegration: true
        }
    } )
    //window.setMenuBarVisibility(false) // get rid of menu bar.
    window.loadFile('index.html')
}

app.on('ready', createWindow)