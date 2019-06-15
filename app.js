var express = require("express");
var app = express();
var port = 3000;
var upload = require("express-fileupload");
// For reading the csv file
var fs = require("fs");
var csv = require("fast-csv");
// Python shell
var ps = require('python-shell')
var spawn = require('child_process').spawn;
// Generating unique id for the files
var uniqid = require('uniqid');
//compressing python output

var pythonOutput = "";
var fileNameTo = "";

var loading = ''

function dataToPyMatrix(fileName) {
  //var scriptExecution = spawn("python.exe", ['./python/nodelink2.py']);
  var scriptExecution = spawn("python.exe", ['./python/AM.py']);
  // Handle normal output
  scriptExecution.stdout.on('data', (data) => {
     //pythonOutput = String.fromCharCode.apply(null, data);
  });

  // Write data (remember to send only strings or numbers, otherwhise python wont understand)
  var data = JSON.stringify([fileName]);
  scriptExecution.stdin.write(data);
  // End data write
  scriptExecution.stdin.end();
}


function dataToPyNodelink(fileName) {
  //var scriptExecution = spawn("python.exe", ['./python/nodelink2.py']);
  var scriptExecution = spawn("python.exe", ['./python/NLD.py']);
  // Handle normal output
  scriptExecution.stdout.on('data', (data) => {
     pythonOutput = String.fromCharCode.apply(null, data);
     console.log(pythonOutput)
  });

  // Write data (remember to send only strings or numbers, otherwhise python wont understand)
  var data = JSON.stringify([fileName]);
  scriptExecution.stdin.write(data);
  // End data write
  scriptExecution.stdin.end();
}

function dataToPyBoth(fileName) {
  //var scriptExecution = spawn("python.exe", ['./python/nodelink2.py']);
  var scriptExecution = spawn("python.exe", ['./python/NLD_and_AM.py']);
  // Handle normal output
  scriptExecution.stdout.on('data', (data) => {
     pythonOutput = String.fromCharCode.apply(null, data);
  });

  // Write data (remember to send only strings or numbers, otherwhise python wont understand)
  var data = JSON.stringify([fileName]);
  scriptExecution.stdin.write(data);
  // End data write
  scriptExecution.stdin.end();
}


// Connect with the upload package
app.use(upload())

// To leave out the ejs extension
app.set("view engine", "ejs")

// Look in the static folder for all the css and js files, img and fonds
app.use(express.static("static"))

//running python files
//ps.PythonShell.run('./python/nodelink.py', null, function (err, results) {
  //if (err) throw err;
  //console.log('python script output!');
  //console.log(results);
//});


// When on just localhost, show landing page
app.get("/", function(req, res){
  res.render("landing");
})

// When on "localhost/upload" show upload page
app.get("/upload", function(req, res){
  res.render("upload");
})

// when on "localhost/grapgs" show the page
app.get("/graphs", function(req, res){
  res.render("graphs")
})

app.get("/nodelink", function(req, res) {
  res.render("nodelink");
})

// Upload file to upload dir
app.post("/", function(req, res){
  if (Object.keys(req.files).length == 0) {
     return res.status(400).send('No files were uploaded.');
   }

   // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
   var csvFile = req.files.csvFile;
   var fileName = csvFile.name;
   var fileName = uniqid() + ".csv";
   var filePath = './upload/' + fileName;

   // Use the mv() method to place the file somewhere on the server
   csvFile.mv(filePath, function(err) {
     if (err)
       return res.status(500).send(err);
       //res.redirect("nodelink");
      //res.redirect("/graphs/" + fileName);
   });
   filenameTo = fileName;
   console.log(fileNameTo + 'adasdad11')

   res.redirect("/choose/" + fileName)
   console.log(fileNameTo + 'adasdad1')

})

app.get("/viewmatrix/choose/:id", function(req, res){
  fileName = req.params.id
  fileNameWithNoExtension = fileName.slice(0, -4)
  fs.writeFile('views/graphs/' + fileNameWithNoExtension + 'matrix.ejs', loading,function (err) {
    if (err) throw err;
  console.log('File is created successfully.');
  res.redirect('/graphs/' + fileNameWithNoExtension + 'matrix.ejs')
  })
  dataToPyMatrix(fileName)
  //res.redirect('/graphs/' + fileNameWithNoExtension + 'matrix.ejs')
})

app.get("/viewnodelink/choose/:id", function(req, res){
  fileName = req.params.id
  fileNameWithNoExtension = fileName.slice(0, -4)
  fs.writeFile('views/graphs/' + fileNameWithNoExtension + 'nodelink.ejs', loading,function (err) {
    if (err) throw err;
  console.log('File is created successfully.');
  res.redirect('/graphs/' + fileNameWithNoExtension + 'nodelink.ejs')
  })
  dataToPyNodelink(fileName)
})

app.get("/viewboth/choose/:id", function(req, res){
  fileName = req.params.id
  fileNameWithNoExtension = fileName.slice(0, -4)
  fs.writeFile('views/graphs/' + fileNameWithNoExtension + 'both.ejs', loading,function (err) {
    if (err) throw err;
  //console.log('File is created successfully.');
  console.log('File is created successfully.');
  res.redirect('/graphs/' + fileNameWithNoExtension + 'both.ejs')
  })
  dataToPyBoth(fileName)

  //res.redirect('/graphs/' + fileNameWithNoExtension + 'both.ejs')
})

app.get("/choose/:id", function(req, res) {
  console.log(req.params.id);
  res.render("choose")
})

app.get("/graphs/:id", function(req, res){
  var id = req.params.id;
  res.render("graphs/" + id)
})

app.get("/about", function(req, res){
  res.render("about")
})



app.listen(process.env.PORT || port, () => console.log("Group 4 server has started!"));
