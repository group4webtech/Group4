var express = require("express");
var router = express.Router();
// For reading the csv file
var fs = require("fs");
var csv = require("fast-csv");
var upload = require("express-fileupload");


// Connect with the upload package
router.use(upload())

// Upload file to upload dir
router.post("/upload", function(req, res){
  if (Object.keys(req.files).length == 0) {
     return res.status(400).send('No files were uploaded.');
   }

   // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
   var csvFile = req.files.csvFile;
   var fileName = csvFile.name;

   // Use the mv() method to place the file somewhere on your server
   csvFile.mv('./upload/' + fileName, function(err) {
     if (err)
       return res.status(500).send(err);

     res.send('File uploaded!');
   });
})

// Reading and translating csv file
fs.createReadStream("upload/SampleCSVFile_2kb.csv")
  .pipe(csv())
  .on("data", function(data){
    console.log(data);
  })
  .on("end", function(data){
    //console.log("Csv file read")
  })

module.exports = router;
