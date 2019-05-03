var express = require("express");
var app = express();
var bodyParser = require("body-parser");
var port = 3000;
var uploadToFolder = require("express-fileupload");
// For reading the csv file
var fs = require("fs");
var csv = require("fast-csv");
// For uploading to mongodb
var path = require("path");
// generating filenames
var crypto = require("crypto");
// interacting with db
var mongoose = require("mongoose");
var multer = require("multer");
var gridFsStorage = require("multer-gridFs-storage");
var grid = require("gridfs-stream");
var methodOverride = require("method-override");

// Middleware
app.use(bodyParser.json());
app.use(methodOverride("_method"));

// Mongo URI
var mongoUri = "mongodb+srv://group4:groupfour@cluster0-365f0.mongodb.net/test?retryWrites=true";

// Create mongo connection
var conn = mongoose.createConnection(mongoUri);

// Init gridFs
var gfs;

conn.once("open", () => {
  // Init stream
  gfs = grid(conn.db, mongoose.mongo);
  gfs.collection("uploads");
})

// Create storage engine
var crypto = require('crypto');
var path = require('path');
var GridFsStorage = require('multer-gridfs-storage');

var storage = new GridFsStorage({
  url: mongoUri,
  file: (req, file) => {
    return new Promise((resolve, reject) => {
      crypto.randomBytes(16, (err, buf) => {
        if (err) {
          return reject(err);
        }
        var filename = buf.toString('hex') + path.extname(file.originalname);
        var fileInfo = {
          filename: filename,
          bucketName: 'uploads'
        };
        resolve(fileInfo);
      });
    });
  }
});
var upload = multer({ storage });

// @route POST /upload
// @desc Uploads file to db
app.post("/upload", upload.single("csvFile"), (req, res) => {
  res.json({file: req.file});
})

// To leave out the ejs extension
app.set("view engine", "ejs")

// Insert post
app.get("/addpost", (req, res) => {
  var post = "";
})

// Connect with the upload package
app.use(uploadToFolder())

// Look in the static folder for all the css and js files, img and fonds
app.use(express.static("static"))


// When on just localhost, show landing page
app.get("/", function(req, res){
  res.render("landing");
})

// When on "localhost/upload" show upload page
app.get("/upload", function(req, res){
  res.render("upload");
})

// Upload file to upload dir
/* app.post("/", function(req, res){
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
}) */

// Reading and translating csv file
fs.createReadStream("upload/GephiMatrix_co-citation.csv")
  .pipe(csv())
  .on("data", function(data){
    //console.log(data);
  })
  .on("end", function(data){
    //console.log("Csv file read")
  })


app.listen(port, () => console.log("Group 4 server has started!"));
