var express = require("express");
var router = express.Router();
// For db
var mysql = require("mysql");

// Create connection
var db = mysql.createConnection({
  host     : 'localhost',
  user     : 'root',
  password : '',
  database : 'csvDb'
});

// Connect db
db.connect((err) => {
  if(err){
    throw err;
  } else {
    console.log("Mysql connected")
  }
})

// Create db
router.get("/createdb", (req, res) => {
  var sql = "CREATE DATABASE csvDb";
  db.query(sql, (err, result) => {
    if(err) throw err;
    console.log(result);
    res.send("Database created")
  })
})

// Create table
router.get("/createtable", (req, res) => {
  var sql = "CREATE TABLE posts(fullName varchar(255))";
  db.query(sql, (err, result) => {
    if(err) throw err;
    console.log(result);
    res.send("Table created")
  })
})

module.exports = router;
