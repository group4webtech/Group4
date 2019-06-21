GROUP 4 - Visualize DAT

Our website is not running online currently. To install the website:
- install Node.js from https://nodejs.org/en/
- (optional) install all the node packages with:
	npm install express, fs, python-shell, child_process, uniqid
- install python from https://www.python.org/downloads/
- install all the python packages, which are imported at the beginning of the python files in folder 'python'.
	You ca do that, by going to the python folder and typing in command prompt: 
	python -m pip install --upgrade pip, pandas, bokeh, networkx, maplotlib, sklearn, bs4, sys, json, os
- open command prompt and go to the directory where you have downloaded the website
- type "node app.js" in the command prompt to run the localhost
- go to "localhost:3000" to view the website

How to use the website:
- click on Upload Dataset -> click on choose file -> select your file -> click on upload
- then pick which visualization you want to see (node-link diagram, adjacency matrix or both)
	a new browser window will open when the visualization is prepared and ready to be shown,
	please be patient, this may take some time, depending on the input data and chosen visualization
- to choose another visualization, you should first press the 'back' button on the initial browser window (not the newly open one)
	to go again on the page with four options
- to see the statistics page, press on the 'Statistics' button, wait some time until calculations are prepared
	and refresh the page in the browser (ctrl+F5)
	
Note:
 - in order to run faster for demonstration, the adjacency matrix is configured in python to run with only part of the input data 
 - currently the adjacency matrix cannot run with other files, different than the provided for the assignment:
 	'GephiMatrix_author_similarity.csv'
	'GephiMatrix_co-authorship.csv'
	'GephiMatrix_co-citation.csv'
	
Backup Plan:
 - if the website is not installed on your PC, then you can see example of visualizations made by our tool,
 	by opening in the browser the following files located in 'backup_plan' folder:
	'indexmatrix.html'
	'indexnodelink.html'
	'indexboth.html'
 - you can also run only the python code without the website from the same folder:
 	'NLD.py'
	'AM.py'
	'NLD_and_AM.py'
	'DBL_stat.py' - see the result at the beginning of the python status window
