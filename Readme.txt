(a) Software that needs to be installed (if any) with URL’s to download it from 
and instructions on how to install them.

Anaconda Python 2.7: http://continuum.io/downloads
Download "Windows 64-Bit Python 2.7 Graphical Installer" from above link

IGraph Python Module: http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph
Download the python-igraph-0.7.0.win-amd64-py2.7.exe file from the above link and install. 

windows powershell


(b) Environment variable settings (if any) and OS it should/could run on.
The program has been tested on Windows 8.0 (64bit)
The program is written in pycharm IDE and executed in pyCharm.



(c) Instructions on how to run the program
The folder where the program is executed should have graph and anomalies folders as the program outputs the graph in graph folder and anomalies.txt in anomalies folder

Change the directory to the project folder which has the Netsimile.py file , graph folder and anomalies folder.
> cd <project directory>
> ls
Netsimile.py graph anomalies
Execute the following command:
>python Netsimile.py <path to input files> <name of output file>
E.g. 
>python Netsimile.py C:/Users/Kristy/PycharmProjects/Netsimile/input_files enron_test

The above command will produce enron_test.txt in anomalies folder and enron-test.png on graph folder.

(d) Instructions on how to interpret the results.
The results are saved in graph and anomalies folder
The output text files are stored in anomalies folder. The output file shows the threshold, number of anomalies and the anomalous time points on each line. 

(e) Sample input files are stored in input_files folder and output files are in anomalies folder while graphs are stored in graph folder in the zip file submitted.
