# Install Anaconda 3.7

- https://docs.anaconda.com/anaconda/install/

- Also install miniconda latest

# For creating an environment using .yml file

- ren file.yml.text file.yml 
(change yml from text to pure yml file)

- conda env create -v -f environment.yml
(environment.yml is tensorflow.yml in our case)

# without .yml file

- conda create --name <name_of_environment> python==3.7.6

# To check how many or what environment we have

- conda env list

# To activate an environment 

conda activate <name_environment>
(tensorflow)>

# To Install Kernel for Jupyter

python -m ipykernel install --user --name tensorflow --display-name "Python 3.7 (tensorflow)"

# To remove an environment

conda env remove -n <ENV_NAME>

# installing package

if .yml then tensorflow is already installed with pip as well. If other method used then

# intsall ujson

conda install ujson

# install tensorflow (latest and updated)

conda install tensorflow

# Download nodejs, Visual studio C++ and npm

https://docs.npmjs.com/downloading-and-installing-node-js-and-npm (For nodejs and npm installation)

npm i -g rasa-nlu-trainer

# install other packages

pip install <name>
 
# To check version of modules

Python
Import tensorflow as tf
Print(tf.__version__)

# Seeing version using pip
  
 pip show <name>
  
  
  
  
  
  
  
  

References

- Jeff Heaton : https://www.youtube.com/watch?v=RgO8BBNGB8w&t=352s
- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
