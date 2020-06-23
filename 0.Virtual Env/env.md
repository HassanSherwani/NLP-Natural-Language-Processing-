### Install Anaconda 3.7

- https://docs.anaconda.com/anaconda/install/

- Also install miniconda latest

### For creating an environment using .yml file
```bash
- ren file.yml.text file.yml 
```
(change yml from text to pure yml file)
```bash
- conda env create -v -f environment.yml

```
(environment.yml is tensorflow.yml in our case)

### without .yml file
```bash
- conda create --name <name_of_environment> python==3.7.6
```
### To check how many or what environment we have
```bash
- conda env list
```
### To activate an environment 
```bash
conda activate <name_environment>
```
(tensorflow)>

### To Install Kernel for Jupyter
```bash
python -m ipykernel install --user --name tensorflow --display-name "Python 3.7 (tensorflow)"
```
### To remove an environment
```bash
conda env remove -n <ENV_NAME>
```
### installing package

if .yml then tensorflow is already installed with pip as well. If other method used then

### intsall ujson
```bash
conda install ujson
```
### install tensorflow (latest and updated)
```bash
conda install tensorflow
```
### Download nodejs, Visual studio C++ and npm

https://docs.npmjs.com/downloading-and-installing-node-js-and-npm (For nodejs and npm installation)
```bash
npm i -g rasa-nlu-trainer
```
(Only if we are working with Rasa environment)
### install other packages
```bash
pip install <name>
 ```
### To check version of modules
```bash
Python
Import tensorflow as tf
Print(tf.__version__)
 ```
### Seeing version using pip
  ```bash
 pip show <name>
 ```
  

### References

- Jeff Heaton : https://www.youtube.com/watch?v=RgO8BBNGB8w&t=352s
- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
