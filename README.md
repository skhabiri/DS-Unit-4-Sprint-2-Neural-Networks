# ML-ANN
Contains four modules on artificial neural network (ANN).
To get started:
Clone the repo to your local: `git clone https://github.com/skhabiri/ML-ANN.git`
For package management we are going to use conda. Then we create a ipykernel based on the conda environment and while the environment is activated, launch jupyter and set the kernel to the one just created. Here are summary of commands for conda and jupyter kernel:

### conda environment
*Create the virtual environment*: `conda create -n <env_name>`
*Activate the virtual environment*: `conda activate <env_name>`
*To list existing conda environments*: `conda env list`
*To remove conda environment*: `conda env remove -n <env_name>`

### Jupyter kernel
*Make sure that ipykernel is installed*: `pip install --user ipykernel`
*Add the new virtual environment to Jupyter*: `python -m ipykernel install --user --name <env_name> --display-name "<jupyter_env_name>"`
*To list existing Jupyter virtual environments*: `jupyter kernelspec list`
*To remove the environment from Jupyter*: `jupyter kernelspec uninstall <env_name>`

### Install required package:

`conda activate ML_NN`
`pip install -r requirements.txt`
√ ML-ANN % cat requirements.txt                            (main)ML-ANN
scikit-learn==0.22.2
seaborn==0.9.0
ipykernel
pandas
scipy
beautifulsoup4
tensorflow==2.4.0
keras==2.4.3
keras-tuner==1.0.2
tensorboard==2.4.1

Now in an activated conda environment launch jupyter lab and select ML_NN as the Ipython kernel. This will add the env/bin to the env PATH.
