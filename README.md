# Pytorch Domain Adaptation Framework

A Pytorch framework deicated to experiments with domain adaptation papers. 

## Installations
To be honest, I have only reached this far in how to build a proper data science project.

## Using Conda

### Creating the Conda environment

After adding any necessary dependencies for your project to the Conda `environment.yml` file 
(or the `requirements.txt` file), you can create the environment in a sub-directory of your 
project directory by running the following command.

```bash
ENV_PREFIX=$PWD/env
conda env create --prefix $ENV_PREFIX --file environment.yml --force
```


Once the new environment has been created you can activate the environment with the following 
command.

```bash
conda activate $ENV_PREFIX
```

Note that the `ENV_PREFIX` directory is *not* under version control as it can always be re-created as 
necessary.

If you wish to use any JupyterLab extensions included in the `environment.yml` and `requirements.txt` 
files then you need to activate the environment and rebuild the JupyterLab application using the 
following commands to source the `postBuild` script.

```bash
conda activate $ENV_PREFIX # optional if environment already active
source postBuild
```

For your convenience these commands have been combined in a shell script `./bin/create-conda-env.sh`. 
Running the shell script will create the Conda environment, activate the Conda environment, and build 
JupyterLab with any additional extensions. The script should be run from the project root directory as 
follows. 

```bash
./bin/create-conda-env.sh
```

### Listing the full contents of the Conda environment

The list of explicit dependencies for the project are listed in the `environment.yml` file. To see 
the full lost of packages installed into the environment run the following command.

```bash
conda list --prefix $ENV_PREFIX
```

### Updating the Conda environment

If you add (remove) dependencies to (from) the `environment.yml` file or the `requirements.txt` file 
after the environment has already been created, then you can re-create the environment with the 
following command.

```bash
$ conda env create --prefix $ENV_PREFIX --file environment.yml --force
```

If you have added any JupyterLab extensions or made any other changes to the `postBuild` script, then you 
should re-create the entire Conda environment by re-running the `bin/create-conda-env.sh` scipt as follows.

```bash
./bin/create-conda-env.sh
```

## Using Docker

In order to build Docker images for your project and run containers you will need to install 
[Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/).

Detailed instructions for using Docker to build and image and launch containers can be found in 
the `docker/README.md`.
