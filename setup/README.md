1. install Anaconda
* `wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh`
* `chmod +x Anaconda3-2022.05-Linux-x86_64.sh`
* `./Anaconda3-2022.05-Linux-x86_64.sh`, and follow the installation steps.
* `source ~/.bashrc`, you are supposed to see the `(base)` leading your CLI
2. Modify prefix in `PATH_TO_ENVIRONMENT_YML/environment.yml`
3. `conda env create -f PATH_TO_ENVIRONMENT_YML/environment.yml`
4. `conda activate rbe549`
