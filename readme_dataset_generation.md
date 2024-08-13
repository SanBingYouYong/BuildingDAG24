1. Clone the repo and `cd BuildingDAG`
2. Run the install script `bash dataset_gen_set_up.sh`
     - the tasks are outlined in the last section
3. Run blender with conda env activated and execute the script
     - if not already following install script: `conda activate dag_distort`
     - ` blender/blender-3.2.2-linux-x64/blender dataset_distortion.blend` and the window should pop up
     - the script `dataset_gen.py` should be opened already and press `alt+p` or click the button to run it
         - optionally, check the `args` at `line 129`: 
             - [100, 100, 5, 0, 1]
             - [batches, samples, vary_params, **gpu index**, use_distortion], adjust which gpu to use as needed and make sure the output log matches config ↓
     - check on the terminal log outputs: e.g. 
         - ```
            Read blend: ~\BuildingDAG\dataset_distortion.blend
            Current working directory: ~\BuildingDAG
            activated gpu NVIDIA GeForce RTX 4070 Laptop GPU
            Using 100 batches, 100 samples per batch, 5 varying params
            Using device: 0
            Using distortion: True
            NVIDIA GeForce RTX 4070 Laptop GPU CUDA True
            13th Gen Intel Core i9-13900HX CPU False
            NVIDIA GeForce RTX 4070 Laptop GPU OPTIX False
            ```
     - and the logs should be rolling by now. 
    
### Optionally, using an `envs.tar` archive instead of configuring the environment
**Known issue: opencv recursive import**
Steps: 
     - download the `envs.tar` file to `./blender` and unzip it: 
         - `tar -xf envs.tar`
     - copy paste the content of `./envs` to `blender-3.2.2-linux-x64/3.2/python/lib/python3.10/site-packages/`
        - `cp -r envs/* blender-3.2.2-linux-x64/3.2/python/lib/python3.10/site-packages/`
     - run the blender file as usual, an opencv import error is expected, if not, nice.

### Install Script tasks: 
If you wish to manually do it: 
1. Download the blender: 
     - `cd ./blender`
     - `wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz`
     - `tar -xf blender-3.2.2-linux-x64.tar.xz`
     - remove blender's own python: 
         - ` rm blender-3.2.2-linux-x64/3.2/python/ -r`
     - `cd ../` go back to project root dir
2. Install dependencies: 
     - conda (pip, take the spirit): 
         - `conda create -n dag_distort python=3.10 -y`
         - `conda activate dag_distort`
         - `pip install opencv-python PyYAML tqdm numpy`
         - optional: check if conda installed its own `freestyle` package: 
             - check in `<conda_home_path>/envs/dag_distort/lib/python3.10/site-packages/freestyle`
                 - if exists, delete everything in it
                 - then copy paste `./blender/blender-3.2.2-linux-x64/3.2/scripts/freestyle/` into it

