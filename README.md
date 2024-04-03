Project material and documentation on usage. 

Preliminary: 
 - install conda environment from environment.yml
 - install [Geometry Scripts](https://carson-katri.github.io/geometry-script/setup/installation.html) if wish to experiment with node tree generation from Python

Shape program code: 
 - basic_building.py, building_mass.py, building4distortion.py, ledge.py, roof.py, shape_primitives.py, window_with_ledge.py
Dataset generation code: 
 - dataset_counter.py, dataset_gen.py, distortion.py, generate_dataset.py, merge_dataset.py, paramgen.py, paramload.py, params.py, render.py
Neural network code:
 - nn_*.py
Evaluation code: 
 - average_performances.py, nn_acc.py, performance.py
User Interface code: 
 - ui_*.py

Blend files: 
 - dataset.blend for generating synthetic datasets
 - interface.blend for user interface
 - distortion.blend for distorted sketches rendering
 - dataset_distortion.blend for generating distortion datasets 

Output files:
 - ./models/*: model training output, including checkpoint, loss records, meta info for backup, loss curve visualization and notes file. 
 - ./datasets/*: dataset directory, containing generated DAGDataset(s) and DAGDatasetDistorted(s)
 - in working directory: results*.yml containing model test outputs, performance*.yml containing model evaluation results, performance*.pdf visualizing model evaluation results
 - ./inference/*: captured sketch and model output files

Pipelines: 
 - Generating dataset: run dataset_gen.py or generate_dataset.py with commandline args. Grammar: python generate_dataset.py batch_num sample_num varying_params_num device distortion;  e.g. python generate_dataset.py 10 10 5 0 0
 - Neural network training: run nn_driver.py; modify config in code as needed. 
 - User Interface: open interface.blend with Blender 3.2, go to Scripts and run ui_interface.py, the panel should appear under tool section; for testing without PyTorch installation and model weight files, switch the import from using ui_external_inference.py to use ui_mock_inference.py; Click the pencil icon to use Blender's annotation tool to draw. Toggle and adjust camera view as needed. To run inference, make sure to have proper model weights in ./models/EncDecModel.pth and a corresponding meta file in ./models/meta.yml and to have created the ./inference folder. 

