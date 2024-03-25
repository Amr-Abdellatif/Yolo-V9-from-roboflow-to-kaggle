<div class="cell markdown">

# This repo gives you the ability to connect a RoboFlow dataset to Kaggle notebok and using the Gpu's for training and evaluation

</div>

<div class="cell markdown">

First let's install needed packages

</div>

<div class="cell code" execution_count="2"
_cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
_uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
execution="{&quot;iopub.execute_input&quot;:&quot;2024-03-25T21:10:54.597763Z&quot;,&quot;iopub.status.busy&quot;:&quot;2024-03-25T21:10:54.597458Z&quot;,&quot;iopub.status.idle&quot;:&quot;2024-03-25T21:11:18.662975Z&quot;,&quot;shell.execute_reply&quot;:&quot;2024-03-25T21:11:18.661761Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2024-03-25T21:10:54.597734Z&quot;}"
trusted="true">

``` python
!pip install roboflow -q
!pip install ultralytics -q
```

</div>

<div class="cell markdown">

Grab the dataset link from roboflow

</div>

<div class="cell code"
execution="{&quot;iopub.execute_input&quot;:&quot;2024-03-25T21:11:24.984381Z&quot;,&quot;iopub.status.busy&quot;:&quot;2024-03-25T21:11:24.984011Z&quot;,&quot;iopub.status.idle&quot;:&quot;2024-03-25T21:11:38.552719Z&quot;,&quot;shell.execute_reply&quot;:&quot;2024-03-25T21:11:38.551809Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2024-03-25T21:11:24.984347Z&quot;}"
trusted="true">

``` python
from roboflow import Roboflow
rf = Roboflow(api_key="your roboflow api key here")
project = rf.workspace("workspace").project("project name") 
version = project.version(4)
dataset = version.download("yolov9")
```

</div>

<div class="cell markdown">

in order to change the yaml file configuration to match your data train
test valid folders you will execute the following code

</div>

<div class="cell code" execution_count="4"
execution="{&quot;iopub.execute_input&quot;:&quot;2024-03-25T21:11:44.828785Z&quot;,&quot;iopub.status.busy&quot;:&quot;2024-03-25T21:11:44.828431Z&quot;,&quot;iopub.status.idle&quot;:&quot;2024-03-25T21:11:44.839282Z&quot;,&quot;shell.execute_reply&quot;:&quot;2024-03-25T21:11:44.838353Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2024-03-25T21:11:44.828756Z&quot;}"
trusted="true">

``` python
import yaml

def update_yaml_content(file_path, new_content):
    try:
        # Write the new content to the YAML file
        with open(file_path, 'w') as file:
            yaml.dump(new_content, file)

        print("File updated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define the yaml file path
file_path = '/kaggle/working/dataset/data.yaml'

# Define the new content
new_content = {
    'names': [
        'Early Blight',
        'Healthy',
        'Late Blight',
        'Leaf Miner',
        'Leaf Mold',
        'Mosaic Virus',
        'Septoria',
        'Spider Mites',
        'Yellow Leaf Curl Virus'
    ],
    'nc': 9,
    'roboflow': {
        'license': 'CC BY 4.0',
        'project': 'proj name',
        'url': 'url',
        'version': 4,
        'workspace': 'workspace'
    },
    'test': '/kaggle/working/dataset/test',
    'train': '/kaggle/working/dataset/train',
    'val': '/kaggle/working/dataset/valid'
}

# Call the function to update the YAML file
update_yaml_content(file_path, new_content)
```

<div class="output stream stdout">

    File updated successfully.

</div>

</div>

<div class="cell markdown">

check if yaml file is updated correctly or not

</div>

<div class="cell code"
execution="{&quot;iopub.execute_input&quot;:&quot;2024-03-25T21:11:46.880650Z&quot;,&quot;iopub.status.busy&quot;:&quot;2024-03-25T21:11:46.880193Z&quot;,&quot;iopub.status.idle&quot;:&quot;2024-03-25T21:11:46.891312Z&quot;,&quot;shell.execute_reply&quot;:&quot;2024-03-25T21:11:46.890157Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2024-03-25T21:11:46.880617Z&quot;}"
trusted="true">

``` python
import yaml

def view_yaml_file(file_path):
    with open(file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
        return yaml_content

file_path = "/kaggle/working/dataset/data.yaml"
yaml_content = view_yaml_file(file_path)
print(yaml_content)
```

</div>

<div class="cell markdown">

### Train code for Yolo-V9

</div>

<div class="cell code"
execution="{&quot;iopub.execute_input&quot;:&quot;2024-03-25T21:20:46.535076Z&quot;,&quot;iopub.status.busy&quot;:&quot;2024-03-25T21:20:46.534402Z&quot;}"
trusted="true">

``` python
from ultralytics import YOLO

# Build a YOLOv9c model from scratch
model = YOLO('yolov9e.yaml')

# Build a YOLOv9c model from pretrained weight
model = YOLO('yolov9e.pt')

# Display model information (optional)
model.info()
# Train the model
results = model.train(data='/kaggle/working/dataset/data.yaml', epochs=300,
                    batch=32,patience=10,cache=True,save_period=30,imgsz=300)
```

</div>
