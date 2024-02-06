## Deep Learning Collection

- Collection repository where all my implemented models are located to make them reusable
- It is not complete yet, the rest of the models will be added in the future.

### Computer Vision
- Object Detection
- Object Segmentation
- Self Supervised Learning
- Diffusion

### Natural Language Processing
- Encoder model
- Recurrent models
- Efficient attention mechanisms
- Electra Pretraining


### Repository structure
Each subdirectory with a different model has its own README with all the details

```
project
│   README.md
│   
│
└───src
│   └─── nlp
|   |    |
|   |    └─── attention
|   |    |     └─── lsh_attention_tf
|   |    └─── electra_pretraining_framework
|   |    |
|   |    └─── reversible_dilated_bert
│   └───computer_vision
|        |
|        └─── object_detection
|        |     |
|        |     └─── center_point_detection
|        |     └─── retinanet
|        |     └─── yolo
|        |     object_detection_utils.py
|        |
|        └─── segmentation
|              |
|              └─── deeplab
|              └─── unet
|              segmenation_utils.py
```
