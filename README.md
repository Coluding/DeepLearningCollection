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

📦src
 ┣ 📂computer_vision
 ┃ ┣ 📂object_detection
 ┃ ┃ ┣ 📂center_point_detection
 ┃ ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┃ ┣ 📜center_point_model.py
 ┃ ┃ ┃ ┣ 📜center_point_model_utils.py
 ┃ ┃ ┃ ┣ 📜img.png
 ┃ ┃ ┃ ┗ 📜img_1.png
 ┃ ┃ ┣ 📂retinanet
 ┃ ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┃ ┣ 📜img.png
 ┃ ┃ ┃ ┣ 📜img_1.png
 ┃ ┃ ┃ ┣ 📜loss.py
 ┃ ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┃ ┣ 📜train.py
 ┃ ┃ ┃ ┣ 📜utils.py
 ┃ ┃ ┃ ┗ 📜validation.py
 ┃ ┃ ┣ 📂yolo
 ┃ ┃ ┃ ┣ 📜yolo_losses.py
 ┃ ┃ ┃ ┗ 📜yolo_models.py
 ┃ ┃ ┗ 📜object_detection_utils.py
 ┃ ┣ 📂segmentation
 ┃ ┃ ┣ 📂deeplab
 ┃ ┃ ┃ ┣ 📜aspp.py
 ┃ ┃ ┃ ┗ 📜deeplabv3.py
 ┃ ┃ ┣ 📂unet
 ┃ ┃ ┃ ┗ 📜unet_models.py
 ┃ ┃ ┗ 📜segmentation_utils.py
 ┃ ┗ 📜model_utils.py
 ┗ 📂nlp
 ┃ ┣ 📂attention
 ┃ ┃ ┗ 📜lsh-attention_tf.py
 ┃ ┣ 📂electra_pretraining_framework
 ┃ ┃ ┗ 📜modelling_electra.py
 ┃ ┗ 📂reversible_dilated_bert
 ┃ ┃ ┣ 📜blocks.py
 ┃ ┃ ┣ 📜embeddings.py
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┣ 📜reversible_layer.py
 ┃ ┃ ┗ 📜sliding_window_attention.py
