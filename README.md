# Deep Learning for Computer Vision: Seasonal Inference Landsat Image Classification

## Overview
In this project, I developed a model for classifying **Landsat 8** satellite images based on their corresponding month. The goal was to use deep learning techniques to process and analyze geospatial data to recognize patterns in satellite imagery. The model was built using the **ResNet18** architecture, fine-tuned for the specific task of multi-class classification using **Landsat OLI (Operational Land Imager)** imagery.

## Requirements
To run this project, you need to install the following Python libraries:

```bash
pip install torchgeo crcmod timm torch torchvision matplotlib
```


## Conclusion
This project demonstrates how deep learning can be used to classify satellite imagery based on specific temporal features. I achieved high accuracy using a ResNet18 architecture, fine-tuned for the Landsat 8 dataset. The project can be extended to other geospatial tasks or enhanced by incorporating more advanced data augmentation techniques.

### Future Work
Some improvements and extensions I plan to explore include:

Data Augmentation: To increase the diversity of the dataset and improve the model's generalization.
Hyperparameter Tuning: Experimenting with different optimizers, learning rates, and batch sizes.
Fine-tuning other layers: Instead of freezing the initial layers, I could fine-tune the entire model to see if it improves performance.


### License
This project is licensed under the MIT License.


