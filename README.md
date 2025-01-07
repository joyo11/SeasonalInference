Deep Learning for Computer Vision: Landsat Image Classification
Overview
In this project, I developed a model for classifying Landsat 8 satellite images based on their corresponding month. The goal was to use deep learning techniques to process and analyze geospatial data to recognize the patterns in satellite imagery. The model was built with the ResNet18 architecture, fine-tuned for the specific task of multi-class classification using Landsat OLI (Operational Land Imager) imagery.

Requirements
To run this project, you need to install the following Python libraries:

bash
Copy code
pip install torchgeo crcmod timm torch torchvision matplotlib
Additionally, ensure that you have access to Google Cloud services to download data from a cloud bucket.

Setup and Data Download
I used Google Cloud Storage to store the dataset. The script allows you to upload and activate a service account key, after which it will download the necessary files.

Upload the Service Account Key: To authenticate and access the cloud storage, you need to upload your Google Cloud service account key file.

Download the Dataset: The dataset is stored in a Google Cloud Storage bucket. The script uses gsutil to download the dataset from the cloud to your local environment.

python
Copy code
uploaded = files.upload()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/content/your-service-account-key.json'
!gsutil -m cp -r gs://your-bucket-name/folder_path /content
Dataset Description
The dataset consists of Landsat 8 images in TIFF format. Each image is associated with a label representing the month of acquisition. The labels are integers from 1 to 12, corresponding to January through December.

I used RasterDataset from the torchgeo library to load and process the Landsat 8 images. The images are processed by selecting specific bands that capture relevant information for classification.

Data Preprocessing
The data is split into three sets: training, validation, and testing. I performed the following steps for preprocessing:

Shuffling: To ensure randomness in the dataset, I shuffled the data.
Normalization: The images were normalized to a range between 0 and 1.
Data Augmentation: I did not use augmentation in this script, but you can add it if needed for further improvements.
python
Copy code
train_data = data[:10000]
val_data = data[10001:11250]
test_data = data[11250:]
Model
I built the model using the ResNet18 architecture, with weights pre-trained on Landsat 8 imagery (MOCO). Here's how I set up the model:

python
Copy code
weights = ResNet18_Weights.LANDSAT_OLI_SR_MOCO
in_chans = weights.meta["in_chans"]
model = timm.create_model("resnet18", in_chans=in_chans, num_classes=12)
I froze the initial layers of the model and only trained the fully connected layer to fine-tune the model for the multi-class classification task.

Training
The model was trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.1. I monitored the validation accuracy and stopped training when the validation accuracy exceeded 96%.

python
Copy code
optimizer = optim.SGD(model.parameters(), lr=0.1)
Training took place for several epochs. After each epoch, I validated the model using the validation set and saved the model weights when the performance improved.

Evaluation
After training, I evaluated the model on the test dataset. The final test accuracy was computed, and I reached an accuracy of over 98%.

python
Copy code
accuracy = 100 * (correct / total)
print(f'Test Accuracy: {accuracy:.2f}%')
Conclusion
This project demonstrates how deep learning can be used to classify satellite imagery based on specific temporal features. I achieved high accuracy using a ResNet18 architecture, fine-tuned for the Landsat 8 dataset. The project can be extended to other geospatial tasks or enhanced by incorporating more advanced data augmentation techniques.

Future Work
Some improvements and extensions I plan to explore include:

Data Augmentation: To increase the diversity of the dataset and improve the model's generalization.
Hyperparameter Tuning: Experimenting with different optimizers, learning rates, and batch sizes.
Fine-tuning other layers: Instead of freezing the initial layers, I could fine-tune the entire model to see if it improves performance.
License
This project is licensed under the MIT License.

