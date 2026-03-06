import tiffile
import pandas as pd

from dataloaderCreator import *

# we have changed this again and again but we are FINALLY using the dataset I actually like
# this is like seeing bloddy handprints on the wall. I changed the dataset again.
# this time to the IU-xray dataset, as its probably the largest one I can get my hands on without causing a plethora of HIPAA violations.

if __name__ == "__main__":
    # this contains the metadata that we want for our MLP
    projections_path = 'IU-xray/indiana_projections.csv'
    # this contains the labelling we want for our image CNN
    reports_path = 'IU-xray/indiana_reports.csv'
    # and of course, the images
    image_dir = 'IU-xray/images/images_normalized'

    # DO REMEMBER THAT THIS ONLY DOES PNEUMONIA ATM
    train_loader, test_loader = get_dataloaders(projections_path, reports_path, image_dir)

    images, labels = next(iter(train_loader))
    print("iamge batch shape ", images.shape)
    print("label batch shape ", labels.shape)
    print("label distribution ", labels.sum().item())