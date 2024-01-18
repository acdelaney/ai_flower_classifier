import argparse
from PIL import Image
import numpy as np
import torch
import json
import os
from torchvision import datasets, transforms

def get_input_args_predict():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    Command Line Arguments:
      1. image_path
      2. checkpoint_path
      3. Top K as --top_k with default value of 3
      4. Category Names as --category_names with default value of 'cat_to_name.json'
      5. GPU as --gpu with default value of False
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None 
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type = str, 
                    help = 'path to the flower image') 
    parser.add_argument('checkpoint_path', type = str, 
                    help = 'path to the model checkpoint to use') 
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = 'number of probabilities to return') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'category mapping to use for inference')  
    parser.add_argument('--gpu', action='store_true', help = 'attempts to use gpu for training when available')

    return parser.parse_args()

def get_input_args_train():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    Command Line Arguments:
      1. data_dir
      2. Save Directory as --save_dir with default path of '/'
      3. CNN Model Architecture as --arch with default value 'alexnet'
      4. Learning Rate as --learning rate with default value 0.01
      5. Hidden Units as --hidden_units with default value of 512
      6. Epochs as --epochs with default value of 20
      7. GPU as --gpu with default value of False
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None 
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help = 'path to the folder where image data is located') 
    parser.add_argument('--save_dir', type = str, 
                    help = 'path to the folder where checkpoints are saved') 
    parser.add_argument('--arch', type = str, default = 'alexnet', 
                    help = 'cnn model architecture to use') 
    parser.add_argument('--learning_rate', type = float, default = 0.01, 
                    help = 'learning rate to use for model hyperparameter')
    parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'hidden units to use for model hyperparameter') 
    parser.add_argument('--epochs', type = int, default = 20, 
                    help = 'epochs to use for training model')
    parser.add_argument('--gpu', action='store_true', help = 'attempts to use gpu for training when available')

    return parser.parse_args()

def print_results(probs, classes, class_to_idx, category_to_names):
    """
    Prints the results from the inference.
    Parameters:
     probs - tensor of probabilities
     classes - tensor of classes
     class_to_idx - dict of classes to indices
     category_to_names - json dict of categories to indices 
    Returns:
     None - results are printed
    """
    np_probs = probs.numpy()
    np_classes = classes.numpy()
    categories = []
    indices = []

    with open(category_to_names, 'r') as f:
        cat_to_name = json.load(f)
    
    for i in np_classes[0]:
        for k,v in class_to_idx.items():
            if v == i:
                indices.append(k)
                break

    
    for i in indices:
        for k,v in cat_to_name.items(): 
            if i == k:
                categories.append(v)
                break

    print(f"The most likely image class is {categories[0]} with a probability of {np_probs[0][0]}")

    if len(categories) > 1:
        print(f"The next likely classes are {categories[1:]} with their probabilities {np_probs[0][1:]}, respectively.")

    


def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    Parameters:
     image_path - path to the image
    Returns:
     tensor object of the image
    '''
    with Image.open(image_path) as im:
        #Resize to 256 using the thumbnail method to keep the aspect ratio
        max_size = (256, 256)
        im.thumbnail(max_size)
        
        #Crop the center portion of the image to 224x224 px.
        new_width, new_height = (224, 224)     
        width, height = im.size
        
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        im_cropped = im.crop((left, top, right, bottom)) # type: ignore
        
        #Convert color channels for the image
        np_image = np.array(im_cropped)
    
        #Normalize image
        reduced_image =  np_image / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_normalized = (reduced_image - mean) / std
        np_normalized = np.float32(np_normalized)
    
        #Transpose image so that color channel is first and convert to Tensor
        transposed_image = np.transpose(np_normalized, axes=[2,0,1])
        
        return torch.from_numpy(transposed_image)


def load_training_data(data_dir):
    '''
    Loads and prepares training data from a director that includes three subdirs: train, valid, and test.  These should include flower pictures that will be used to train the model.
    Parameters:
     data_dir - dir that has the flower images
    Returns:
     dataloaders - dict of dataloaders for each subdir
     image_datasets - dict of image datasets that were used to create each dataloader
    '''
    #check directory for sub directors
    sub_dirs = os.listdir(data_dir)
    dataloaders = {}
    image_datasets = {}

    if 'train' not in sub_dirs or 'valid' not in sub_dirs or 'test' not in sub_dirs:
        raise ValueError("Directory must contain subfolders: train, valid, and test")
     
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

    dataloaders['train'] = trainloader
    dataloaders['valid'] = validloader
    dataloaders['test'] = testloader

    image_datasets['train'] = train_dataset
    image_datasets['valid'] = valid_dataset
    image_datasets['test'] = test_dataset

    return dataloaders, image_datasets