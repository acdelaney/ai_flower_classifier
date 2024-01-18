import torch
from torch import nn, optim
from torchvision import models
import torch.nn.functional as F
from helper import process_image
import time

class Classifier(nn.Module):
    ''' 
    Class for a neural network classifier.  This network will be added as the classifier to a
    pretrained model for feature detection.  There is only one hidden layer whereas the input
    units and hidden units are dynamic.
    Parameters:
     input_units - number of input units
     hidden_units - number of hidden_units
    '''
    def __init__(self, input_units, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 102)
        
        # Define proportion of neurons to dropout
        self.dropout = nn.Dropout(0.50)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

def create_network(cnn_arch, hidden_units):
    ''' 
    Creates a network based on a pretrained network and number of hidden units. The model
    is ready to train.
    Parameters:
     cnn_arch - pretrained network to use: either alexnet or vgg13
     hidden_units - number of hidden units to add to the classifier
    Returns:
     model - a neural network
    '''
    archs = ['alexnet', 'vgg13']

    if cnn_arch not in archs:
        raise ValueError("CNN Arch must be either alexnet or vgg13")
    
    if cnn_arch == 'alexnet':
        model = models.alexnet(weights='IMAGENET1K_V1')
        input_units = 9216
    else:
        model = models.vgg13(weights='IMAGENET1K_V1')
        input_units = 25088

    #Freeze model parameters so I don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = Classifier(input_units, hidden_units)
    model.classifier = classifier # type: ignore

    return model

def predict(image_path, model, topk, gpu=False):
    ''' 
    Predict the class (or classes) of a flower image using a trained deep learning model.
    Parameters:
     image_path - path to the image to predict
     model - the trained model to use
     topk - the number of classes to return predictions for
     gpu - a bool on whether to use gpu or cpu
    Returns:
     top_p - the tensor of probabilities
     top_class - the tensor of classes
    '''
    image = process_image(image_path)
    image = image.view([1,3,224,224])
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        
        return top_p, top_class
    

def train(dataloaders, model, learning_rate, epochs, gpu):
    ''' 
    Trains a deep learning model and prints out important stats such as
    loss and accuracy. 
    Parameters:
     dataloaders - dict of dataloaders
     model - neural network to train
     learning_rate - the learning rate to use during training
     epochs - the number of epochs to train for
     gpu - a bool on whether to use gpu or cpu
    Returns:
     None - the model is trained and the relevant information printed
    '''
    #Training the network
    train_losses, valid_losses = [], []
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        print("Training starting")
        start = time.time()
        for images, labels in dataloaders['train']:
            # Move images and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                print("Validation Starting")
                #Clear memory cache if on gpu
                if gpu:
                    torch.cuda.empty_cache()

                for images, labels in dataloaders['valid']:
                    # Move images and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)
                    
                    log_ps = model.forward(images)
                    loss = criterion(log_ps, labels)
                    valid_loss += loss.item()
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            train_losses.append(running_loss/len(dataloaders['train']))
            valid_losses.append(valid_loss/len(dataloaders['valid']))
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders['train'])),
                "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloaders['valid'])),
                "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))
            print(f"Time per batch: {(time.time() - start):.3f} seconds")

            model.train()

def save_checkpoint(train_dataset, model, arch, hidden_units, save_dir):
    ''' 
    Saves a checkpoint.pth for a model.
    Parameters:
     train_dataset - the image dataset that the model was trained on
     model - the model to save
     arch - the pretrained network that was used for the model - alexnet or vgg13
     hidden_units - the number of hidden units used for the model
     save_dir - the directory to save the checkpoint.pth in
    Returns:
     None - the checkpoint is saved and stored as a file
    '''
    path = save_dir + '/checkpoint.pth' if save_dir else 'checkpoint.pth'
    checkpoint = {'class_to_idx': train_dataset.class_to_idx,
              'state_dict': model.state_dict(),
              'pretrained_model': arch,
              'hidden_units': hidden_units}
    print("Saving the model checkpoint.")
    torch.save(checkpoint, path)

def load_checkpoint(filepath):
    ''' 
    Loads a checkpoint.pth for a model.
    Parameters:
     filepath - the path to the checkpoint.pth
    Returns:
     model - the model loaded from the checkpoint
    '''
    checkpoint = torch.load(filepath)
    model = create_network(checkpoint['pretrained_model'], checkpoint['hidden_units'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
