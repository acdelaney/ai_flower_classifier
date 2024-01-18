from helper import get_input_args_train, load_training_data
from model import create_network, save_checkpoint, train

def main():
    ''' 
    This function creates and trains a deep learning neural network. To execute
    you must provide the following command line arguments: 
    Command Line Arguments:
      1. data_dir
      2. Save Directory as --save_dir with default path of '/'
      3. CNN Model Architecture as --arch with default value 'alexnet'
      4. Learning Rate as --learning rate with default value 0.01
      5. Hidden Units as --hidden_units with default value of 512
      6. Epochs as --epochs with default value of 20
      7. GPU as --gpu with default value of False
    The trained model will be saved as a checkpoint.pth that can be used for inference.
    '''
    #Retrieve the input arguments
    input_args = get_input_args_train()
    
    #Load Dataset
    dataloaders, datasets = load_training_data(input_args.data_dir)
    
    #Create Network
    model = create_network(input_args.arch, input_args.hidden_units)
    
    #Train Network
    train(dataloaders, model, input_args.learning_rate, input_args.epochs, input_args.gpu)

    #Save Network
    save_checkpoint(datasets['train'], model, input_args.arch, input_args.hidden_units, input_args.save_dir) # type: ignore


# Call to main function to run the program
if __name__ == "__main__":
    main()