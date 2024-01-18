from helper import get_input_args_predict, print_results
from model import load_checkpoint, predict

def main():
    ''' 
    This function takes a flower image and predicts what flower it is using
    inference from a trained deep learning network.  To execute you must provide
    the following command line arguments: 
    Command Line Arguments:
      1. image_path
      2. checkpoint_path
      3. Top K as --top_k with default value of 3
      4. Category Names as --category_names with default value of 'cat_to_name.json'
      5. GPU as --gpu with default value of False
    The results of the prediction will be printed out in the terminal.
    '''
    #Retrieve the input arguments
    input_args = get_input_args_predict()

    #Load the checkpoint and return the model
    model = load_checkpoint(input_args.checkpoint_path)

    #Perform inference on the image and return the probabilities and classes
    probs, classes = predict(input_args.image_path, model, input_args.top_k, input_args.gpu)

    #Print the results
    print_results(probs, classes, model.class_to_idx, input_args.category_names)



# Call to main function to run the program
if __name__ == "__main__":
    main()