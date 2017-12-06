import sys
import os 
sys.path.insert(0, 'C:/Users/Mavericks/Documents/Simple_LSTM/RNN-Implementation/src')

import tf_train_ctc as main_file

class raiseExcept(Exception):
    pass

def training_param(train_option):

        if train_option == 1:
            return True            #you're training a model
        else:
            return False
       
def main(config, name, debug, isTrain, started_by_server):

        tf_train = main_file.Tf_train_ctc(
            config_file=config, model_name=name, debug=debug, train_option=isTrain, server_start = started_by_server)

        # run the model
        tf_train.run_model()

if __name__ == "__main__":
    started_by_server = False
    try: 
        numb_of_args=len(sys.argv)-1

        if numb_of_args > 3 or numb_of_args < 2: 
            raise raiseExcept()
                
        train_option = int(sys.argv[1]) 
        
        if train_option != 0 and train_option != 1:
            raise raiseExcept()

        temp = sys.argv[1:4]
        if len(temp) == 3:
            if temp[2] == 'S117':
                started_by_server = True
        
    except(raiseExcept):
        print("Number of input parameters wrong! Correct format(0 or 1 for train option, string for model_name): 'train_option', 'model_name'")
        sys.exit()
     
    try:
        config='neural_network.ini'
        name=sys.argv[2]
        debug=False
        isTrain=training_param(train_option)        
        main(config,name,debug,isTrain,started_by_server)
       
    except(raiseExcept): 
        print("Error while starting TF model!")