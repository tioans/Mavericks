import sys
sys.path.insert(0, 'C:/Users/Mavericks/Documents/Simple_LSTM/RNN-Implementation/src')

import tf_train_ctc as main_file

def main(config='neural_network.ini', name='develop_LSTM_20171201-141710', debug=False):

        tf_train = main_file.Tf_train_ctc(
            config_file=config, model_name=name, debug=debug)

        # run the training
        tf_train.run_model()

main()