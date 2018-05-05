import tensorflow as tf
import json

from data_utils import Data
from models.char_cnn_zhang import CharCNNZhang
from models.char_cnn_kim import CharCNNKim

tf.app.flags.DEFINE_string("model", "char_cnn_zhang", "Specifies which model to use: char_cnn_zhang or char_cnn_kim")
FLAGS = tf.app.flags.FLAGS

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))
    # Load training data
    training_data = Data(data_source=config["data"]["training_data_source"],
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()
    # Load validation data
    validation_data = Data(data_source=config["data"]["validation_data_source"],
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Load model configurations and build model
    if FLAGS.model == "kim":
        modelConfig = config['char_cnn_kim']
        model = CharCNNKim(input_size=config["data"]["input_size"],
                           alphabet_size=config["data"]["alphabet_size"],
                           embedding_size=modelConfig["embedding_size"],
                           conv_layers=modelConfig["conv_layers"],
                           fully_connected_layers=modelConfig["fully_connected_layers"],
                           num_of_classes=config["data"]["num_of_classes"],
                           dropout_p=modelConfig["dropout_p"],
                           optimizer=modelConfig["optimizer"],
                           loss=modelConfig["loss"])
    else:
        modelConfig = config['char_cnn_zhang']
        model = CharCNNZhang(input_size=config["data"]["input_size"],
                             alphabet_size=config["data"]["alphabet_size"],
                             embedding_size=modelConfig["embedding_size"],
                             conv_layers=modelConfig["conv_layers"],
                             fully_connected_layers=modelConfig["fully_connected_layers"],
                             num_of_classes=config["data"]["num_of_classes"],
                             threshold=modelConfig["threshold"],
                             dropout_p=modelConfig["dropout_p"],
                             optimizer=modelConfig["optimizer"],
                             loss=modelConfig["loss"])
    # Train model
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])
