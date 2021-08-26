import os

import yaml


class Configuration:

    def __init__(self, conf_file="config.yml"):
        self.root_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../")
        try:
            with open(os.path.join(self.root_path, "common", conf_file), 'r') as yml_file:
                cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
        except Exception as e:
            print('ERROR - Configuration::__init__::{0}::{1}'.format(conf_file, str(e.args)))
            exit(1)

        cnn = cfg['cnn']

        self.logs_path = cfg['logs_path']

        self.cnn_name = cnn['name']
        self.cnn_data_path = cnn['data_path']
        self.cnn_epochs = cnn['epochs']
        self.cnn_batch_size = cnn['batch_size']
        self.cnn_model_path = cnn['model_path']

        # doc classification model
        doc_class = cfg['doc_class']
        self.doc_class_model_path = doc_class['model_path']
        self.doc_class_img_name = doc_class['img_class_name']
        self.doc_class_txt_name = doc_class['txt_class_name']
        self.doc_class_txt_tokenizer = doc_class['txt_class_tokenizer']
        self.doc_class_mixed_name = doc_class['mixed_class_name']
