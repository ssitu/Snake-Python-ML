from tensorflow import keras
import AI.ai as ai
import os


class Ai_tf(ai.Ai):

    model_name = "default_name"
    model_save_files_folder_path = os.path.dirname(os.path.realpath(__file__)) + "/savefiles/"

    # To be implemented in inheriting classes
    def model_create(self):
        pass

    def model_save(self):
        save_file_path = self.model_save_files_folder_path + self.model_name + ".h5"
        self.model.save(save_file_path)

    def model_load(self):
        try:
            save_file_path = self.model_save_files_folder_path + self.model_name + ".h5"
            self.model = keras.models.load_model(save_file_path)
        except IOError as error:
            print("IOError:", error)
