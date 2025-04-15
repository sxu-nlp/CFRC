import torch
from transformers import ViTImageProcessor,ViTForImageClassification,ViTModel
import pandas as pd
from PIL import Image
from tqdm import tqdm

class Vision_Encoder_Class:
    def __init__(self,pretrained_model_path,device):
        self.pretrained_model_path = pretrained_model_path
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(self.pretrained_model_path)
        # self.pretrained_model = ViTForImageClassification.from_pretrained(self.pretrained_model_path)
        self.pretrained_model = ViTModel.from_pretrained(self.pretrained_model_path)

    def encoder(self,data_path,save_path,image_path,index_value,column_value):
        data = pd.read_excel(data_path)
        data_list = []
        for index,row in data.iterrows():
            text_index = row[index_value]
            image_index = row[column_value]
            data_list.append([text_index,image_index])
        for row in tqdm(data_list,total=len(data_list),desc="处理进度"):
            text_index = row[0]
            image_index = row[1]
            image = Image.open(image_path+str(image_index)+".jpg")

            image = image.convert("RGB")

            inputs = self.processor(images=image,return_tensors="pt")
            outputs = self.pretrained_model(**inputs)

            image_encoder = outputs.last_hidden_state

            image_vision_encoder = {"text_index":text_index,"image_index":image_index,"image_encoder":image_encoder}
            torch.save(image_vision_encoder,save_path+str(text_index)+".pkl")

if __name__ == '__main__':

    pretrained_model_path = "../../pretrained_model/vit-base-patch16-224"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vision_encoder_class = Vision_Encoder_Class(pretrained_model_path,device)
    index_value = "id"
    column_value = "image"
    train_data_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/train.xlsx"
    train_save_path = "./result/SarcNet/SarcNet-Image-Text/image_encoder/train/"
    train_image_path = "./DataSet/SarcNet/SarcNet-Image-Text/Image/"
    vision_encoder_class.encoder(train_data_path,train_save_path,train_image_path,index_value,column_value)
    test_data_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/test.xlsx"
    test_save_path = "./result/SarcNet/SarcNet-Image-Text/image_encoder/test/"
    test_image_path = "./DataSet/SarcNet/SarcNet-Image-Text/Image/"
    vision_encoder_class.encoder(test_data_path,test_save_path,test_image_path,index_value,column_value)
    valid_data_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/valid.xlsx"
    valid_save_path = "./result/SarcNet/SarcNet-Image-Text/image_encoder/valid/"
    valid_image_path = "./DataSet/SarcNet/SarcNet-Image-Text/Image/"
    vision_encoder_class.encoder(valid_data_path,valid_save_path,valid_image_path,index_value,column_value)



