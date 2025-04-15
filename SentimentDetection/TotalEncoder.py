import torch
import pandas as pd
from tqdm import tqdm

class Total_Encoder_Class:
    def __init__(self,data_path,text_encoder_path,chatglm_generate_text_encoder_path,image_encoder_path,cogvlm_generate_text_encoder_path,save_path):
        self.data_path = data_path
        self.text_encoder_path = text_encoder_path
        self.chatglm_generate_text_encoder_path = chatglm_generate_text_encoder_path
        self.image_encoder_path = image_encoder_path
        self.cogvlm_generate_text_encoder_path = cogvlm_generate_text_encoder_path
        self.save_path = save_path

    def generate(self):
        data = pd.read_excel(self.data_path)
        for index,row in tqdm(data.iterrows(),total=data.shape[0],desc="处理进度"):
            id_value = row["id"]
            text_value = row["text"]
            image_value = row["image"]
            label_value = row["label"]
            text_encoder_data = torch.load(self.text_encoder_path+str(id_value)+".pkl")
            text_encoder = text_encoder_data["text_encoder"]
            chatglm_generate_text_encoder_data = torch.load(self.chatglm_generate_text_encoder_path+str(id_value)+".pkl")
            chatglm_generate_text_encoder = chatglm_generate_text_encoder_data["text_encoder"]
            image_encoder_data = torch.load(self.image_encoder_path+str(id_value)+".pkl")
            image_encoder = image_encoder_data["image_encoder"]
            cogvlm_generate_text_encoder_data = torch.load(self.cogvlm_generate_text_encoder_path+str(id_value)+".pkl")
            cogvlm_generate_text_encoder = cogvlm_generate_text_encoder_data["text_encoder"]
            text_data = {"id":id_value,"text_encoder":text_encoder,"chatglm_generate_text_encoder":chatglm_generate_text_encoder,"image_encoder":image_encoder,"cogvlm_generate_text_encoder":cogvlm_generate_text_encoder,"label":label_value}
            torch.save(text_data,self.save_path+str(id_value)+".pkl")

if __name__ == '__main__':
    train_data_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/train.xlsx"
    train_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/text_encoder/train/"
    train_chatglm_generate_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate_encoder/train/"
    train_image_encoder_path = "./result/SarcNet/SarcNet-Image-Text/image_encoder/train/"
    train_cogvlm_generate_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate_encoder/train/"
    train_save_path = "./result/SarcNet/SarcNet-Image-Text/total_encoder/train/"
    train_total_encoder_class = Total_Encoder_Class(train_data_path,train_text_encoder_path,train_chatglm_generate_text_encoder_path,train_image_encoder_path,train_cogvlm_generate_text_encoder_path,train_save_path)
    train_total_encoder_class.generate()
    test_data_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/test.xlsx"
    test_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/text_encoder/test/"
    test_chatglm_generate_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate_encoder/test/"
    test_image_encoder_path = "./result/SarcNet/SarcNet-Image-Text/image_encoder/test/"
    test_cogvlm_generate_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate_encoder/test/"
    test_save_path = "./result/SarcNet/SarcNet-Image-Text/total_encoder/test/"
    test_total_encoder_class = Total_Encoder_Class(test_data_path,test_text_encoder_path,test_chatglm_generate_text_encoder_path,test_image_encoder_path,test_cogvlm_generate_text_encoder_path,test_save_path)
    test_total_encoder_class.generate()
    valid_data_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/valid.xlsx"
    valid_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/text_encoder/valid/"
    valid_chatglm_generate_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate_encoder/valid/"
    valid_image_encoder_path = "./result/SarcNet/SarcNet-Image-Text/image_encoder/valid/"
    valid_cogvlm_generate_text_encoder_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate_encoder/valid/"
    valid_save_path = "./result/SarcNet/SarcNet-Image-Text/total_encoder/valid/"
    valid_total_encoder_class = Total_Encoder_Class(valid_data_path,valid_text_encoder_path,valid_chatglm_generate_text_encoder_path,valid_image_encoder_path,valid_cogvlm_generate_text_encoder_path,valid_save_path)
    valid_total_encoder_class.generate()





