import torch
import torchvision
import open_clip
import numpy as np

class OpenCLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ConvertImageDtype(torch.float32),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        # 强行指定用缓存模型，不联网
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained="/home/zht/github_play/OpenGaussian/open_clip_cache/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/snapshots/7288da5a0d6f0b51c4a2b27c624837a9236d0112/open_clip_pytorch_model.bin", # self.clip_model_pretrained,
            precision="fp16",
            cache_dir="/home/zht/github_play/OpenGaussian/open_clip_cache"
        )
        model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)
        self.device = device
        
    @torch.no_grad()
    def encode_text(self, text_list):
        text = self.tokenizer(text_list).to(self.device)
        return self.model.encode_text(text)
    
    @torch.no_grad()
    def encode_image(self, image):
        # 确保输入为3通道（丢弃Alpha通道）
        if image.shape[0] == 4:
            image = image[:3, :, :]
        # 应用预处理流程
        processed_img = self.process(image.to(self.device)).half()  # 添加设备转移和精度转换
        return self.model.encode_image(processed_img.unsqueeze(0))
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    open_clip_network = OpenCLIPNetwork(device)
    
    # Example usage
    text_list = ["sofa"]
    text_features = open_clip_network.encode_text(text_list)
    print(text_features.shape)  # Should print the shape of the encoded text features
    print(text_features)  # Print the encoded text features
    
    img_path = "./groundtruth_data/00display cabinet/00display cabinet.png"
    img = torchvision.io.read_image(img_path).float() / 255.0
    print(img.shape)  # Should print the shape of the image tensor
    # shape: 4*480*640
    
    img_features = open_clip_network.encode_image(img)
    print(img_features.shape)  # Should print the shape of the encoded image features
