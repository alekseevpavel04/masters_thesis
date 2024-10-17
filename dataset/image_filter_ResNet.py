import torch
from torchvision import transforms, models
from PIL import Image

class ImageFilter:
    def __init__(self, model_name='resnet18'):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def evaluate_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image)

        score = outputs.mean().item()
        return score

    def get_files_to_remove(self, image_files):
        scores = {}
        for image_file in image_files:
            score = self.evaluate_image(image_file)
            scores[image_file] = score

        # Сортировка изображений по оценкам
        sorted_images = sorted(scores.items(), key=lambda x: x[1])

        # Выбор % наименее информативных изображений для удаления
        num_to_remove = len(sorted_images) // 10
        files_to_remove = [image_file for image_file, _ in sorted_images[:num_to_remove]]

        return files_to_remove