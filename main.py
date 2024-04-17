import torch
import os
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
"""
-зібрати невеликий датасет із 10-20 зображень різних класів (в кожному класі може бути кілька зображень).
наприклад 10 зобр. котів, 3 зображення лабрадора, 1 автомобіль і тд
-проіндексувати всі зображення в датасеті
-зібрати тестовий датасет (2-3 зображення)
-для них знайти найближчі зображення із побудованої бази. перевірити чи модель працює задовільно
"""


def load_preprocess_images(image_folder):
    """
loading and preprocessing
    """
    images = []
    image_paths = []
    file_names = os.listdir(image_folder)
    for name in file_names:
        full_path = os.path.join(image_folder, name)
        if not name.startswith('.') and os.path.isfile(full_path):
            processed_img = preprocess(Image.open(full_path)).unsqueeze(0).to(device)
            images.append(processed_img)
            image_paths.append(name)
    return torch.cat(images, dim=0), image_paths


train_images, train_path = load_preprocess_images("dataset/train")


# print(train_path)
# print(train_images.shape)
# # torch.Size([16, 3, 224, 224])

def encode_image(images):
    """
encoding images
    """
    with torch.no_grad():
        encode = model.encode_image(images)
    return encode


train_encoded = encode_image(train_images)
# print(train_encoded.shape)
# # torch.Size([16, 512])

test_images, test_path = load_preprocess_images("dataset/test")
# print(test_path)
# print(test_images.shape)
# # torch.Size([3, 3, 224, 224])

test_encoded = encode_image(test_images)


# print(test_encoded.shape)
# # torch.Size([3, 512])

def find_closest_images(test_encodings, train_encodings, train_paths, top_k=3):
    """
used matmul for dot product of normalized vectors
used topk for search indices of max value in each raw
creating list of lists where inner list contains names of training images corresponding to indices
    """
    similarities = torch.matmul(test_encodings, train_encodings.T)
    top_k_indices = torch.topk(similarities, k=top_k, dim=1).indices
    closest_paths = [[train_paths[i] for i in indices] for indices in top_k_indices]
    return closest_paths, top_k_indices


# search nearest image
closest_paths, closest_indices = find_closest_images(test_encoded, train_encoded, train_path, top_k=3)
for test_path, paths, indices in zip(test_path, closest_paths, closest_indices):
    print(f"for test image {test_path} nearest image:")
    print(", ".join(paths))
    print(f"nearest indices image: {indices.tolist()}")
    print()

# for test image car.jpg nearest image:
# car1.jpg, car3.jpg, car2.jpg
# nearest indices image: [0, 2, 1]
#
# for test image cat.jpg nearest image:
# cat7.jpg, cat6.jpg, cat5.jpg
# nearest indices image: [9, 8, 7]
#
# for test image husky.jpg nearest image:
# husky4.jpg, husky2.jpg, husky3.jpg
# nearest indices image: [15, 13, 14]
