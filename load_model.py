# Gerekli kütüphanelerin import edilmesi
from transformers import ViTForImageClassification, ViTImageProcessor  # Vision Transformer modelini ve görüntü işleyicisini yükler
from PIL import Image  # Görüntü işleme kütüphanesi
import torch  # PyTorch, modelin çalışması için gerekli
import requests  # URL üzerinden veri almak için kullanılır

# Model ve işlemcinin önceden eğitilmiş hali yükleniyor
vit_model = ViTForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
image_processor = ViTImageProcessor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
# 'vit_model' değişkeni, CIFAR-10 sınıflandırması için önceden eğitilmiş Vision Transformer modelini içeriyor.
# 'image_processor' ise, bu modelin anlayabileceği formata görüntüleri dönüştürmek için kullanılan işleyiciyi içeriyor.

# Görüntünün URL'den indirilmesi
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cifar10-airplane.png"
input_image = Image.open(requests.get(image_url, stream=True).raw)
# Burada, 'image_url' adlı değişkenle belirtilen URL'den bir görsel indiriliyor ve 'input_image' olarak PIL formatında açılıyor.

# Görüntünün işlenmesi, modele uygun hale getirilmesi
processed_inputs = image_processor(images=input_image, return_tensors="pt")
# 'input_image' işlenip, modelin anlayabileceği formata (PyTorch tensörü) dönüştürülüyor ve 'processed_inputs' adlı değişkene aktarılıyor.

# Model tahmini yapılıyor (gradyanlar hesaplanmıyor)
with torch.no_grad():  # Bu blok içinde gradyanlar hesaplanmadığı için belleği verimli kullanıyoruz.
    model_outputs = vit_model(**processed_inputs)  # Modelin çıktısı alınıyor.
    class_logits = model_outputs.logits  # Modelin verdiği raw çıktı (logits), sınıf olasılıklarını temsil eder.
    predicted_class_idx = class_logits.argmax(-1).item()  # En yüksek olasılığa sahip sınıf index'i alınıyor.

# Tahmin edilen sınıf etiketinin yazdırılması
predicted_class_label = vit_model.config.id2label[predicted_class_idx]  # Sınıf numarasından etiketine dönüştürme
print(f"Predicted class: {predicted_class_label}")
# Son olarak, 'predicted_class_label' değişkenindeki sınıf etiketi ekrana yazdırılıyor.
