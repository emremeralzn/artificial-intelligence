# Gerekli modülleri yüklüyoruz
import gradio as gr
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# CIFAR-10 sınıflarının Türkçeye çevrilmiş halleri (tuple listesi kullanılarak)
etiket_listesi = [
    ("airplane", "Uçak"),
    ("automobile", "Araba"),
    ("bird", "Kuş"),
    ("cat", "Kedi"),
    ("deer", "Geyik"),
    ("dog", "Köpek"),
    ("frog", "Kurbağa"),
    ("horse", "At"),
    ("ship", "Gemi"),
    ("truck", "Kamyon")
]

# Listeyi sözlüğe çeviriyoruz
sinif_cevirisi = dict(etiket_listesi)

# Vision Transformer modelini ve ön işleyicisini yüklüyoruz
vit_model = ViTForImageClassification.from_pretrained(
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
)
gorsel_isleyici = ViTImageProcessor.from_pretrained(
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
)

# Görsel üzerinden sınıf tahmini yapan fonksiyon
def gorsel_siniflandir(gorsel):
    gorsel = gorsel.convert("RGB")  # Görseli RGB formatına çeviriyoruz
    veriler = gorsel_isleyici(images=gorsel, return_tensors="pt")  # Modelin anlayacağı formata dönüştürüyoruz

    # Model ile tahmin yapıyoruz
    with torch.no_grad():
        sonuc = vit_model(**veriler)
        en_yuksek_skor = sonuc.logits.argmax(-1).item()
        sinif_ingilizce = vit_model.config.id2label[en_yuksek_skor]
        sinif_turkce = sinif_cevirisi.get(sinif_ingilizce, "Sınıf Bilinmiyor")

    return f"Tespit Edilen Sınıf: {sinif_turkce}"

# Gradio arayüzünü tanımlıyoruz
arayuz = gr.Interface(
    fn=gorsel_siniflandir,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Görsel Sınıflandırıcı - CIFAR-10",
    description="Yüklenen resmi CIFAR-10 sınıflarına göre sınıflandıran ViT tabanlı bir modeldir."
)

# Arayüzü başlatıyoruz
arayuz.launch()
