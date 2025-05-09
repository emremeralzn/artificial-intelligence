# Gerekli kütüphanelerin import edilmesi
import torch  # PyTorch, modelin çalışması için gerekli
from torchvision.datasets import CIFAR10  # CIFAR-10 veri seti
from torchvision import transforms  # Görüntü dönüşümleri için
from transformers import ViTForImageClassification, ViTImageProcessor  # Vision Transformer modelini ve işleyicisini yüklemek için
from sklearn.metrics import classification_report  # Modelin performansını ölçmek için
from tqdm import tqdm  # İlerleme çubuğu için

# Cihazın belirlenmesi (CUDA varsa GPU'yu kullan, yoksa CPU'yu kullan)
def cihaz_belirle():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Bu fonksiyon, modelin hangi cihazda çalıştırılacağını (GPU veya CPU) belirler. CUDA, GPU üzerinde hesaplama yapılmasını sağlar.

# Veri yükleme fonksiyonu
def veri_yukle(klasor="./data"):
    donusum = transforms.Compose([  # Görüntü işleme adımları
        transforms.Resize((224, 224)),  # Görüntü boyutunu 224x224'e ölçeklendirir
        transforms.ToTensor()  # Görüntüyü PyTorch tensörüne dönüştürür
    ])
    test_verisi = CIFAR10(root=klasor, train=False, download=True, transform=donusum)
    veri_yukleyici = torch.utils.data.DataLoader(test_verisi, batch_size=32, shuffle=False)
    # CIFAR-10 test veri setini indirir ve verilen dönüşümleri uygular. DataLoader, verileri küçük gruplara ayırarak (batch) yükler.
    return test_verisi, veri_yukleyici
# 'test_verisi' gerçek veri kümesini, 'veri_yukleyici' ise verileri yüklemek için kullanılan veri yükleyicisini döndürüyor.

# Modeli yükleme fonksiyonu
def model_yukle(cihaz):
    print("\n🔄 Model ve görüntü işleyici yükleniyor...")
    model = ViTForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    isleyici = ViTImageProcessor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    # ViT (Vision Transformer) modelini ve işleyicisini Hugging Face üzerinden yükler.
    return model.to(cihaz), isleyici
# Modeli belirtilen cihaza (GPU veya CPU) gönderiyor ve model ile işleyiciyi geri döndürüyor.

# Test fonksiyonu
def test_et(model, processor, yukleyici, cihaz):
    gercek_etiketler = []  # Gerçek etiketlerin tutulacağı liste
    tahminler = []  # Modelin tahminlerinin tutulacağı liste

    print("\n🚀 Test verisi üzerinde tahminler yapılıyor...\n")
    model.eval()  # Modeli test moduna alır
    with torch.no_grad():  # Gradyanları hesaplamadan işlem yapılır (bellek tasarrufu sağlar)
        for batch_gorseller, batch_etiketler in tqdm(yukleyici, desc="İşleniyor"):
            pil_gorseller = [transforms.ToPILImage()(img) for img in batch_gorseller]
            # Tensorlerden PIL görüntüleri oluşturuluyor.
            girisler = processor(images=pil_gorseller, return_tensors="pt").to(cihaz)
            # Görüntüler işlenip PyTorch tensörlerine dönüştürülür ve cihaza (GPU/CPU) gönderilir.
            cikti = model(**girisler)  # Model tahminini yapar
            tahmin = cikti.logits.argmax(dim=-1).cpu().numpy()  # Modelin logits çıktısının argmax'ini alarak tahmini sınıfı belirler

            tahminler.extend(tahmin)  # Tahminler listesine eklenir
            gercek_etiketler.extend(batch_etiketler.numpy())  # Gerçek etiketler listesine eklenir

    return gercek_etiketler, tahminler
# Test fonksiyonu, modelin tüm test verisi üzerinde tahmin yapmasını sağlar. Tahminler ve gerçek etiketler döndürülür.

# Sonuç raporunun yazdırılması
def raporu_yazdir(etiketler, tahminler, siniflar):
    print("\n📈 CIFAR-10 Test Sonuçları:\n")
    print(classification_report(etiketler, tahminler, target_names=siniflar))
    # Gerçek etiketler ve tahminler arasındaki farkı gösteren bir rapor yazdırır.
    # classification_report, doğruluk, hassasiyet, duyarlılık gibi istatistikleri hesaplar.

# Ana fonksiyon
def calistir():
    cihaz = cihaz_belirle()  # Cihazı belirler (GPU veya CPU)
    print(f"\n🖥️ Kullanılan cihaz: {cihaz}")

    veri_seti, yukleyici = veri_yukle()  # Veri setini ve yükleyiciyi yükler
    model, processor = model_yukle(cihaz)  # Model ve işleyiciyi yükler
    y_true, y_pred = test_et(model, processor, yukleyici, cihaz)  # Test fonksiyonunu çalıştırır ve tahminleri alır
    raporu_yazdir(y_true, y_pred, veri_seti.classes)  # Sonuçları yazdırır

# Ana kontrol bloğu, script olarak çalıştırıldığında 'calistir' fonksiyonu çağrılır
if __name__ == "__main__":
    calistir()
