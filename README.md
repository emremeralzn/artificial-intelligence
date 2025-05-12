CIFAR-10 Görsel Tanıma Uygulaması (ViT Tabanlı)
Bu proje, Vision Transformer (ViT) mimarisi kullanılarak eğitilmiş bir model aracılığıyla CIFAR-10 veri kümesindeki nesneleri sınıflandıran etkileşimli bir görsel tanıma uygulamasıdır. Kullanıcılar, uygulamaya görsel yükleyerek hangi sınıfa ait olduğunu anında öğrenebilir.

🔍 Proje Hakkında
Bu uygulama, önceden eğitilmiş ViT modeli sayesinde 10 farklı CIFAR-10 sınıfını yüksek doğruluk oranıyla tahmin eder. Gradio kütüphanesi kullanılarak oluşturulan web tabanlı arayüz ile görseller kolayca yüklenip sınıflandırma sonucu görüntülenebilir.

📌 Temel Özellikler
---

## 🛠️ Kullanılan Teknolojiler

| Bileşen        | Açıklama                          |
|----------------|-----------------------------------|
| Model          | Vision Transformer (ViT)          |
| Framework      | PyTorch, Transformers (Hugging Face) |
| Arayüz         | Gradio                            |
| Görüntü İşleme | Pillow (PIL)                      |
| Değerlendirme  | Scikit-learn                      |
| Python         | 3.10 veya 3.11 (önerilen)         |

---

🛠️ Kullanılan Teknolojiler
Bileşen Açıklama
Model Vision Transformer (ViT)
Framework PyTorch, Transformers (Hugging Face)
Arayüz Gradio
Görüntü İşleme Pillow (PIL)
Değerlendirme Scikit-learn (metric hesaplama)
Python Sürümü 3.10 veya 3.11 önerilir

🧪 Sınıf Etiketleri
Aşağıdaki 10 kategoriye göre sınıflandırma yapılır:

🛫 Uçak

🚗 Araba

🐦 Kuş

🐱 Kedi

🦌 Geyik

🐶 Köpek

🐸 Kurbağa

🐴 At

🚢 Gemi

🚚 Kamyon

⚙️ Kurulum Adımları

```bash
1. Gerekli Kütüphanelerin Yüklenmesi
   pip install -r requirements.txt
```
```bash
2. Modelin Yüklenmesi
   python load_model.py
```
```bash
3. Uygulamanın Başlatılması
   python main.py
```

🎯 Model Doğrulama
Modelin doğruluk, hassasiyet (precision), duyarlılık (recall) ve F1 skorunu görüntülemek için:

```bash
python model_evaluate.py
```

Bu script model_evaluate.py dosyasında yüklenen modelin performansını değerlendirir.

<pre><code>## 📁 Klasör Yapısı ``` 
   ├── main.py # Gradio arayüzü ve tahmin sistemi 
   ├── load_model.py # Modelin yüklenmesi 
   ├── model_evaluate.py # Performans metriği hesaplamaları 
   ├── requirements.txt # Gerekli pip paketleri 
   ├── results/ # Tahmin sonuçları (görsellerle) 
   ├── metrics/ # Doğruluk, precision, recall, F1-score görselleri 
   ├── demo/ # Tanıtım videosu açıklamaları ve bağlantı 
   └── README.md # Proje açıklamaları (bu dosya) ``` </code></pre>


🔍 Test Bilgileri
Uygulama, CIFAR-10 veri kümesinden her bir sınıfa ait 5 rastgele görselle test edilmiştir. Her görsel doğru şekilde sınıflandırılmıştır. Bu testlerin sonuçlarına results/ klasöründen ulaşabilirsiniz. Değerlendirme metrikleri ise metrics/ klasöründe yer almaktadır.

▶️ Demo
Uygulamanın nasıl çalıştığını gösteren bir tanıtım videosuna demo/ klasöründe yer alan .txt dosyasında belirtilen Drive bağlantısı üzerinden ulaşabilirsiniz.

📜 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır. Dilediğiniz gibi kullanabilir, değiştirebilir ve paylaşabilirsiniz.

---

---
 
Bu projede yaptığım çalışmalarla ilgili başka sorularınız varsa, bana her zaman ulaşabilirsiniz!  
 
 
[![LinkedIn](https://img.icons8.com/ios-filled/50/0A66C2/linkedin.png)](https://www.linkedin.com/in/emre-meralsm)  &nbsp;&nbsp; 
[![Website](https://img.icons8.com/ios-filled/50/8e44ad/domain.png)](https://emremeralzn.github.io/myblog/)
 
