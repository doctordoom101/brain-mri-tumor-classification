# Brain Tumor MRI Classification with Deep Learning

Proyek ini merupakan sistem klasifikasi tumor otak menggunakan citra MRI. Model mampu mengidentifikasi empat jenis kondisi:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

Aplikasi ini menggunakan *Transfer Learning* (DenseNet121) yang digabungkan dengan **Conv2D + Pooling** layer secara eksplisit sesuai requirement, dilatih pada dataset MRI 4 kelas. Model kemudian di-deploy secara lokal menggunakan antarmuka **Gradio**.

---

## 📂 Dataset
Dataset terdiri dari empat kelas:

| Set   | Glioma | Meningioma | No Tumor | Pituitary |
|------|:------:|:----------:|:--------:|:---------:|
| Train | 1316 | 1298 | 1588 | 1416 |
| Test  | 305 | 347 | 412 | 341 |

Dataset dibagi menggunakan `ImageDataGenerator` dengan augmentasi untuk meningkatkan generalisasi.

---

## 🧠 Arsitektur Model

Model menggunakan **DenseNet121 (pretrained ImageNet)** sebagai feature extractor + tambahan layer berikut:

- Conv2D (ReLU)
- Batch Normalization
- Pooling Layer
- Dense Layer (Fully Connected)
- Dropout untuk regularisasi
- Output softmax (4 kelas)

Training dilakukan dalam **2 Stage**:

| Stage | Base Model | Learning Rate | Tujuan |
|------|------------|--------------|--------|
| 1 | Frozen | 1e-3 → ReduceLROnPlateau | Stabilize high-level features |
| 2 | Unfrozen (partial) | 1e-4 | Fine-tune deeper representation |

Model training & notebook tersedia di Google Drive:

🔗 https://drive.google.com/file/d/1OlW9uRURRegyGVCkdOjI8Bj4HidsuQrf/view?usp=sharing


