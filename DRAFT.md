BAB I
PENDAHULUAN

Latar Belakang

Kecelakaan lalu lintas yang menyebabkan cedera hingga kematian masih menjadi tantangan besar dalam kesehatan dan pembangunan global. Masalah ini semakin diperparah dengan meningkatnya volume kendaraan di jalan, yang secara substansial meningkatkan risiko kecelakaan dan kematian. Pertumbuhan kendaraan lalu lintas yang pesat dalam beberapa tahun terakhir telah menyebabkan peningkatan kecelakaan lalu lintas, terutama kecelakaan kendaraan berat [1]. Menurut World Health Organization (WHO), diperkirakan terdapat 1,19 juta kematian akibat kecelakaan lalu lintas pada tahun 2021 dan tetap menjadi penyebab kematian utama bagi anak-anak dan remaja berusia 5-29 tahun, dan penyebab kematian ke-12 jika semua usia digabungkan [2]. Kecelakaan lalu lintas di Indonesia juga merupakan masalah yang berkembang dimana angka korban jiwa dan kematian akibat kecelakaan lalu lintas yang tinggi dan terus memburuk, dari 25.266 korban jiwa sepanjang tahun 2021 menjadi 94.617 korban jiwa hanya dalam sembilan bulan pertama tahun 2022 [3]. Apabila tidak ada tindakan pencegahan yang dilakukan, kecelakaan lalu lintas akan menjadi penyebab kematian terbesar pada tahun 2030 [4] . Mengidentifikasi faktor-faktor utama kecelakaan lalu lintas dapat menjadi langkah awal yang penting dalam mencegah dan mengurangi angka korban jiwa.

Salah satu faktor utama dari penyebab kecelakaan lalu lintas yaitu adalah menurunnya konsentrasi pengemudi akibat rasa kantuk dan kelelahan saat berkendara. Kondisi mengantuk dapat menyebabkan keterlambatan reaksi, hilangnya fokus, hingga microsleep yang sangat berbahaya bagi keselamatan pengemudi maupun pengguna jalan lainnya. Hal ini terbukti memiliki dampak signifikan terhadap keselamatan jalan raya, mencerminkan perilaku pengemudi yang tidak aman [5]. Sebuah studi di Yogyakarta menemukan korelasi signifikan antara rasa kantuk saat mengemudi dan peningkatan risiko kecelakaan, terutama pada larut malam dan pagi hari ketika pengemudi lebih cenderung kelelahan [6]. Sehingga, tingginya fatalitas atau tingkat keparahan cedera dalam kecelakaan akibat kantuk berdampak dengan hilangnya kemampuan preventif pengemudi dalam menghindari tabrakan.

Perkembangan teknologi Artificial Intelligence (AI), khususnya Deep Learning dan Convolutional Neural Network, memungkinkan pengembangan sistem deteksi kantuk secara otomatis berbasis citra wajah. Konsep dari Deep Learning adalah mempelajari fitur-fitur yang ada pada sebuah data baru ketika terdapat kesamaan pada data lama atau yang telah dipelajari [7]. Convolutional Neural Network merupakan salah satu algoritma Deep Learning yang berperan dalam proses klasifikasi citra, banyak digunakan untuk menyelesaikan permasalahan yang memiliki tingkat kompleksitas tinggi [8]. Convolutional Neural Network memiliki kemampuan yang baik dalam mengenali pola visual seperti kondisi mata tertutup, frekuensi kedipan, dan aktivitas menguap pada wajah pengemudi.

Beberapa penelitian sebelumnya telah mengembangkan sistem deteksi kantuk menggunakan CNN dan transfer learning seperti MobileNet, ResNet, dan YOLOv8. Penelitian tersebut menunjukkan tingkat akurasi yang tinggi, namun sebagian besar masih memiliki kendala pada kebutuhan komputasi yang cukup besar, kurang optimal pada implementasi realtime, serta sensitivitas terhadap kondisi pencahayaan yang berbeda.

Berdasarkan permasalahan tersebut, penelitian ini bertujuan untuk mengimplementasikan model CNN berbasis MobileNetV2 untuk mendeteksi kantuk pengemudi secara realtime menggunakan kamera. Pemilihan MobileNetV2 dilakukan karena arsitektur ini memiliki ukuran model yang ringan, proses inferensi cepat, dan tetap mampu menghasilkan akurasi yang baik pada perangkat dengan spesifikasi terbatas. Penelitian ini diharapkan dapat membantu meningkatkan keselamatan berkendara melalui sistem peringatan dini terhadap kondisi kantuk pengemudi.

Rumusan Masalah

Berdasarkan latar belakang penelitian tersebut, diperoleh beberapa rumusan masalah sebagai berikut:
1.	Bagaimana mengimplementasikan metode CNN untuk mendeteksi kantuk pengemudi?
2.	Bagaimana performa arsitektur MobileNetV2 dalam klasifikasi kondisi kantuk pengemudi?
3.	Bagaimana tingkat akurasi sistem deteksi kantuk berbasis CNN secara realtime?
4.	Bagaimana pengaruh preprocessing citra terhadap performa model CNN?

Tujuan Penelitian

Tujuan dari penelitian ini adalah mengimplementasikan CNN berbasis MobileNetV2 untuk deteksi kantuk pengemudi, mengevaluasi performa model CNN berdasarkan accuracy, precision, recall, dan FPS realtime, dan menganalisis pengaruh processing terhadap hasil klasifikasi.

Batasan Masalah

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

BAB II
TINJAUAN PUSTAKA

Studi Literatur

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

Convolutional Neural Network

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

Deteksi Kantuk

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

BAB III
METODOLOGI PENELITIAN

Dataset

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

MobileNetV2

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

Evaluasi Model

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

DAFTAR PUSTAKA

[1]	R. E. Al Mamlook, K. M. Kwayu, M. R. Alkasisbeh, and A. A. Frefer, “Comparison of Machine Learning Algorithms for Predicting Traffic Accident Severity,” in 2019 IEEE Jordan International Joint Conference on Electrical Engineering and Information Technology (JEEIT), Amman: IEEE, Apr. 2019, pp. 272–276. doi: 10.1109/JEEIT.2019.8717393.
[2]	“Global status report on road safety 2023,” Geneva, 2023.
[3]	S. F. E. Mubalus, “ANALISIS FAKTOR-FAKTOR PENYEBAB KECELAKAAN LALU LINTAS DI KABUPATEN SORONG DAN PENANGGULANGANNYA,” SOSCIED, vol. 6, no. 1, pp. 182–197, Jun. 2023, Accessed: Nov. 14, 2025. [Online]. Available: https://www.poltekstpaul.ac.id/jurnal/index.php/jsoscied/article/view/624
[4]	P. Kohli and A. Chadha, “Enabling Pedestrian Safety Using Computer Vision Techniques: A Case Study of the 2018 Uber Inc. Self-driving Car Crash,” in Advances in Information and Communication, vol. 69, K. Arai and R. Bhatia, Eds., Cham: Springer International Publishing, 2020, pp. 261–279. doi: 10.1007/978-3-030-12388-8_19.
[5]	S. Soares, T. Monteiro, A. Lobo, A. Couto, L. Cunha, and S. Ferreira, “Analyzing Driver Drowsiness: From Causes to Effects,” Sustainability, vol. 12, no. 5, p. 1971, Mar. 2020, doi: 10.3390/su12051971.
[6]	N. R. Widyastuti and D. F. Brilianti, “The Impact of Drowsiness on Road Traffic Accidents in Yogyakarta,” Journal of Scientific Research, Education, and Technology (JSRET), vol. 3, no. 4, pp. 1651–1661, Dec. 2024, doi: 10.58526/jsret.v3i4.555.
[7]	Muhammad Nur Ichsan, Nur Armita, Agus Eko Minarno, Fauzi Dwi Setiawan Sumadi, and Hariyady, “Increased Accuracy on Image Classification of Game Rock Paper Scissors using CNN,” Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), vol. 6, no. 4, pp. 606–611, Aug. 2022, doi: 10.29207/resti.v6i4.4222.
[8]	N. A. Ujilast, N. S. Firdausita, C. S. K. Aditya, and Y. Azhar, “MRI Image Based Alzheimer’s Disease Classification Using Convolutional Neural Network: EfficientNet Architecture,” Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), vol. 8, no. 1, pp. 18–25, Jan. 2024, doi: 10.29207/resti.v8i1.5457.
