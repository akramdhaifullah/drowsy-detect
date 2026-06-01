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

1.	Dataset yang digunakan berupa data citra mata manusia "MRL Eye Dataset" dengan dua kelas: Drowsy dan Alert.
2.	Evaluasi performa model dilakukan dengan metrik akurasi, precision, recall, F1-score, dan confusion matrix.
3.	Implementasi pemrograman dilakukan menggunakan bahasa pemrograman Python dengan framework PyTorch.


BAB II
TINJAUAN PUSTAKA

Studi Literatur

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas arcu est, faucibus nec molestie vitae, scelerisque vitae leo. Quisque elit urna, posuere id lorem a, suscipit laoreet erat. Phasellus dolor neque, posuere et quam at, auctor tincidunt diam. Vestibulum auctor quam vel magna bibendum porta.

Deteksi Kantuk

Deteksi kantuk pada pengemudi merupakan upaya untuk mengidentifikasi tanda-tanda penurunan kewaspadaan secara otomatis guna mencegah kecelakaan lalu lintas. Sistem deteksi kantuk secara umum dapat diklasifikasikan menjadi empat kategori berdasarkan jenis data yang digunakan, yaitu berbasis citra (*image-based*), berbasis sinyal fisiologis (*biological-based*), berbasis perilaku kendaraan (*vehicle-based*), dan pendekatan gabungan (*hybrid-based*) [9]. Pendekatan berbasis citra menjadi yang paling banyak diadopsi karena sifatnya yang non-intrusif dan hanya memerlukan kamera sebagai sensor utama, sehingga lebih praktis untuk diterapkan pada kendaraan komersial [9].

Pada pendekatan berbasis citra, indikator kantuk umumnya diidentifikasi melalui analisis kondisi mata, aktivitas menguap, dan pergerakan kepala pengemudi. Dua metrik yang sering digunakan untuk mengukur tingkat kantuk adalah Eye Aspect Ratio (EAR) dan Percentage of Eye Closure (PERCLOS). EAR merupakan rasio jarak vertikal terhadap jarak horizontal titik-titik *landmark* pada kelopak mata yang dapat menunjukkan kondisi mata terbuka atau tertutup secara kuantitatif [10]. Maior *et al.* mengembangkan sistem klasifikasi kantuk secara *real-time* menggunakan EAR dengan menggabungkan informasi temporal dari *frame* video berturut-turut, sehingga mampu menangkap pola kedipan mata yang mengindikasikan kantuk dan menghasilkan akurasi rata-rata sebesar 94,9% menggunakan *Support Vector Machine* (SVM) [10]. Sementara itu, PERCLOS mengukur proporsi waktu mata dalam kondisi tertutup selama periode tertentu dan telah diakui sebagai indikator kantuk yang andal dalam berbagai studi [11].

Perkembangan *deep learning*, khususnya Convolutional Neural Network (CNN), membawa perubahan signifikan dalam pendekatan deteksi kantuk. CNN memiliki kemampuan untuk mengekstraksi fitur spasial dari citra wajah secara otomatis tanpa memerlukan rekayasa fitur manual (*hand-crafted features*), sehingga dapat mengenali pola visual yang lebih kompleks [12]. Majeed *et al.* mengembangkan model CNN mendalam (*deep CNN*) untuk mendeteksi kantuk berdasarkan kondisi mata dan mulut pengemudi, dengan hasil evaluasi menunjukkan performa klasifikasi yang tinggi pada dataset citra wajah pengemudi [12]. Pendekatan serupa juga dikembangkan oleh Venkateswarlu dan Ch yang merancang arsitektur ringan bernama DrowsyDetectNet, sebuah model CNN yang dioptimalkan untuk pelatihan dengan data terbatas namun tetap mampu mencapai akurasi tinggi pada klasifikasi kondisi kantuk [13].

Penggunaan teknik *transfer learning* semakin memperkuat efektivitas CNN dalam deteksi kantuk. Melalui *transfer learning*, model yang telah dilatih pada dataset berskala besar seperti ImageNet dapat diadaptasi untuk tugas klasifikasi kantuk, sehingga mempercepat proses pelatihan dan meningkatkan akurasi meskipun dengan jumlah data pelatihan yang terbatas [13]. Arsitektur ringan seperti MobileNet dan MobileNetV2 menjadi pilihan yang populer karena menggunakan *depthwise separable convolution* yang mampu mengurangi jumlah parameter dan beban komputasi secara substansial tanpa kehilangan kemampuan klasifikasi [13]. Keunggulan ini memungkinkan model untuk dijalankan secara *real-time* pada perangkat dengan sumber daya terbatas seperti *embedded system* dan perangkat *edge computing* [9].

Tinjauan sistematis yang dilakukan oleh Fonseca dan Ferreira terhadap 81 studi mengenai deteksi kantuk berbasis *deep learning* yang diterbitkan antara tahun 2015 hingga 2025 menunjukkan bahwa CNN, Recurrent Neural Network (RNN), dan arsitektur hibrid merupakan model yang paling umum digunakan, dengan akurasi median yang dilaporkan melebihi 95% [14]. Meskipun demikian, tinjauan tersebut juga menggarisbawahi bahwa penerapan di kondisi nyata masih menghadapi tantangan berupa variabilitas lingkungan, keterbatasan transparansi dataset, dan pertimbangan etis terkait pemantauan pengemudi secara berkelanjutan [14]. Selain itu, tren terbaru menunjukkan peningkatan penggunaan pendekatan multimodal yang menggabungkan data visual dengan sinyal fisiologis seperti EEG dan ECG untuk meningkatkan keandalan sistem deteksi pada kondisi yang beragam [9], [14].

MobileNetV2

MobileNetV2 merupakan arsitektur Convolutional Neural Network (CNN) ringan yang dirancang untuk aplikasi *computer vision* pada perangkat dengan sumber daya komputasi terbatas. Arsitektur ini dikembangkan oleh Sandler *et al.* sebagai penyempurnaan dari MobileNet generasi pertama yang diperkenalkan oleh Howard *et al.* [15]. MobileNet generasi pertama memperkenalkan konsep *depthwise separable convolution*, yaitu teknik dekomposisi operasi konvolusi standar menjadi dua tahap terpisah: *depthwise convolution* yang melakukan pemfilteran spasial pada setiap kanal input secara independen, dan *pointwise convolution* berupa konvolusi 1×1 yang menggabungkan keluaran antar kanal [15]. Teknik ini mengurangi jumlah parameter dan operasi komputasi secara drastis dibandingkan konvolusi konvensional.

MobileNetV2 memperkenalkan dua inovasi arsitektural utama yang membedakannya dari pendahulunya, yaitu *inverted residual block* dan *linear bottleneck* [16]. Pada blok residual konvensional seperti yang digunakan dalam arsitektur ResNet, struktur yang diterapkan mengikuti pola *wide-narrow-wide*, di mana data dikompresi terlebih dahulu kemudian diekspansi kembali. MobileNetV2 membalik pola tersebut menjadi *narrow-wide-narrow*: lapisan input berdimensi rendah (*bottleneck*) diekspansi ke dimensi yang lebih tinggi menggunakan konvolusi 1×1 untuk menangkap fitur yang lebih kaya, kemudian dilakukan pemfilteran spasial melalui *depthwise convolution*, dan akhirnya diproyeksikan kembali ke representasi berdimensi rendah [16]. Koneksi *shortcut* ditempatkan langsung antara lapisan *bottleneck* berdimensi rendah, sehingga mengurangi kebutuhan memori selama proses inferensi secara signifikan [16].

Inovasi kedua, yaitu *linear bottleneck*, menghilangkan fungsi aktivasi non-linear (ReLU) pada lapisan *bottleneck* berdimensi rendah. Sandler *et al.* menunjukkan bahwa penerapan aktivasi non-linear pada ruang berdimensi rendah menyebabkan kehilangan informasi karena ReLU dapat menghancurkan *manifold* data [16]. Dengan mempertahankan lapisan *bottleneck* dalam bentuk linear, kapasitas representasi model tetap terjaga meskipun dimensi fitur dikompres. Kombinasi kedua teknik ini menghasilkan arsitektur dengan sekitar 3,4 juta parameter dan memerlukan hanya 300 juta operasi *Multiply-Adds* untuk resolusi input 224×224, menjadikannya jauh lebih efisien dibandingkan arsitektur seperti VGG16 yang memiliki lebih dari 138 juta parameter [16].

Selain efisiensi arsitektural, MobileNetV2 menyediakan *width multiplier* (α) dan *resolution multiplier* sebagai hiperparameter yang memungkinkan penyesuaian ukuran model sesuai kebutuhan perangkat keras [16]. Fleksibilitas ini memungkinkan pengembang untuk melakukan *trade-off* antara ukuran model, latensi inferensi, dan akurasi klasifikasi secara langsung. Gulzar menunjukkan bahwa MobileNetV2 yang dikombinasikan dengan *deep transfer learning* mampu mencapai akurasi klasifikasi yang kompetitif pada tugas pengenalan citra, dengan ukuran model dan kecepatan inferensi yang jauh lebih unggul dibandingkan arsitektur berat seperti VGG16 dan ResNet [17]. Keunggulan ini menjadikan MobileNetV2 sebagai *backbone* yang ideal untuk sistem deteksi kantuk secara *real-time*, di mana kecepatan pemrosesan *frame* per detik menjadi faktor kritis dalam memastikan respons sistem yang tepat waktu.


BAB III
METODOLOGI PENELITIAN

Dataset

Penelitian ini menggunakan MRL Eye Dataset, sebuah dataset berskala besar yang berisi citra mata manusia yang dikembangkan oleh Machine Recognition Laboratory, VŠB – Technical University of Ostrava, Republik Ceko [18]. Dataset ini dirancang untuk mendukung penelitian di bidang deteksi mata, estimasi arah pandang, dan analisis frekuensi kedipan mata, khususnya dalam konteks perilaku pengemudi. Seluruh citra dalam dataset diakuisisi menggunakan kamera inframerah (*near-infrared*) pada kondisi nyata, sehingga merepresentasikan variasi lingkungan yang realistis.

Dataset terdiri dari 84.898 citra mata yang dikumpulkan dari 37 subjek berbeda (33 laki-laki dan 4 perempuan). Setiap citra dilengkapi dengan anotasi yang dikodekan langsung pada nama *file* dengan format terstruktur, mencakup delapan atribut: identitas subjek, nomor citra, gender, penggunaan kacamata, kondisi mata, tingkat refleksi, kondisi pencahayaan, dan jenis sensor yang digunakan. Kondisi mata dianotasi dalam dua kelas, yaitu mata terbuka (*open*) dan mata tertutup (*closed*), yang menjadi label utama untuk tugas klasifikasi deteksi kantuk dalam penelitian ini.

Berdasarkan hasil eksplorasi data, distribusi kelas kondisi mata relatif seimbang dengan 42.952 citra mata terbuka (50,6%) dan 41.946 citra mata tertutup (49,4%). Distribusi gender menunjukkan 63.173 citra (74,4%) berasal dari subjek laki-laki dan 21.725 citra (25,6%) dari subjek perempuan. Sebanyak 60.897 citra (71,7%) diambil tanpa kacamata, sedangkan 24.001 citra (28,3%) dengan kacamata. Terkait kondisi pencahayaan, 53.630 citra (63,2%) diperoleh pada pencahayaan buruk dan 31.268 citra (36,8%) pada pencahayaan baik. Variasi ini memungkinkan model yang dilatih untuk lebih *robust* terhadap perbedaan kondisi akuisisi.

Citra dalam dataset diakuisisi menggunakan tiga jenis sensor dengan resolusi yang berbeda, yaitu Intel RealSense SR300 dengan resolusi 640×480 piksel (70.142 citra; 82,6%), IDS Imaging dengan resolusi 1280×1024 piksel (11.992 citra; 14,1%), dan Aptina Imaging dengan resolusi 752×480 piksel (2.764 citra; 3,3%). Penggunaan beragam sensor ini menambah variasi resolusi dan karakteristik citra dalam dataset, sehingga model yang dilatih diharapkan mampu menggeneralisasi pada berbagai kondisi perangkat keras. Jumlah citra per subjek bervariasi antara 382 hingga 10.257 citra dengan rata-rata 2.295 citra per subjek, yang mengindikasikan adanya ketidakseimbangan jumlah sampel antar individu yang perlu dipertimbangkan dalam pembagian data pelatihan dan pengujian.


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
[8]	N. A. Ujilast, N. S. Firdausita, C. S. K. Aditya, and Y. Azhar, "MRI Image Based Alzheimer's Disease Classification Using Convolutional Neural Network: EfficientNet Architecture," Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), vol. 8, no. 1, pp. 18–25, Jan. 2024, doi: 10.29207/resti.v8i1.5457.
[9]	Y. Albadawi, M. Takruri, and M. Awad, "A Review of Recent Developments in Driver Drowsiness Detection Systems," Sensors, vol. 22, no. 5, p. 2069, Mar. 2022, doi: 10.3390/s22052069.
[10]	C. B. S. Maior, M. J. das C. Moura, J. M. M. Santana, and I. D. Lins, "Real-time classification for autonomous drowsiness detection using eye aspect ratio," Expert Systems with Applications, vol. 158, p. 113505, Nov. 2020, doi: 10.1016/j.eswa.2020.113505.
[11]	W. Deng and R. Wu, "Real-Time Driver-Drowsiness Detection System Using Facial Features," IEEE Access, vol. 7, pp. 118727–118738, 2019, doi: 10.1109/ACCESS.2019.2936663.
[12]	F. Majeed, U. Shafique, M. Safran, S. Alfarhood, and I. Ashraf, "Detection of Drowsiness among Drivers Using a Novel Deep Convolutional Neural Network Model," Sensors, vol. 23, no. 21, p. 8741, Oct. 2023, doi: 10.3390/s23218741.
[13]	M. Venkateswarlu and V. R. R. Ch, "DrowsyDetectNet: Driver Drowsiness Detection Using Lightweight CNN With Limited Training Data," IEEE Access, vol. 12, pp. 110476–110491, 2024, doi: 10.1109/ACCESS.2024.3440585.
[14]	T. Fonseca and S. Ferreira, "Drowsiness Detection in Drivers: A Systematic Review of Deep Learning-Based Models," Applied Sciences, vol. 15, no. 16, p. 9018, 2025, doi: 10.3390/app15169018.
[15]	A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv preprint arXiv:1704.04861, 2017, doi: 10.48550/arXiv.1704.04861.
[16]	M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.
[17]	Y. Gulzar, "Fruit Image Classification Model Based on MobileNetV2 with Deep Transfer Learning Technique," Sustainability, vol. 15, no. 3, p. 1906, Jan. 2023, doi: 10.3390/su15031906.
[18]	R. Fusek, "Pupil Localization Using Geodesic Distance," in Advances in Visual Computing: 13th International Symposium, ISVC 2018, Las Vegas, NV, USA, Nov. 2018, pp. 433–444, doi: 10.1007/978-3-030-03801-4_38.
