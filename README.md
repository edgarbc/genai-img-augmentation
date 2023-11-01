# genai-img-augmentation
![GitHub](https://img.shields.io/github/license/edgarbc/genai-img-augmentation)

![pan xray](https://github.com/edgarbc/genai-img-augmentation/blob/main/img/IMG_0010.png)

Deep learning has proven to be an effective method to improve condition identification in healthcare using x-ray images [1]. However, developing deep learning models can be difficult because accessing appropriate training datasets can be complicated due to high costs of expert labeling or privacy restrictions due sensitive patient information [2]. Generative AI has the potential to overcome these limitations by producing synthetic medical images that closely resemble real patient data [3]. In this project I review some generative AI models that have been used to generate images and explain how one of them, diffusion models, can augment an x-ray dataset using Medical Open Network for Artificial Intelligence (MONAI) platform.

This repository is part of a [medium blog](https://medium.com/@viajesubmarino/synthetic-x-ray-dataset-augmentation-using-generative-ai-178ebc15a074) post I wrote about using Generative AI to augment panoramic xrays datasets with monai. 

# Code

I adapted a tutorial from MONAI to train a variational autoencoder to generate panoramic xrays:  

- [panoramic xray augmentation](https://github.com/edgarbc/genai-img-augmentation/blob/main/my_monai_panxray_autoencoder.ipynb)

# References
[1] Litjens, G., Kooi, T., Ehteshami Bejnordi, B., Adiyoso Setio, AA., Ciompi, F., Ghafoorian, M., van der Laak, J.A.W.M., van Ginneken, B., Sánchez, CI. (2017) A survey on deep learning in medical image analysis. Medical Image Analysis.

[2] Masayuki Tsuneki (2022). Deep learning models in medical image analysis. Journal of Oral Biosciences.

[3] Koohi-Moghadam, M., Bae, K.T. (2023) Generative AI in Medical Imaging: Applications, Challenges, and Ethics. J Med Syst 47, 94.


