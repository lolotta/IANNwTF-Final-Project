# IANNwTF-Final-Project

Wintersemester 2022/2023

Lotta Piefke, Berit Reise, Ruth Verdugo

## General Information about the Project:

## Idea:

Image Colorization (estimating L*A*B colors for grayscale images) with CNNs

* Train CNN on colorization and classification task
* How does colorization affect classification and vice versa?
* Does colorization profit from high-level concepts like body parts, background, etc. (as this would theoretically also help with classification)? 
* Try on a small/easier dataset first to verify if the model architecture works and then train on a bigger/harder dataset
* Try different methods to improve results (i.e. different optimzer, dropout layers, residual connections) 


## Datasets:

Natural-Color dataset (NCD) from Anwar et al. (2020) can be found here: 

color groundtruth: https://drive.google.com/file/d/1k_UvYzdrHbphW4UcbDb9jWB0ZQIAGEAo/view

* Number of categories (dog breeds): 20
* Number of images: 721

The Stanford Dog Dataset can be found here: 
http://vision.stanford.edu/aditya86/ImageNetDogs/ 

Images can be downloaded via this link: http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar 

* Number of categories (dog breeds): 120
* Number of images: 20.580


## Architecture: 
![architecture](https://user-images.githubusercontent.com/46372060/223740729-c5bd76aa-8112-4910-9c81-14c922ac47a2.jpg)
(Iizuka et al., 2016)


## Repository Information: 

Code can be found in the folder 'src' - since the datasets (at least the Stanford Dog Dataset) are very large, they are not uploaded in this repository! If you want to run the code, you must download the data first (see Datasets above) and save them in the designated folder of the script to be run.

There are multiple models and hence multiple scripts one can run. Utility functions are in python scripts but for convenience and better visualization the main file to run is always a jupyter notebook. 

Note that notebooks including 'colorful-dataset' in their names refer to the Natural-Color dataset.

The final report as well as meeting notes and the explanation video can be found in the folder 'report'. 
The final report includes detailed descriptions of what was done in this project. 


## References:

Iizuka, S., Simo-Serra, E., & Ishikawa, H. (2016). Let there be color! Joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification. ACM Transactions on Graphics (ToG), 35(4), 1-11.

Anwar, S., Tahir, M., Li, C., Mian, A., Khan, F. S., & Muzaffar, A. W. (2020). Image colorization: A survey and dataset. arXiv preprint arXiv:2008.10774.
