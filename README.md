My implementation of Histogram of Oriented Gradients. Here I was learnig how does
one of the earlier feature descriptors work. I was using Carlo Tomasi's paper and some 
other sources. This HOG algorith can only work with 128x64 images or it will rescale any to this size
just like an original HOG. I belive it contains a bug where you can't actually choose number 
of cells and cells' size and it works well with default settings: 8 pixel cells and 32 blocks. It can be
fixed easily but I'm long past this project.

Goal for this project was to use it in SVM training.