# Generating Images of dogs, using generative adversarial networks. 
> How does Florida look like under a heavy snow? How will London look like under a sand tornado? These question usually don’t come to mind, but what if we want to simulate Elizabeth Tower under heavy snow with its current construction going on. We need someone with good photoshop experience to help us out. These solutions need a huge amount of labour power and usually time-consuming. We can imagine Trafalgar Square in Christmas, with its amazing festive decoration, but how will it look like if there are 10,000 tourists bursting into the square and taking photos. That’s hard to imagine but these kinds of data and simulations can be useful for security and control organizations.



### Generative Networks 
**Generative** networks are a method of deep learning where the network can learn and mimic the data. Its aim is to model the distribution that a given set of data (e.g. images, audio) came from. Normally this is an unsupervised problem, in the sense that the models are trained on a large collection of data.


### Cycle-GANs
The idea of  Cycle Consistency is using transitivity as a way to regularize structured data has a long history. In visual tracking, enforcing simple forward-backwards consistency has been a standard trick for decades. In the language domain, verifying and improving translations via “back translation and reconciliation” is a technique used by human translators, as well as by machines. More recently, higher-order cycle consistency has been used in structure from motion, 3D shape matching, co-segmentation, dense semantic alignment, and depth estimation. 
We use cycle consistency loss as a way of using transitivity to supervise CNN training.
Our goal here is to find a mapping function between two datasets that following each path,  we get to the same entry data. Mathematically, if we have a network G: X -> Y and another network F: Y -> X, then G and F should be inverse of each other and both networks must be bijections. 
We apply this structural assumption by training both the mapping G and F simultaneously and adding cycle consistency loss that encourages F(G(x)) = x and G(F(y)) = y. Combining this loos with adversarial losses on domain X and Y yields our full objective for unpaired translation. 

 L(G, F, DX, DY ) =LGAN(G, DY , X, Y ) + LGAN(F, DX, Y, X) + λLcyc(G, F),

where λ controls the relative importance of the two objectives. We aim to solve:
G ∗, F∗ = arg min G, F max Dx, DY L(G, F, DX, DY ).
### Wasserstein GAN
Training two distribution networks, means we have two cost functions that we need to find an optimal point. But one of the main problems with generative networks is that this process can go forever because the two cost functions never converge. This means we never know when our leaning has finished. 


### Implementation 
The focus of this project is to find a mapping function between a raw image input and a dataset of images and vice versa. In other words, we are aiming for image-to-image translation, where given an input, we change its features to be a fit in our trained dataset. 
As we are training our networks on an unpaired dataset, we must use feature learning. 
Feature learning or representation learning is a set of machine learning techniques that allow a system to automatically discover the feature set of raw datasets. 

The best method we can use here is Cycle-consistant Generative adversarial networks or Cycle-GANs.
To optimize my cost function I use Wasserstein distance. This will make my network a Wasserstein Cycle-GAN. 

### Run 
```
    python doggoGan.py
```


### Results 
