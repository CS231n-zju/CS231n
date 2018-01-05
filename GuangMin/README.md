# Cs231n Learning Note
##### author: Baldr 
##### orgnization: ZJU
##### email: YGM_Baldr@163.com

## Assignment 1


### Q1: KNN

#### Refrence
>>https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

#### Steps:
- Training (Lazily)
- Compute distance 
- get K min-distance point`s class 
- vote

### Q2: Support Vector Machine & Q3 Softmax 
#### Refrence
>>SVM   
      https://en.wikipedia.org/wiki/Support_vector_machine
>>Softmax
https://en.wikipedia.org/wiki/Softmax_function

#### Formula derivation
##### SVM
>$f(x_i;W)=Wx_i$
>
>$L=\frac{1}{N}\sum_iL_i+\lambda\sum_k||W_k^2||$
>
>$L_i=max(0,s_j-s_{yi}+\Delta)$ $normally:\Delta = 1$
> 
>$\frac{\partial }{\partial w_j}L_{i}= \begin{cases}  -x_i & j!=y_i\\ x_i & y=y_i\end{cases}$
>
>so:
>
>![](https://pic2.zhimg.com/50/9cf5d79f58ca3c63ea21b6a9c75940f5_hd.jpg)
>
>![](https://pic4.zhimg.com/50/c6149bcc0b824799d33528f9afd01fd4_hd.jpg)
>![](http://img.blog.csdn.net/20170327152910327?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGppYV8xMDA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##### Softmax
>![](http://img.blog.csdn.net/20170327154010150?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGppYV8xMDA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>
>![](http://img.blog.csdn.net/20170327154601836?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGppYV8xMDA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>
>![](http://img.blog.csdn.net/20170327154836262?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGppYV8xMDA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>
>
>$j!=y_i$
>
>![](http://img.blog.csdn.net/20170327155946785?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGppYV8xMDA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>
>
>$j=y_i$![](http://img.blog.csdn.net/20170425152237405?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGppYV8xMDA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>
##### Steps
- initial $W$ 
- splite tarin set to mini-batch, train them one by one
- compute loss for every step and global loss
- compute gradient of weight
- update $W=W-\alpha dW$ 
- iterate ...


### Q4: Two-Layer Neural Network
### Q5: Higher Level Representations: Image Features

## Assignment 2
### Q1: Fully-connected Neural Network
### Q2: Batch Normalization
### Q3: Dropout
### Q4: ConvNet on CIFAR-10
### Q5: Do something extra!

## Assignment 3
### Q1: Image Captioning with Vanilla RNNs
### Q2: Image Captioning with LSTMs
### Q3: Image Gradients: Saliency maps and Fooling Images
### Q4: Image Generation: Classes, Inversion, DeepDream
### Q5: Do something extra!




