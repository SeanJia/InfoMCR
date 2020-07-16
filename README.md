# Information-Theoretic Local Minima Characterization and Regularization

This repo contains an elegant TensorFlow 2 (re-)implementation of the method introduced in the paper:  
[**Information-Theoretic Local Minima Characterization and Regularization**](https://arxiv.org/pdf/1911.08192.pdf)  
[Zhiwei Jia](https://zjia.eng.ucsd.edu), [Hao Su](https://cseweb.ucsd.edu/~haosu/)  
ICML 2020

<img src="algorithm_illustration.png"
     width=400px />


#### Abstract
Recent advances in deep learning theory have evoked the study of generalizability across different local minima of deep neural networks (DNNs). While current work focused on either discovering properties of good local minima or developing regularization techniques to induce good local minima, no approach exists that can tackle both problems. We achieve these two goals successfully in a unified manner. Specifically, based on the observed Fisher information we propose a metric both strongly indicative of generalizability of local minima and effectively applied as a practical regularizer. We provide theoretical analysis including a generalization bound and empirically demonstrate the success of our approach in both capturing and improving the generalizability of DNNs. Experiments are performed on CIFAR-10, CIFAR-100 and ImageNet for various network architectures.

#### Code
##### Dependencies (Python >=3.6 & pip)
```
numpy, tensorflow>=2.2, absl-py, tqdm
```
##### Examples
To run the regularized SGD (or the baseline) on Wide ResNet for CIFAR-10, execute
```bash
cd InfoMCR
python src/train_cifar10.py --model_id=$MODEL_NAME --use_local_min_reg=True
```
To evalute the proposed metric on models obtained with or without applying the regularizer, run
```bash
cd InfoMCR
python src/eval_cifar10.py --model_id=$MODEL_NAME
```
Notice that, as mentioned in the paper, to compute and compare the proposed metric on local minima obtained with or without applying the regularizer, we normlize each softmax prediction to satisfy the assumption that all local minima in comparison are of similarly small training loss. This is achieved by the following line in `src/eval_cifar10.py`:
```Python
predictions /= tf.expand_dims(tf.linalg.norm(predictions, axis=-1), 1) * scale
```
When computing our metric to compare different local minima in other experiments (different batch size, data aug., etc.), we don't perform this operation as the assumption already holds. 
#### BibTex

#### Contact
I can be reached by zjia@{ucsd.edu, eng.ucsd.edu}.
