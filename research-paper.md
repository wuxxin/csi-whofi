# **WhoFi: Deep Person Re-Identification via Wi-Fi Channel Signal Encoding**

**Danilo Avola, Emad Emam, Dario Montagnini, Daniele Pannone, and Amedeo Ranaldi**

Department of Computer Science, La Sapienza University of Rome
{avola, emam, montagnini, pannone, ranaldi}@di.uniromal.it

### **Abstract**

Person Re-Identification is a key and challenging task in video surveillance. While traditional methods rely on visual data, issues like poor lighting, occlusion, and suboptimal angles often hinder performance. To address these challenges, we introduce WhoFi, a novel pipeline that utilizes Wi-Fi signals for person re-identification. Biometric features are extracted from Channel State Information (CSI) and processed through a modular Deep Neural Network (DNN) featuring a Transformer-based encoder. The network is trained using an in-batch negative loss function to learn robust and generalizable biometric signatures. Experiments on the NTU-Fi dataset show that our approach achieves competitive results compared to state-of-the-art methods, confirming its effectiveness in identifying individuals via Wi-Fi signals.

**Keywords:** Person Re-Identification, CSI, Deep Neural Networks, Transformers, Wi-Fi Signals, Radio Biometric Signature

## **1\. Introduction**

Person Re-Identification (Re-ID) plays a central role in surveillance systems, aiming to determine whether two representations belong to the same individual across different times or locations. Traditional Re-ID systems typically rely on visual data such as images or videos, comparing a probe (the input to be identified) against a set of stored gallery samples by learning discriminative biometric features. Most commonly, these features are based on appearance cues such as clothing texture, color, and body shape. However, visual-based systems suffer from a number of known limitations, including sensitivity to changes in lighting conditions \[4\], occlusions \[6\], background clutter \[20\], and variations in camera viewpoints \[12\]. These challenges often result in reduced robustness, especially in unconstrained or real-world environments. To overcome these limitations, an alternative research direction explores non-visual modalities, such as Wi-Fi-based person Re-ID. Wi-Fi signals offer several advantages over camera-based approaches: they are not affected by illumination, they can penetrate walls and occlusions, and most importantly, they offer a privacy-preserving mechanism for sensing. The core insight is that as a Wi-Fi signal propagates through an environment, its waveform is altered by the presence and physical characteristics of objects and people along its path. These alterations, captured in the form of Channel State Information (CSI), contain rich biometric information. Unlike optical systems that perceive only the outer surface of a person, Wi-Fi signals interact with internal structures, such as bones, organs, and body composition, resulting in person-specific signal distortions that act as a unique signature.

Earlier wireless sensing methods primarily relied on coarse signal measurements such as the Received Signal Strength Indicator (RSSI) \[11\], which proved insufficient for fine-grained recognition tasks. More recently, CSI has emerged as a powerful alternative \[17\]. CSI provides subcarrier-level measurements across multiple antennas and frequencies, enabling a detailed and time-resolved view of how radio signals interact with the human body and surrounding environment. By learning patterns from CSI sequences, it is possible to perform Re-ID by capturing and matching these radio biometric signatures. Despite the promising nature of Wi-Fi-based Re-ID, the field remains underexplored, especially in terms of developing scalable deep learning methods that can generalize across individuals and sensing environments. In this paper, we propose WhoFi, a deep learning pipeline for person Re-ID using only CSI data. Our model is trained with an in-batch negative loss to learn robust embeddings from CSI sequences. We evaluate multiple backbone architectures for sequence modeling, including Long Short-Term Memory (LSTM), Bidirectional LSTM (Bi-LSTM), and Transformer networks, each designed to capture temporal dependencies and contextual patterns. The main contributions of this work are:

* We propose a modular deep learning pipeline for person Re-ID that relies solely on Wi-Fi CSI data, without requiring visual input;
* We perform a comparative study across three widely used backbone architectures (LSTM, Bi-LSTM, and Transformer networks) to assess their ability to encode biometric signatures from CSI;
* We adopt an in-batch negative loss training strategy, which enables scalable and effective similarity learning in the absence of labeled pairs;
* We conduct extensive experiments on the public NTU-Fi dataset to demonstrate the accuracy and generalizability of our approach;
* We perform an ablation study to evaluate the impact of preprocessing strategies, input sequence length, model depth, and data augmentation.

By leveraging non-visual biometric features embedded in Wi-Fi CSI, this study offers a privacy-preserving and robust approach for Wi-Fi-based Re-ID, and it lays the foundation for future work in wireless biometric sensing.

## **2\. Related Work**

### **2.1 Person Re-Identification via Visual Data**

In the field of computer vision, person Re-ID has long been of major importance. Earlier methods primarily relied on RGB images or videos to track people across camera views. Handcrafted descriptors such as Local Binary Patterns (LBP), color histograms, and Histograms of Oriented Gradients (HOG) were widely used to capture low-level visual cues like texture and silhouette. With the advent of deep learning, Convolutional Neural Networks (CNNs) became the dominant approach, enabling hierarchical spatial feature learning \[7\]. Training strategies like triplet loss, cross-entropy with label smoothing, and center loss were adopted to optimize embedding space separability \[15, 19\]. Recent models often integrate attention mechanisms \[10\] and part-based representations \[13\] to handle misalignment and occlusion. Despite strong benchmark performance, these systems rely heavily on high-quality visual input and careful manual tuning, limiting their applicability in uncontrolled environments.

### **2.2 Person Identification and Re-ID via Wi-Fi Sensing**

Several works have extensively investigated human identification and authentication through Wi-Fi CSI, focusing on features such as amplitude, phase, and heatmap variations \[3\]. Early methods include line-of-sight waveform modeling combined with PCA or DWT for classification \[15\], or gait-based identification through handcrafted features \[18\]. CAUTION \[14\] introduced a dataset and few-shot learning approach for user recognition via downsampled CSI representations. More recent methods leverage deep learning models to enhance generalization capabilities \[16\]. A recent approach \[1\] proposed a dual-branch architecture that combines CNN-based processing of amplitude-derived heatmaps with LSTM-based modeling of phase information for re-identification. However, the use of private datasets in such work limits replicability and hinders direct comparison. In contrast, our study relies on a widely available public benchmark, enabling reproducibility and fair evaluation across different architectures.

## **3\. Method**

In this section, details about data pre-processing and augmentation, together with the proposed deep architecture, are presented.

### **3.1 Data Pre-processing**

Data extracted from the CSI complex matrix must first be pre-processed to remove noise and sampling offsets to extract meaningful biometric features.

Channel State Information (CSI): Wi-Fi transmission relies on electromagnetic waves that carry information from a transmitting antenna (TX) to a receiving one (RX). Modern systems adopt Multiple-Input Multiple-Output (MIMO), involving multiple TX/RX antennas, and Orthogonal Frequency-Division Multiplexing (OFDM), a modulation technique that transmits data across orthogonal subcarriers spanning nearly the entire frequency band. The integration of MIMO and OFDM enables sampling of the Channel Frequency Response (CFR) at subcarrier granularity in a CSI matrix. The CSI measurement for each subcarrier k∈K represents the CFR H(θ,γ) between the receiving antenna (RX) θ∈Θ and the transmitting antenna (TX) γ∈Γ and is given by:

Hk(θ,γ)​=∣Hk(θ,γ)​∣ej∠Hk(θ,γ)​

where ∣Hk(θ,γ)​∣ denotes the signal amplitude and ∠Hk(θ,γ)​ the signal phase. By collecting the responses across all TX/RX antenna pairs, a CSI complex matrix of size Θ×Γ×K is formed, representing the CFR across all subcarriers in K.
Amplitude Filtering: Signal amplitude represents the strength of the received signal. For a subcarrier k∈K, receiver antenna θ∈Θ, and transmitter antenna γ∈Γ, the signal amplitude Ak(θ,γ)​ is defined as:
$$A\_{k}^{(\\theta,\\gamma)} \= |H\_{k}^{(\\theta,\\gamma)}| \= \\sqrt{real(H\_{k}^{(\\theta,\\gamma)})^{2} \+ img(H\_{k}^{(\\theta,\\gamma)})^{2}}$$which corresponds to the magnitude of the CSI measurement. In this work, signal amplitudes are cleaned of outliers using the Hampel filter \[2\], which identifies outliers based on the median of a local window and the Median Absolute Deviation (MAD). Given a sequence of amplitude values across p packets, the local window Wp,k of size w (set to 5\) centered on packet p is defined. An amplitude value is classified as an outlier if its deviation from the local median exceeds a fixed threshold. Specifically, any value outside the range:
limitp,k​=median(Wp,k)±ξ⋅MAD(Wp,k)

with ξ set to 3, is considered an outlier and removed.
Phase Sanitization: Signal phase represents the temporal shift of a signal. It is calculated as the arctangent of the imaginary and real parts of the CFR:

Pk(θ,γ)​=tan−1(real(Hk(θ,γ)​)img(Hk(θ,γ)​)​)

To remove any possible phase shifts caused by imperfect synchronization between the transmitter and receiver hardware components, we apply a standard linear phase sanitization technique. The calibrated phase ∠H′(f)k​ for each subcarrier k∈K can be estimated by subtracting a linear term from the raw phase.

### **3.2 Data Augmentation**

To enhance model sensitivity and overall robustness against noise or minor signal fluctuations, we apply several data augmentation techniques during training. These transformations are performed on the extracted amplitude features rather than directly on the raw CSI data. For each amplitude entry, one augmentation is applied with a 90% probability, leaving the remaining 10% unmodified.

* The first augmentation adds **Gaussian noise** n(t)∼N(0,σ2) to the amplitude value, where σ=0.02.
* The second augmentation **scales** the amplitude by a random factor uniformly sampled in \[0.9, 1.1\].
* Finally, a **time shift** is applied by offsetting the amplitude sequence forward or backward by a random integer t′∈\[−5,5\] within a sequence of length P=100.

### **3.3 Deep Neural Network Architecture**

In the proposed pipeline, a DNN is designed to generate a biometric signature from the processed CSI features. The architecture is composed of an Encoder module (Me​) and a Signature Module (Ms​) as shown in Figure 1\.

*Fig. 1: Overview of the proposed framework. The system takes an input signal and processes it through an encoder that extracts latent representations. These features are passed to a signature model that computes a compact signature vector s. The output signature is normalized through l2-normalization.*

**Encoder Module:** The encoder module produces a fixed-size vector that contains human signature relevant information from the provided CSI measurements. This work evaluates three types of encoding architectures:

1. **LSTM Encoder:** Consists of stacked hidden units with interleaved dropout layers. The final hidden state serves as the encoded output.
2. **Bi-LSTM Encoder:** Processes the sequence in both forward and backward directions. The last hidden states from both passes are concatenated to form the output encoding.
3. **Transformer Encoder:** Contains identical layers, each with a multi-head self-attention sub-layer and a position-wise feed-forward network. Sinusoidal positional encodings are added to the input. The output of the final Transformer layer acts as the encoded representation.

**Signature Module:** The Signature module takes the fixed-size vector output from the encoder module and generates a final biometric signature. It consists of a linear layer and a l2 normalization function. The linear layer maps the encoder output to the desired signature s-dimensional space. Normalization ensures that the signatures lie on a hypersphere, which facilitates similarity computations.

### **3.4 Loss Function**

The training phase requires a loss function that facilitates signatures from the same person to be close together in the embedding space, and increases the distance of signatures from different people. The pipeline utilizes in-batch negative loss \[8\]. During training, a custom batch sampler constructs batches, each composed by a query list Bq​={Xi​}i=0N​ and a gallery list Bg​={Xj​}j=0N​. The i-th sample in Bq​ and the j-th sample in Bg​ belong to the same person if and only if i=j. The two lists of biometric signatures are computed by the model: Sq​=DNN({Xi​}i=0N​) and Sg​=DNN({Xj​}j=0N​). A similarity matrix sim(q,g) of size N×N is computed between the query and gallery signatures using cosine similarity. Due to the l2 normalization, this is simplified to the dot product:

sim(q,g)=Sq​SgT​

In the similarity matrix, diagonal elements indicate similarities between positive pairs (same person), while off-diagonal elements correspond to negative pairs (different people). We apply cross-entropy loss across each row to maximize diagonal (positive) scores and minimize off-diagonal (negative) ones.
*Fig. 2: Similarity Matrix example used in in-batch negative loss function.*

## **4\. Experimental Results and Discussion**

### **4.1 Dataset**

Experiments are conducted on the **NTU-Fi dataset** \[14, 16\]. We utilize only the Human Identification (HID) part. The dataset collects the CSI measurements of 14 different subjects, with 60 samples each. Samples were collected in three different scenarios (T-shirt, T-shirt and coat, T-shirt, coat, and backpack). Data was recorded using two TP-Link N750 routers (1 TX antenna, 3 RX antennas). CSI amplitude data were collected across 114 subcarriers per antenna pair and recorded over 2000 packets per sample. Each sample has a dimensionality of 3×114×2000. The dataset is pre-divided into training (546 samples) and test sets (294 samples).

### **4.2 Implementation Details**

* **Hardware:** AMD Ryzen 7 CPU, 64GB RAM, NVIDIA GeForce RTX 3090 GPU.
* **Framework:** PyTorch.
* **Training:** 300 epochs, batch size of 8\.
* **Optimizer:** Adam \[9\] with a starting learning rate of 0.0001.
* **Scheduler:** StepLR, decreases LR by a factor of 0.95 every 50 epochs.

### **4.3 Person Re-Identification Evaluation**

Performance is evaluated using mean Average Precision (mAP) and Rank-k accuracy.

Table 1: Results of each model on the NTU-Fi test set.
| Model | Rank-1 | Rank-3 | Rank-5 | mAP |
| :--- | :--- | :--- | :--- | :--- |
| LSTM | 0.777±0.032 | 0.897±0.014 | 0.933±0.005 | 0.568±0.010 |
| Bi-LSTM | 0.845±0.045 | 0.934±0.022 | 0.958±0.013 | 0.612±0.026 |
| Transformer | 0.955±0.013 | 0.981±0.006 | 0.991±0.000 | 0.884±0.012 |
The model utilizing the Transformer encoder exceeds in performance both LSTM and Bi-LSTM ones, achieving a 95.5% Rank-1 score and an 88.4% mAP score.

### **4.4 Ablation Study**

* **Amplitude Filtering:** Table 2 shows that models trained **without** amplitude filtering achieved better performance, suggesting the filter may have removed useful signal variations.
* **Data Augmentation:** Table 4 indicates that augmentation improved generalization for LSTM/Bi-LSTM. The Transformer did not benefit significantly but still outperformed the others.
* **Packet Size:** Table 3 reveals that LSTM performance was stable or slightly degraded with longer sequences. The Transformer benefited from extended input sequences due to its self-attention mechanism.
* **Encoder Depth:** Table 5 shows the Transformer achieved its best performance with a single layer. For LSTM/Bi-LSTM, stacking layers resulted in marginal gains but slower convergence.

## **5\. Conclusion**

In this paper, we presented a pipeline to address the problem of person Re-ID using Wi-Fi CSI. The proposed approach leverages a DNN that generates biometric signatures from CSI-derived features. We evaluated three encoder architectures (LSTM, Bi-LSTM, and Transformer) on the public NTU-Fi dataset, with the Transformer-based model delivering the best overall performance. By applying a unified and reproducible pipeline to a public benchmark, this work establishes a valuable baseline for future research in CSI-based person re-identification. The encouraging results confirm the viability of Wi-Fi signals as a robust and privacy-preserving biometric modality.

### **Acknowledgements**

This work was supported by the "Smart unmanned AeRial vehicles for Human 1".

### **References**

1. Avola, D., et al. (2022). Person re-identification through wi-fi extracted radio biometric signatures. *IEEE TIFS*.
2. Davies, L., & Gather, U. (1993). The identification of multiple outliers. *JASA*.
3. Duan, P., et al. (2023). A comprehensive survey on wi-fi sensing for human identity recognition. *Electronics*.
4. Feng, Z., et al. (2019). Learning modality-specific representations for visible-infrared person re-identification. *IEEE TIP*.
5. Hermans, A., et al. (2017). In defense of the triplet loss for person re-identification. *arXiv*.
6. Hou, R., et al. (2019). Vrstc: Occlusion-free video person re-identification. *CVPR*.
7. Jalali, A., et al. (2017). Sensitive deep convolutional neural network for face recognition... *Expert Systems with Applications*.
8. Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. *EMNLP*.
9. Kingma, D.P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
10. Li, W., et al. (2018). Harmonious attention network for person re-identification. *CVPR*.
11. Oguchi, K., et al. (2014). Human positioning estimation method using rssi... *Procedia Computer Science*.
12. Sun, X., & Zheng, L. (2019). Dissecting person re-identification from the viewpoint of viewpoint. *CVPR*.
13. Sun, Y., et al. (2018). Beyond part models: Person retrieval with refined part pooling... *ECCV*.
14. Wang, D., et al. (2022). Caution: A robust wifi-based human authentication system... *IEEE IoT Journal*.
15. Xin, T., et al. (2016). Freesense:indoor human identification with wifi signals.
16. Yang, J., et al. (2023). Sensefi: A library and benchmark on deep-learning-empowered wifi human sensing. *Patterns*.
17. Yang, Z., et al. (2013). From rssi to csi: Indoor localization via channel response. *ACM CSUR*.
18. Zeng, Y., et al. (2016). Wiwho: Wifi-based person identification in smart spaces. *IPSN*.
19. Zheng, Z., et al. (2019). Joint discriminative and generative learning for person re-identification. *CVPR*.
20. Zhou, S., et al. (2019). Discriminative feature learning with consistent attention regularization... *ICCV*.
