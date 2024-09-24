## Socially-Aware-Trajectory-Prediction
<div align="center">
<img src="picture/model architecture.png">
</div>
The upcoming prediction of road agent positions is a crucial task in intelligent systems, requiring consideration of various factors such as the multiple categories of road agents, their social interactions, and their uncertain future trajectories. To fully leverage the potential of vision-based deep learning architectures, researchers have focused on trajectory prediction tasks. However, most previous studies have considered these factors separately. To address this issue, this paper proposes a novel social-goal attention networks that combines the strengths of numerous well-known models by considering the social interactions between heterogeneous road agents to predict uncertain future trajectories. The model consists of graph attention network (GAT) and long short-term memory (LSTM) network to respectively encode socially-aware by achieving the influence weights of target road users towards nearby neighbors and encode temporal inputs, goal-directed forecaster (GDF) module to predict each of road users' coarse goals, conditional variational autoencoder (CVAE) module to produce multimodal trajectory prediction, multi-head attentions and feed-forward networks to decode the predicted trajectories. Our model is evaluated on the TITAN dataset, and the results show that the model outperforms previous state-of-the-art models. Additionally, extensive experiments are conducted to show the influence-level of socially-aware interaction correlation, effectiveness of each neural network module, and various predicted trajectory horizons results.

### **How it works:**

## Dataset
```
The data set that is used in this project are [TITAN](https://usa.honda-ri.com/titan)
```
## Data processing
```
python data_preprocessing.py
```
## Train
```
python train_code.py
```
## Evaluate
```
python eval_code.py
```
