
# RISC-V Edge AI with VSDSquadron Pro Workshop


<img width="1665" height="908" alt="image" src="https://github.com/user-attachments/assets/1c0ecb67-d7a3-493a-939c-447d7e6b92fb" />




## Overview


RISC-V Edge AI with VSDSquadron Pro is a hands-on workshop for memory-constrained AI deployment. Built directly on open-source toolchains with SiFive-based cores, you will work in Python and TensorFlow Lite using RISC-V device libraries. The curriculum follows an industry-aligned loop: train, quantize, deploy, benchmark—emphasizing practical methods, reproducible results, and trusted deployment pipelines meeting commercial edge-device standards.

 

You’ll begin with ML foundations: linear regression and gradient descent, implementing models from scratch while profiling memory usage. You’ll then advance to KNN/SVM classifiers, generate decision boundaries, and quantify accuracy/RAM tradeoffs. Each lab includes pre-configured datasets, memory analysis scripts, and VSDSquadron deployment checklists to focus on optimization rather than setup. Results are validated through on-device inference.

 

The program extends to neural networks on RISC-V, guided by instructor-supplied quantization scripts and deployment templates. You’ll reproduce and analyze edge results—model accuracy post-quantization, inference latency, and 16KB memory footprints—while learning compression techniques like weight pruning and 8-bit quantization. Automated memory profiling, optional layer-wise analysis, and real-time validation help benchmark outcomes against industry constraints.

 

Graduates leave with a coherent portfolio: quantized models, deployment scripts and optimization narratives demonstrating readiness for edge AI roles. We include templates for Neural network model conversion, VSDSquadron deployment, and memory reports to accelerate product development. Whether targeting IoT development, embedded systems, or edge AI research, this workshop delivers practical RISC-V deployment experience with globally portable evidence of competence.

## Installation and Settings for Freedom Studio
<details>
<summary>Installation and Settings - Freedom Studio</summary>

### Install needed USB Driver

- Download Zadig from "https://zadig.akeo.ie/"
  Open Zadig from the location of the folder where you downloaded it. Click on ”Options” tab and select ”List All Devices”. Then select ”Dual RS-232-HS (Interface 0). Choose ”libusb-win32” software. Finally click on ”Install or Reinstall Driver”.

&nbsp;
<img width="578" height="256" alt="image" src="https://github.com/user-attachments/assets/76c895ef-0082-4666-9fe1-f278584602bb" />

- Download Freedom Studio via this link "https://vsd-labs.sgp1.cdn.digitaloceanspaces.com/vsd-labs/VSDSquadronPRO.tar.gz"

- extract Freedom Studion & folder structure shouldlook like this

&nbsp;
<img width="857" height="648" alt="image" src="https://github.com/user-attachments/assets/63e49a26-34ee-4ad6-a3a5-48ac0332d88d" />

- start "FreedomStudio-3-1-1.exe"

- create a new project "Validation Software Project"

&nbsp;
 <img width="551" height="391" alt="image" src="https://github.com/user-attachments/assets/de82bfca-5825-49e0-8885-2f12df0e0ee4" />

fill fields with below content and press "Finish" button

&nbsp;
<img width="965" height="1080" alt="image" src="https://github.com/user-attachments/assets/e3511292-8ff3-4617-ac1b-458845b4c82c" />

After some time you will get below window where you select "OpenOCD" tab" and press "Debug"

&nbsp;
<img width="1406" height="1180" alt="image" src="https://github.com/user-attachments/assets/f23f1f47-d321-4c62-a6d9-c88f8cf468a7" />


debug window look like this

&nbsp;
<img width="2447" height="1178" alt="image" src="https://github.com/user-attachments/assets/1112f6c4-f4ba-49d2-b797-4b3a2a2d9d61" />

- press "run" buttoen 

as result youget the SiFive logo on com-port and blue Led is blinking.

&nbsp;
<img width="1915" height="1152" alt="image" src="https://github.com/user-attachments/assets/74eb7b68-1757-45b0-b807-0a6c41cab317" />

&nbsp;
https://github.com/user-attachments/assets/10b7d359-9f3e-4748-a09a-9690f148bcb7


</details>
 
## Edge AI Orientation & Hardware Primer
<details>
<summary>Edge AI Orientation & Hardware Primer</summary>

- AI On A Microchip - Edge Computing With VSDSquadron Pro RISC-V Board

**key components of the VSDSquadron PRO RISC-V development board**

&nbsp;
<img width="1700" height="944" alt="image" src="https://github.com/user-attachments/assets/393a1734-444e-4686-9fc5-2871af46743e" />


- Understanding Your RISC-V Board - Prerequisites to AI on 16KB RAM

for our model we use the 32Mbit SPI Flash.

detail spec of the board that we use:

&nbsp;
<img width="1725" height="965" alt="image" src="https://github.com/user-attachments/assets/a4cd8a68-50fd-4358-9e67-71b1347fb46f" />

**what we need addition is a google colab account!**

&nbsp;
<img width="1934" height="991" alt="image" src="https://github.com/user-attachments/assets/025ab45e-263e-4015-aed5-ff6e090da00a" />

</details>

## ML Foundations (Regression & Optimization)
<details>
<summary>ML Foundations (Regression & Optimization)</summary>

- Best-Fitting Lines 101 - Getting Started With ML

 **we try to minimze errors as much as possible**

&nbsp;
<img width="2784" height="1200" alt="image" src="https://github.com/user-attachments/assets/59b77fb1-bef3-43cf-8c20-c345f9960ec8" />

  
- Gradient Descent Unlocked - Build Your First AI Model From Scratch

  now we start im colab. "https://colab.research.google.com/drive/1S7MID5a163q5FS3oWriPtIATu4E-qOOa#scrollTo=VKW5VEUKKmQR"


  first we import dataset "studentscored.csv"
  second we import pyton libs and initialize them

``` py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

```

&nbsp;
<img width="1503" height="262" alt="image" src="https://github.com/user-attachments/assets/d9b623e5-91e5-479d-ab40-d410c1fd8184" />

**import now dataset and print dataset**

``` py
dataset = pd.read_csv('studentscores.csv')
print(dataset)
```

&nbsp;
<img width="818" height="568" alt="image" src="https://github.com/user-attachments/assets/dc1bcedb-bf4c-47a2-a217-52d14bf7dedd" />

**plot scatter of dataset**

``` py

dataset = pd.read_csv('studentscores.csv')
plt.scatter(dataset['Hours'], dataset['Scores'])
plt.show()

```

&nbsp;
<img width="993" height="529" alt="image" src="https://github.com/user-attachments/assets/6f8a6cb3-5ab6-45fb-a72b-cda1dcb0c8c3" />

**now we split dataset into x & y column**

``` py
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

```

**and we print x & y**

&nbsp;
<img width="835" height="691" alt="image" src="https://github.com/user-attachments/assets/accf00e5-eba1-4947-8927-95e69605b39b" />

**Model definition**

``` py
class Model():
  def __init__(self,learning_rate, interations):
    self.learning_rate = learning_rate
    self.interations = interations
    
  def predict(self,x):
    return x.dot(self.slope) + self,const

  def fit(self,x,y):
    self.m, self.n = x.shape
    self.slope = np.zeros(self.n)
    self.const = 0
    self.X = X
    self.y = y
    

    for i in range(self.interations):
      self.update_weights()
      return self

  def update_weights(self):
    Y_pred = self.predict(self.X)
    dW = - (2 * (self.X.T).dot(self.y - Y_pred)) / self.m
    db = - 2 * np.sum(self.y - Y_pred) / self.m

    self.slope -= self.learning_rate * dW
    self.const -= self.learning_rate * db
    return self
```

&nbsp;
<img width="940" height="619" alt="image" src="https://github.com/user-attachments/assets/e3d8cfcb-878f-4595-ae50-02ceb3080ac6" />

**load dataset into the model**

``` py
model = Model(learning_rate=0.01, interations=1000)
model.fit(X,y)
```

**red line show prediction**

&nbsp;
<img width="852" height="765" alt="image" src="https://github.com/user-attachments/assets/79d718d0-dadc-4f42-9aa2-f257f5538b13" />


- Visualizing Gradient Descent in Action
  
- Predicting Startup Profits – AI for Business Decisions
  
- Degree Up - Fitting Complex Patterns for Edge AI
  
- From Python to Silicon - Your Model Runs on RISC-V (Need VSDSQ Board)

</details>

## From Regression to Classification (KNN → SVM)
<details>
<summary>From Regression to Classification (KNN → SVM)</summary>

- From Regression to Classification - Your First Binary AI Model

- Implementing KNN Classifier in Python - Smarter Decision Boundaries

- From KNN to SVM - Smarter Models for Embedded Boards

- Deploying SVM Models on VSDSquadron PRO Boards - From Python to C (Need VSDSQ Board)

- Handwritten Digit Recognition with SVM - From MNIST to Embedded Boards

- Running MNIST Digit Recognition on the VSDSquadron PRO Board

</details>

## Memory-Constrained ML & Quantization Basics
<details>
<summary>Memory-Constrained ML & Quantization Basics</summary>

- Beating RAM Limits - Quantizing ML Models for Embedded Systems

- Quantization Demystified - Fitting AI Models on Tiny Devices

- Post-Training Quantization - From 68KB Overflow to MCU-Ready AI

- Fitting AI into 16KB RAM - The Final Embedded ML Optimization (Need VSDSQ Board)

- Regression to Real-Time Recognition - A Complete Embedded ML Recap

</details>

## Neural Networks on RISC-V Microcontrollers
<details>
<summary>Neural Networks on RISC-V Microcontrollers</summary>

</details>

## Advanced Quantization & Deployment
<details>
<summary>Advanced Quantization & Deployment</summary>

</details>


## Capstone & Next Steps
<details>
<summary>Capstone & Next Steps</summary>

</details>


## Acknowledgments

- ### Kunal Ghosh, Director, VSD Corp. Pvt. Ltd.
