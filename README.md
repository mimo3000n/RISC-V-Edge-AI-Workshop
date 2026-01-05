
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

**this is slop and const for our line**

``` py
print(model.slope, model.const=
[9.77890599] 2.4644522714760995
```

&nbsp;
<img width="892" height="169" alt="image" src="https://github.com/user-attachments/assets/c0eab9a1-4728-4ff3-b6b6-da510bdfad68" />



- Visualizing Gradient Descent in Action

 **LinearRegression Model**

 ``` py
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation
class LinearRegression:
    def __init__(self):
      self.parameters = {}

    def forward_propagation(self, train_input):
      m = self.parameters['m']
      c = self.parameters['c']
      predictions = np.multiply(m, train_input) + c
      return predictions

    def cost_function(self, predictions, train_output):
      cost = np.mean((train_output - predictions) ** 2)
      return cost

    def backward_propagation(self, train_input, train_output, predictions):
      derivatives = {}
      df = (predictions-train_output)
      dm = 2 * np.mean(np.multiply(train_input, df))
      dc = 2 * np.mean(df)
      derivatives['dm'] = dm
      derivatives['dc'] = dc
      return derivatives

    def update_parameters(self, derivatives, learning_rate):
      self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
      self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']

    def train(self, train_input, train_output, learning_rate, iters):
      self.parameters['m'] = np.random.uniform(0,1) * -1
      self.parameters['c'] = np.random.uniform(0,1) * -1

      self.loss = []

      fig, ax = plt.subplots()
      x_vals = np.linspace(min(train_input), max(train_input), 100)
      line, = ax.plot(x_vals, self.parameters['m'] * x_vals + self.parameters['c'], color='red', label='Regression Line' )
      ax.scatter(train_input, train_output, marker='o', color='green', label='Training Data')

      ax.set_ylim(0, max(train_output) + 1)

      def update(frame):
        predictions = self.forward_propagation(train_input)
        cost = self.cost_function(predictions, train_output)
        derivatives = self.backward_propagation(train_input, train_output, predictions)
        self.update_parameters(derivatives, learning_rate)
        line.set_ydata(self.parameters['m'] * x_vals + self.parameters['c'])
        self.loss.append(cost)
        print("Interation = {}, Loss = {}".format(frame + 1, cost))
        return line,

      ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
      ani.save('linear_regression_A.gif', writer='ffmpeg')

      plt.xlabel('Input')
      plt.ylabel('Output')
      plt.title('Linear Regression')
      plt.legend()
      plt.show()

      return self.parameters, self.loss


```

run model with diffent interation and see how loss is getting smaller if interations are increasing:

``` py

model = LinearRegression()
parameters, loss = model.train(X, y, learning_rate=0.01, iters=100)

```

&nbsp;
<img width="828" height="1075" alt="image" src="https://github.com/user-attachments/assets/3fa81d15-6704-43a1-b8c2-5daed757f0e5" />

  
- Predicting Startup Profits – AI for Business Decisions

Now let import another dataset "50_Startups.csv"

``` py

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

```

and print X values:

&nbsp;
<img width="511" height="919" alt="image" src="https://github.com/user-attachments/assets/6d7bafed-e4c8-410b-a1f5-c7b189cd3986" />

Now we import our model:

``` py

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

```

we split now dataset into dataset for testing and training.

``` py

from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

```

now lets train our training model

``` py

regressor.fit(X_train, Y_train)
```

&nbsp;
<img width="953" height="289" alt="image" src="https://github.com/user-attachments/assets/c2c4d20e-7175-487a-b258-82530b2b4034" />

now let run prediction and print out prediction

``` py

y_pred = regressor.predict(X_test)
print(y_pred)
```

&nbsp;
<img width="807" height="163" alt="image" src="https://github.com/user-attachments/assets/5aaafbca-9aaa-4a54-a6e5-d504309d93f4" />


now we look into actual an predicted values

``` py

for i, (pred, actual) in enumerate(zip(y_pred, Y_test)):
  print(f"Sample {i+1}: Predicted {pred:.2f}, Actual {actual:.2f}")

```

&nbsp;
<img width="749" height="302" alt="image" src="https://github.com/user-attachments/assets/2001e204-4ab5-4747-8cf2-dc675426b5fe" />

now es print "Coeffficints" and "Intercept"

``` py

print("Coefficients", regressor.coef_)
print("Intercept", regressor.intercept_)

```

&nbsp;
<img width="720" height="185" alt="image" src="https://github.com/user-attachments/assets/1cdcfe1c-9f6c-49ef-959c-a961e91b8390" />

``` txt

Y = 0.77884104x1 + 0.0293919x2 0.03471025*x3

```

  
- Degree Up - Fitting Complex Patterns for Edge AI

Now we load our student dataset again:

``` py

dataset =  pd.read_csv('studentscores.csv')
plt.scatter(dataset['Hours'], dataset['Scores'])
plt.show()

```

&nbsp;
<img width="710" height="537" alt="image" src="https://github.com/user-attachments/assets/10f28183-e83e-4b7e-844c-871f9fb69ab4" />


now generate train dataset

``` py

from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

```

now we import a polynomial preprocessor and print poly out

``` py

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train) 
print(X_poly)

```

&nbsp;
<img width="755" height="783" alt="image" src="https://github.com/user-attachments/assets/2b33763c-35db-4116-b811-909bc3a6e512" />

now we transform

``` py

dataset =  pd.read_csv('studentscores.csv')
plt.scatter(dataset['Hours'], dataset['Scores'])
plt.show()

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

```

&nbsp;
<img width="902" height="1142" alt="image" src="https://github.com/user-attachments/assets/378657d1-cf18-447c-98df-1d38c1b1c6da" />

``` py

generate the model

``` py
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X) 
print(X_poly)

model = LinearRegression()
model.fit(X_poly, Y)
```

&nbsp;
<img width="867" height="778" alt="image" src="https://github.com/user-attachments/assets/64091e31-71d5-4ab5-b1ca-a675f3510fbb" />







- From Python to Silicon - Your Model Runs on RISC-V (Need VSDSQ Board)

</details>

## From Regression to Classification (KNN → SVM)
<details>
<summary>From Regression to Classification (KNN → SVM)</summary>

- From Regression to Classification - Your First Binary AI Model

 We open new colab session, call it "Socail_Network_Ads" and import data "Socal_Network_Ads.csv"

 Colab: "https://colab.research.google.com/drive/157WT1Ue6nqG-jtQlkTm1O-R6OD0fk1Og#scrollTo=JPdk7_eGvmez"
 
 Data:
 
 &nbsp;
 <img width="967" height="698" alt="image" src="https://github.com/user-attachments/assets/3a1249c0-d6f5-4f15-bc77-2b22580840e6" />

 Colab session:

 &nbsp;

import needed libs

 ``` py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

```

next we import dataset

``` py

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [1,2]].values
y = dataset.iloc[:, 3].values

```

we print X,Y

``` py

print(X)
print(Y)

```

&nbsp;

<img width="938" height="775" alt="image" src="https://github.com/user-attachments/assets/8663b98e-6937-4754-8ab5-1794d6a7b304" />

&nbsp;

result of "print(y)"

&nbsp;

<img width="672" height="382" alt="image" src="https://github.com/user-attachments/assets/529be847-c839-4077-9e76-64b2cdd8cea1" />

&nbsp;

now we divide in train & test dataset:

``` py

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

```

&nbsp;

<img width="913" height="205" alt="image" src="https://github.com/user-attachments/assets/96ade8d4-3d04-495e-914e-85ad578a174f" />

&nbsp;

now let visualize data:

``` py
import seaborn as sns
plt.figure(figsize=(8,6))
sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=y_train, palette={
                0: "blue", 1: "red"}, marker='o')
plt.xlabel("Age")
plt.ylabel("Estimated Salary")

plt.show()

```

&nbsp;
<img width="978" height="804" alt="image" src="https://github.com/user-attachments/assets/866200b9-4062-437d-b724-e1acc13069b8" />
&nbsp;

now we want to separte the plot in two differen parts, but befor we do that we scale the model down.

``` px

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 

```
and print x_test & x_train

``` py

print(x_train)
print(x_test)

```

&nbsp;
<img width="468" height="671" alt="image" src="https://github.com/user-attachments/assets/de898673-85d9-4776-93bf-597f130d6df1" />
&nbsp;

let plot it again

``` py

import seaborn as sns
plt.figure(figsize=(8,6))
sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=y_train, palette={
                0: "blue", 1: "red"}, marker='o')
plt.xlabel("Age")
plt.ylabel("Estimated Salary")

plt.show()

```

&nbsp;
<img width="898" height="759" alt="image" src="https://github.com/user-attachments/assets/f820716f-7b73-4658-a4a2-557736e53f2f" />
&nbsp;

let create a logistic regression model to separate dataset by line

``` py

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

```

now lets import some metrics

``` py

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix

```

&nbsp;
<img width="930" height="265" alt="image" src="https://github.com/user-attachments/assets/aab702e0-a66a-4c2a-a978-8b667b9a1b7e" />
&nbsp;

lets create prediction

``` py

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

```

&nbsp
<img width="602" height="170" alt="image" src="https://github.com/user-attachments/assets/7148a165-ec0a-4072-9e73-3f0a70bf9958" />

&nbsp;

before we plot we need classiefier coefficeient and intercep value

``` py

print("Coefficient:", classifier.coef_)
print("Intercept:", classifier.intercept_)

```

&nbsp;
<img width="550" height="208" alt="image" src="https://github.com/user-attachments/assets/3f85d500-bb18-4705-a3d6-735c5cd423dc" />
&nbsp;

we have 2 different codefficient's

line equation will be:

y = (-2.07665837 * x + 0.95217247)/1.11008221

``` py

import seaborn as sns
x1 = np.linspace(-3, 3, 100)
x2 = (-2.07665837 * x1 + 0.95217247)/1.11008221

plt.figure(figsize=(8,6))
plt.plot(x1, x2, color='green')
sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=y_train, palette={
                0: "blue", 1: "red"}, marker='o')
plt.xlabel("Age")
plt.ylabel("Estimated Salary")

plt.show()

```

&nbsp;
<img width="959" height="868" alt="image" src="https://github.com/user-attachments/assets/576d2144-1a3a-45a7-87ab-446404b9fbd8" />

&nbsp;

this is predition line plot dividing data into two sectionsabout line people purchase.

<img width="959" height="868" alt="image" src="https://github.com/user-attachments/assets/7015cd43-cb22-4bb1-aafb-86045f79fdb9" />
&nbsp;


- Implementing KNN Classifier in Python - Smarter Decision Boundaries

now lets try to seperate data with a polygon or a specificshape to get better accuracy
for that we implement a KNN classiefier meaning nearest neighbour principle.

``` py


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

```

now let train dataset

``` py

knn.fit(x_train, y_train)

```

now we check accuracy

``` py

y_pred = knn.predict(x_test)
print(f"Test Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")

```

&nbsp;
<img width="824" height="409" alt="image" src="https://github.com/user-attachments/assets/681e69f4-4c7f-46ac-9af6-74bac6407e3f" />

&nbsp;

let create a linspace

``` py

x1_vals = np.linspace(-3, 3, 400)
x2_vals = np.linspace(-3, 3, 400)
x1, x2 = np.meshgrid(x1_vals, x2_vals)

Z = knn.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)
print(Z)

```

&nbsp;
<img width="682" height="399" alt="image" src="https://github.com/user-attachments/assets/7b01805f-8247-4de5-a239-032281cf568e" />

&nbsp;

now we plot it out

``` py

import seaborn as sns
plt.figure(figsize=(8,6))
plt.contourf(x1, x2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=y_train, palette={
                0: "blue", 1: "red"}, marker='o')
plt.xlabel("Age")
plt.ylabel("Estimated Salary")

plt.show()

```

&nbsp;
<img width="935" height="798" alt="image" src="https://github.com/user-attachments/assets/1fc62693-d3d4-4bfb-9762-8391d1ddc18b" />

&nbsp;

now we have a boundery with a better accuracy.
thats how KNN classifier works!


- From KNN to SVM - Smarter Models for Embedded Boards

now we use SVC classifier using "linear" classifier

``` py

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

```

<img width="631" height="222" alt="image" src="https://github.com/user-attachments/assets/7a3237d3-2fc6-4e78-8cb6-50e329f9110d" />

&nbsp;

now lets carry out prediction and accouracy

``` py

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100)) 

```

&nbsp;
<img width="668" height="184" alt="image" src="https://github.com/user-attachments/assets/b56338fe-1cf2-474b-ba80-980d1d4d961b" />

&nbsp;

let's plot it out using a meshgrid and a contour

```py

from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green'))) 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j) 
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    

```

&nbsp;
<img width="1038" height="848" alt="image" src="https://github.com/user-attachments/assets/671e2583-53d3-4ce5-b0db-f4b01598b468" />

&nbsp;

now lets find out the coefficient of the plane

``` py

print(classifier.coef_)
print(classifier.intercept_)

[[1.60291291 0.97138722]]
[-0.76862169]
```

&nbsp;
<img width="556" height="163" alt="image" src="https://github.com/user-attachments/assets/a9d62ee0-42d4-417a-92c5-084bd993536f" />

&nbsp;

now lets change SVC classifier from "linear" to "rbf"

``` py

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

```

&nbsp;
<img width="976" height="1128" alt="image" src="https://github.com/user-attachments/assets/ccef5750-6954-4241-9492-51d1055576d4" />

&nbsp;

we have higther accuracy and dual coefficient

``` py

print(classifier.dual_coef_)
print(classifier.intercept_)

```

&nbsp;
<img width="832" height="385" alt="image" src="https://github.com/user-attachments/assets/51aa6987-91d0-452f-9ab5-d744a234f3c8" />

&nbsp;

now we will look how we can implement this in a C basedcode


- Deploying SVM Models on VSDSquadron PRO Boards - From Python to C (Need VSDSQ Board)

create "svm_model.h"

``` py

print(classifier.coef_)
print(classifier.intercept_)

[[1.60291291 0.97138722]]
[-0.76862169]

weights = classifier.coef_
bias = classifier.intercept_

with open("svm_model.h", "w") as f:
    f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
    f.write(f"#define NUM_FEATURES {weights.shape[1]}\n")
  
    f.write("double weigths[NUM_CLASSSES][NUM_FEATURES] = {\n")
    for row in weights:
        f.write("    {"+ ", ".join(f"{v:.10f}" for v in row) + "},\n")
    f.write("};\n\n")

    f.write("double bias[NUM_CLASSES] = {" + ", ".join(f"{b:.10f}" for b in bias) + "};\n")

print(" Exported SVM model to svm_model.h")

```

&nbsp;
<img width="1185" height="717" alt="image" src="https://github.com/user-attachments/assets/fd92593a-6411-4f9c-af33-81c96f4d507d" />

&nbsp;

``` h
#define NUM_CLASSES 1
#define NUM_FEATURES 2
double weigths[NUM_CLASSSES][NUM_FEATURES] = {
    {1.6029129051, 0.9713872195},
};

double bias[NUM_CLASSES] = {-0.7686216858};

```

code to create "scaler.h"

``` py

mean =   sc.mean_
scale = sc.scale_
with open("scaler.h", "w") as f:
    f.write(f"#define NUM_FEATURES {len(mean)}\n\n)")

    f.write("double mean[NUM_FEATURES] = {\n")
    f.write("    " + ", ".join(f"{m:.10f}" for m in mean) + "\n};\n\n")

    f.write("double scale[NUM_FEATURES] = {\n}")
    f.write("    " + ", ".join(f"{s:.10f}" for s in scale) + "\n};\n")

print("Exported scaler parameter to scaler.h")

```

&nbsp;
<img width="1170" height="868" alt="image" src="https://github.com/user-attachments/assets/ab398c17-a607-4ad0-b6d1-7029ca7d3f74" />

&nbsp;

scaler.h

``` h

#define NUM_FEATURES 2

)double mean[NUM_FEATURES] = {
    38.1266666667, 69583.3333333333
};

double scale[NUM_FEATURES] = {
}    10.0977203148, 34490.9126518211
};

```

now we go to SiFive Freedom Studio

paste scaler.h & svm_model.h into src tree

&nbsp;
<img width="540" height="892" alt="image" src="https://github.com/user-attachments/assets/a86872ec-04c2-46ea-90d3-255cd32b35d5" />

&nbsp;

C program:

``` c

/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <math.h>
#include "svm_model.h"
#include "scaler.h"


void scale_input(float *x) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        x[i] = (x[i] - mean[i]) / scale[i];
    }
}

int predict(float *x){
	float score = bias[0];
	for (int i = 0; i < NUM_FEATURES; i++) {
		score+= weights[0][i] * x[i];
	}
	if (score <= 0) {
		return 0;
	}
	return 1;
}

void print_float(float val)
{
	int int_part = (int)val;
	int frac_part = (int)((val - int_part) * 100);     // 2 deciamal places
	if (frac_part < 0) frac_part *= -1;
	printf("%d.%02d \n", int_part, frac_part);
}

int main () {
	float input[2] = {
			0, 19000
	};

	scale_input(input);
	int label = predict(input);
	printf("predicted output : - %d", label);
    return 0;
}

```

&nbsp;
<img width="2452" height="1206" alt="image" src="https://github.com/user-attachments/assets/45e82d24-9beb-41b1-9fdc-1fa3ae3a7a34" />

if you change something in source code you have to "Terminate and Remoce" the current session. To do so, do a left mouseclick on Debud OpenOCD Launch.

&nbsp;
<img width="829" height="644" alt="image" src="https://github.com/user-attachments/assets/1bad7b84-a26d-48ac-818f-4b125ceab622" />

&nbsp;

here you find "Terminate and Remove" after right mouse click!

then change code and rebuild !


- Handwritten Digit Recognition with SVM - From MNIST to Embedded Boards

now we start working on something complex using a dataset of handwritten images.
for that we start opening a new colab notebook - 

here we load tensorflow classes & dataset

``` py

from tensorflow.keras.datasets import mnist

```
now we have to import libs & classes

``` py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

```

&nbsp:
<img width="869" height="555" alt="image" src="https://github.com/user-attachments/assets/76b64b70-2969-4a7f-8ab1-d94d30c5c5d1" />

&nbsp;

now we split data ino training and test

``` py

(X_train, y_train), (X_test, y_test) = mnist.load_data()

```

&nbsp;
<img width="982" height="215" alt="image" src="https://github.com/user-attachments/assets/32bb4497-cbff-4a8a-bc2f-2d35ed9d8148" />

&nbsp;

let print data

``` py

print(X_train.shape)
print(X_test.shape)

```

&nbsp;

<img width="469" height="149" alt="image" src="https://github.com/user-attachments/assets/a5b45a67-5681-4414-8303-13e28fce40fd" />

we have 60000 training images and 10000 test images, each image is 28x28

now we flatten all images

``` py

X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) 
X_test = X_test.reshape(-1, 28 *28).astype(np.float32)

```

print result

``` py

print(X_train.shape)
print(X_test.shape)

```

&nbsp;
<img width="692" height="207" alt="image" src="https://github.com/user-attachments/assets/c1c026a4-2c7d-4e09-81a7-a76ece8471bb" />

&nbsp;

no we have to scale using standard scaler

``` py

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

```

&nbsp;
<img width="652" height="124" alt="image" src="https://github.com/user-attachments/assets/5e348f87-8975-44c5-ba56-423d530c44d1" />

&nbsp;

let pot images

``` py

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_train[i].reshape(28, 28), cmap = 'gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

```

&nbsp;

<img width="1235" height="725" alt="image" src="https://github.com/user-attachments/assets/3b097dee-750b-4ec1-b24d-421821b95ee8" />

&nbsp;

now we create our classifierwith 10interations cause we have a large dataset

``` py

clf = LinearSVC(dual=False, max_iter=10)

```

now lets train data,it will take some time!

``` py

clf.fit(X_train_scaled, y_train)  

```
&nbsp;
<img width="1440" height="178" alt="image" src="https://github.com/user-attachments/assets/f6f88feb-8fc7-4ba9-9ef3-986200d0a194" />

&nbsp;

now we do prediction and print it

``` py

y_pred = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

&nbsp;
<img width="597" height="185" alt="image" src="https://github.com/user-attachments/assets/3d4c4f00-2cbb-409e-a564-86b5171c4860" />

&nbsp;

we print now classification report:

``` py

print(classification_report(y_test, y_pred))

```

%nbsp;
<img width="875" height="400" alt="image" src="https://github.com/user-attachments/assets/673ed679-7ff6-41b1-8b7d-e4b32d4fafbb" />

&nbsp;

let create a heatmap:

``` py

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Lable')
plt.ylabel('True Lable')
plt.show()

```

&nbsp;
<img width="1049" height="990" alt="image" src="https://github.com/user-attachments/assets/fe7548e0-e362-4881-ba78-3c2616350028" />

&nbsp;

now let plot erverthing with prediced and actual labels

``` py

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(28, 28), cmap = 'gray')
    ax.set_title(f"Label: {y_test[i]}, Predicted {y_pred[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()    

```

&nbsp;
<img width="1127" height="707" alt="image" src="https://github.com/user-attachments/assets/4fea9cae-4f87-478f-ba63-d813df87a62f" />

&nbsp;

now lets plot images which were misidentified

``` py

misclassified_indices = np.where(y_test != y_pred)[0]
plt.figure(figsize=(15,5))
for i, index in enumerate(misclassified_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap = 'gray')
    plt.title(f"True: {y_test[index]}\nPred: {y_pred[index]}")
    plt.axis('off')
plt.tight_layout()
plt.show

```

&nbsp;
<img width="1488" height="742" alt="image" src="https://github.com/user-attachments/assets/8d3cff52-29fa-472d-bdb0-4f03ca97c556" />

&nbsp;

now we move to a much faster implementation of this recognition on our bouard


- Running MNIST Digit Recognition on the VSDSquadron PRO Board

before we upload to our board we have to find out our coeficiants, weihgts, biases

``` py

weights = clf.coef_
biases = clf.intercept_
print(weights.shape, biases)

```

&nbsp;
<img width="994" height="187" alt="image" src="https://github.com/user-attachments/assets/a176d386-734d-4cea-9e02-da8ee84f99fa" />

now we export to svm_model.h

``` py

with open("svm_model.h", "w") as f:
    f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
    f.write(f"#define NUM_FEATURES {weights.shape[1]}\n")

    f.write("double weigths[NUM_CLASSES][NUM_FEATURES] = {\n")
    for row in weights:
        f.write("    {"+ ", ".join(f"{v:.10f}" for v in row) + "},\n")
    f.write("};\n\n")

    f.write("double bias[NUM_CLASSES] = {" + ", ".join(f"{b:.10f}" for b in biases) + "};\n")

print(" Exported SVM model to svm_model.h")

```

&nbsp;
<img width="1221" height="741" alt="image" src="https://github.com/user-attachments/assets/241ce954-7c4c-40f1-ba48-773093b03853" />

&nbsp;

similare we dowload our scales in "scales.h"

``` py

mean =   scaler.mean_
scale = scaler.scale_
with open("scaler.h", "w") as f:
    f.write(f"#define NUM_FEATURES {len(mean)}\n\n)")

    f.write("double mean[NUM_FEATURES] = {\n")
    f.write("    " + ", ".join(f"{m:.10f}" for m in mean) + "\n};\n\n")

    f.write("double scale[NUM_FEATURES] = {\n}")
    f.write("    " + ", ".join(f"{s:.10f}" for s in scale) + "\n};\n")

print("Exported scaler parameter to scaler.h")

```

&nbsp;
<img width="1117" height="899" alt="image" src="https://github.com/user-attachments/assets/ffe61c64-3958-4e7e-a6ca-1da06b4ea35e" />

&nbsp;

now we import .h files into FreedomStudio

&nbsp;
<img width="958" height="719" alt="image" src="https://github.com/user-attachments/assets/f38b0aae-f3a6-4cde-b17b-49d073eb643d" />

&npsp;

since we have multiple classes we have to modify our predic funcion

``` c

/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <math.h>
#include "svm_model1.h"
#include "scaler1.h"
#include "test_images.h"


void scale_input(float *x) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        x[i] = (x[i] - mean[i]) / scale[i];
    }
}

int predict(float *x){
	int best_class = 0;
	float max_score = -INFINITY;
	for (int c = 0; c > NUM_CLASSES; ++c){
		float score = bias[0];
		for (int i = 0; i < NUM_FEATURES; i++) {
			score+= weights[0][i] * x[i];
		}
		if (score > max_score) {
					 max_score = score;
					 best_class = c;
		}

	}
	return best_class;
}

void print_float(float val)
{
	int int_part = (int)val;
	int frac_part = (int)((val - int_part) * 100);     // 2 deciamal places
	if (frac_part < 0) frac_part *= -1;
	printf("%d.%02d \n", int_part, frac_part);
}

int main () {
	for (int i=0; i<NUM_TEST_IMAGES; i++) {
		scale_input(test_images[i]);
		int predicted = predict(test_images[i]);
		int actual = test_labels[i];
		printf("Image %d: Predicted = %d, Actual = %d\n, i, predicted, actual");
	}


}


```

svm_model1.h

´´´ h

#define NUM_CLASSES 10
#define NUM_FEATURES 784
double weights[NUM_CLASSES][NUM_FEATURES] = {
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0037683721, 0.0045697757, 0.0029002457, 0.0029002465, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0179307276, -0.0454586502, 0.0063822057, 0.0256679830, 0.0006374738, 0.0436086338, 0.0001321549, 0.0016679154, 0.0382218620, 0.0457499208, -0.0743927691, 0.0624075681, 0.0033401127, 0.0592544973, -0.0989808420, 0.0065580806, 0.0161480681, 0.0083299659, 0.0048091220, 0.0078475470, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0039983401, 0.0106587164, 0.0148603566, 0.0231127851, -0.0186272991, 0.0354119632, -0.0049802520, -0.0683157197, 0.0315599221, -0.0225567615, -0.1482194100, 0.0464053618, -0.0879546724, -0.1340810798, -0.0383641881, -0.1364720680, -0.1104840805, 0.0295720192, -0.0907139369, 0.0072024434, 0.0141800772, -0.0037246612, 0.0019959385, -0.0000005118, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0074796934, -0.0203506841, -0.0042796484, 0.0488705997, -0.0599611275, -0.0779526372, 0.0441110308, -0.0826407784, -0.0827041903, 0.0276235741, -0.0685220913, 0.0366285855, -0.0366467234, -0.0310409351, -0.0085159383, -0.0165353569, -0.0397106747, -0.0561222401, 0.0129750726, -0.0365070940, -0.0275741393, 0.0118364030, -0.0068943628, 0.0128430090, -0.0282487759, 0.0000000000, 0.0000000000, 0.0097696275, 0.0154362005, -0.0186947909, -0.0103470925, -0.0817453504, -0.0026593911, -0.0190682327, -0.0134010537, 0.0444045692, -0.0664682295, 0.0491814991, -0.0744723804, 0.0825554235, -0.0823041926, 0.0892421288, -0.1179034388, 0.0612601022, -0.0063891860, 0.0098409721, 0.0057065425, -0.0082772620, 0.0001196279, 0.0085277111, -0.0328919370, 0.0257043371, -0.0104409594, 0.0058203728, 0.0000000000, 0.0000000000, -0.0049129644, 0.0103282823, 0.0060074075, -0.0818605872, 0.1241189375, -0.0403824951, 0.0231565035, -0.0232110149, 0.0968215089, 0.0070849583, -0.0711720120, 0.0386567109, 0.0060765344, 0.0533367813, 0.0072374863, 0.0077127920, 0.0215907111, -0.0124427077, -0.0158436969, -0.0286817944, -0.0285609359, -0.0096536566, 0.0078715747, -0.0556619242, 0.0131372487, 0.0003487636, 0.0000000000, 0.0046929227, -0.0254056683, 0.0164612503, -0.0045874593, -0.0347449621, -0.0782730650, -0.0398972326, 0.0561955602, -0.1361411566, -0.0300173896, -0.0289928518, -0.0205206316, 0.0631996624, -0.0166627027, 0.0018215038, 0.0015007516, 0.0026242759, 0.0316734864, -0.0119482638, -0.0021018924, 0.0703524341, 0.0545116654, -0.0132566605, -0.0667017478, -0.0776350027, -0.0291346407, -0.0049554839, -0.0028526914, 0.0016112567, -0.0221207269, -0.0181819921, 0.0023472774, 0.0542181642, 0.0699620610, 0.0072217992, -0.1103185386, 0.0598508131, -0.0117432542, 0.0708676596, -0.0631243093, 0.0239127969, 0.0455875961, -0.0368814071, -0.0136331342, 0.1260407088, 0.0037725706, 0.0679050400, -0.0142345785, -0.0419202949, -0.0508576841, 0.0236861301, -0.0226601136, -0.0082803504, -0.0366588723, 0.0229105649, 0.0297087037, -0.0059475612, 0.0200419690, 0.0490937286, -0.0454191056, -0.1022258968, 0.0099382025, -0.0035256412, 0.0562961733, -0.0093287549, 0.0041114931, -0.0674553273, 0.1044104501, -0.0694237312, 0.0709055969, 0.0253300886, 0.1075708103, 0.0164557024, 0.0436808730, -0.0188904954, 0.0476254095, 0.0486280706, 0.0487423362, 0.0033221438, -0.0447149439, -0.0889355626, -0.0342717245, -0.0038961163, -0.0044954351, 0.0009428155, -0.0358400997, 0.0235247649, -0.0538642109, 0.0502493017, -0.0248150831, 0.0084904707, 0.0018714025, -0.0063009370, 0.0824497049, -0.0689112203, -0.0856428003, 0.1084570414, 0.0070485668, -0.0430843230, 0.0235933727, 0.0616289619, 0.0028342847, 0.0757770985, 0.0492815524, -0.0256860473, -0.0781579254, 0.0215805890, -0.0225179785, -0.0295851749, -0.0378476056, 0.0209994526, 0.0185384584, -0.0918789875, 0.1817164384, -0.2104991709, 0.0613982278, 0.0213943144, 0.0218724665, 0.0120331598, -0.0737772290, 0.0635181043, -0.0364520673, 0.1073253289, 0.0049989072, -0.0583924503, 0.0351042075, 0.0533379153, 0.0560938925, 0.0476972139, 0.1317842053, -0.0282448782, 0.0021206573, 0.0143884000, 0.0922225939, 0.0366780219, 0.0059385497, -0.0143276300, -0.0474900834, 0.0212170534, -0.0193545124, 0.1711270816, -0.2435373969, 0.1400165484, -0.0702149452, -0.0672611677, 0.0197006147, 0.0030691433, 0.0429835636, -0.0864210465, -0.0292126761, 0.0224347519, -0.0894996238, 0.0182671127, -0.0979969946, -0.0784233043, -0.1186536692, 0.0765679616, -0.0909250615, 0.0446159249, 0.0641866345, 0.0360300944, 0.0045425977, 0.0171071518, -0.0191151078, -0.0343780198, -0.0029480788, 0.0171143150, 0.0128677040, -0.0525820568, -0.0178479188, -0.0726358131, -0.0652668078, 0.0417924964, 0.0229075995, -0.0350219693, 0.0195724998, 0.0476561079, -0.0279835527, 0.0189022896, 0.0857193562, 0.0151630494, -0.1083662766, -0.1255424461, -0.0153063164, -0.0154089646, -0.0049337594, 0.0084331169, 0.0545302573, 0.0210703806, 0.0290069075, -0.0191604774, 0.0557550364, -0.0037839073, -0.0871490990, 0.0194169668, 0.0083693343, 0.0206516239, 0.0065610253, 0.0255633599, 0.1032177686, -0.0978603359, 0.0918387212, -0.0052067850, 0.0234800249, 0.0200390788, 0.0794678483, 0.0035831978, -0.0537803031, -0.1786654986, 0.0120888676, -0.1594008494, -0.0839007078, -0.0209907679, 0.0090571285, 0.0161953640, -0.1414735333, -0.0239388708, 0.0527924729, 0.0392826756, -0.0145818289, 0.0099143360, -0.0523359029, 0.0087403820, 0.0031470021, 0.0055703752, 0.0052986607, -0.0513973648, -0.0639094471, 0.0926748549, -0.0065639487, 0.1359430726, -0.0293866940, 0.0150705861, -0.0365729332, 0.0260773148, 0.0951437965, 0.0629820722, -0.3561070565, 0.1150649844, -0.1943780884, 0.0308534913, -0.0173792135, -0.0464298773, 0.0883606022, 0.0492307958, -0.0340867593, 0.0210486125, -0.0189555983, 0.0052024224, -0.0041971956, 0.0018031893, 0.0036800655, -0.0029652565, 0.0023648291, -0.0228788208, 0.0289021010, -0.0108764155, -0.0698540152, 0.0680771012, 0.0502882357, 0.0676045203, 0.0898135824, -0.0618186242, -0.0027982696, -0.0628285357, -0.2380191589, -0.0077623178, -0.1124409018, -0.0213439966, -0.0083793363, -0.0265025158, 0.0198604712, -0.0266166984, 0.0372257193, 0.0160888690, 0.0531821371, -0.0064096867, 0.0063024923, 0.0073868320, 0.0031626092, 0.0048983544, 0.0193887717, -0.0182943230, -0.0157311339, 0.0513353096, 0.0135900322, 0.0353377404, 0.0527985799, 0.0499767558, 0.0186155315, 0.0724883670, -0.0327764031, -0.2370790392, -0.1213217917, -0.0105976260, -0.0717295580, 0.0253935042, -0.0483995182, 0.0021893020, -0.0145620163, 0.0391357008, -0.0698603429, -0.0124277525, -0.0241478635, -0.0486668670, -0.1212812419, 0.0324557669, 0.0000000000, 0.0129503262, -0.0088889812, -0.0099116013, 0.0274638413, 0.0055442585, 0.0588921863, 0.0198164709, -0.0295765694, 0.0439071841, 0.0474328186, -0.0983273975, 0.1241671205, -0.1794649672, -0.0690028932, -0.0612504640, -0.0324372397, 0.0301528066, 0.0132102136, 0.0207640041, 0.0730472627, -0.0461100656, 0.1114797278, -0.0121789690, 0.0340657045, 0.0217315657, -0.0158917172, 0.0154660104, 0.0030114787, -0.0008082960, 0.0116421468, -0.0262735401, -0.0140702172, -0.0135190730, -0.0440718907, -0.0260621389, 0.0785310487, 0.0528799716, 0.0427908946, 0.0506775951, -0.0596501882, -0.1737993400, -0.0353736034, -0.0086273660, -0.0736082940, -0.0188714489, -0.0351403067, -0.0269353530, -0.0955561445, -0.0121531690, -0.0389316936, -0.0177168283, -0.0141163832, -0.0178681599, -0.0103870702, -0.0300709747, 0.0071554561, 0.0030845092, 0.0068007441, -0.0436951132, -0.0376266510, -0.0276964519, 0.0418972934, 0.0545771814, 0.0122291665, -0.0557313995, 0.0290998644, -0.0114663435, 0.1283891769, -0.0783550452, -0.0323874861, -0.0175450871, 0.0123255976, -0.0263778590, -0.0033232464, -0.0684470574, 0.0496133130, -0.0174410533, 0.0375611373, -0.0343971701, 0.0152737845, -0.0454910556, 0.0183753279, 0.0175920022, 0.0000000000, 0.0143097688, -0.0368400630, -0.0021923226, 0.0486552080, 0.0403337267, -0.0326110131, -0.0174447476, 0.0532617164, -0.0336786786, 0.0700235558, 0.0695326830, 0.0199545415, 0.0430719206, -0.1059453538, 0.0416503634, -0.1117014209, 0.0051666221, 0.0540624965, -0.0101362524, -0.0346742635, 0.0488220747, -0.0198496349, -0.0166382062, -0.0131057899, -0.0151987266, -0.0210990617, 0.0093675409, 0.0065746314, 0.0165545974, -0.0222871801, 0.0065210443, -0.0087152713, -0.0326499493, 0.0101504415, 0.0036669550, -0.0727411847, 0.0489956253, 0.0381069460, 0.0793285183, 0.0182310651, -0.0051213391, -0.0170650007, -0.0398985328, 0.0869938393, -0.1999805279, -0.0269503804, -0.0311414120, -0.0612953960, -0.0999215260, 0.0330949879, -0.0775770400, 0.0538348047, -0.0924918131, 0.0573990679, 0.0100959900, 0.0065746328, -0.0096774857, 0.0061739823, -0.0133052661, -0.0415201865, 0.0099631731, 0.0201454797, -0.0181031177, 0.0255016242, 0.0520076727, -0.0209921869, 0.0023094502, -0.0071908996, 0.1610362028, -0.0623984109, -0.0155407482, 0.0222677535, 0.0728097803, -0.1061985901, 0.0522466169, 0.0737748043, -0.0323470300, 0.0713382415, 0.0097823958, -0.0375143923, 0.0231615771, 0.0074347159, 0.0100959913, 0.0000000000, 0.0000000000, 0.0171000581, -0.0122980342, -0.0692589535, 0.0034994531, -0.0020232266, 0.0300299361, -0.0667578169, 0.0209807542, 0.0545850183, 0.0385935586, 0.0398304612, -0.0141056830, 0.0826248746, -0.0032847169, -0.0946765199, 0.0917981134, -0.0800271216, -0.0796105011, -0.0215499352, -0.0731538746, 0.0013295106, -0.0073967976, -0.0132205793, 0.0249515530, -0.0058122798, 0.0000000000, 0.0000000000, 0.0000000000, -0.0182542008, 0.0039607842, 0.0350057226, 0.0137547574, -0.0179960896, -0.0057794185, -0.0207185991, 0.0361583588, 0.0458732623, -0.1190717852, 0.0514266334, -0.0180266953, 0.0137706091, 0.0199311194, -0.1069223218, 0.0860136994, 0.0429168068, -0.0468155331, -0.0514570311, -0.0421041462, 0.0400941066, -0.0405265220, 0.0228855478, -0.0616321262, 0.0076791596, 0.0000000000, 0.0000000000, 0.0000000000, 0.0119706968, 0.0164319274, -0.0437701747, -0.0356049541, -0.0969810665, -0.0048738219, -0.0740981234, -0.0385774554, -0.2745771371, 0.1312220522, -0.2405107057, -0.0958776835, -0.1268628980, -0.1336559674, -0.1030525538, -0.0123584626, -0.2747769192, -0.0700660983, -0.0452420216, -0.0253355101, -0.0384101674, -0.0171130521, 0.0026668071, 0.0100961335, 0.0059729470, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0007434902, 0.0304157881, 0.0293225031, -0.0071219950, -0.0521744347, -0.0243225460, -0.0542491495, -0.0653778979, -0.1011913769, 0.0063695293, -0.0969150278, -0.0473838348, -0.1124828906, -0.0992734747, -0.1107784480, -0.0636337831, 0.0653973371, -0.0056625388, -0.0175418401, 0.0100089280, 0.0147017961, 0.0000667147, 0.0087361664, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0139465645, -0.0095009515, 0.0089032340, 0.0012833199, 0.0159371970, -0.0065202998, -0.0032971566, 0.0099164154, 0.0083256505, -0.0164087068, 0.0048942510, 0.0107450431, 0.0065008983, 0.0357710834, 0.0067083353, 0.0272714689, 0.0103968214, -0.0151913625, -0.0049643495, 0.0090530513, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0041414762, 0.0063410577, 0.0054528243, 0.0054528254, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0067846595, 0.0071909514, 0.0047391960, 0.0132723636, -0.0039651478, 0.0099175922, 0.0082144941, 0.0172837041, 0.0102350207, 0.0143277270, -0.0046603950, 0.0148427636, -0.0330282659, 0.0010922729, 0.0196051737, 0.0124750413, 0.0057920121, 0.0068479349, 0.0146446759, 0.0062650205, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0009894492, 0.0037235539, 0.0099746299, 0.0204569566, 0.0242785798, 0.0239818031, -0.0097074038, -0.0108778817, -0.0105061830, -0.0087570636, -0.0668074657, 0.1178520695, -0.0886763761, 0.0932270678, -0.0028434563, 0.0710309903, -0.0452441575, 0.0646396421, -0.0405805342, 0.0122022139, -0.0036090317, -0.0136335284, 0.0170820831, -0.0053058053, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0109327445, 0.0101186090, 0.0156710953, -0.1426127093, 0.0140861532, 0.0460760620, -0.0546075131, -0.0355592765, -0.0373124431, -0.1114279645, -0.0319536122, -0.0091643014, -0.0929409819, 0.0012571522, -0.0448623563, 0.0096011825, -0.0893350016, 0.0109581422, -0.0327072731, -0.0116990029, -0.0021702352, -0.0080662969, -0.0067153096, 0.0409386003, 0.0100851869, 0.0000000000, 0.0000000000, 0.0120860648, 0.0329999372, -0.0404910247, -0.0045590604, 0.0368654134, -0.0275674035, -0.1071827274, 0.1069496980, -0.0300122391, 0.0178076182, -0.0132243405, -0.0223085330, 0.0076750746, 0.0149543601, -0.0469704952, -0.0053700052, -0.0076135549, -0.0298565315, 0.0163993339, -0.0085961947, 0.0131264807, -0.0233702211, 0.0117672024, -0.0479902740, -0.1181983115, 0.0363010171, -0.0105628072, 0.0000000000, 0.0000000000, 0.0006080892, 0.0343910526, 0.0058581128, 0.0614698391, 0.0089773102, -0.0436557482, -0.0606338061, -0.0677818845, -0.0291785551, -0.1062700589, -0.0902584030, -0.0350853401, -0.0803755104, 0.0407618147, -0.0428308230, -0.0708932312, -0.0203163343, -0.0136776174, -0.0163243492, 0.0143727268, 0.0468674169, -0.0000481533, 0.0622816693, -0.0546702289, 0.0026952394, 0.0076324931, 0.0000000000, 0.0139841870, -0.0287619167, -0.0019345629, -0.0137688552, -0.1211440701, 0.0928293234, -0.0616180304, -0.0489560127, -0.1019449969, -0.0088011881, 0.0404193934, -0.0601196524, -0.0230716807, -0.0341968653, -0.0988378477, -0.0678005748, -0.0767674268, 0.0104105403, -0.0453907953, -0.0175532395, -0.0144233832, -0.0524785114, 0.0384244608, -0.0647166826, -0.1111525876, 0.0195974395, 0.0161828737, 0.0128094055, -0.0328241571, -0.0223902982, -0.0809381047, 0.1675515014, -0.0454523728, -0.0996487858, 0.1001376226, -0.1379401732, 0.0093306533, -0.0225468600, -0.0998516708, 0.0414088532, -0.0770513577, -0.0023964977, -0.0798160305, 0.0312035002, -0.0194931066, -0.0469003270, 0.0046135468, 0.0580158716, 0.0470356700, -0.0372718249, -0.0201472060, -0.0551552039, -0.0866047326, 0.0560897970, -0.0162227437, 0.0033781120, 0.0054386626, 0.0319630422, -0.0371034552, -0.0094045272, 0.0065213673, -0.1313250884, -0.0350661030, -0.0263099698, -0.1339824643, -0.0019631964, -0.0626447149, 0.0145921326, 0.0214688019, -0.0164649361, 0.0048589208, -0.0908365195, 0.0623695467, -0.0039012749, 0.0219228824, -0.0928666535, -0.0357023615, 0.0266428360, -0.0209982857, -0.1364102773, -0.0698261780, 0.0336553275, 0.0108750660, 0.0019393537, -0.0699564252, 0.0349531678, 0.0194111951, 0.0268075501, -0.1425360145, -0.0034502653, 0.1233999762, -0.0060546998, -0.0084852231, -0.0257763034, 0.0018260009, -0.0191278942, 0.0223466926, 0.0332920523, 0.0708847248, 0.0168987749, -0.0661035369, 0.0307284623, -0.0517940551, -0.0292145974, -0.0114802722, 0.0020492163, -0.0994501334, 0.0468875950, -0.1601536506, 0.0291561595, -0.0090909429, 0.0063343546, 0.0488600123, 0.0193162812, -0.0478280602, -0.0677523188, -0.0630438928, -0.1182863411, -0.0723582587, 0.0373848164, -0.0262112329, 0.0204757474, 0.0252751406, 0.0145606775, 0.0447449812, 0.1007992991, 0.0829696672, 0.0529259012, 0.0552699550, -0.0998227317, -0.0064650158, -0.0693484709, -0.0106012633, 0.0161853054, 0.0075658512, -0.1412087636, 0.0189920884, -0.0391938023, 0.0460587985, 0.0149947098, -0.0041170640, 0.0368116383, -0.0540435359, 0.0514659765, -0.0123957769, -0.0962398863, -0.1080344673, -0.0700721412, -0.0970356940, -0.0206490807, -0.0290790191, -0.0673256158, 0.0512342333, 0.0461837641, 0.1251117522, 0.0317606222, -0.0115961933, 0.0075492619, -0.0153507751, 0.0228537947, 0.0142398527, -0.0813268270, 0.0096825263, -0.0512333210, -0.0325048133, -0.0037453032, -0.0001209256, 0.0109489477, 0.0194287503, -0.1043377345, 0.0313174164, -0.0039378733, -0.0949509072, 0.0882512338, 0.0627066940, -0.0658585704, 0.0361917945, -0.0620318191, -0.0632068802, 0.0211101116, 0.0217337276, 0.1253180852, 0.0905263097, -0.0271924872, 0.0260402021, 0.0204778828, -0.0742221898, -0.0047683477, -0.0361948484, -0.0599715341, 0.0890689356, 0.0441720929, -0.0273200052, -0.0479055244, 0.0033722422, 0.0138264869, 0.0156911545, -0.0203133529, 0.0027766557, -0.0015064547, -0.0539056830, 0.1130642736, 0.0524963175, 0.0018846325, 0.0257651945, -0.1296477280, -0.0659560728, 0.0023257659, 0.0036793731, 0.0873041522, 0.0221250580, 0.0520386069, 0.0693953611, -0.0711279705, -0.0094539858, -0.0074078197, 0.0021229455, -0.0411978857, 0.0418049715, -0.0981839998, -0.0889634528, 0.0051597150, -0.0001332931, 0.0028689827, 0.0076595864, -0.0075626544, -0.0260784733, 0.0982690325, 0.0045192074, 0.0381985312, -0.0807798870, 0.0333868182, -0.0654170543, -0.0843691373, -0.0749781859, -0.0334113061, 0.0108521575, 0.0787245683, 0.0511465823, -0.0128214946, -0.0085591874, -0.1715099504, -0.0092390761, 0.0026407312, -0.0337908944, 0.0309474746, -0.0374045096, 0.1229411073, 0.0064198493, 0.0000462581, 0.0048836733, 0.0040233460, 0.0057845401, 0.0225059535, 0.0350210671, -0.0016581594, -0.1126483728, -0.0739557738, -0.0100767763, 0.0500599220, -0.0450683794, -0.1402212011, -0.0156160458, 0.0597848361, 0.0577756853, 0.0747696136, -0.0202322621, 0.0971405946, -0.0665290321, -0.0016427797, -0.0935942252, -0.0328276433, -0.0303292556, -0.0354857315, -0.0473005443, -0.0155216857, -0.0486095187, 0.0186475271, -0.0018246153, 0.0064788041, 0.0070886041, 0.0036804149, -0.0221461120, 0.0592004322, -0.0170692069, 0.0092162032, -0.0274358961, -0.1061196590, 0.0850712831, -0.0412782436, -0.0068279385, -0.0939437010, -0.0132205943, 0.0650839329, -0.0826744847, 0.0161559988, -0.1470404604, -0.0851827079, -0.0789702277, -0.0433005676, 0.0155358206, -0.0417426464, -0.0605512518, 0.0755672163, -0.0496900067, -0.0210063421, 0.0178328841, 0.0000000000, 0.0117966787, -0.0075747934, -0.0307657509, 0.0198996161, -0.0554386013, -0.0034718840, 0.0668414194, 0.0192976460, -0.0606695247, -0.0007020209, -0.0616206575, 0.0463825880, 0.0765634030, 0.0751899386, 0.0397990942, -0.0014099248, -0.1140157494, -0.0184101455, 0.0323861126, 0.0149827723, -0.0279365012, 0.0937430952, -0.0580961627, 0.0505670382, -0.0807211915, 0.0032364350, -0.0027766333, 0.0000494662, -0.0047324837, 0.0112688307, 0.0012036191, -0.0389868490, -0.0778435062, -0.1766984699, -0.1209465682, 0.0031508935, -0.0137829079, -0.1091277520, -0.0278003524, -0.0133898166, -0.0324719803, 0.0617265644, -0.0578335839, -0.1903549821, 0.0640989205, -0.0053061390, -0.0548689627, 0.0047108148, 0.0068781299, -0.0407014853, -0.1114202627, 0.0700208340, -0.0124224185, -0.0008305786, 0.0072705696, 0.0126534984, 0.0033029263, 0.0017311080, 0.0154243386, 0.0044614192, -0.1636764178, 0.0313353731, 0.0595306523, -0.0692908419, 0.0416016084, 0.0145004845, -0.0427238095, -0.0552225447, 0.0256036256, -0.0431863690, 0.0558098027, 0.0883556813, -0.1258741932, -0.0313301002, 0.0306158993, -0.0054866191, 0.0444977365, -0.0329933654, -0.0199674684, 0.0195437163, -0.0194821666, 0.0045906023, 0.0079067900, 0.0000000000, 0.0003931135, 0.0189543219, -0.0654456166, 0.0310367588, -0.2942586516, 0.0037331571, -0.0607361943, 0.0104910763, 0.0342867631, -0.0097575169, 0.0813810588, 0.0695633179, -0.0230234575, -0.0156786787, -0.1031588435, 0.0262719797, 0.0245833716, 0.1022939869, 0.0588096638, 0.0055408496, -0.0597873259, 0.0425099843, -0.0020652886, 0.0078854851, -0.0204003418, 0.0325068875, 0.0072956281, 0.0130510128, 0.0074284120, -0.0157005572, 0.0524358518, -0.0042805138, 0.0598975526, 0.0669518346, 0.0511657569, 0.0366837416, -0.0430548125, 0.0780058250, -0.0622669887, -0.0137340773, 0.0107606709, 0.0196225748, 0.0320727863, 0.0348115193, -0.0167827622, 0.0517331925, -0.0962006409, -0.0095579556, 0.1135655959, -0.0606755045, -0.2012910174, 0.1468079135, -0.0155122874, -0.0066685913, 0.0095443879, 0.0130510145, -0.0061998032, 0.0065552054, -0.0167126303, 0.0180067633, 0.1366388639, -0.0469155678, -0.0390612796, -0.0029760607, 0.0311426606, -0.0418788889, -0.0237745483, -0.0872836366, -0.0944616279, 0.0264716577, -0.0195348057, -0.0141465854, -0.0022041748, 0.1385292905, -0.0080885920, 0.0658759165, 0.0435984650, -0.1001559384, 0.0481844874, -0.0650787950, 0.0865019841, 0.0009256275, 0.0095443896, 0.0000000000, 0.0000000000, 0.0272831361, -0.0531946687, -0.0562787758, -0.0140606664, 0.0279672641, 0.0193793301, -0.0306593981, -0.0065368310, -0.0418568096, -0.0475491391, 0.0275858708, -0.0054105338, -0.0953489507, 0.0359266881, 0.0133170635, 0.0695241534, -0.1140842887, 0.0090095108, -0.0368519531, -0.0302204951, 0.0935430495, -0.1049849502, -0.0335384032, 0.0049291660, 0.0203260460, 0.0000000000, 0.0000000000, 0.0000000000, 0.0106744383, -0.0024199824, -0.0125009805, -0.0078449901, -0.0065870464, 0.0109601888, -0.0503125482, 0.0072555039, -0.0244536308, -0.0270483751, -0.0381788613, -0.0752597199, -0.0478238261, 0.0013166255, -0.1276996181, 0.0496752337, -0.0266233435, 0.0945664721, -0.0086581097, 0.0048477252, -0.0536174631, 0.0497682760, -0.0056422482, 0.0117790292, 0.0488435640, 0.0000000000, 0.0000000000, 0.0000000000, 0.0049104439, -0.0049043917, 0.0172398446, -0.0622036168, 0.0223858071, -0.0971347249, -0.0551984209, -0.0863218553, -0.1826247418, -0.2085333363, -0.0860868967, -0.0980562499, -0.0348251990, -0.1604803526, -0.1260202259, -0.1006806971, -0.1183132259, -0.0748657854, -0.0308726997, -0.0300182725, -0.0042242509, 0.0090197179, 0.0086003861, -0.0072997559, -0.0299929380, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0002620733, 0.0123943626, -0.0220548872, -0.0198849521, 0.0012617427, -0.0240663260, -0.0500173348, -0.0152047557, -0.0403361689, -0.1658783900, -0.1165561161, -0.0883691565, -0.0370881332, 0.0135683655, -0.0417453693, -0.0033439851, -0.0003031482, -0.0129711961, 0.0360502555, 0.0058791098, -0.0351004829, 0.0134252861, 0.0049074821, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0050310177, 0.0189121630, -0.0310672041, 0.0333264355, -0.0086962402, 0.0144711814, -0.0045035114, -0.0238224066, 0.0436814085, 0.0162348610, 0.0205016015, 0.0250358426, -0.0157875733, 0.0296062503, -0.0579988592, 0.0286308335, -0.0052061546, -0.0016181131, -0.0066711220, 0.0100798239, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0015401104, 0.0051990504, 0.0069078557, 0.0069078565, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0031873134, 0.0040328204, -0.0076011227, 0.0026785000, 0.0027103858, -0.0097072037, 0.0055936208, 0.0057643517, -0.0193615919, -0.0158626421, 0.0051611549, -0.0093676195, 0.0147738993, -0.0119061350, 0.0040414218, -0.0690412740, 0.0001050474, 0.0012681119, 0.0045188078, -0.0014393920, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0022750833, -0.0032890544, 0.0037024360, -0.0095937819, 0.0158767340, -0.0133606085, -0.0033199514, 0.0079833062, -0.0296078428, -0.0045441089, 0.0005100187, -0.0173485340, -0.0164117090, 0.0005997949, -0.0256379521, -0.0127324673, -0.0103410909, -0.0250174155, -0.0238531187, 0.0034011999, -0.0346017280, 0.0223203459, 0.0137347280, 0.0027034122, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0034178056, 0.0086473108, -0.0050809809, 0.0021321409, -0.0095318690, 0.0118624557, -0.0193051657, 0.0252692313, 0.0059431784, 0.0041753481, 0.0003706837, 0.0158934275, -0.0059906899, -0.0177290959, 0.0085778562, -0.0059049538, -0.0277782014, -0.0004688883, 0.0003582557, -0.0105289444, 0.0231316239, -0.0426876333, -0.0351400891, 0.0140076648, 0.0002550255, 0.0000000000, 0.0000000000, 0.0009412028, -0.0043462740, -0.0069357043, -0.0064478610, -0.0069062390, 0.0252876723, 0.0147830679, -0.0184629606, 0.0317010646, 0.0091648365, 0.0017858961, 0.0283744783, -0.0010254408, 0.0415658059, -0.0157917935, 0.0196678551, -0.0293146401, 0.0003323218, -0.0264170221, -0.0157895026, -0.0211757236, -0.0450318486, -0.0287665505, -0.1075639309, 0.0482784098, -0.0155737945, -0.0038605079, 0.0000000000, 0.0000000000, -0.0006983245, 0.0044319056, 0.0087339344, -0.0033889848, -0.0151662727, -0.0011916987, 0.0206510682, -0.0211678542, 0.0085216840, 0.0215554201, 0.0127233034, 0.0429888095, 0.0030215042, 0.0193765686, 0.0487797087, 0.0163189710, 0.0265433830, -0.0100877985, 0.0037636028, 0.0085995285, 0.0091769497, 0.0286429953, -0.0522736287, 0.0175894766, 0.0037974085, 0.0003326724, 0.0000000000, 0.0044439483, -0.0070951291, -0.0095159057, -0.0086379382, -0.0027180487, -0.0152169294, 0.0091291639, -0.0111921243, 0.0101818838, 0.0248379844, -0.0077131209, -0.0070240761, 0.0178927590, -0.0128612786, 0.0268451967, -0.0033645548, 0.0182788962, -0.0139787562, -0.0028583228, -0.0382536406, -0.0103100244, -0.0128770564, -0.0283400541, -0.0253319186, -0.0371161790, 0.0163950786, -0.0108000508, 0.0041985854, -0.0149410420, 0.0227419018, 0.0280682826, 0.0058164060, 0.0074051139, 0.0352861208, -0.0335033409, 0.0307432714, -0.0282822395, -0.0308582674, 0.0362803740, -0.0183064405, 0.0329126504, 0.0224318795, -0.0309084905, 0.0206459507, -0.0593614164, -0.0089597616, -0.0277971515, -0.0255091353, 0.0170547356, 0.0296241521, -0.0795890551, -0.0572072603, 0.0865602798, -0.0356183487, 0.0189244046, -0.0045620460, 0.0162716851, -0.0119265543, -0.0506099657, 0.0016636378, -0.0253781848, -0.0026807853, 0.0286637670, 0.0121309521, -0.0028136405, 0.0372871398, -0.0054629006, -0.0111879194, -0.0107020787, 0.0289233947, -0.0054961250, -0.0224372459, 0.0261925551, -0.0194895416, 0.0215762681, 0.0323078462, -0.0064761691, -0.0300776967, 0.0592395411, -0.0868463558, -0.1428854620, 0.0104741027, 0.0170159917, 0.0046113709, -0.0196557738, 0.0023013380, 0.0094877738, 0.0014036385, 0.0649865463, -0.0963699591, 0.0176404892, -0.0497815625, 0.0220452473, 0.0095509044, 0.0303520504, -0.0205435327, 0.0004792531, -0.0180642819, 0.0238887535, 0.0447976405, -0.0264144014, 0.0199240599, -0.0016253248, -0.0214712837, -0.0352702701, 0.0323558887, 0.0146054662, -0.0857674720, -0.0107922935, -0.0104757826, 0.0028506826, 0.0037408588, -0.0042865576, -0.0286958354, 0.0466808125, -0.0432498127, 0.0043218531, 0.0835788371, -0.0518568410, 0.0247207485, 0.0074429582, -0.0265890359, 0.0228041253, 0.0131663161, -0.0109379018, 0.0356203966, -0.0440564711, -0.0232161981, 0.0596824668, 0.0357541559, -0.0289124393, 0.0186028195, 0.0182877429, -0.0210823828, -0.0244660950, -0.0112282549, -0.0479874718, 0.0484152197, -0.0094747872, 0.0039521662, -0.0442688465, -0.0381951254, -0.0440281798, 0.0095199226, 0.0226795056, 0.0167181770, 0.0267077477, -0.0867977503, -0.1109002498, -0.0904448811, -0.0862830462, -0.1128802002, -0.1018120599, -0.0757815267, -0.0167763318, -0.0074745471, -0.0514101297, -0.0118563481, 0.0303671632, -0.0482986410, -0.0110808182, -0.0008060394, 0.0476538080, -0.0424779503, 0.0189818397, -0.0552294576, 0.0027257638, 0.0016077072, 0.0185391148, 0.0476825345, -0.0231345303, 0.0229645771, 0.0100627482, -0.1669848183, -0.1435663806, -0.0244949302, -0.0417383756, -0.0442060735, -0.1113764649, -0.0584926235, -0.0397600908, -0.1283974944, -0.0874162526, 0.0064367078, -0.0425486516, -0.0309481406, 0.0613137981, 0.0218321540, 0.0291327253, -0.0089320379, -0.1391079394, -0.0007373267, 0.0178496221, 0.0210665070, -0.0218372874, 0.0011979367, 0.0030817375, -0.0042642780, -0.0018910217, -0.0836734307, -0.0366910648, -0.0143077833, -0.0993941011, -0.0743572354, -0.0421817607, -0.0389113473, 0.0023807057, -0.0227449437, -0.0066091053, -0.0079319707, -0.0255299424, -0.0617168182, 0.0056174955, -0.0124258985, -0.0596548954, -0.0304193282, -0.0340247234, -0.0281963907, -0.0166473727, 0.0485063422, 0.0034361739, 0.0172634552, 0.0096615942, 0.0020488591, -0.0217666970, 0.0157309615, -0.0232970482, -0.0042220064, -0.1415732827, -0.0897511139, 0.0602715157, -0.0302901482, -0.0205047043, -0.0189837501, -0.0017449881, 0.0218199777, -0.0337077169, 0.0242351315, -0.0209897415, 0.0221077398, -0.0353968782, 0.0161520755, -0.0163049510, 0.0174146679, -0.0196121461, -0.0225703686, 0.0164224888, -0.0504890569, 0.0055065966, 0.0237162404, 0.0108242860, 0.0034936376, 0.0024187482, -0.0007707441, 0.0202913631, 0.0202502415, 0.1391706117, 0.0352118333, -0.0725689425, 0.0071539167, 0.0041234123, 0.0100065874, 0.0175001747, -0.0112214967, 0.0357019053, -0.0118615296, -0.0116085826, 0.0142846103, -0.0321673538, -0.0473905336, 0.0273619839, -0.0553479853, -0.0221440904, -0.0264541812, -0.0675633279, 0.0538623534, -0.0009510430, 0.0096027806, 0.0092529458, 0.0079641999, -0.0011825369, -0.0318412544, 0.0078754927, -0.0054151277, -0.0549590475, 0.0599784549, 0.0425513650, -0.0213419839, 0.0171655734, 0.0206250419, 0.0099225764, 0.0407211449, 0.0327382361, -0.0169995192, 0.0083771312, 0.0291977479, 0.0610336952, -0.0328155166, 0.0054328099, -0.0244261557, 0.0646100697, -0.0047417200, 0.0486393657, -0.0165183448, 0.0193380645, 0.0264194110, 0.0018450340, 0.0000000000, -0.0034654586, 0.0283849940, 0.0044516585, 0.0120508086, 0.0336640779, 0.0475477229, -0.0321201116, 0.0319287880, 0.0068833812, 0.0151343835, 0.0169705617, 0.0301566975, 0.0410391164, -0.0005995600, 0.0575086771, 0.0014165964, -0.0331193803, 0.0378420521, -0.0233232861, 0.0015690137, -0.0395220324, 0.0015385500, 0.0005889205, -0.0142122523, 0.0347242344, 0.0024162775, 0.0119792640, 0.0059456543, 0.0032475555, -0.0213536315, 0.0015310672, -0.0014784002, 0.0558528974, 0.0136277213, 0.0605367184, 0.0220572968, -0.0004284727, 0.0329689229, -0.0261461476, 0.0507262624, 0.0320041879, 0.0328060759, 0.0109107526, 0.0359019446, 0.0223146783, -0.0165829333, 0.0223236208, 0.0025990847, 0.0309111834, 0.0163568880, 0.0228697430, 0.0483463244, 0.0086724985, -0.0019991971, 0.0013723889, 0.0001400523, -0.0022718018, -0.0168526082, -0.0256007798, 0.0127600367, -0.0374037654, 0.0060500358, 0.0282708978, -0.0086304788, 0.0492971175, 0.0296869808, 0.0248026672, 0.0040868942, 0.0495294134, 0.0350340526, 0.0192299361, -0.0004807710, 0.0356542349, 0.0103211440, -0.0213584464, 0.0047391743, 0.0022089282, 0.0191055376, 0.0150003371, -0.0192524479, 0.0337999186, 0.0214063184, -0.0010019479, 0.0000000000, 0.0007782071, 0.0197860412, 0.0120360916, 0.0266278218, 0.0330291928, 0.0317409312, 0.0558281973, 0.0523262806, 0.0187566208, -0.0016999873, 0.0660528625, -0.0185720366, -0.0098684037, -0.0091932865, 0.0093359027, 0.0268045052, -0.0556707152, 0.0247536694, 0.0267442481, 0.0133166442, 0.0274890293, -0.0112333568, 0.0064551570, 0.0491127559, 0.0021360092, -0.0181338589, -0.0009726833, 0.0003267042, -0.0026387345, -0.0086156550, 0.0021690477, -0.0414733426, -0.0036514217, -0.0174369888, -0.0001357175, -0.0491978273, 0.0537832464, 0.0166387634, -0.0114647102, 0.0005030790, -0.0109674922, -0.0086267614, -0.0356209629, 0.0167644063, 0.0444351767, -0.0226780938, 0.0560264256, -0.0094660551, 0.0265480250, 0.0094103788, 0.0380310074, -0.0073221889, 0.0192901071, 0.0144848164, 0.0013876994, 0.0003267050, -0.0003994484, -0.0019120892, -0.0063453721, 0.0258488232, 0.0088926470, 0.0168963930, 0.0279782399, 0.0552244502, 0.0211999007, -0.0051608669, 0.0279515007, -0.0001731430, 0.0150644627, -0.0219959008, -0.0292146813, -0.0020860889, -0.0162605300, 0.0443557026, -0.0180428313, 0.0345092281, -0.0046843257, 0.0168857914, -0.0017503682, 0.0195663352, -0.0094017719, -0.0216993299, 0.0013877003, 0.0000000000, 0.0000000000, -0.0083591359, 0.0080699619, 0.0059629909, -0.0177589396, 0.0195129081, -0.0211730098, -0.0276486903, -0.0239555629, -0.0223193637, -0.0265621992, -0.0043420652, -0.0505511246, 0.0095639646, -0.0583458309, 0.0208404567, -0.0210172846, 0.0137189069, 0.0266111979, -0.0053305953, 0.0256833678, 0.0050301636, 0.0166221050, -0.0072404840, 0.0009886570, -0.0010782130, 0.0000000000, 0.0000000000, 0.0000000000, -0.0004277715, -0.0542412246, -0.0040177352, -0.0075207298, -0.0180043496, -0.0291250204, 0.0051148872, -0.0103367077, -0.0182498252, -0.0080575365, 0.0040717068, -0.0379733765, 0.0463951307, -0.0404632678, -0.0311832489, 0.0115001037, 0.0038676809, 0.0077683045, -0.0104990696, 0.0154125994, 0.0080140519, -0.0214633946, 0.0244448322, 0.0280884827, -0.0230328725, 0.0000000000, 0.0000000000, 0.0000000000, 0.0044906875, 0.0090686527, 0.0089909237, -0.2609388454, 0.0624049025, -0.0857210122, -0.0121777819, -0.1241015828, -0.0535508240, -0.0803233590, -0.0551101240, -0.0540073271, -0.0468732387, 0.0464622847, -0.0724749608, 0.0322666975, -0.0241167122, -0.0329386077, 0.0195273755, -0.0194157919, -0.0265417141, -0.0245508535, -0.0380478579, -0.0159125076, 0.0197687161, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0046818015, 0.0113594631, -0.0050839533, 0.0093479820, -0.0500749620, -0.0409331427, -0.0183889251, -0.0268761965, -0.0930793721, -0.0204490608, -0.0740695387, -0.0784869470, -0.0887446557, -0.0966960383, -0.1184402941, -0.0567519465, 0.0352359223, -0.0344946861, -0.0277820992, -0.0314616894, -0.0004775572, 0.0176565263, 0.0021609853, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0074075695, 0.0092403555, 0.0024941281, 0.0034612427, 0.0147207351, 0.0085606712, -0.0159890283, -0.0370306860, 0.0054947387, 0.0203700826, 0.0332413129, 0.0171246325, 0.0388147541, 0.0379832691, 0.0162387568, 0.0019375920, 0.0183454921, -0.0050764758, 0.0119738096, 0.0097277640, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0000216450, 0.0014855787, 0.0025802860, 0.0025802866, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0023821841, 0.0057866181, 0.0098210989, 0.0012153189, 0.0133202663, 0.0122608579, 0.0093258405, 0.0212723557, 0.0006173420, 0.0205532659, -0.0129143083, -0.0042949588, 0.0105115224, -0.0075862072, 0.0251758849, 0.0108242103, -0.0034706069, 0.0170918499, -0.0044766741, 0.0061931801, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0050247020, 0.0072434778, 0.0158754440, 0.0015714701, 0.0018007295, 0.0100645849, 0.0069227654, -0.0250593878, -0.0427101090, 0.0258830838, 0.0055926108, -0.0305514779, 0.0801535266, -0.0853893775, 0.0413572579, 0.0014207302, -0.0394580429, -0.0118435627, -0.0069759715, -0.0037255383, -0.0260851687, 0.0388868650, 0.0255652411, -0.0177238469, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0138365922, 0.0082728333, -0.0093008574, 0.0391073610, -0.0586048103, 0.0134856703, 0.0278097517, -0.0385689572, 0.0056027200, -0.0021983533, 0.0540588718, -0.0737357998, 0.0678240933, -0.0001744120, 0.0288442821, -0.0088836011, 0.0482888043, -0.0233444171, 0.0725493537, -0.0070152106, -0.0723703161, -0.2090243183, 0.0153760393, 0.0274022042, 0.0031458810, 0.0000000000, 0.0000000000, 0.0025266109, 0.0052356132, 0.0035014076, -0.0080991452, 0.0124499125, 0.0104688258, -0.0109179515, 0.0073863448, 0.0063123606, 0.0290929888, 0.0016325042, 0.0074564101, 0.0154370931, 0.0022946076, 0.0053455241, 0.0268689081, -0.0240803171, 0.0393017594, -0.0287744562, 0.0232478411, -0.0473199022, 0.0688577874, -0.1359906622, 0.0217534388, 0.0195474158, -0.0007732161, 0.0041245821, 0.0000000000, 0.0000000000, -0.0073371249, 0.0050617447, 0.0073234376, 0.0044970877, 0.0152688875, 0.0221413761, 0.0010774923, 0.0111062016, -0.0016734602, 0.0183883470, 0.0272886912, 0.0037113727, 0.0079400690, 0.0027572566, 0.0364663296, -0.0462746028, -0.0136432463, 0.0170172857, -0.0198111071, -0.0341636460, -0.0526267331, 0.0290743031, -0.0404595507, 0.0268575928, 0.0066408007, 0.0006014616, 0.0000000000, 0.0115261443, -0.0011481205, 0.0103201356, 0.0020486347, -0.0159181690, 0.0241007988, -0.0240454621, 0.0110666738, 0.0158847093, 0.0362628121, 0.0214383479, -0.0120782629, 0.0483089504, -0.0191376725, 0.0261026602, 0.0190906741, -0.0084377323, 0.0594157154, -0.0711294299, 0.0148654384, 0.0333948029, 0.0119911687, -0.0323633341, 0.0044179465, -0.1026160318, -0.0340626962, 0.0174833804, 0.0018784530, 0.0125487689, -0.0101905111, 0.0092556159, -0.0018626507, 0.0215036961, 0.0071380625, 0.0022021767, 0.0058701567, 0.0047526311, -0.0156853136, -0.0110861966, 0.0170569964, -0.0214240013, 0.0508367165, 0.0278341772, -0.0191874953, 0.0373519843, -0.0589393325, 0.0138876923, 0.0130395409, -0.0815029632, -0.0602414815, 0.0204801616, -0.0308020646, -0.0468365535, 0.0285091638, 0.0062443016, 0.0036801279, -0.0332522735, -0.0113971025, -0.0073958775, 0.0176924518, -0.0070236464, -0.0084148738, -0.0061699426, -0.0058470674, -0.0480706192, 0.0038677017, -0.0147089301, 0.0248728823, 0.0074887900, -0.0056161167, 0.0035523257, 0.0167223355, -0.0041244018, 0.0652444983, 0.0046386582, 0.0028824401, -0.0047993344, 0.0323319570, -0.0841729653, -0.0671522200, -0.0729883806, 0.0653901959, 0.0099133715, -0.0035227538, 0.0372393497, -0.0015320599, 0.0043255089, 0.0224091200, -0.0066385801, 0.0206658656, -0.0000266355, -0.0166018690, 0.0153864897, 0.0017255489, -0.0648382445, -0.0344282332, -0.0404789226, 0.0062050157, 0.0531882834, 0.0251510112, 0.0271334291, 0.0002769724, 0.0334519252, 0.0524044747, 0.0366597200, 0.0660850288, -0.0356314960, -0.2899247873, 0.0696309141, -0.0247353040, 0.0105731296, -0.0012340731, -0.0227695639, -0.0396484166, -0.0028203371, -0.0224129693, 0.0221173629, -0.0102365068, 0.0334239398, -0.0179626317, -0.0759559408, -0.1038204811, -0.0535179481, -0.0790674176, 0.0502685424, 0.0375542471, 0.0163621842, 0.0459852557, 0.0296497907, 0.0203564730, -0.0029360419, 0.0504055267, 0.0297310336, 0.0172768111, 0.1421669017, -0.1818717122, 0.0685616569, -0.0430484095, 0.0018150125, 0.0005934935, -0.0022370286, -0.0015114651, 0.0452001378, 0.0231797610, -0.0178196768, -0.0016733197, -0.0818393226, -0.0620369818, -0.0530011045, -0.0594211299, -0.0253645006, 0.0059115670, 0.0023035494, 0.0310554674, 0.0511436116, -0.0087107273, 0.0447859813, 0.0234693434, 0.0101442173, -0.0069442539, -0.0476092647, -0.0080434841, 0.0486651450, -0.3156220148, 0.0045953038, -0.0003681343, -0.0000490023, -0.0016657132, 0.0145502042, -0.0327884337, -0.0182836672, -0.0177853253, -0.0094993379, -0.0700622918, -0.0512369523, -0.0482030214, -0.0138737133, 0.0260517775, -0.0397809006, 0.0569068185, -0.0155344113, 0.0730131706, 0.0351484128, 0.0435061863, 0.0017602780, 0.0115505395, 0.0078216731, 0.0369088965, -0.0658258725, -0.0759601156, -0.2646937184, -0.2257612294, 0.1186228272, 0.0206474325, 0.0027717264, 0.0004084735, -0.0022523978, -0.0020985444, -0.0037856619, -0.0246909947, 0.0066658071, 0.0451327946, -0.0104617749, 0.0029547414, -0.0260399942, -0.0474010375, 0.0152521921, 0.0434606140, 0.0106871315, 0.0555465093, -0.0511527194, 0.0460086069, 0.0081074900, -0.0040604188, -0.0217217667, -0.0801100093, -0.0598203661, -0.1701875715, 0.0971544633, 0.1328241283, -0.2070097337, 0.0184062701, 0.0065982538, -0.0005195153, 0.0109023082, -0.0192641308, 0.0411241314, -0.0048926217, -0.0198263855, -0.0674729721, -0.0230947977, -0.0136613418, -0.0022857521, -0.0281104660, -0.0316383820, 0.0683848291, -0.0214086201, 0.0113598206, 0.0029047929, 0.0217070857, 0.0028872744, 0.0208896138, -0.0525988208, -0.0486759112, 0.0469376981, 0.0120828007, -0.0233258255, -0.0336157033, -0.0353205337, 0.0084066353, 0.0007257065, 0.0001417785, 0.0060616307, -0.0112466730, -0.0242286259, -0.0179968154, -0.0066829558, -0.0011700204, -0.0322091596, -0.0390948294, -0.0267986897, -0.0292752314, 0.0017451459, 0.0253149033, -0.0028960430, 0.0016195240, 0.0334945376, 0.0049349610, -0.0011274411, -0.0622907542, 0.0132144652, 0.0934427174, -0.0013101013, 0.0468424160, 0.0796641265, -0.0006889270, -0.1507324363, 0.0807157482, 0.0055288406, 0.0035086989, 0.0038772578, 0.0276793161, 0.0132745348, 0.0367155832, 0.0194087762, -0.0285011270, -0.0158262064, -0.1036757806, 0.0228646265, -0.0221079179, 0.0129473365, -0.0186724573, 0.0264403202, -0.0134741175, -0.0245014304, -0.0655067097, -0.0469648068, 0.0702462781, 0.0354403195, 0.0166486219, 0.0296367985, -0.0136770183, -0.0110952447, 0.0407622484, -0.1963795213, -0.0240342444, 0.0063828267, 0.0000000000, 0.0021452769, -0.0163393654, -0.0019981264, 0.0214076329, -0.0266004725, -0.0329592267, 0.0541018641, 0.0000334634, -0.1267178541, -0.0924473139, -0.0867199193, -0.0213999114, -0.0491957355, -0.0570990435, -0.0676041869, -0.0151564613, 0.0863555592, -0.0229706165, 0.0695721603, -0.0167953710, -0.0007273248, 0.0344045190, 0.0413850471, -0.0252582757, -0.1484935726, 0.0091035689, 0.0021001639, -0.0008424860, 0.0016037292, 0.0159187386, 0.0340887542, -0.0396809667, 0.0386216612, 0.0185859129, -0.0453816985, 0.0145842706, -0.0542516145, -0.0805461447, -0.0699715631, -0.0864196288, -0.0838132669, -0.0892316109, 0.0456110120, 0.0765282189, 0.0035513895, 0.0189066702, 0.0067586737, 0.0187376502, 0.0355067237, 0.0651226804, 0.0048858423, -0.0465127946, -0.0820207638, 0.0156183734, 0.0000907990, 0.0026560148, -0.0066915581, -0.0072093944, 0.0205533450, 0.0099964982, 0.0406037007, -0.0300399180, -0.0110636343, -0.0063390315, -0.0245245921, -0.0065195260, -0.0615813392, 0.0037914424, -0.0514425260, -0.0233159160, -0.0474070162, -0.0024942092, 0.0535084507, 0.0161627284, 0.0394219032, -0.0161192205, 0.0129373607, -0.0324107324, -0.0284694265, 0.0038530948, -0.0897391298, 0.0215148215, -0.0040530616, 0.0000000000, 0.0081506896, -0.0144449662, 0.0437883667, -0.0029713046, 0.0297947722, 0.0131598779, 0.0278577793, -0.0122297230, 0.0481871971, -0.0238103992, 0.0103543819, -0.0503562449, -0.0080434308, 0.0233033163, -0.0240173536, -0.0151045615, -0.0066880074, 0.0391031595, -0.0448851051, 0.0274592296, -0.0037214940, -0.0078817559, 0.0179125880, -0.0140335168, -0.0914131597, 0.0198090924, 0.0013032562, -0.0003164365, -0.0018514665, 0.0142491145, -0.0047271837, 0.0244442017, -0.0119298888, 0.0166675424, -0.0094296848, 0.0432404949, -0.0607757645, 0.0143820765, -0.0400561221, -0.0083222101, -0.0114860648, -0.0185293813, -0.0002804821, 0.0550412776, -0.0394213961, 0.0351984567, 0.0373538101, -0.0073149695, -0.0163687551, -0.0111031846, -0.0228431667, -0.0713720158, 0.0534050902, -0.0080074708, 0.0031099126, -0.0003164355, 0.0043833465, -0.0128642811, 0.0121510840, -0.0021248835, -0.0171832309, 0.0082666131, 0.0063571446, -0.0382504182, 0.0265891262, -0.0108629030, 0.0354492522, 0.0037535688, 0.0002104484, -0.0281415060, 0.0012197169, -0.0016904117, 0.0365334421, -0.0207087256, -0.0020647959, 0.0249631972, -0.0196864128, 0.0078604455, 0.0166941974, -0.0989326377, -0.0301582563, -0.0104199057, 0.0031099135, 0.0000000000, 0.0000000000, 0.0236928651, -0.0017543168, 0.0172915725, 0.0333199393, 0.0194010203, 0.0339191168, 0.0039705128, 0.0327156404, 0.0072546137, 0.0235073450, -0.0211399326, 0.0498190340, -0.0281458800, 0.0021143556, -0.0413586725, -0.0145988800, 0.0545100979, -0.0009977309, -0.0017383988, 0.0104923636, -0.0363897824, -0.0178257891, 0.0332214146, -0.0261948989, 0.0056733103, 0.0000000000, 0.0000000000, 0.0000000000, -0.0237246378, 0.0325420459, 0.0165990533, -0.0272557384, 0.0386685418, 0.0165427251, 0.0127672302, -0.0065370427, 0.0323600939, 0.0267533342, -0.0222431326, 0.0179372289, 0.0120700702, -0.0055791290, 0.0132053243, 0.0178126781, -0.0534933267, -0.0108885969, -0.0109343995, 0.0276357060, -0.0012121234, 0.0501178458, -0.0646132130, 0.0079618268, -0.0135055761, 0.0000000000, 0.0000000000, 0.0000000000, 0.0180817607, -0.0212318944, -0.0171291115, 0.0295798822, -0.0001251474, -0.0125918158, 0.0426183044, 0.0135043559, 0.0298417021, -0.0137598462, 0.0610659267, -0.0020001099, 0.0266038244, -0.0041108018, 0.0088836605, -0.0053183874, -0.0135989573, 0.0309445609, -0.0226122610, -0.0292348048, 0.0208008106, -0.2202062337, 0.0397047116, 0.0055199099, 0.0113504516, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0025541684, -0.0968714962, 0.0052723891, -0.0153007114, 0.0035600503, 0.0040851994, -0.0496523945, 0.0069568775, -0.0213783759, -0.0369901191, -0.0348468012, -0.0205321453, -0.0447852353, -0.0183868816, 0.0010864232, -0.0125945400, -0.0157047682, 0.0221869083, -0.0081033347, -0.0000576439, -0.0028027606, 0.0023748793, 0.0052243565, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0019404159, 0.0116857658, 0.0035090228, 0.0015125254, 0.0054605563, 0.0043024016, -0.0150953212, 0.0039421036, -0.0063031044, -0.0023376802, -0.0023927145, -0.0101167376, -0.0453526795, -0.0227753554, 0.0005974997, 0.0042880824, -0.0065858285, 0.0039118632, 0.0046024376, 0.0036229043, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0071746577, 0.0066593991, 0.0020157792, 0.0020157799, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0085816366, 0.0085140320, -0.0237727623, 0.0219886219, -0.0043014584, 0.0267709633, 0.0071392185, 0.0114527143, -0.0120830030, -0.0977437826, 0.0481393086, -0.0733097889, -0.0120283727, -0.0010500913, -0.0077556681, 0.0073027846, -0.0450738667, -0.0003237293, -0.0128360543, 0.0035554180, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0000449929, -0.0077923384, -0.0072482393, -0.0089166237, 0.0162537136, -0.0636355002, -0.0216504706, -0.0401864791, -0.1207629370, -0.0952215046, -0.0410578921, -0.0051637669, -0.0008465618, -0.0248815765, -0.0677988593, -0.0387077169, -0.0914236952, -0.0109817997, -0.0239226745, -0.0181974500, -0.0006097958, -0.0440978724, 0.0112770226, 0.0019936202, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0031700066, 0.0037545681, -0.0069982698, 0.0325547080, -0.0564959563, 0.0458268284, -0.0799752807, -0.0880292184, -0.0159373538, -0.0850477851, -0.0473482569, -0.2838145558, -0.0463259511, -0.1037233675, -0.0307645655, -0.0212124666, -0.1186992797, 0.0292322156, -0.0383918378, -0.0065506764, -0.0163909923, 0.0135664107, -0.0463547894, -0.0081798731, -0.0010542060, 0.0000000000, 0.0000000000, 0.0051716282, 0.0143919419, -0.0404239610, -0.0031619983, 0.0210423675, -0.0756066842, 0.0719151681, -0.0554450148, 0.0384977585, -0.1029652941, -0.0217225571, -0.0475729960, 0.0753968668, -0.1039143580, -0.0781586272, -0.0265738031, -0.0299233858, -0.0618406431, -0.0075165418, 0.0262012277, 0.0014550142, 0.0363880418, -0.0003591007, 0.0247697923, 0.0050674787, -0.0054716728, 0.0040335422, 0.0000000000, 0.0000000000, -0.0192690065, 0.0260986310, -0.0138691470, 0.0065238940, 0.0099911945, -0.0115108926, -0.0118932339, -0.0048635952, -0.0127460922, -0.0201554398, 0.0036693914, -0.0746089454, 0.0490652630, 0.0025751185, 0.0612676717, -0.0017533320, 0.0329245708, 0.0356955842, 0.0079632790, -0.0020215059, 0.0043583931, 0.0018602353, -0.0190731467, -0.0041189365, -0.0198921808, 0.0037939587, 0.0000000000, 0.0085911222, 0.0016253851, -0.0359595808, 0.0299407316, 0.0005519350, -0.0300363368, 0.0083739203, -0.0099442951, -0.0292516424, -0.0268203005, -0.0933426532, -0.0724044853, -0.0891703343, -0.1099003570, -0.1256126034, -0.0978830035, -0.0801299952, -0.1027334198, -0.0409296650, -0.0398536625, -0.0027140043, -0.0092916545, 0.0516056589, -0.0127006193, -0.0115675228, 0.0103235621, -0.0164907650, 0.0079326567, 0.0044419264, -0.0790713227, -0.0148495191, -0.0395332741, 0.0645718012, -0.0157327117, 0.0454851446, -0.0214944606, 0.0289136237, -0.0306080082, 0.0387860093, -0.0787818537, -0.0392657475, -0.0837368825, -0.0878135944, -0.0813978548, -0.0143026180, -0.0392793318, -0.0019453676, 0.0000553408, 0.0134035312, 0.0191963198, 0.0446412495, -0.0231816506, -0.0247053092, 0.0006551553, -0.0411781490, -0.0033934563, -0.0662446782, -0.0030063799, 0.0602374302, 0.0163247278, 0.0173323113, -0.0400169220, 0.0316730874, -0.0923683459, 0.0270602368, -0.0414579488, -0.0386459987, -0.0307545840, -0.0370035154, -0.0893962442, -0.0424073566, -0.0961288068, -0.0285419253, -0.0063699242, -0.0309726329, -0.0300560302, -0.0024197759, -0.0651275252, -0.0277820213, 0.0094878227, 0.0082438286, -0.0096919467, -0.0095242632, -0.0015453950, -0.0164067900, 0.0448637336, -0.0436955872, -0.0701101446, 0.0453298134, 0.0030010521, -0.0039371013, -0.0124882129, -0.0356878942, -0.0612208225, 0.0121766529, -0.0817332402, -0.0045435660, -0.1119981984, -0.0721078514, 0.0235359738, -0.0436790311, 0.0141028138, 0.0015791050, -0.0147475545, 0.0192919668, 0.0004807105, 0.0401376523, -0.0786447331, -0.0283713194, 0.0024463452, -0.0111103136, 0.0009828254, 0.0078987226, -0.0079157590, -0.0595410010, 0.0619389189, -0.0787161604, -0.0784225673, 0.0384804773, 0.0275313480, 0.0364448628, -0.0308099811, -0.0157920916, 0.0251174357, -0.0319698090, -0.0985080016, -0.0009414113, -0.0839286648, 0.0347766950, -0.0381130887, -0.0315806811, -0.0192854526, 0.0050032826, 0.0348038108, -0.0679291619, 0.0547550315, -0.0133838513, -0.0478779337, 0.0091116252, 0.0170832738, -0.0840375326, -0.0215258899, 0.0499137207, -0.0485555940, 0.0219209963, 0.0340936908, -0.0494058598, -0.0511349248, 0.0422849902, 0.0484317669, -0.0246334091, 0.0506872812, -0.1410986930, -0.1502673150, 0.0528202391, 0.0507820643, 0.0056805327, -0.0025442212, 0.0630729169, -0.0547289242, -0.0456758896, -0.0588126599, -0.0285678976, -0.1081729645, -0.0120203651, -0.0262807423, 0.0168878623, 0.0164199498, -0.0071338790, -0.0017088924, -0.0134475703, -0.0309365726, 0.0020258332, -0.0287659446, 0.0465232908, 0.0423413077, -0.0161540204, 0.0841127368, 0.0381456793, 0.1740366472, -0.0070818499, -0.0768025250, 0.0345280650, 0.0616113024, -0.0236027161, -0.0142865119, -0.0005506601, -0.0002927952, 0.0323968140, 0.0148158564, -0.0595755864, 0.0663644972, -0.0096144931, -0.0275167037, -0.0000456093, 0.0052919468, 0.0137435247, -0.0075538128, -0.0224707311, -0.0225083262, -0.0226933654, 0.0205353990, 0.0065830715, 0.0497026939, 0.0865805180, 0.0685413157, 0.1013917573, 0.0554205384, -0.0639287119, -0.0406342807, 0.0439941297, 0.0083340047, 0.0532573615, 0.0423124276, -0.0635622531, 0.0299888696, 0.0560266772, -0.0454482271, -0.0055290984, -0.0854239280, 0.0040822107, 0.0266197842, -0.0153769111, 0.0017697736, 0.0335735857, -0.1136555125, 0.0004300647, 0.0505425670, 0.0364591416, 0.0309868957, 0.0939896007, 0.0152494657, 0.0304864473, 0.0275591534, 0.0642265824, 0.0566606783, -0.0609377263, 0.0052064980, -0.0013115030, 0.0686048173, 0.0588699778, 0.0010081949, 0.0927925743, 0.0030744121, 0.0083795599, 0.0461508182, -0.0597786379, 0.0214736604, 0.0180931123, -0.0678004119, 0.0010233459, 0.0019940975, 0.0132574026, 0.0667972993, -0.0148971966, 0.0417447009, -0.0261521503, 0.0118797342, 0.0060319351, 0.0420798010, 0.0499354298, 0.0405290809, 0.0448544055, 0.0440841761, -0.0448983433, 0.0153503400, 0.0437838730, 0.0631896988, 0.0430308473, -0.0304477108, 0.0626600937, -0.0111305317, -0.0697395879, 0.0663873259, 0.0851995889, -0.1478719235, -0.0109472586, 0.0482727466, -0.0136049583, 0.0013726478, -0.0115126605, -0.1193902972, -0.0105265095, -0.0562174994, -0.0103505234, 0.0020202755, 0.0425043484, 0.0699756226, 0.0179898123, 0.0122911491, -0.0306360639, 0.0670150826, -0.0329361739, 0.0525365307, 0.0734747526, 0.0254021204, 0.0534312491, 0.0190264016, -0.0006515916, 0.0904422583, 0.0386474995, -0.0402868839, -0.0329370787, 0.0323973666, -0.0376950774, -0.0223310437, 0.0192828305, 0.0000000000, 0.0088204499, 0.1247327390, -0.0360599639, 0.0298116232, -0.0236534327, -0.0015620568, 0.0371823161, 0.0150806629, 0.0462282524, -0.0221641571, 0.0084183704, -0.0441040778, 0.0292510828, 0.0855600693, 0.0261409387, 0.0524923464, 0.0456694350, 0.0917492480, -0.0471476526, -0.0395842148, 0.0423403654, -0.0453103828, 0.0859429263, -0.1107149389, -0.0123457592, 0.0132348745, -0.0186274311, -0.0065964660, 0.0108424099, -0.1055104406, -0.0260968755, -0.0323943553, -0.0243787353, 0.0328512780, 0.0072085439, 0.0168908403, 0.0647439989, -0.0996946988, -0.0807469690, 0.0992988144, 0.1009271329, 0.0158751586, 0.0168583045, 0.0376996430, -0.0897994751, -0.0602957038, 0.0523934013, 0.0048671632, 0.0196085212, -0.0420155431, -0.0179574098, 0.0715795004, -0.0881219163, -0.0465734065, 0.0291344347, 0.0082469446, -0.0147580578, 0.0242735347, 0.0676261824, 0.0039700514, 0.0865349197, -0.0668062062, -0.0446633850, -0.0321300175, -0.0937041050, -0.0642659852, 0.0700163722, -0.1639234183, 0.0110094930, 0.0772538704, -0.0218735120, -0.0280024761, 0.0170602271, 0.0010892045, -0.0537352599, -0.0656347964, 0.1167194684, -0.1937251396, 0.0107692349, -0.0426902028, -0.0464971669, 0.0256927674, -0.0056459866, 0.0000000000, 0.0119603391, 0.0230514457, -0.0299039927, -0.0020312271, -0.2047204878, 0.0607594462, 0.0015518106, -0.1779051555, 0.0290906062, -0.1248143021, -0.0113373851, -0.0048904310, -0.0796655496, 0.0228202149, 0.0173884901, 0.0694066949, 0.0048933874, 0.0137917666, -0.0292903045, -0.0454818080, -0.0260627575, 0.0223396197, -0.0150504576, -0.0387507657, -0.0358470730, 0.0354645057, -0.0015492621, 0.0036411997, 0.0209581575, -0.0419755033, 0.0276643059, -0.0639847929, 0.2407343652, -0.1561363297, -0.1766911845, 0.1532587780, -0.0512589723, 0.0232775268, -0.0445775469, -0.0336904731, -0.0405608080, -0.0417406414, 0.0286672754, -0.0763659880, 0.0449833497, -0.0245758908, 0.0041452481, 0.1130319637, -0.0782054883, -0.0026823974, 0.0289656764, 0.0059976593, 0.0289608951, -0.0110151707, 0.0055435055, 0.0036412010, -0.0067040215, 0.0905974346, -0.0722314262, 0.0177996099, -0.0952098193, -0.0522575302, -0.0004022847, -0.0614476969, -0.0881254770, -0.0305465882, -0.0167694982, -0.0502263935, -0.0219421975, -0.0676903485, -0.0364575872, -0.0348699951, 0.0141061471, -0.0207926362, 0.0581233152, -0.0984296221, 0.0736057599, 0.0488657202, -0.0199489994, -0.0008925261, -0.0286804676, 0.0186763999, 0.0055435068, 0.0000000000, 0.0000000000, -0.0146202470, -0.0293613359, 0.0017997188, -0.0735154011, 0.0046571508, -0.0075165850, 0.0054850849, -0.0315828824, -0.0043321675, 0.0027369254, -0.0350602294, -0.0051293712, -0.0098311712, 0.0109274990, 0.0327412116, -0.0194852348, 0.0199015489, -0.0259387146, 0.0178642640, 0.0229339524, -0.0673251932, -0.0021909425, -0.0024714754, 0.0063137547, 0.0119991964, 0.0000000000, 0.0000000000, 0.0000000000, 0.0063696443, 0.0323755524, -0.1111260445, 0.0084629145, 0.0699669979, -0.0580717117, 0.0523502289, 0.0505371228, -0.0533850522, 0.0523083424, -0.0300433258, -0.0181423394, 0.0062854275, -0.0363547120, -0.0062789183, -0.0129822368, -0.0211375548, 0.0291900095, -0.0215831258, 0.0024776690, 0.0082609990, -0.0035422301, -0.0051234014, 0.0340188873, -0.0026669773, 0.0000000000, 0.0000000000, 0.0000000000, 0.0054697595, 0.0116688050, 0.0338069438, -0.0266781778, -0.0142404263, -0.0329347765, -0.0285002693, -0.0578934098, -0.0057686639, -0.0537334514, -0.0548564060, -0.0327192902, -0.0508265141, -0.0266662978, -0.0497382781, -0.0407384584, -0.0196140728, -0.0540127834, -0.0013887310, -0.0377802000, 0.0016166417, 0.0071374656, -0.0234106755, -0.0049357316, -0.0043676911, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0029895510, -0.0126721765, 0.0205695097, 0.0075042433, -0.0974958643, -0.0241580163, -0.0638674479, -0.0517642971, -0.0920855806, -0.0558754308, -0.0921665234, -0.0815155653, -0.0699846145, -0.0594839568, -0.0593465140, -0.0396551615, -0.0266279939, -0.1480432425, -0.0214303808, -0.0173806205, -0.0061260214, 0.0080494053, 0.0073707377, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0019144557, 0.0149349902, 0.0015600800, 0.0058665231, -0.0324816024, -0.0076335865, -0.0332413023, 0.0107017188, -0.0316502976, 0.0145449290, -0.1042757331, 0.0022459997, -0.0219508655, 0.0164431391, -0.0152196402, -0.0059733606, 0.0530806970, -0.0027389298, -0.0017131575, 0.0104467989, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0030696457, 0.0036935676, 0.0023128666, 0.0023128671, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0053592090, 0.0060989902, -0.0106529417, 0.0043900383, 0.0096627586, 0.0360631756, -0.0088602330, 0.0035454858, -0.0096136281, -0.0008593247, 0.0252334865, 0.0036948716, -0.0062799484, 0.0156322935, 0.0053308616, -0.0121399848, 0.0087981265, 0.0060488065, 0.0060571198, -0.0017048052, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0021477157, -0.0423401592, -0.0546681897, 0.0133678889, 0.0222012459, 0.0244947993, -0.0432782902, -0.0376986612, 0.0158670319, -0.1213474247, -0.0511387124, -0.1028834219, -0.1077342320, -0.0160331185, -0.0164572121, -0.0152270671, -0.0071893804, -0.0073972700, -0.0306397096, -0.0050702890, -0.0156022650, 0.0209744918, -0.0818899697, 0.0037853382, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0039506940, 0.0005184480, 0.0356903755, -0.0459114182, -0.0060167818, -0.0478066230, 0.0477697719, -0.0945368224, 0.0788421423, -0.1074724189, 0.0470881300, -0.0412981157, -0.0055506734, -0.0283748209, -0.0358552232, -0.0106749227, -0.0217518222, -0.0113975030, 0.0029774374, -0.0152975176, 0.0078255219, -0.0190000542, -0.0035440894, -0.0002172989, 0.0005847326, 0.0000000000, 0.0000000000, 0.0016294729, -0.0082786393, -0.0084596811, -0.0678179613, -0.0357984983, 0.0001369514, 0.0104422429, -0.0545475696, -0.0047923262, -0.0287450472, 0.0043343605, -0.0079274240, -0.0664669556, -0.0083889034, 0.0059802041, 0.0038618312, -0.0408638609, 0.0258595653, -0.0039500658, -0.0401150490, 0.0286519003, -0.0114127353, 0.0075847611, 0.0200681989, -0.0062705506, 0.0036571768, -0.0054036183, 0.0000000000, 0.0000000000, 0.0018883850, 0.0276312844, -0.0894848820, 0.0072202333, -0.0417977843, 0.0085595504, -0.0090596039, -0.0030070093, -0.0017013611, -0.0107200150, -0.0272907745, -0.0101099865, -0.0276940192, -0.0267625580, -0.0150358931, 0.0194061143, 0.0135216180, -0.0226427568, 0.0558117180, -0.0085494292, 0.0403878007, 0.0019919113, 0.0025242947, 0.0007199575, 0.0054061543, 0.0000996028, 0.0000000000, -0.0001216848, 0.0138362676, -0.0825600286, 0.0357213350, 0.0247089872, -0.0069683821, -0.0106704471, -0.0317832923, 0.0010431918, -0.0118483180, -0.0165892948, -0.0056584393, -0.0260215106, 0.0320202101, -0.0399026094, -0.0241330668, -0.0202515082, -0.0008673698, 0.0295024651, 0.0162279375, -0.0110353676, 0.0115238974, 0.0130497128, 0.0424648412, -0.0076617144, 0.0058673400, -0.0006701466, 0.0036905911, -0.0015182385, -0.0112141728, 0.0094360345, -0.0850407084, -0.0220752970, -0.0035540834, -0.0205348222, 0.0297489925, -0.0225014635, 0.0428277422, 0.0160326696, 0.0087761852, -0.0405839752, 0.0342602866, -0.0706266149, 0.0117259209, 0.0014626440, 0.0147384587, -0.0146083639, -0.0023774865, 0.0245750667, 0.0387357964, 0.0064741833, 0.0184407268, 0.0448210690, -0.0201132497, 0.0079015123, 0.0055950566, 0.0138634233, -0.0196389965, -0.0538740068, -0.0300499868, -0.0013199579, -0.0290846570, -0.0111153880, 0.0204487097, -0.0093679094, 0.0154519326, 0.0216789473, -0.0120904235, -0.0199454615, -0.0315773798, -0.0017882999, -0.0222886847, 0.0033436206, -0.0297807472, 0.0150218914, 0.0259232818, 0.0312336055, -0.0466016328, 0.0586084468, 0.0420878549, 0.0160442143, 0.0099906434, -0.0055354682, -0.0012944548, 0.0062992110, -0.0141650520, -0.1071914980, -0.0540058932, -0.0258850306, 0.0277531995, -0.0177461508, -0.0074553337, 0.0121478129, -0.0093672475, 0.0567806428, 0.0076468281, -0.0126680540, -0.0235248695, -0.0732881618, -0.0655254159, -0.0163899801, -0.0262580119, -0.0051385835, -0.0370884423, 0.0622006328, 0.0604576166, -0.0227634106, 0.0523839093, 0.0278974074, -0.0026965189, 0.0044475908, -0.0000768700, 0.0044692398, 0.0140518172, -0.1031916526, 0.0325280464, -0.0214688366, -0.0460982639, 0.0521387175, 0.0722567636, 0.0587493230, 0.0416949765, 0.0563219519, 0.0592361534, 0.0255491514, -0.0180327205, 0.0134041048, -0.0579420896, -0.0689802471, -0.0830691624, -0.0840431335, -0.0642413673, -0.0218472698, 0.0259582595, 0.0455566117, 0.0807844164, 0.0202976020, 0.0239786397, -0.0113729057, 0.0093489491, -0.0258948087, 0.0674377800, -0.1098220907, -0.0095416916, -0.0076050415, 0.0804179568, -0.0263517719, -0.0065878894, 0.0340559300, 0.0256363045, 0.0293187091, 0.0166492667, 0.0450217970, -0.0176291847, -0.0153482828, -0.0144750253, -0.0295253030, -0.0179559687, -0.0511375555, -0.1601568047, -0.1634613300, -0.1767827023, -0.0372247169, 0.1272706878, 0.0356743979, -0.0209882408, 0.0064476021, 0.0046425627, -0.0061013259, -0.0224188236, 0.0400917891, -0.0007420049, -0.0064014480, 0.0234408804, 0.0172660328, 0.0275131052, -0.0313754427, 0.0318477635, 0.0386555949, 0.0230623039, 0.0495710721, -0.0345684004, 0.0260339660, -0.0420036769, -0.0451304158, -0.0458369812, -0.0632913775, -0.0096329062, -0.0936847785, -0.1355960936, -0.1538971554, -0.3909688472, 0.0844728783, 0.0552744940, -0.0133703387, 0.0020770258, -0.0089077924, -0.0427743626, -0.0146836099, 0.0130260101, 0.0339534235, -0.0207259949, 0.0411399824, 0.0034972515, 0.0337735301, 0.0004857095, 0.0312122872, 0.0519437933, 0.0175617862, -0.0584848210, -0.0168574024, -0.0649126444, -0.0458336791, -0.0115070962, 0.0198090374, 0.0023132577, -0.0212864662, -0.0304177553, -0.1576999773, -0.0567564185, -0.0631970994, 0.0264881279, -0.0000795866, 0.0081351519, -0.0177507415, 0.0145265146, 0.0314566703, -0.0219223791, -0.0062636488, -0.0367137275, -0.0290411940, -0.0262862963, 0.0008959647, 0.0651993717, -0.0110949262, 0.0387205488, -0.0123421445, -0.0244477751, 0.0124724239, -0.0507479554, -0.0327664744, -0.0336433790, -0.0623415164, 0.0018935249, -0.0432562773, 0.0215422715, -0.0023561223, 0.0945603467, -0.1796376500, -0.0296285961, 0.0025822956, 0.0070657114, 0.0049042220, -0.0093940920, -0.0397985317, 0.0676086580, -0.0433689129, 0.0128827440, -0.0248811090, 0.0480450715, 0.0350215564, -0.0089805208, 0.0162781616, 0.0162449514, -0.0484464403, -0.0517110318, -0.0405005238, -0.0164430165, -0.0500246915, 0.0537561234, -0.0199301777, -0.0121835359, -0.0125267696, -0.0550155693, 0.0140827285, 0.0458876662, 0.0196737086, -0.0158134804, 0.0012539714, -0.0049903755, 0.0046181284, 0.0230474575, 0.0283855378, -0.0353058358, 0.0147000161, -0.0689382914, -0.1024831914, -0.0601057650, -0.0345120264, 0.0143450970, -0.0253428586, 0.0243389849, 0.0056725420, -0.0565695122, -0.0095366037, -0.0674017869, -0.0092431928, -0.0710801562, -0.0089908030, 0.0058805978, -0.0443592987, 0.0822093526, -0.0017248004, -0.0256641683, -0.0138311885, -0.0914485131, 0.0068063798, 0.0000000000, 0.0088532489, -0.0317026013, -0.0148895765, 0.0009115362, -0.0149323121, 0.1159223470, -0.0367856017, -0.0306050130, -0.0793170612, -0.0844716981, -0.0609765835, -0.0972969781, -0.0734864014, -0.0755904193, -0.0225801878, 0.0094987713, -0.0042027227, 0.0592074545, -0.0064268950, 0.0151341646, 0.0709174107, -0.0617152967, 0.0200636575, 0.0479409538, -0.0457459436, 0.0045312338, -0.0013460701, 0.0008389433, -0.0071342470, -0.0352803136, -0.0102801149, 0.0258616045, 0.0183763167, 0.0242127934, 0.0288850825, -0.0351991046, -0.0089808412, -0.0758045363, -0.0540113708, 0.0188733261, -0.0047288008, 0.0300760825, -0.0295039595, -0.0159728010, 0.0454173588, -0.0420576237, 0.0132691478, -0.0013425786, -0.0400899625, 0.0643428715, -0.0000369557, -0.0069387048, -0.0429216784, 0.0174744279, -0.0045903143, 0.0048964233, -0.0076144037, 0.0162383890, -0.0049821461, 0.0249105590, -0.0311565398, 0.0419313937, -0.0086249321, 0.0560383262, -0.0224343982, 0.0280463206, -0.0233344729, 0.0014732395, 0.0097367948, -0.0256506918, 0.0647925162, -0.0028824134, -0.0425657989, 0.0227111640, -0.0154240764, 0.0367398589, 0.0317305926, -0.0267804467, -0.0228347616, 0.0543866686, -0.0448778532, -0.0068844083, 0.0026747430, 0.0000000000, 0.0061059151, 0.0031128145, -0.0144955581, -0.0060094173, 0.0149118928, 0.0014882427, -0.0168526296, 0.0248921721, 0.0443485252, -0.0106198017, 0.0709546521, 0.0005233676, 0.0206991099, -0.0402870393, -0.0420605770, 0.0149682575, 0.0360322823, -0.0053051346, 0.0246970744, -0.0331705499, -0.0474809055, 0.0542284124, 0.0264242224, -0.0573243382, 0.0356759681, 0.0003669296, 0.0033929125, 0.0018343497, 0.0009927623, -0.0162209632, 0.0044622371, 0.0179406341, 0.0083184058, 0.0146215056, 0.0008499237, 0.0068282641, 0.0201586502, 0.0146125576, 0.0343859746, -0.0303068494, 0.0095205157, 0.0095696148, 0.0084747485, -0.0320000920, 0.0271753059, -0.0136569917, 0.0001851115, 0.0586580910, 0.0554183340, 0.0176942615, 0.0057870201, 0.0150488131, -0.0099915368, -0.0112947382, 0.0033869580, 0.0018343506, 0.0028726004, -0.0273673923, -0.0355722776, 0.0034124551, -0.0187388391, -0.0401049057, 0.0342848631, -0.0047076252, 0.0329253096, -0.0246566489, 0.0159041908, 0.0431052693, 0.0143473339, -0.0001406735, -0.0140701772, 0.0195374369, -0.0594825765, 0.0726438944, 0.0173667664, -0.0071946580, -0.0121517952, 0.0141364424, 0.0059445181, 0.0028188769, -0.0004790099, -0.0052970258, 0.0033869588, 0.0000000000, 0.0000000000, 0.0228385448, 0.0098243125, 0.0113361760, 0.0278145887, 0.0151511841, -0.0531894509, 0.0261834189, -0.0277488868, 0.0436847641, -0.0073814833, 0.0231024981, -0.0224001516, 0.0404237503, -0.0286924502, 0.0408582856, -0.0179891364, -0.0545092731, -0.0198251375, 0.0355048259, -0.0018418455, -0.0054729613, 0.0174437830, -0.0550292611, 0.0451007504, 0.0004830581, 0.0000000000, 0.0000000000, 0.0000000000, -0.0906537724, -0.0025894727, -0.0274255387, 0.0134961973, -0.0236851889, 0.0196052530, -0.0211807422, 0.0511651932, -0.0191946408, 0.0093350334, -0.0114395181, 0.0169305266, 0.0138877111, 0.0098794515, -0.0272193330, 0.0175273235, 0.0214408304, -0.0279305941, 0.0028600066, -0.0116855001, 0.0059634873, -0.0366408385, 0.0158504634, -0.1102083562, -0.0050278619, 0.0000000000, 0.0000000000, 0.0000000000, 0.0149400152, 0.0056590217, 0.0139465452, -0.0363150278, 0.0067484320, 0.0069803038, -0.0219082500, 0.0201718622, -0.0206133867, 0.0195666314, 0.0017053059, 0.0115552945, -0.0274771754, 0.0198951040, -0.0070937741, -0.0029164818, -0.0035313689, 0.0204675078, -0.0567260025, 0.0048579521, 0.0156543480, 0.0025513547, 0.0304617794, 0.0068009324, 0.0045224068, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0038400171, 0.0029046263, -0.0182682402, 0.0176556017, -0.0955291153, -0.0561398926, -0.0005429240, -0.0291826070, 0.0009804513, -0.0324118540, 0.0001143161, -0.0409295169, -0.0104568679, -0.0116084282, -0.0171356644, -0.0249360482, -0.0248527458, 0.0420432083, -0.0266961572, -0.0068910954, -0.0007574169, -0.0067387994, 0.0051496055, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0029395833, 0.0249647871, -0.0055989352, 0.0199875326, 0.0005040187, -0.0188928082, 0.0086721664, -0.0013957816, 0.0011142985, -0.0165817187, -0.0294355837, -0.0099164022, -0.0226368550, -0.0043423800, -0.0174754774, 0.0120440397, -0.0027151621, -0.0174788382, 0.0041836252, 0.0054143988, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0095240804, 0.0070630264, -0.0003766959, -0.0003766953, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0069758679, 0.0081419726, 0.0246785442, -0.0197573595, 0.0132456646, 0.0039458622, -0.0026665983, 0.0021209860, 0.0039271393, 0.0090872306, -0.0105513214, 0.0086453538, -0.0095988757, 0.0060900321, 0.0180929286, 0.0113374545, 0.0113789092, 0.0151144166, 0.0136175113, 0.0060575305, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0080184014, -0.0086815092, 0.0196436322, 0.0085031125, 0.0006086316, -0.0039425073, 0.0037583656, -0.0039338724, 0.0296269722, -0.0078683243, 0.0255487710, 0.0155907594, 0.0111218994, -0.0005400598, 0.0299265002, 0.0047627048, 0.0229217470, 0.0078473158, 0.0117991976, 0.0350602303, -0.0001149005, 0.0352698987, -0.0135490286, -0.0028056873, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0034627434, 0.0040281060, 0.0107151636, -0.0214852346, 0.0178236331, 0.0195004351, -0.0032247196, -0.0017862591, -0.0052993978, 0.0082868465, -0.0190514197, 0.0041252679, -0.0098731713, 0.0254464208, -0.0141632870, 0.0114820891, 0.0130047179, 0.0309416100, -0.0028268592, 0.0060719529, 0.0053785731, -0.0024683120, 0.0286762124, -0.0113316041, 0.0127942935, 0.0000000000, 0.0000000000, 0.0080556323, -0.0136744939, -0.0077649099, 0.0000243978, 0.0196627959, -0.0521168783, 0.0186820449, 0.0050977320, -0.0367385602, -0.0050629873, -0.0244182194, -0.0583168982, -0.0197877382, -0.0181753731, -0.0557105031, -0.0206787661, 0.0239541677, -0.0572658345, -0.0054895337, -0.0039354111, -0.0147899686, 0.0026926781, 0.0004470395, -0.0407741893, 0.0067653593, -0.1237485121, 0.0529520687, 0.0000000000, 0.0000000000, 0.0044459579, -0.0302843140, -0.0009692819, -0.0306296903, 0.0419026578, -0.1036786837, 0.0879785814, -0.0974267098, 0.0593308549, -0.1055112026, 0.0335252572, -0.1132683046, -0.0200740307, -0.0411070277, -0.0656247683, -0.0423375664, 0.0526672322, 0.0121052689, 0.0461789120, 0.0305607544, -0.0409720616, 0.0319127785, 0.0346580682, -0.0221578677, 0.0023716243, -0.0163493275, 0.0000000000, -0.0013724663, -0.0025953358, 0.0137114143, -0.0105081010, -0.0043022833, -0.0206875894, 0.0294388872, -0.0467492903, 0.0185477781, -0.0782940655, -0.0282397062, -0.0129861101, -0.0073404654, -0.0788579541, -0.0034130170, -0.0709215592, -0.0080431616, -0.0986826659, -0.0635624800, -0.0824185414, -0.0417334317, 0.0179081133, 0.0019370849, -0.0868705692, 0.0348076914, -0.0296445655, 0.0389131468, 0.0024915902, 0.0207311348, -0.0075353991, -0.0703903858, 0.0221421360, 0.0877921777, -0.1096183084, 0.0665632330, -0.0473176427, 0.0311423286, -0.0018660738, -0.0390666138, -0.0510851577, -0.0597386878, 0.0033403118, -0.0339720061, -0.0644498317, -0.0358964654, -0.0291347248, -0.0230043807, -0.0459675573, -0.0166549246, -0.0949449032, -0.0645937884, 0.1194310059, -0.0071523449, -0.0172949795, -0.0236043317, 0.0044382834, 0.0046411045, 0.0148631339, -0.0566945499, -0.0797232470, -0.0400786691, -0.0130052779, -0.0238805052, -0.0769344498, -0.0681543257, -0.0249188364, 0.0017844628, -0.0466925336, -0.0183020274, -0.0606239881, -0.0735970987, -0.0282207836, -0.0510802956, -0.1383181875, -0.1191984939, -0.1603758019, -0.0558032613, -0.0149839629, -0.1366709848, -0.1595882754, -0.0298784831, 0.0489781125, -0.0012514951, 0.0009715711, 0.0059962779, 0.0561444358, -0.0285086421, 0.1173021169, -0.0260109197, -0.0552989814, 0.0678424985, 0.0677301289, 0.0011217192, -0.0308121956, -0.0268055679, 0.0077657049, -0.0866712662, -0.0042316283, -0.0928301061, -0.1017455033, -0.1328538775, -0.1055138206, -0.0639417870, -0.0125078597, -0.0107821149, -0.0701122346, 0.0397844753, -0.0396536346, -0.0644948457, -0.0581041098, 0.0190602131, 0.0020609911, 0.0152295358, -0.0490900286, -0.0506517933, -0.0420468886, -0.0644608551, -0.0168042211, -0.0569116491, -0.0152232936, -0.0444221333, -0.0393192521, -0.0054921330, -0.0509566280, 0.0122158773, -0.0485495341, -0.0169774481, -0.0453213326, -0.1036385702, 0.0065326215, -0.1465101352, -0.0190670460, -0.0597715946, -0.0993016145, -0.0002140534, 0.0352863755, -0.0309655305, 0.0146873731, -0.0829496829, 0.0081608019, -0.0028487237, 0.0799377273, 0.0170337529, 0.0016329397, 0.1659485540, -0.0204120371, 0.0270347067, -0.0327423631, 0.0080604088, 0.0296131230, -0.0616083129, 0.0195196361, 0.0532789957, 0.0086810502, -0.1147120854, 0.0415989136, -0.0758276377, -0.0499102335, 0.0560303439, -0.0492611767, 0.0263452576, -0.0161381428, 0.0559425538, -0.0174143255, -0.0142901033, -0.0241812093, 0.0627593655, 0.0061901281, 0.0038339644, -0.0540947434, -0.0524724051, -0.0340782156, -0.1364478987, 0.0056346938, -0.0059997202, 0.0686174043, 0.0473466240, -0.0622787554, 0.1017621709, 0.0061161641, 0.0060739905, -0.0008361957, -0.0629916393, -0.0128587331, 0.0454524233, -0.0155382427, -0.0595577501, -0.0469865460, 0.0284112450, 0.0592656565, -0.0083182487, 0.0079766735, 0.0117324834, -0.0238744994, -0.0482057844, 0.0027034404, -0.0014571121, -0.0147450333, -0.0655614916, 0.0192998631, 0.0844465721, 0.0951272011, -0.0228268657, -0.0344551559, 0.0001651582, 0.0880612516, -0.0409944066, 0.0433316588, 0.0045839945, -0.0336983079, 0.0953477794, -0.0462644958, -0.0218157082, -0.0198095595, 0.0080294935, 0.0019171553, 0.0626049610, -0.0685952681, 0.1058041318, 0.0086130808, -0.0255942024, -0.0158800545, 0.0157547971, -0.0000169789, -0.0090462042, 0.0690963759, 0.0881818119, -0.0641346782, -0.0083298560, 0.0102290780, 0.0421522120, 0.0517951072, 0.0431172979, 0.0168625338, 0.0898395419, 0.0221842471, -0.0135576750, -0.0202319723, -0.0297441547, 0.0582528642, -0.0557125207, -0.0023880506, 0.0030614854, -0.0099952044, -0.0454863870, 0.1126091848, -0.0168910809, -0.0051356526, 0.0215998888, -0.0156866913, -0.0258979577, 0.0014250612, -0.0010777350, -0.0338073414, -0.0199426654, -0.1208380809, -0.0322370195, -0.0091255262, 0.0381281949, 0.0157807256, 0.0159123617, 0.0334841098, 0.0298362740, 0.0195993429, -0.0038720135, 0.0824361910, 0.0207860223, -0.0283243258, 0.0300144387, -0.0037104170, -0.0589621650, 0.0910008704, 0.0712334988, -0.0747411921, 0.0339756623, -0.0359363061, 0.0175438281, -0.0143816812, 0.0066715066, 0.0077558032, 0.0057706126, 0.0371569654, -0.0621820406, 0.1267720064, -0.0313501118, 0.0421471699, 0.0454587801, -0.0741958377, 0.0936891983, -0.0277316432, 0.1186695407, -0.0515282529, -0.0297824639, -0.0084237403, 0.0254357573, 0.0061376392, -0.0571230811, -0.0138280763, 0.0293883687, -0.0608877915, -0.0113906622, 0.0376673047, -0.0502011400, 0.0177720987, -0.0182224052, -0.0172109816, -0.0012329201, 0.0000000000, 0.0083877442, -0.0015704865, 0.0533635554, -0.1976879366, 0.0877502482, -0.0662037461, -0.0497079383, 0.1352902316, -0.0282622213, 0.0998359083, 0.0815641717, -0.0097091112, 0.0173278299, 0.0779622821, -0.0680220224, -0.0468277760, 0.0298474012, 0.0203903237, 0.0398155476, 0.0669209114, -0.0255145369, 0.0269738980, 0.0189904411, 0.0138878454, -0.0169711692, -0.0223837632, -0.0117962865, 0.0078856943, 0.0027809033, -0.0126954124, -0.0053648287, 0.0298730477, -0.0942091833, -0.0283383545, 0.1209112848, -0.0582996547, 0.0627426779, 0.0277792534, 0.0093436033, 0.1096296712, 0.0193738345, -0.0293908892, 0.0392187195, 0.0497407381, -0.0020462242, 0.0555772147, -0.0389745600, -0.0323776176, 0.0253874582, -0.0404705288, 0.0193328964, -0.0618244337, 0.0216635414, 0.0166649508, 0.0016972153, 0.0008421226, 0.0077086693, 0.0815286030, -0.2800549241, -0.0080353894, 0.0215367665, 0.0144942464, -0.0447503438, 0.0268822986, 0.0305170209, 0.0499290809, 0.0987069231, 0.1013542189, 0.0070740774, 0.0360326723, 0.0445046907, 0.0158942282, 0.0446795372, -0.0007949016, 0.0499311971, 0.0234089949, 0.0049976659, 0.0249693172, -0.0099318256, -0.0048241903, -0.0219697021, -0.0412255392, 0.0025398737, 0.0000000000, -0.0036909981, 0.0248450306, -0.0277727570, 0.0559841033, -0.0033063155, -0.0559826247, 0.0207989602, -0.0286018603, 0.0219151127, 0.1008732710, -0.0274982631, 0.0899395638, -0.0235070408, 0.1260811509, 0.0013772064, 0.0655378549, -0.0476491888, 0.0383068317, -0.0155233446, -0.0239013933, -0.0001865586, -0.0423769931, 0.0142451262, -0.0442324195, 0.0063690444, 0.0134438514, 0.0053188860, 0.0028026893, 0.0181795092, -0.0131495038, -0.0160518944, -0.0445239701, -0.0739931942, -0.0817645369, 0.0048742811, 0.0362101379, -0.0029608106, -0.0098152845, 0.0777386238, 0.1014519719, 0.0814650947, 0.1009368691, 0.0436966512, 0.0124962248, 0.0762769733, 0.0250110788, -0.0012664356, 0.0606594371, -0.0938905369, 0.0408279963, -0.0406420647, 0.0482857422, -0.0321985437, 0.0087957270, 0.0035861379, 0.0028026906, 0.0054147683, -0.0233736855, -0.0227056716, 0.0155690209, 0.0076008052, -0.2004866120, 0.1188831575, -0.0335039818, 0.0198493927, 0.0651817367, -0.0177796883, 0.0197264258, 0.0183885464, -0.0250366335, 0.0913169482, -0.0032445406, 0.0564278216, -0.0360982834, -0.0111047259, 0.0218284552, 0.0550232694, -0.0104813028, -0.0102605856, -0.1325283186, -0.0271830065, 0.0257584602, 0.0035861391, 0.0000000000, 0.0000000000, 0.0252041252, 0.0182914978, 0.0428340519, 0.0591478422, -0.0154540349, -0.0919248958, -0.0395095940, 0.0241313737, -0.0808749578, -0.0593884563, -0.0234999692, 0.0079734129, -0.0224948709, -0.0234226095, 0.0175135050, -0.1005280995, 0.0875905533, 0.0136962539, -0.0027765899, -0.0338592058, 0.0091014419, -0.0773045110, 0.0261333855, -0.0608007422, -0.0514359065, 0.0000000000, 0.0000000000, 0.0000000000, -0.0197396249, 0.0258717988, 0.0232340753, 0.0061669569, 0.0013533787, -0.0699437976, -0.0984211218, -0.0385781822, -0.0936983474, -0.1204199271, -0.0315390979, -0.0879871088, -0.0325051044, -0.0990572445, 0.0401443998, 0.0028612620, 0.0012257612, -0.1054571533, -0.0080967723, -0.0658485858, 0.0001909345, 0.0261422008, -0.0032703054, 0.0116083090, -0.0002787299, 0.0000000000, 0.0000000000, 0.0000000000, 0.0102059792, 0.0034171815, 0.0059257115, 0.0322712671, 0.0472386022, 0.0512633382, 0.0181083297, -0.0102197255, 0.0481828489, 0.0392262606, 0.0047328491, -0.0854262900, -0.0805156927, -0.0397865650, -0.0909511385, -0.0201666679, -0.0156332336, 0.0197566152, 0.0084762584, 0.0516478447, 0.0255178809, 0.0094030035, 0.0042276289, 0.0077946417, -0.0015196435, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0052713073, 0.0209865352, -0.0093427070, 0.0235789628, 0.0584906086, 0.0074491042, 0.0532295717, -0.0196421783, 0.0026527076, 0.0999079697, 0.0992438865, 0.0674025115, -0.0422933574, 0.0598427589, 0.0110656577, 0.0126348939, 0.0116924887, 0.0074391538, -0.0050202657, -0.0179894326, 0.0056896499, 0.0011824989, 0.0081214312, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0022795566, 0.0079711641, 0.0106745637, 0.0051228413, 0.0265747956, 0.0187739979, -0.0229522623, 0.0356434994, 0.0331634115, 0.0065200253, 0.0196171329, 0.0247640622, 0.0520900528, -0.0194984303, 0.0254384168, -0.0139877260, 0.0232883728, 0.0031439342, 0.0040575690, 0.0088914723, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0091464622, 0.0070137040, 0.0000345561, 0.0000345567, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0080652822, -0.0029047035, 0.0149956357, -0.0091031172, -0.0141468237, -0.0078158192, 0.0102861123, 0.0349211149, -0.0453279932, -0.0200940360, 0.0328370086, -0.0167752301, 0.0055785482, 0.0138388394, 0.0146713390, -0.0059019567, 0.0104664768, -0.0091075078, 0.0127989492, -0.0045657089, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0115787244, -0.0073757403, 0.0131378822, 0.0123072610, -0.0101305577, 0.0092736913, 0.0356343025, 0.0488337287, -0.0157506937, -0.0317079660, 0.0778878919, 0.0036303849, 0.0379376756, 0.0372190803, -0.0096547110, 0.0200839776, 0.0414617423, 0.0165640044, 0.0477328772, 0.0101093698, -0.0015568316, 0.0056073209, 0.0093307988, 0.0192302624, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0100381191, -0.0129687404, 0.0306524584, -0.0311187338, 0.0570472000, -0.0290407857, 0.0241022301, -0.0508308438, 0.0516640034, -0.0607838159, -0.0817018410, -0.1501481933, 0.0156790602, -0.1051906146, 0.0140703955, -0.0013387922, -0.0471375357, 0.0702095218, -0.0070698725, -0.0419511694, 0.0654658783, -0.0293428726, 0.0067236614, 0.0135239104, 0.0122111404, 0.0000000000, 0.0000000000, 0.0094107409, 0.0051208556, 0.0285356693, 0.0361746826, -0.1040504380, -0.0203777687, -0.0693268941, -0.0444975392, -0.0641475627, -0.1976886166, -0.0637304468, 0.0122927320, 0.0159675718, -0.0692175132, -0.0766369942, -0.1105509098, -0.0620856249, -0.1242150573, -0.0331493427, -0.0061990586, 0.0677153395, -0.0373897854, 0.0233328973, 0.0105010132, 0.0003254517, 0.0143600226, 0.0015857940, 0.0000000000, 0.0000000000, 0.0064986639, -0.0337943231, -0.1140931460, 0.1315169883, -0.0341292799, -0.0149619061, 0.0342459269, 0.0475541519, -0.0222330773, -0.0463158587, -0.0111563160, -0.0254050027, 0.0044054088, -0.0429043964, -0.0017113800, -0.1719178139, -0.0263030802, -0.1946757788, -0.1097065901, -0.0523301250, -0.1074625772, -0.0976959725, 0.0785889068, -0.0083852132, -0.0095500861, 0.0103823737, 0.0000000000, 0.0014177190, -0.0019326713, 0.0287014554, 0.0038433590, -0.0220863188, 0.0399582410, -0.0343985602, 0.0743275600, -0.0316135576, 0.0390478485, 0.0493557348, 0.0451972187, -0.0445087709, 0.0265186340, -0.0427308301, 0.0165418302, 0.0221637793, -0.0151595786, 0.0306925088, -0.0560905868, 0.0229246793, 0.0102970701, -0.1167928486, 0.0019439290, -0.0391159609, -0.0874688099, 0.0183297513, 0.0085733937, 0.0073825219, -0.0116794843, 0.0084676151, -0.0031797190, -0.0046247596, 0.0269314281, 0.0344579440, -0.0479064981, 0.0655423077, 0.0597017011, 0.0069683923, 0.0489504365, 0.0350092173, -0.0423879620, -0.0160904782, -0.0035079335, 0.0343200951, 0.0414160424, -0.0209654973, 0.0470711324, 0.0715695017, -0.0112100504, 0.0207251511, 0.0147790584, -0.0240151082, 0.0017500893, -0.0098964212, -0.0133141469, 0.0193070121, 0.0207755884, 0.0136514458, -0.0063642932, 0.0017071493, -0.0116886473, 0.1131685777, 0.0477432027, -0.0075971761, 0.0329877483, 0.0450591185, 0.0633312391, -0.0083027138, 0.0057071505, 0.0575742682, 0.0153261596, 0.0100726569, 0.0393844638, 0.0268079436, 0.0403670940, -0.0465941560, 0.0110302666, 0.0436191269, -0.0484749673, -0.0331013944, -0.0185345148, 0.0104419034, 0.0165096870, 0.0113606141, -0.0139046726, 0.0061581481, 0.0387248080, 0.0146522994, 0.0016634031, -0.0362006450, 0.0350561351, 0.0448118522, -0.0176724608, -0.0077796478, 0.1411883749, -0.0133829747, 0.0953924443, 0.0082100758, 0.0180135560, 0.0879206477, 0.0426715916, 0.0119828437, -0.0260321517, 0.0530910601, -0.0185107683, 0.0199617769, 0.0222675355, -0.0339533369, -0.1800485476, 0.0557033461, 0.0156242975, -0.0122039584, 0.0365127559, -0.0205746135, 0.0275282554, 0.0199851409, 0.0148360801, 0.0439565497, -0.0261663636, 0.0347639076, -0.0022311993, -0.0256783667, -0.0610965093, 0.1412532763, -0.0103737168, 0.0688228021, 0.1048562668, 0.0100544633, 0.0159302336, 0.0483882969, 0.0677375974, 0.0043566711, 0.1021865474, -0.0223771288, -0.0396411617, 0.0326379648, -0.0066004256, 0.0202290317, 0.0087267667, -0.0202238232, 0.0372046293, -0.0040974461, -0.0044453111, 0.0194512604, -0.0048069765, 0.0115466915, -0.0880191991, 0.0070255391, 0.0546373616, 0.0562904804, -0.0511887671, 0.0750284761, 0.0506097396, 0.0518389040, 0.0053187141, 0.0345806686, 0.0475492126, 0.0078034339, 0.0433333892, -0.0207481118, 0.0315426801, -0.0431180423, 0.0479306725, -0.0589310359, -0.0380947850, -0.0992519677, 0.0052725750, 0.1013212681, 0.0098945480, 0.0208276500, 0.0242784569, 0.0069609966, -0.0721732077, 0.0524752785, 0.0911885926, -0.0247107212, -0.0979170361, -0.0038678353, -0.0609309575, -0.1503436897, -0.0765040482, 0.0625158060, 0.0620439989, 0.0389415946, 0.0453758580, 0.0026739559, -0.0193505153, -0.0757698657, 0.0487942139, -0.0574257152, -0.1068878750, 0.0086456833, -0.0257937474, 0.1027116656, 0.0004203824, -0.0559100474, 0.0043150425, 0.0186868073, -0.0164856050, -0.0036995764, 0.0614834893, -0.0546727226, -0.0883184140, -0.0376665302, 0.0087410610, -0.0025280862, -0.1212383015, -0.1197746294, -0.1239835513, -0.0128583933, -0.0070556176, 0.0026009024, -0.0693102332, -0.1091790270, 0.0096546560, 0.0537792485, 0.0102643357, -0.0743726471, 0.0738625122, -0.0202538977, 0.0380242308, -0.1599491282, 0.0015187262, 0.0279783677, 0.0347406750, -0.0211575016, 0.0203948876, 0.0301195071, -0.0886190243, 0.0258034988, 0.0415875424, -0.0444622848, -0.0355560470, -0.0702855774, -0.1302762801, -0.0377041460, -0.0607486442, -0.0325019756, 0.0126903269, -0.0213049520, 0.0377431467, 0.1691787657, 0.0204887570, -0.0336451561, 0.0185232416, 0.0653898979, -0.0346606009, 0.0048256570, -0.1434388239, 0.0191186593, 0.0034355740, -0.0031472946, -0.0249716785, 0.0492859542, -0.0297603999, -0.0168597188, -0.0007677249, -0.0499616714, -0.0800205848, -0.0450869626, -0.0322533646, -0.0594142830, -0.0446311559, -0.0874920933, -0.0477882835, 0.0211487611, -0.0730362387, 0.0986319503, 0.0374416968, -0.0484993012, 0.0556157152, 0.0505296933, 0.0765161814, 0.0159674159, -0.0199976729, -0.0216721980, -0.0131321556, 0.0100475871, 0.0104539786, -0.0017482345, -0.0269744991, -0.0522263704, 0.0176958150, -0.0539317245, -0.0081940993, 0.0822864965, -0.0572121340, 0.0335287499, -0.0403825573, -0.1647011896, 0.0289237115, -0.0113312379, -0.0914161352, 0.0444605065, 0.0072868575, 0.0180193238, -0.0038298141, 0.0505922005, 0.0321810278, 0.0079641982, -0.0128821280, -0.0143416836, 0.0026999350, -0.0831992846, -0.0600462272, -0.0075564505, 0.0000000000, 0.0059403013, -0.0217024254, 0.0330960846, -0.0004130184, 0.0223964079, 0.0161136123, -0.0708134225, -0.0500527301, 0.0213334577, -0.1722073075, 0.0456921376, -0.0168465420, -0.0371245136, 0.0038005291, 0.0038270388, -0.0286283813, 0.0194758791, 0.0743755684, -0.0560729618, 0.0846194593, -0.0742659895, -0.0099065977, -0.0238514867, 0.0385334167, 0.0081341389, 0.0203358542, -0.0129184539, 0.0030473132, -0.0000227027, 0.0086754679, -0.0314346430, -0.0258021736, 0.1019390125, -0.1705676228, 0.0355641682, -0.0013623392, -0.0740351983, -0.0292592025, 0.0042219877, -0.0555461705, 0.0701985727, -0.0322776384, -0.0405154017, -0.0149783771, -0.0695024108, 0.0368479517, -0.0267266814, -0.1024224572, 0.0445108599, 0.0162798229, -0.0188283051, -0.1036664757, -0.0439820398, 0.0298746936, 0.0153601552, -0.0043766110, 0.0228908894, 0.0489030129, 0.0322860705, 0.1661675270, -0.1580985815, -0.0047112667, -0.1488623883, 0.0177137218, -0.0270745764, -0.1366910103, 0.0254381552, 0.0372084726, -0.0281567045, 0.0372147672, -0.0487290918, -0.0501461828, -0.0002286457, -0.1048756355, 0.0006501590, 0.0622002300, -0.0686046469, -0.0007809592, -0.2169153437, 0.1198107868, -0.0564477463, 0.0100582362, -0.0156614812, 0.0000000000, 0.0012588766, -0.0008100892, -0.1007177407, -0.3328224648, -0.0445910628, -0.0513153451, -0.2179198140, 0.0443952291, -0.1437204529, 0.0165149706, -0.0162964764, 0.0367677421, -0.0558276892, 0.0112774911, 0.0038017644, -0.0970426327, -0.0977698763, -0.1522418161, -0.0326009366, -0.2070292646, 0.0381518702, -0.1320247639, -0.0067696813, 0.0488776432, -0.0319359800, 0.0142177923, 0.0062310489, 0.0019508619, 0.0168874592, -0.0115147401, -0.0290475784, 0.1149747367, 0.1120925276, -0.1421192438, -0.0613708167, -0.0008116626, -0.0610503370, -0.0009100874, -0.0014276072, -0.0017913373, -0.0049303803, -0.0485000951, -0.0437715653, -0.0039117172, -0.0442659581, 0.0265419396, -0.1506856386, -0.0364452845, -0.0937532265, -0.0009060304, -0.0939136087, -0.0439458513, 0.0824729536, -0.0078497126, 0.0069905599, 0.0019508634, -0.0016328090, 0.0207169455, -0.0844656127, -0.1349426981, 0.0607494595, -0.0816877956, 0.0388370901, 0.0567008556, -0.0080033374, -0.0086798806, -0.0926630388, 0.0388173128, -0.0408292010, -0.0030023900, 0.0419206891, 0.0045220083, -0.0432447829, -0.0468522229, 0.0974205851, -0.0399957009, 0.0495792706, -0.0321298245, -0.0613908630, -0.0271590878, 0.0057706879, -0.0395418952, 0.0069905614, 0.0000000000, 0.0000000000, 0.0051051099, -0.0935569659, 0.0597227732, -0.0355135715, -0.0171664670, 0.0664356706, -0.1453652683, -0.0190722902, -0.0353712876, -0.0356863680, -0.0440231288, -0.0318043534, -0.0376925378, -0.0820001360, -0.0138725188, -0.0260099218, -0.0340947245, -0.1005820172, -0.0114372565, -0.1415801981, 0.0033729080, -0.0834478300, 0.0049463929, 0.0224052069, 0.0055913838, 0.0000000000, 0.0000000000, 0.0000000000, 0.0040541645, -0.0092609113, -0.0177127248, 0.0426405873, 0.0298623978, 0.0047500376, 0.0278459944, -0.0279583958, -0.0063358269, -0.0265453866, -0.0639296287, 0.0196035739, -0.0125556458, 0.0273135650, -0.0203633505, 0.0318059144, -0.0561975330, 0.0228847461, 0.0183788909, -0.0277088195, 0.0140134361, -0.0059446579, -0.0087927082, -0.0107101358, 0.0117980220, 0.0000000000, 0.0000000000, 0.0000000000, -0.0187871381, 0.0145527346, 0.0105029784, -0.0126144925, -0.0110229842, -0.0046599893, 0.0004748813, 0.0243007165, -0.0153985899, 0.0371431370, 0.0346777801, -0.0041865921, 0.0136452148, 0.0034172380, -0.0076549927, -0.0028434902, 0.0072300886, -0.0271742047, -0.0027375683, 0.0172054450, -0.0056909927, -0.0481730594, 0.0106887868, -0.0008693476, 0.0102546764, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0021463989, -0.0029907411, -0.0127578589, -0.0004722612, 0.0005187183, 0.0006146886, 0.0010842531, 0.0048467088, -0.0040293927, -0.0166657846, 0.0101465057, -0.0094412447, 0.0151250348, 0.0056827450, 0.0005529300, 0.0104861150, 0.0136119335, -0.0224175558, 0.0029851233, 0.0136402072, -0.0000366001, -0.0578864419, 0.0049272475, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0128603607, 0.0143133786, -0.0029104615, -0.0053765539, 0.0077564955, -0.0137307572, 0.0166544001, -0.0055740659, -0.0060185489, 0.0331595028, -0.0056842754, 0.0196885289, -0.0151849489, 0.0434863769, -0.0072215627, 0.0174462789, -0.0109451599, 0.0157805374, -0.0101409740, 0.0210297911, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0019273557, 0.0022437977, 0.0013228396, 0.0013228400, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0019720971, 0.0025383226, 0.0089075702, 0.0098202830, -0.0032691828, 0.0017887097, 0.0071320132, 0.0095281497, 0.0064762466, 0.0012553988, 0.0075216583, -0.0016238334, 0.0059614486, 0.0037295231, -0.0069369498, 0.0031093507, -0.0116427276, 0.0058236624, 0.0000411808, 0.0032793978, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0036816359, 0.0058350044, 0.0040192981, 0.0056777665, -0.0054013904, 0.0098198215, -0.0061203071, 0.0023515158, -0.0051527799, -0.0060722471, -0.0073195719, -0.0342957941, -0.0412260600, -0.0031241761, -0.0527625608, -0.0808732038, 0.0614970211, -0.1009001588, 0.0006924390, -0.0049430182, -0.0107433713, -0.0038530794, -0.0102357691, 0.0021742197, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0656048134, -0.0054027518, 0.0018788290, 0.0044775081, -0.0213913594, -0.0740954058, 0.0054922042, 0.0022171614, -0.0112030264, -0.0088566229, -0.0563611434, 0.0322999716, -0.0934441889, 0.0145046959, -0.0395927652, -0.0388900437, 0.0052778194, -0.0296280441, -0.0213873140, 0.0219365332, -0.0157597913, 0.0384660330, -0.0153040728, -0.0038652838, -0.0016589766, 0.0000000000, 0.0000000000, 0.0043352936, 0.0213991460, -0.0308929029, -0.0166122510, 0.0124888235, -0.0319750344, -0.0055705024, 0.0101862530, 0.0150679137, 0.0151365352, -0.0083131903, 0.0167818225, 0.0006936262, 0.0200821686, -0.0032926777, -0.0000062644, -0.0108898686, -0.0144957256, -0.0052440416, 0.0025227105, -0.0140133849, 0.0036116129, -0.0147208859, 0.0135531967, 0.0067698254, -0.0018380588, 0.0064889015, 0.0000000000, 0.0000000000, 0.0015549164, -0.0101878165, -0.0601221911, -0.0346365681, 0.0116181899, 0.0211877002, -0.0482188082, 0.0089692075, -0.0491577262, 0.0220372128, 0.0376425820, -0.0173551885, 0.0452886867, 0.0092836116, 0.0170846763, 0.0513771110, -0.0130303461, 0.0104191168, -0.0137194963, 0.0001199577, -0.0088299359, -0.0149593817, -0.0111030112, 0.0065119379, 0.0070310115, -0.0062670380, 0.0000000000, 0.0055873042, -0.0042411779, -0.0031730640, -0.0059622262, 0.0128036497, -0.0101240758, -0.0126859912, -0.0182199955, -0.0009490068, -0.0195248545, -0.0295671601, 0.0404701596, 0.0412986961, -0.0215417104, 0.0558399361, 0.0033630165, 0.0468287128, -0.0440143640, 0.0349199091, -0.0008208482, 0.0012357561, -0.0098716784, 0.0037843687, -0.0294051334, 0.0118177672, -0.0065660247, 0.0049890913, 0.0066044400, -0.0305807258, -0.0269323720, -0.0867081613, -0.0198274757, -0.0321529918, -0.0271987941, -0.0229254645, 0.0447389703, -0.0229161010, 0.0238363321, -0.0043046830, -0.0483497685, -0.0171818407, -0.0423224058, 0.0453703386, -0.0029181628, -0.0367966445, 0.0298092637, -0.0568714318, 0.0080763994, 0.0160352620, -0.0263402134, 0.0376195617, 0.0086038518, -0.0132818963, -0.0006736890, -0.0054010526, 0.0073963091, 0.0172084463, 0.0216076250, 0.0041118290, 0.0564979170, 0.0111284060, 0.0228991756, 0.0366673312, -0.0389759916, 0.0124683109, 0.0159203493, -0.0210093384, 0.0555163964, 0.0213419910, -0.0343683400, -0.0328103955, 0.0307689315, -0.0441230182, -0.0062740331, 0.0554288545, -0.0213757385, 0.0170451633, 0.0357777259, -0.0050568134, 0.0178009257, -0.0154460992, -0.0052366742, 0.0107557007, 0.0047277003, -0.0209390871, -0.0464514241, 0.0522775277, -0.0873311737, 0.0012034502, 0.0037265582, 0.0081449908, 0.0420676906, 0.0457199347, -0.0078505551, 0.0300787085, -0.0199865245, 0.0071044659, -0.0224131862, -0.0605014076, -0.0077089039, 0.0214931608, -0.0106343074, -0.0144480199, 0.0092140347, 0.0183581356, 0.0045405415, 0.0111179277, 0.0094189872, -0.0077424054, 0.0196302564, -0.0120974516, 0.0051889746, -0.0037029447, 0.0025281656, -0.1121378619, 0.1091478107, -0.0407832004, 0.0522077704, 0.0084252611, 0.0111021724, 0.0045765166, 0.0850910605, 0.0236283602, 0.0547384451, -0.0125313266, -0.0456712929, -0.0792934248, -0.0376970934, -0.0029991850, 0.0155947698, 0.0652837501, 0.0108818082, 0.0570372100, 0.0244021135, 0.0462104040, -0.0235865554, -0.0151550646, -0.0070386051, 0.0045551281, -0.0009144014, 0.0041537851, 0.0017853915, 0.0033040435, -0.0409783708, 0.0133078286, 0.0091215301, 0.0401342162, 0.0608952826, 0.0621705497, 0.0207645337, 0.0277205211, 0.0507565169, 0.0705232313, 0.0557268549, -0.0083685044, -0.0359528024, -0.0161432743, -0.0143371746, -0.0116999157, 0.0004209220, -0.0176302389, 0.0343130481, 0.0120356604, 0.0499725301, -0.0039951408, 0.0096956094, -0.0015340770, 0.0049754764, 0.0185118515, -0.0924603400, 0.0059073790, 0.0182503278, 0.0508906412, 0.0028308441, 0.0186160583, -0.0399922407, 0.0227791278, -0.0209690063, 0.0036776970, 0.0416555026, 0.0168991939, 0.0361626433, 0.0216568274, 0.0067009937, 0.0440197672, -0.0146929131, 0.0742861430, 0.0254844806, 0.0525204181, 0.0176357164, 0.0144726462, 0.0488788497, -0.0132995200, 0.0015076138, -0.0065645745, 0.0032468860, 0.0110189441, -0.0573362682, 0.0172904732, 0.0051031190, -0.0453508813, 0.0249099714, 0.0051649947, 0.0298099240, 0.0262806004, -0.0611060031, 0.0442470857, 0.0491859507, 0.0614929459, -0.0379763886, 0.0729502529, 0.0259373024, -0.0342665804, -0.0105072365, -0.0225182479, 0.0356933584, -0.0318366750, 0.0432488907, 0.0018436907, -0.0086889872, -0.0053829697, 0.0059221935, 0.0074206030, 0.0017904674, 0.0045264437, 0.0228709742, -0.0082113068, -0.0919025598, 0.0969315761, -0.0588679925, -0.0957834716, -0.0713828031, -0.0879146503, -0.0389149868, 0.1011034628, -0.0139821040, 0.0627171161, 0.0401022651, -0.0068530130, 0.0335105921, 0.0294922375, -0.0248575313, -0.0341003567, -0.0451620739, 0.0149446142, -0.1122278162, -0.0397146459, -0.0601395922, 0.0150245606, -0.0034357790, -0.0188668102, 0.0021314900, 0.0012000651, 0.0102146570, -0.0543795052, 0.0620148071, -0.1262678353, -0.1069441809, 0.0003751955, -0.0437733455, 0.0142210081, 0.0019271755, 0.0206407921, -0.0062772967, 0.0933588755, 0.0577048295, -0.0244932705, 0.0282769869, -0.0449099473, -0.0449757212, 0.0152334110, -0.0208018615, -0.0323720830, 0.0824338932, -0.0675388526, -0.0062826767, 0.0116116280, -0.0895515065, 0.0098043385, 0.0020023653, -0.0015339183, -0.0188143619, 0.0692517972, -0.0712488527, -0.0109854544, 0.0482835751, -0.0125067082, -0.0253530189, -0.0184751119, 0.0185605715, 0.0203283621, 0.0350815689, 0.0351322478, -0.0247285088, 0.0121784550, 0.0287878536, -0.0329477893, -0.0389662516, -0.0732927299, -0.0385071470, -0.0332750100, -0.1037825581, 0.0884760119, -0.0675178388, -0.0028796720, 0.0077282735, -0.0019473770, 0.0000000000, -0.0050746436, 0.0358674602, -0.0345409493, 0.0696369083, -0.0179852137, -0.0456299467, 0.0167119979, 0.0157325701, 0.0878602163, -0.0262518082, 0.0383564952, 0.0802284299, 0.0267905296, 0.0290917207, -0.0286849391, 0.0182544572, -0.0162715674, -0.0369428253, 0.0169209072, 0.0048733361, -0.0050417827, -0.0209259776, -0.0168898049, 0.1011911007, -0.0901383200, -0.0069882721, 0.0091899512, 0.0004866344, 0.0049215972, -0.0033623971, -0.1089749102, -0.0459114269, -0.0134607477, 0.0151586800, 0.0256856002, 0.0063458620, 0.0206886066, 0.0596189441, 0.0431430134, 0.0383799219, 0.0144599773, -0.0347141375, -0.0030641811, -0.0340030928, -0.0484567155, 0.0098888416, -0.0318307327, 0.0165797963, -0.0387178390, 0.0585021923, -0.0480427250, -0.0255086052, -0.0099312255, 0.0087915886, -0.0057507476, -0.0033612560, 0.0070202069, -0.0491243962, -0.0190055875, 0.0047685151, -0.0065096457, -0.0259327615, -0.0396161896, 0.0291941164, 0.0517396255, 0.0228764639, 0.0092735383, 0.0131507237, -0.0610293305, -0.0251695698, -0.0133743420, 0.0230516941, 0.0318689398, -0.0172422645, -0.0207093343, -0.0053179312, -0.0500000580, 0.0353724368, 0.0129722739, -0.0150523597, -0.0965300540, 0.0004429056, -0.0029775356, 0.0000000000, 0.0027330717, 0.0220650478, 0.0116937171, -0.0114663556, -0.0281855569, -0.0042679924, 0.0418123728, -0.0237817073, -0.0579217603, 0.0336738041, -0.0478041043, 0.0204568409, -0.0156659368, -0.0574834453, 0.0176323361, -0.0515975498, -0.0530485217, -0.0027693763, -0.0160944610, -0.0082243868, 0.0311801066, -0.0372301061, 0.0197619698, 0.0026408542, -0.0166284075, -0.0267814396, -0.0001754449, -0.0001414259, 0.0019869497, -0.0080336481, -0.0779113436, -0.0202660655, 0.0168184921, -0.0034189752, 0.0039179375, -0.0007724903, 0.0262602363, -0.0249893924, -0.0266961262, -0.0463402105, 0.0203211765, -0.0131692786, -0.0346035452, 0.0040005260, -0.0071892149, -0.0093772269, -0.0108877420, 0.0212523059, -0.0383761083, 0.0374399054, -0.0168946824, -0.0479299591, -0.0518232638, -0.0293789101, 0.0000534941, -0.0001414252, 0.0046131712, -0.0351432609, 0.0559104727, -0.0117593232, 0.0026123702, 0.0184675636, 0.0193821087, -0.0354812569, -0.0340058335, 0.0043088468, 0.0341250607, 0.0405881729, -0.0025988063, 0.0378006371, -0.0103789123, -0.0118906623, -0.0074687313, 0.0130766091, -0.0268870258, -0.0222278832, 0.0201150039, -0.0158142826, -0.0078476182, 0.0003363786, 0.0051987846, 0.0061922920, 0.0000534948, 0.0000000000, 0.0000000000, 0.0036138646, -0.0298682714, 0.0070560575, -0.0377869472, -0.0084806533, -0.0504923130, 0.0099457528, -0.0009346875, -0.0018329242, 0.0152403052, 0.0291247329, 0.1071927375, 0.0113427117, 0.0639097821, 0.0113030924, 0.0143610166, 0.0237733653, -0.0024755463, -0.0025539028, 0.0123429397, -0.0238598541, 0.0210772284, 0.0041539191, -0.0279648575, -0.0104228985, 0.0000000000, 0.0000000000, 0.0000000000, 0.0111793726, -0.0113527575, -0.0373919693, -0.0081786964, -0.0024105639, -0.0012743465, 0.0117560908, 0.0122344534, -0.0089833871, 0.0054704900, 0.0029056469, -0.0072841390, 0.0040307512, 0.0012646016, 0.0638523714, -0.0356466063, 0.0274009080, 0.0659771174, -0.0066763519, 0.0080027942, -0.0146102987, 0.0217740384, -0.0739494057, -0.0415747380, 0.0024984091, 0.0000000000, 0.0000000000, 0.0000000000, 0.0002626445, 0.0140588063, -0.0230792673, -0.0262241025, 0.0173176932, -0.0338136202, -0.0189422150, -0.0457660684, -0.0059601667, -0.0232613632, -0.0354378887, -0.0299661289, -0.0011518107, 0.0019122017, -0.0174512157, 0.0296385148, -0.0500545070, 0.0049748301, 0.0126543119, -0.0277010934, -0.0369372509, -0.0077440354, -0.0358451821, 0.0042622695, 0.0026252789, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0023567911, 0.0135459018, -0.0456367991, -0.1112620965, 0.0378749389, -0.0198016217, -0.0377301377, -0.0437724522, 0.0131261118, -0.0595887302, 0.0131237683, -0.0373828959, -0.1297790088, -0.0577247694, -0.0165984472, 0.0500267092, -0.0721404598, -0.0392104166, 0.0173763309, -0.0050457731, 0.0241023104, -0.0430038006, 0.0044989630, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0047368764, 0.0066094652, 0.0137510942, 0.0006697004, -0.0215071185, 0.0213063876, -0.0422299502, 0.0146959538, -0.0287603117, -0.0135646545, -0.0125521471, -0.0247477505, 0.0252241310, -0.0227418390, -0.0127671567, 0.0031791943, -0.0072523974, -0.0124533601, -0.0161637558, 0.0035176541, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
    {0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0008117753, 0.0014104483, 0.0034890127, 0.0034890134, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0055813995, 0.0014314599, 0.0076862996, -0.0148200207, 0.0127517980, -0.0048197879, 0.0082191321, -0.0144130006, 0.0099158715, -0.0100653548, 0.0049536426, 0.0012826960, -0.0277913632, -0.0008140728, 0.0169088870, -0.0282531971, -0.0568091607, -0.0033625025, -0.0054549952, 0.0026527394, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0011690858, 0.0029185328, 0.0112577540, -0.0048795151, 0.0029389674, 0.0002648849, 0.0272005975, 0.0181116054, -0.0177553108, 0.0283915155, 0.0359083655, -0.0212035234, 0.0159542995, -0.0092220827, 0.0303560974, -0.0395065834, 0.0036892270, -0.0397101458, 0.0296026191, -0.0310701175, -0.0264685670, 0.0153918678, 0.0013253781, 0.0073431247, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0023945043, -0.0047510727, 0.0063129193, -0.0126791291, 0.0040794818, 0.0122855413, -0.0182476547, -0.0500195797, 0.0211685641, -0.0872242485, -0.1912954007, -0.0094630543, -0.1691299757, -0.1117532815, -0.0710346692, -0.0777303597, -0.0134176891, -0.0784822224, -0.0448266866, 0.0245287910, 0.0302459285, 0.0096613262, -0.0022638065, -0.0133178382, 0.0076476839, 0.0000000000, 0.0000000000, 0.0072887019, 0.0048496146, 0.0118892385, -0.0023457612, -0.0050476809, 0.0648710792, -0.1303580219, -0.0302234488, -0.1213039177, 0.0341809999, -0.0381649875, -0.0499697722, -0.0120982630, -0.0639542834, -0.0028469910, -0.1169847727, -0.0663775714, -0.0163363221, -0.1053090086, 0.0207807565, -0.1075778201, -0.0222794358, 0.0109163670, 0.0433560058, 0.0211186005, 0.0010556230, 0.0004416983, 0.0000000000, 0.0000000000, 0.0102087092, -0.0185801893, -0.0035554283, 0.1301168514, -0.1664674943, -0.0447910503, 0.0265385332, -0.0201967748, -0.0740704378, 0.0771390348, -0.0328495771, 0.0385935831, -0.1046681769, 0.0098047191, -0.1407079288, -0.0481639075, -0.0980983969, -0.0548876531, -0.0549857607, -0.0525877530, -0.0657206146, -0.1539602474, -0.0683414234, 0.0047860854, 0.0185981246, 0.0096562936, 0.0000000000, -0.0090022967, 0.0001283104, 0.0217238139, -0.0618113046, -0.2236637425, 0.0420670444, -0.0872684903, -0.0359867209, 0.0103522615, -0.0294751311, 0.0183356767, 0.0348883700, 0.0690189385, 0.1119400505, 0.0117572385, 0.1034276967, 0.0507937946, 0.0808174396, 0.0033404355, 0.0494648627, -0.0124376814, -0.0407080382, -0.0192373665, -0.0207906922, -0.0075090973, -0.0216911635, 0.0223320875, 0.0066929722, -0.0006098611, 0.0243257534, 0.0297674882, -0.0884306140, -0.1213056809, -0.0229370638, -0.0322036045, -0.0092780239, -0.0630958999, 0.0029427596, -0.0074250882, 0.0393416405, 0.0635468110, 0.0892110834, 0.1231455848, 0.1163384099, 0.0234473792, -0.0072391831, -0.0217599996, 0.0030712963, -0.0239985304, -0.0386437122, -0.0014373661, -0.0693924391, -0.0357676931, -0.0341259627, -0.0072187248, -0.0079185716, 0.0133470293, -0.0408379554, -0.1115710653, 0.0442721574, 0.0726980718, -0.0455671790, -0.0596843942, 0.0186602108, -0.0340209764, -0.0197553120, -0.0528380697, -0.0566150189, 0.0445918492, 0.0469301262, 0.0449093247, 0.0277202650, -0.0111123245, 0.0075285812, -0.0158228441, -0.0427853497, 0.0530595645, -0.0173160638, -0.0219134358, 0.0218188956, 0.0030331558, -0.0345413110, 0.0087125786, -0.0056378649, 0.0226726275, -0.0068146027, 0.0677428299, -0.1133930160, -0.0437639611, 0.0432872700, -0.0258375994, -0.0386998222, -0.0150073480, 0.0227014088, -0.0138449224, 0.0059366517, -0.0135573633, 0.0238183616, 0.0512460801, -0.0531672918, -0.0196835067, -0.0111418665, -0.0482399664, 0.0339263485, -0.0618956048, -0.0303031490, -0.0605740125, -0.0000830269, -0.0559829463, -0.0201063880, 0.0126685762, 0.0022932324, 0.0004699638, -0.0952845271, 0.0137124831, 0.0230568975, -0.0106471114, 0.0079609303, -0.0166067979, 0.0267674271, -0.0348006158, 0.0180275262, 0.0268592938, -0.0207267737, -0.0590982965, 0.0255002104, -0.0687500184, -0.0028415404, -0.0336084968, -0.0077111898, 0.0084278275, -0.0530120392, -0.0314497940, 0.0151080596, 0.0365001132, -0.0985466436, 0.0358194638, -0.0102216700, -0.0140310073, 0.0091764547, -0.0804247910, -0.0135097563, -0.0416024242, -0.0140430847, 0.0212056570, -0.0175143894, 0.0342656681, 0.0826646925, 0.0170894122, 0.0343006659, 0.0293379568, -0.0066550522, 0.0030375776, -0.0622195463, 0.0380562265, 0.0085145045, 0.0522307876, 0.0216684126, -0.0099041478, 0.1035899623, 0.0456150180, 0.0038806746, -0.0813623909, 0.0312363952, -0.0057215980, -0.0441076488, -0.0137408199, 0.0065708036, -0.0194042391, 0.0024966112, -0.0079760208, -0.0037351652, 0.0206183830, 0.0432041811, 0.0078814448, -0.0222961574, 0.0351146738, 0.0389138087, 0.0066200168, -0.0314812549, 0.0090877073, 0.0686685947, 0.0503460876, -0.0006382968, 0.0459322267, 0.0371026096, 0.0800823603, 0.0117977521, 0.0217730615, 0.0549280339, 0.1515765994, -0.1265980621, -0.1151233514, -0.0257277303, 0.0190650746, -0.0006925006, -0.0932332628, -0.0150960187, -0.0095981142, 0.0257747075, 0.0323153452, -0.0117677099, 0.0328807560, 0.0686043781, 0.0156304282, 0.0227859958, 0.0147200822, 0.0035390215, -0.0028425002, 0.0488583487, 0.0288352777, 0.0471441098, 0.0316597666, 0.0303252844, 0.0450123409, 0.0236323191, 0.0652402863, -0.0272574034, -0.0569646890, 0.0549504254, -0.0568946555, 0.0204517661, -0.0005478322, -0.0017386064, -0.0364875521, 0.0197745931, 0.0094939423, -0.0077677294, 0.0007186057, 0.0517126311, 0.0012740036, -0.0239431145, 0.0305718223, 0.0142501612, 0.0230857549, 0.0176153666, -0.0033809742, 0.0291866886, 0.0264763541, 0.0303016694, 0.0426050464, 0.0006018751, -0.0228907589, -0.0015320682, -0.0311122474, 0.0619574579, -0.0507109987, -0.1411988082, 0.0464677118, -0.0166424072, -0.0056184431, -0.0007895694, 0.0079610249, -0.0571290681, 0.0031257339, -0.0021474284, 0.0030895397, -0.0542753178, 0.0097173978, 0.0255514031, -0.0169804685, 0.0195984974, 0.0208980602, 0.0137669452, -0.0001181804, -0.0377524162, 0.0040968370, 0.0524512329, 0.0222581536, -0.0317150654, 0.0149872875, 0.0082059763, -0.0088578148, -0.0624606195, -0.0099652376, -0.0586368894, -0.0227137524, 0.0219081791, -0.0132360998, 0.0049423162, 0.0047257309, -0.0273531094, -0.0155051349, 0.0311080952, 0.0097200291, -0.0132041604, -0.0407921303, 0.0276644567, -0.0123281353, 0.0687354606, 0.0188085171, 0.0259644714, -0.0374614257, -0.0085129405, -0.0418631864, 0.0345758029, -0.0060808545, 0.0112468528, -0.0610406806, 0.0322701104, -0.1498445804, 0.1143514507, -0.0007954640, -0.0487138696, -0.1031613910, 0.0646740636, -0.0232594891, 0.0000000000, -0.0052186567, -0.0160995507, 0.0485892578, -0.0478166437, -0.0024523802, -0.0572096059, 0.0555049483, 0.0112364997, -0.0007704851, 0.0087148377, -0.0039989913, 0.0262847842, 0.0090155684, -0.0704035121, 0.0534044549, -0.0255175472, -0.0206500406, 0.0199224921, -0.0223002057, 0.0092259066, 0.0053675752, -0.0893546924, -0.0233833920, -0.0468958459, -0.0653870410, -0.0242579924, -0.0401852359, 0.0041579560, 0.0053279884, 0.0068855753, -0.0847673030, 0.0379870094, -0.1550934788, 0.1231673713, -0.0389235880, -0.0542889873, 0.0012011670, 0.0755562517, 0.0388479759, -0.0669460073, -0.0302680321, 0.0700980341, -0.0141918202, -0.0130755955, 0.0253586821, -0.0468956923, -0.0224599812, 0.0028393744, 0.0165394433, 0.0734264148, -0.0937279034, -0.0229360164, -0.0143790862, -0.1339922736, -0.0066081493, 0.0063018740, -0.0075498808, -0.1583282589, 0.0240224574, -0.0676344650, -0.0686935118, -0.0196021287, 0.0441980088, 0.0520265939, -0.0526214907, -0.0344740418, -0.0100570010, 0.0524020201, -0.0013208824, -0.0749394626, -0.0405553891, 0.0258433162, -0.0522011444, 0.0217819599, -0.0128096932, 0.0167618596, 0.0021294061, -0.0638211940, -0.0128763365, -0.0287305367, -0.0641203513, 0.0162386182, 0.0094812526, 0.0000000000, 0.0016520746, 0.0229944130, -0.0228845499, -0.0173267088, 0.1115274129, -0.0689323709, -0.1028971701, -0.0862808120, -0.0943968741, -0.1091515386, -0.0231024849, -0.1005514342, -0.0332778832, -0.0066470661, -0.0410703857, -0.0671499603, -0.0025177511, 0.0140423293, 0.0268729667, -0.0340542747, -0.1272228558, 0.0158191216, -0.0257051172, -0.0190369596, 0.0669660899, -0.0385811236, 0.0072373369, 0.0004382490, -0.0039377732, 0.0172340420, -0.0933940740, -0.0527651178, 0.0284616498, 0.0660275129, -0.0195419485, -0.0009351652, 0.0115537060, -0.0252343507, -0.0931063779, -0.0254858710, -0.0190925780, -0.0498250893, -0.0041160334, -0.0078898878, -0.0903627530, -0.1035501173, -0.0040466868, 0.0025718469, 0.0707719252, -0.0720777113, -0.0342532641, 0.0682514731, -0.0369229491, -0.0457260575, 0.0034288208, 0.0004382501, 0.0174220657, -0.0287005838, 0.0085322444, 0.0945867129, -0.1004962120, -0.0896247175, -0.0269475868, 0.0028009659, -0.0348952935, 0.0347106774, 0.0082701252, 0.0014879799, -0.0229333678, -0.0042920584, -0.0948349508, -0.0051784377, -0.0220033961, 0.0590966249, -0.1273072882, -0.0009653045, -0.0296888780, -0.0226214905, 0.0280646597, -0.0055239426, 0.0078782981, 0.0035248833, 0.0034288219, 0.0000000000, 0.0000000000, -0.0250349667, 0.0520950114, -0.1369176629, 0.0144749777, 0.0224158300, -0.0957250501, 0.0657030441, -0.0660171440, -0.0487333354, -0.0442429338, -0.0195585788, -0.0352968936, 0.0030671426, -0.0336511960, -0.0415742528, -0.0516306939, -0.0449742733, 0.0198354145, -0.0266416942, -0.0147357561, 0.0804981539, -0.0139792818, 0.0352411373, -0.0214972355, -0.0105601585, 0.0000000000, 0.0000000000, 0.0000000000, 0.0028528184, -0.2776755860, 0.0574987759, -0.0022850930, -0.0006768164, 0.0245307659, -0.0352230896, -0.0115638847, -0.0067969030, -0.0322235009, -0.0397153709, -0.0642122353, -0.0640205662, -0.0044574966, -0.0173094472, 0.0196139109, -0.0149130122, 0.0142911913, 0.0205603419, 0.0063567434, -0.0133823712, 0.0366761893, -0.0161039639, -0.0188851397, -0.0084802460, 0.0000000000, 0.0000000000, 0.0000000000, 0.0148146673, -0.0222928115, -0.0209631385, 0.0212953053, -0.0102774890, -0.0133848056, 0.0049139024, -0.0081175254, 0.0128234728, -0.0077137928, 0.0188533378, 0.0239740366, 0.0106951013, 0.0078047657, 0.0231348346, 0.0271717679, 0.0327923682, 0.0077518682, 0.0300467199, 0.0267969443, -0.0045838547, -0.0064308558, -0.0045366985, 0.0316304524, -0.0100259851, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0111212907, 0.0203559187, -0.0153880839, 0.0312591954, 0.0066448063, 0.0221136324, 0.0165183874, 0.0217558650, 0.0045539735, 0.0408516741, 0.0026605288, 0.0544135970, -0.0025474489, 0.0337363629, 0.0010283288, 0.0355163010, 0.0028840215, 0.0161788511, 0.0023539117, 0.0001778706, 0.0050296249, 0.0058194811, 0.0025521934, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0043038242, -0.0067054912, -0.0036387021, 0.0057365889, -0.0021643228, -0.0081930953, 0.0098241351, -0.0149970686, 0.0156002060, -0.0466214089, 0.0223095615, -0.0260665832, 0.0130334095, -0.0295103194, 0.0095729851, -0.0152278070, 0.0135754182, -0.0149610515, 0.0174539874, -0.0105841194, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000},
};

double bias[NUM_CLASSES] = {-2.9525892761, -3.6934993680, -1.8095658856, -1.9831172869, -2.7205156918, -1.8805774781, -2.7108441697, -3.0740618807, -1.5869174001, -2.3630234768};


´´´

scaler1.h

``` h

#define NUM_FEATURES 784

double mean[NUM_FEATURES] = {
    0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0021000000, 0.0078333333, 0.0036000000, 0.0001500000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0002666667, 0.0009166667, 0.0092833333, 0.0242833333, 0.0437166667, 0.0641000000, 0.1201333333, 0.1607333333, 0.1741833333, 0.1774333333, 0.1893166667, 0.1741500000, 0.1869333333, 0.1536500000, 0.1001166667, 0.0712333333, 0.0538166667, 0.0213666667, 0.0100833333, 0.0035333333, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0010666667, 0.0007000000, 0.0069500000, 0.0054833333, 0.0471000000, 0.1384000000, 0.2641833333, 0.5066166667, 0.8668000000, 1.2900833333, 1.8703500000, 2.5299500000, 3.2016166667, 3.6255500000, 3.7219833333, 3.3925500000, 2.8029333333, 2.0443833333, 1.2021166667, 0.6334500000, 0.2961666667, 0.0939833333, 0.0352166667, 0.0086333333, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0032333333, 0.0058500000, 0.0120166667, 0.0697666667, 0.2120833333, 0.5461000000, 1.1544166667, 2.2159000000, 3.6396333333, 5.4380000000, 7.4070500000, 9.6967333333, 11.8830833333, 13.2373666667, 13.1246166667, 11.8135166667, 9.5436666667, 6.8630500000, 4.1936500000, 2.2746333333, 1.0610333333, 0.4130666667, 0.1620833333, 0.0277666667, 0.0028000000, 0.0000000000, 0.0000000000, 0.0006333333, 0.0052166667, 0.0143500000, 0.0808833333, 0.4102166667, 1.0432333333, 2.4192833333, 4.7757500000, 8.3944166667, 13.3102666667, 19.4773000000, 27.0366833333, 35.2121666667, 41.8408000000, 45.2526000000, 44.3388333333, 39.1414500000, 31.3282166667, 22.9239333333, 14.8310166667, 8.6614000000, 4.5479666667, 2.1370166667, 0.8629000000, 0.2077500000, 0.0296500000, 0.0020333333, 0.0000000000, 0.0000000000, 0.0103166667, 0.0633500000, 0.3954666667, 1.4634000000, 3.5877833333, 7.2278166667, 13.0171166667, 21.2179166667, 31.5004000000, 44.2063666667, 58.9038500000, 73.8041500000, 85.1027333333, 90.5997833333, 88.9239333333, 80.1812500000, 65.9428166667, 49.7938333333, 34.3498333333, 21.5100000000, 12.3903166667, 6.6391166667, 2.9927166667, 0.8436666667, 0.1416333333, 0.0047000000, 0.0000000000, 0.0001833333, 0.0275000000, 0.2214333333, 1.1195166667, 3.2899000000, 7.4364666667, 14.2544333333, 24.1377500000, 37.2885166667, 53.1154000000, 70.9128166667, 89.2784666667, 106.2329666667, 118.5061166667, 124.0981500000, 121.7113333333, 112.0333333333, 96.1081166667, 75.1653666667, 54.0371166667, 35.3124833333, 20.9919000000, 11.4424333333, 5.3877666667, 1.8555833333, 0.3727000000, 0.0303000000, 0.0007833333, 0.0197333333, 0.1076666667, 0.5956500000, 2.3124000000, 5.9287666667, 12.4360500000, 22.4518000000, 36.2947833333, 53.8984166667, 73.7647166667, 94.1030833333, 111.5794833333, 124.9999166667, 132.7591166667, 135.3987000000, 133.3386500000, 126.6775333333, 113.5466666667, 93.7442666667, 69.7779333333, 46.8328833333, 28.1439166667, 15.2010500000, 7.0336666667, 2.5830333333, 0.5131000000, 0.0318833333, 0.0040666667, 0.0508166667, 0.3293666667, 1.3173500000, 3.7481333333, 8.4834666667, 16.8181833333, 29.5974000000, 46.9301500000, 68.1070000000, 90.2913833333, 108.4757166667, 119.4354166667, 123.3938500000, 123.0380500000, 122.3406500000, 122.7300500000, 122.1441000000, 116.2609333333, 100.6106000000, 77.3180333333, 52.8976000000, 31.8106166667, 16.3576666667, 7.1856666667, 2.6066166667, 0.4815833333, 0.0271833333, 0.0048666667, 0.0794666667, 0.5120166667, 1.7193833333, 4.4458166667, 9.8066333333, 19.5264833333, 34.5048166667, 54.7009000000, 78.1380333333, 99.0387000000, 110.6004666667, 110.7875500000, 104.5536166667, 99.0472000000, 99.3883833333, 104.3332166667, 110.8820666667, 111.0782666667, 99.1469166667, 77.0405500000, 52.7057666667, 31.4484166667, 15.3927166667, 5.9611333333, 1.8979666667, 0.3518333333, 0.0282166667, 0.0066666667, 0.0985666667, 0.5230333333, 1.6745833333, 4.2890833333, 9.8764333333, 20.5393166667, 37.1890333333, 59.7151166667, 83.3987333333, 100.2259000000, 103.1688166667, 93.8921333333, 82.9528166667, 79.4228333333, 84.5250166667, 93.8685833333, 104.4275333333, 106.2811833333, 94.2218666667, 71.8490666667, 48.3172500000, 28.4781833333, 13.4981666667, 4.6005000000, 1.1501000000, 0.2218333333, 0.0187333333, 0.0059333333, 0.0764000000, 0.4208833333, 1.2803666667, 3.6460833333, 9.5539000000, 21.1332333333, 39.8545166667, 64.0892166667, 87.2096666667, 98.7805833333, 94.8668500000, 81.6835333333, 73.3427666667, 76.1450666667, 85.4255500000, 97.2928000000, 107.4825833333, 105.5094333333, 89.0295000000, 64.9443500000, 42.4968833333, 25.3411000000, 12.5802166667, 3.9898833333, 0.6045333333, 0.1245000000, 0.0081666667, 0.0038000000, 0.0436833333, 0.2420333333, 0.8788166667, 3.0688333333, 9.6074500000, 22.7779833333, 43.5750333333, 68.9713333333, 90.2567000000, 97.8938166667, 91.0827333333, 79.7067500000, 79.3691166667, 89.3387000000, 101.7145166667, 113.0378666667, 117.5811000000, 107.4267166667, 84.5188833333, 58.5871666667, 38.2634500000, 23.7392333333, 12.7121000000, 4.2947166667, 0.4432500000, 0.0797833333, 0.0101666667, 0.0005333333, 0.0182500000, 0.1226166667, 0.6073333333, 2.9178333333, 10.6341666667, 25.5538000000, 47.5837166667, 72.6322333333, 91.6736500000, 96.8884166667, 91.0977333333, 86.8673000000, 96.9665500000, 111.3718333333, 123.9720666667, 129.8065500000, 126.5995500000, 108.8803333333, 81.1576333333, 55.2258666667, 37.2046000000, 23.9728000000, 13.6437500000, 5.1026833333, 0.5847000000, 0.0829833333, 0.0108666667, 0.0018833333, 0.0082500000, 0.0541666667, 0.4661833333, 3.0655833333, 12.3512666667, 28.5280666667, 50.5083833333, 73.7506166667, 90.3691833333, 95.0769833333, 93.2885333333, 97.8289500000, 115.4213166667, 130.2672500000, 139.5536000000, 137.1006333333, 128.0857500000, 106.9945166667, 79.5539500000, 56.0965666667, 38.9430666667, 25.5374666667, 14.6162500000, 5.7215833333, 0.8203000000, 0.0924500000, 0.0022166667, 0.0007333333, 0.0038666667, 0.0451000000, 0.5055833333, 3.5605666667, 14.4081500000, 30.9883166667, 51.2096333333, 71.4404333333, 85.7322166667, 91.3810333333, 94.2424333333, 105.0471166667, 123.2042000000, 135.6894333333, 139.1100500000, 131.8045000000, 121.4365666667, 101.3403000000, 78.3792666667, 58.2829000000, 41.2509333333, 26.9801166667, 14.8906000000, 5.8282833333, 1.0795666667, 0.1482166667, 0.0108333333, 0.0006666667, 0.0039666667, 0.0743333333, 0.6229666667, 4.4699666667, 16.5289666667, 32.5549166667, 49.9447333333, 66.3117166667, 77.8862666667, 83.7070500000, 89.4484166667, 101.1360166667, 115.8574000000, 126.4734166667, 127.4266500000, 121.0568166667, 111.2181000000, 95.3723333333, 77.3547833333, 59.6073333333, 42.0403333333, 26.7287666667, 14.2431166667, 5.6155500000, 1.2871000000, 0.1915000000, 0.0123000000, 0.0000000000, 0.0072833333, 0.1122000000, 0.9533500000, 5.9522333333, 18.5244000000, 33.4512000000, 48.1763500000, 60.4605333333, 69.2180333333, 74.6814166667, 80.7367000000, 89.5865500000, 101.9504166667, 112.6165000000, 115.4369166667, 112.2448333333, 104.5241500000, 92.6265833333, 77.2614833333, 59.3285166667, 40.8083000000, 24.9427500000, 12.8348000000, 5.1566500000, 1.4166666667, 0.2100333333, 0.0088833333, 0.0019000000, 0.0053166667, 0.1805000000, 1.5201500000, 7.6214500000, 20.4178666667, 34.9108333333, 48.1825666667, 58.2768500000, 65.9944666667, 71.7466333333, 76.5832833333, 83.1222000000, 95.1299333333, 106.5572000000, 112.2013833333, 111.3479333333, 105.0182833333, 93.6185833333, 76.6274333333, 56.7010833333, 37.7625666667, 22.5467166667, 11.3823333333, 4.5790500000, 1.2985500000, 0.1539333333, 0.0121833333, 0.0002500000, 0.0121833333, 0.2863166667, 2.0438166667, 8.7235166667, 21.7344666667, 36.8331666667, 50.9790666667, 62.3501500000, 71.2715166667, 77.8803000000, 82.6346666667, 89.9674333333, 101.4063166667, 112.4454833333, 117.8034500000, 115.9246166667, 107.4325000000, 92.2773833333, 71.8833000000, 50.8002500000, 32.6939666667, 18.7729666667, 9.1563500000, 3.6400500000, 1.0685000000, 0.1467000000, 0.0068333333, 0.0000000000, 0.0152666667, 0.3320000000, 2.2556666667, 8.5130500000, 20.6940833333, 37.0852166667, 53.7940333333, 68.8695000000, 81.1596000000, 90.4059166667, 97.8863000000, 106.7345666667, 117.3395166667, 124.9524166667, 125.5557500000, 118.2162500000, 103.8058333333, 83.8763833333, 61.2356500000, 40.9525000000, 24.9440333333, 13.6286166667, 6.5375000000, 2.7138833333, 0.7491666667, 0.1116833333, 0.0016833333, 0.0005333333, 0.0127000000, 0.2842333333, 1.8392000000, 6.6325833333, 16.9004833333, 32.8094666667, 51.5708833333, 70.4237333333, 87.2520000000, 100.9851333333, 112.4557500000, 122.5940000000, 130.3723000000, 131.8449000000, 124.8680333333, 110.0258000000, 89.4660000000, 66.5912666667, 45.5973833333, 28.5566666667, 16.2958166667, 8.5114833333, 4.1090833333, 1.7029833333, 0.4438166667, 0.0590500000, 0.0006500000, 0.0005166667, 0.0009833333, 0.1870833333, 1.0879666667, 3.9444333333, 10.8453666667, 23.3906333333, 40.6385333333, 60.9691833333, 81.0054000000, 99.1114333333, 113.1005000000, 122.5243166667, 125.3022500000, 120.6135666667, 107.1858666667, 87.7064166667, 65.9730500000, 45.3337166667, 28.7308166667, 16.9034500000, 9.1614500000, 4.7089666667, 2.2250166667, 0.8351333333, 0.1891166667, 0.0178333333, 0.0012000000, 0.0000000000, 0.0000000000, 0.0641666667, 0.4187666667, 1.7327000000, 5.0147333333, 12.0309166667, 23.9401333333, 40.2565833333, 58.8924833333, 77.2688666667, 92.0322000000, 99.7043833333, 99.1646666667, 90.5921000000, 75.5835000000, 57.5861333333, 40.1938000000, 25.4972333333, 15.2398333333, 8.5236833333, 4.4192333333, 2.2050666667, 0.9816500000, 0.3103666667, 0.0579666667, 0.0096166667, 0.0000000000, 0.0000000000, 0.0000000000, 0.0158500000, 0.1222000000, 0.5436333333, 1.6008500000, 4.2033166667, 9.1435833333, 16.8276166667, 27.0768166667, 38.1033833333, 47.0437833333, 51.6087500000, 50.9622500000, 45.4375833333, 36.7445333333, 27.4417833333, 19.1085833333, 12.1141500000, 7.2149000000, 3.9593333333, 1.9931833333, 0.9512333333, 0.3996333333, 0.1018166667, 0.0221833333, 0.0019333333, 0.0000000000, 0.0000000000, 0.0000000000, 0.0015666667, 0.0177833333, 0.1277833333, 0.4745000000, 1.4055000000, 3.1807833333, 6.1434000000, 9.8287000000, 13.8693333333, 16.7183333333, 18.1000166667, 17.7754000000, 15.8076666667, 13.1235500000, 10.4980500000, 7.7911666667, 5.2219333333, 3.1621333333, 1.6913333333, 0.8253833333, 0.3705833333, 0.1398166667, 0.0312666667, 0.0035666667, 0.0017333333, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0006333333, 0.0354500000, 0.1630833333, 0.5382833333, 1.1993500000, 2.3160833333, 3.5114333333, 4.8502166667, 5.9802333333, 6.4448166667, 6.2592333333, 5.5705166667, 4.4507666667, 3.5349333333, 2.5918166667, 1.7011666667, 1.0086000000, 0.5402833333, 0.2384000000, 0.0752666667, 0.0161666667, 0.0005166667, 0.0009833333, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0025333333, 0.0155833333, 0.0416333333, 0.0892000000, 0.1282166667, 0.1967333333, 0.3360166667, 0.4299666667, 0.5259833333, 0.5906833333, 0.6880333333, 0.5920666667, 0.4827333333, 0.3435166667, 0.2004333333, 0.0888666667, 0.0456333333, 0.0192833333, 0.0151166667, 0.0020000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000
};

double scale[NUM_FEATURES] = {
    1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 0.4753198116, 1.3611779111, 0.8818089589, 0.0367420400, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 0.0653191821, 0.1946342546, 1.0550026002, 2.0950959373, 2.9620385187, 3.3238518604, 4.8692574364, 5.5289870798, 5.6936318374, 5.6935329464, 5.9230180257, 5.7660721851, 5.9703620155, 5.4133761810, 4.3304572491, 3.7165296284, 3.2086633406, 1.9430671970, 1.2112314641, 0.8065693908, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 0.2612767286, 0.1297414480, 0.8154968000, 0.4575149539, 2.6680295332, 4.9944247690, 7.1362144843, 9.7143668289, 12.9303270554, 15.5399807484, 18.8203181219, 21.9643977761, 24.7931227169, 26.1971780388, 26.7992945815, 25.7316560129, 23.2744702352, 20.0644556398, 15.2931509350, 11.1140132759, 7.4680777517, 3.9607975375, 2.4997019395, 1.1024784800, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 0.6022924640, 0.6594814459, 0.8540134267, 3.1501903560, 6.0284965505, 9.9248177879, 14.5951089579, 20.5844898048, 26.5337540226, 32.5422651742, 37.7893833455, 43.1942858411, 47.7892499818, 50.5015609369, 49.9800895759, 47.6628416480, 42.9997545716, 36.6593584418, 28.4045398310, 21.0294153477, 13.8133597747, 8.7745945355, 5.3666481463, 1.8218110986, 0.5299297060, 1.0000000000, 1.0000000000, 0.1551330576, 0.6719544526, 0.9635234355, 3.3753776904, 8.1487753039, 13.2444691933, 20.8233022874, 29.6175020656, 39.6034718868, 49.8701009416, 59.8053887041, 69.2819190289, 77.8898623184, 83.7824768196, 86.4542439284, 85.6592249866, 81.3916679308, 74.0480271434, 64.2594269131, 52.2801252354, 39.9443523339, 28.6893830397, 19.6444812377, 12.1667964939, 5.3332844731, 2.0803535463, 0.3523670419, 1.0000000000, 1.0000000000, 0.8668199927, 2.8729955524, 7.6288841112, 16.1416808431, 25.6881054333, 36.6741368846, 49.2355091070, 62.2718550310, 74.5508517714, 85.7695477008, 95.6176495485, 103.0808265042, 107.1660245252, 108.4963348534, 107.8930650252, 105.1089682113, 98.9301282728, 89.5169397115, 76.7344282790, 61.8770913451, 47.5014322896, 34.8865238531, 22.8312000769, 11.2720285023, 4.4357269226, 0.5725480271, 1.0000000000, 0.0329135188, 1.7558503401, 6.0890777035, 14.1667380073, 24.6824409515, 37.3402057240, 51.7534375381, 66.4336965824, 80.5227471472, 92.3994860168, 101.4774236751, 107.7235781510, 111.0187579940, 112.1714962126, 112.3346550116, 112.3308524177, 112.1509072138, 109.8695373043, 103.7393648551, 93.0702165340, 78.0975315711, 61.8639860317, 45.9664009839, 31.1540185265, 17.8242819536, 7.7135418611, 2.0601007848, 0.1918750975, 1.4914458071, 4.1025530127, 10.0686982481, 20.4628380137, 33.2252633671, 48.2008455361, 63.9813174041, 79.0566842174, 92.2688121604, 101.7508636740, 107.4678660057, 109.9518762113, 110.4075138596, 110.4312127913, 110.1216510878, 110.2247694464, 110.6731134855, 110.9801896236, 108.6318759256, 101.0571040867, 87.8828435401, 70.7938571111, 52.8398999705, 35.7026184832, 21.3070685955, 9.0320814355, 2.0716016637, 0.8092075108, 2.9338997369, 7.6453701196, 15.4529858704, 26.8763600558, 40.2282511839, 56.0572519189, 72.6706174914, 88.0674883502, 100.0647304715, 107.2358863087, 110.4147931378, 111.1088091122, 111.3682279596, 111.1865643961, 110.8012611582, 110.7647259299, 111.0194687215, 111.3838073830, 110.2004039661, 104.5981356819, 92.6393871287, 75.0832994122, 54.8553814548, 36.3488651068, 21.5987260323, 8.9288536494, 2.0127537686, 0.8240608688, 3.7644678750, 9.6626905639, 18.4724020533, 29.5602001149, 43.3555372396, 60.3329661017, 78.0420276313, 93.6262493777, 104.3822883130, 109.9880875473, 111.4488479368, 111.3343748130, 110.1078301420, 108.9742500417, 108.7476436912, 109.3290242798, 110.5407232278, 111.1353289508, 109.9788902418, 104.5778241584, 92.8589657854, 74.8465490576, 53.0463965815, 32.8469038021, 18.0862467785, 7.4520162801, 1.9968109124, 1.1548400563, 4.3551369606, 9.9053387692, 18.1613652385, 29.0880562114, 43.4540389141, 61.7857013195, 80.3746317756, 96.4650775747, 106.3375934390, 110.3554390860, 110.7706777579, 107.9744983694, 104.0735526302, 102.9577227732, 105.1881712654, 107.4085850681, 109.7417312690, 110.4643505960, 109.1196116280, 102.4477152135, 89.9098602440, 71.7748560247, 49.7678460117, 28.8750382352, 14.1536121417, 6.0034676068, 1.5777565493, 0.8409903659, 3.6781738730, 8.9163468166, 15.6134971910, 26.5252098891, 42.4274670639, 62.4334091830, 82.5256337021, 98.8591413260, 107.6921396383, 110.1895595765, 108.8190155920, 104.0586252176, 100.1209390904, 102.7667307822, 106.4877026884, 107.9516411863, 109.9315361031, 110.4177285479, 107.9272483192, 99.0023694316, 85.1019846437, 68.5685339699, 48.6729171643, 26.6532902469, 9.8923239323, 4.4445921916, 0.8949673954, 0.7163929741, 2.6901254803, 6.6531761737, 12.8140989253, 24.0941312766, 42.4838917061, 64.6728999808, 85.8480077618, 101.5105893239, 108.9726360382, 110.1378359531, 108.1869993200, 103.9832481113, 103.0493724850, 108.9039848474, 110.8383421141, 109.8419129148, 111.1059564385, 111.2453186860, 106.3830778057, 94.9934773303, 81.3989730326, 66.8476521708, 49.7718265044, 27.9195855763, 8.4502157431, 3.7098406911, 1.0246609060, 0.1306383643, 1.6690467152, 4.8000640780, 10.6413272773, 23.4059525614, 44.5880424849, 68.4613285407, 89.2524807770, 103.4508468518, 109.4676034831, 110.0911846266, 108.3188910034, 106.3519363123, 108.7161528834, 113.8130593268, 111.6057651724, 109.9468544059, 112.1483635924, 111.5496709089, 104.6931313493, 92.9029894616, 80.7198385705, 67.4339330517, 51.7692750184, 31.0186563130, 9.8247388045, 3.5510510791, 1.4676109551, 0.4613167239, 1.0327545388, 2.9605009664, 9.0804142582, 23.7005699402, 47.9620993952, 72.2208821066, 91.7630631739, 104.2672265126, 109.3991929602, 109.5542296015, 108.3813113434, 108.2135939946, 111.5501333135, 113.7272001800, 109.5218455851, 109.8392427122, 112.3780918608, 111.0727243143, 104.2631326152, 93.9598425297, 82.5030031288, 69.3255099482, 53.4756538306, 32.9989873707, 11.8982831217, 3.7513201673, 0.3173511510, 0.1537296183, 0.9229581332, 2.4184566683, 9.4905796535, 25.3564560815, 51.9810865948, 75.1762239463, 92.4355942654, 103.2368084478, 108.1681054752, 108.5361696100, 108.3536570345, 109.5765342733, 112.6075320262, 112.6151935354, 109.3266834111, 111.3557848209, 112.6130536078, 109.8439181865, 104.1610860965, 95.5790833163, 84.7469944351, 70.8331451699, 53.5057828492, 33.2111903338, 14.0155878392, 5.0028740226, 1.1891380515, 0.1632979554, 0.5913974402, 3.4411298371, 10.2617418534, 28.4670616327, 55.6225280583, 77.0883083493, 91.5196795536, 100.6094486089, 105.0741431786, 106.1760777999, 107.2565044143, 110.0005905263, 112.9263271573, 112.6435472038, 111.2835118954, 112.1457488652, 111.2307454456, 108.5111221698, 104.0297139224, 96.8246227959, 85.2607151031, 70.3621382502, 52.0738486282, 32.3589861017, 15.3134768616, 5.6055711350, 1.3758931802, 1.0000000000, 0.8154121369, 4.4177118316, 12.9005106014, 33.4658256287, 59.1109747676, 77.9724512369, 90.0342854177, 97.3941489125, 101.3902511494, 103.2266749505, 105.4611103983, 108.5804887742, 110.8891662494, 111.9158916080, 111.7692041090, 111.0595296375, 109.8508972051, 108.2762111915, 104.3621296726, 96.7419302068, 84.1137189630, 67.7714812619, 49.2166202513, 30.6334872992, 15.9998246518, 5.8446117064, 0.8293799409, 0.3217085482, 0.8369916764, 5.4171320595, 16.6491679665, 38.2779638508, 62.2557787474, 79.5243668065, 90.1528639002, 96.1746881326, 100.0818272018, 102.8112077483, 104.8478681895, 106.4133512950, 109.1329387368, 111.1760705435, 110.9535319916, 110.1965072453, 110.2039204493, 109.0204928962, 104.6377729475, 95.3102596724, 81.4850914303, 64.6461642395, 46.2913356316, 28.9445190741, 15.1987472476, 4.8777970980, 1.2482393867, 0.0612367333, 1.0799235620, 7.2016483831, 19.6800261712, 41.1523572468, 64.4026391181, 81.5430796163, 92.3313677024, 98.6730250794, 102.8116113515, 105.4738523928, 107.2159958443, 108.5554316440, 110.3114620371, 111.4639210444, 111.0485573286, 110.6020346438, 111.0924159296, 109.0102000204, 102.5113625301, 91.4684230574, 76.5532910914, 59.2561416440, 41.4904350987, 25.8054771834, 13.6687444345, 5.0269718297, 0.8514223231, 1.0000000000, 1.4761662945, 7.6721102703, 20.6625297230, 40.7259495043, 62.6666843838, 81.5283209777, 94.5086702076, 102.3406716303, 106.7045348670, 109.0860923690, 110.4178180321, 111.2003269411, 111.6048181984, 111.4745997339, 111.2152704980, 111.2372919451, 110.5101642353, 106.5303140061, 97.6659223360, 83.9082636599, 67.6079576263, 50.6090238108, 34.9201879589, 22.3130855610, 11.2667112610, 4.2900672294, 0.2915713173, 0.1306383643, 1.2687941953, 7.0183268243, 18.3948129471, 35.2418840065, 56.4042629574, 76.7924733533, 92.8784456995, 103.2943209961, 108.8490005895, 111.1367110619, 111.4587252840, 111.0832346966, 110.3604048533, 110.5909107657, 110.9637190773, 110.6111588751, 107.5011775626, 99.8181267960, 87.2352473669, 71.5588589127, 54.9234289033, 40.0302614880, 27.6345010960, 17.5754221694, 8.7140620130, 2.9211806570, 0.1592155065, 0.1265559154, 0.1637549991, 5.5503888596, 13.7765342230, 26.6982305096, 44.5852231335, 64.9115406195, 83.5422011632, 97.9566284995, 107.0502880465, 111.5194795359, 112.8555367114, 112.4496624066, 112.1335120661, 112.2153841322, 111.2957250762, 107.2640706799, 99.2589533679, 86.5816322891, 71.4444515480, 55.7742332512, 41.3272063403, 29.5481381184, 20.2162686344, 11.9237026820, 5.0925944488, 1.4615797295, 0.2939363196, 1.0000000000, 1.0000000000, 2.8219111678, 8.0678828994, 17.3290752987, 29.7731106246, 46.2175229485, 64.7315658391, 82.1398653619, 95.7024527211, 104.5617966119, 109.1494663134, 110.5429647729, 110.2067084387, 107.8800277357, 102.8829110903, 94.0065567627, 81.6828966689, 67.1106687421, 52.7663280888, 39.9346228950, 28.7583595397, 20.2066098343, 13.2478607057, 7.1649777296, 2.9966547792, 0.9244588614, 1.0000000000, 1.0000000000, 1.0000000000, 1.5310014514, 4.2504196452, 9.4432637084, 16.7961492395, 27.3636037892, 40.9247072947, 55.4309910368, 69.6993886812, 81.0372630043, 88.2418005616, 91.1000960855, 90.5538912744, 86.1923867723, 78.5043321742, 69.3315652798, 58.8337569739, 47.3754854305, 37.0733082508, 27.3861670597, 19.5844939565, 13.3178910422, 8.3644222673, 3.9536712938, 1.9494250177, 0.4067713472, 1.0000000000, 1.0000000000, 1.0000000000, 0.2350834155, 1.6219434083, 4.6290842960, 9.2040126983, 16.1976501305, 24.7739930987, 34.9501841165, 44.0641311762, 52.0151458028, 56.7022406720, 58.8608028601, 58.2456023076, 54.6539758348, 49.8555849636, 44.9165188195, 38.7566582061, 31.9011532539, 24.7687985064, 17.9995830939, 12.5661141840, 8.0690469073, 4.8684701190, 1.9923911251, 0.6284536145, 0.4245746839, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 0.1551330576, 2.3443606017, 5.1987774678, 10.0933988851, 15.1379052793, 21.3706857165, 26.2088204481, 30.7924495337, 34.1521703939, 35.2895261345, 34.6704047387, 32.7915689479, 29.0600988083, 26.1009510618, 22.2511161450, 18.1069074841, 13.8969862215, 10.3177683595, 6.5525108755, 3.5002764075, 1.2417482188, 0.1149626014, 0.2408644841, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 0.4375236171, 1.7211838444, 2.6011984415, 4.0325934616, 4.7760071000, 6.1778283128, 7.6806603101, 9.0934827578, 9.8982906706, 10.4318954749, 11.4029181995, 10.5172155882, 9.3978172215, 8.0576658924, 6.0424216844, 3.9561558845, 2.8398211679, 1.6867557084, 1.6782693923, 0.3465967878, 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000
};



```

now we have to export images

``` py

NUM_IMAGES_TO_EXPORT = 10

assert X_test.shape[1] == 784
X_sample = X_test[:NUM_IMAGES_TO_EXPORT]
y_sample = y_test[:NUM_IMAGES_TO_EXPORT]

def export_test_images_to_c(images, labels, filename="test_image.h"):
  num_images, num_features = images.shape
  with open(filename, "w") as f:
    f.write(f"ifndef TEST_IMAGES_H\n#define TEST_IMAGES_H\n\n")
    f.write(f"#define NUM_TEST_IMAGES {num_images}\n")
    f.write(f"#define NUM_FEATURES {num_features}\n\n")

    # Write image data
    f.write("float test_images[NUM_TEST_IMAGES][NUM_FEATURES] = {{\n")
    for img in images:
        f.write("    { " + ", ".join(f"{px:.6f}" for px in img) + " },\n")
    f.write("};\n\n")

    # Write labels (as integer)
    f.write(f"int test_labels[NUM_TEST_IMAGES] = {{ ")
    f.write(", ".join(str(label) for label in labels))
    f.write(" };\n\n")

    f.write("#endif // TEST_IMAGES_H\n")

  print(f"Exported {num_images} test images to {filename}")

export_test_images_to_c(X_sample, y_sample, "test_images.h")

```

&nbsp;
<img width="1086" height="749" alt="image" src="https://github.com/user-attachments/assets/a2af95a9-93ee-4928-88c3-0fd256d42149" />

&nbsp;

aferter build we go a ram overvlow error

&nbsp;
<img width="1947" height="872" alt="image" src="https://github.com/user-attachments/assets/c2609cd4-cfdd-4882-b855-d83469ca1f68" />

next we will try to get running it on our board

</details>

## Memory-Constrained ML & Quantization Basics
<details>
<summary>Memory-Constrained ML & Quantization Basics</summary>

- Beating RAM Limits - Quantizing ML Models for Embedded Systems

&nbsp;
<img width="1458" height="884" alt="image" src="https://github.com/user-attachments/assets/9e1a10af-caec-4a80-9c21-08ddf107acc4" />

&nbsp;

<img width="1458" height="884" alt="image" src="https://github.com/user-attachments/assets/c7968603-fcbd-4433-b04d-7a3eba65dc65" />

&nbsp;
<img width="1522" height="969" alt="image" src="https://github.com/user-attachments/assets/f5148a2d-1ec9-49d1-aa9d-1eaa36c927a3" />

&nbsp;
<img width="1526" height="962" alt="image" src="https://github.com/user-attachments/assets/e48784a0-03e6-4155-ba31-455963fae9ce" />

&nbsp;
<img width="1530" height="889" alt="image" src="https://github.com/user-attachments/assets/894f9a92-e8ce-4303-b625-c5095e9ac467" />

&nbsp;

- Quantization Demystified - Fitting AI Models on Tiny Devices

now lets start to modify export of our test images from float to int

```py

def export_test_images_to_c(images, labels, filename="test_image.h"):
  num_images, num_features = images.shape
  images_int8 = images.astype(np.uint8) 
  with open(filename, "w") as f:
    f.write("ifndef TEST_IMAGES_H\n#define TEST_IMAGES_H\n\n")
    f.write(f"#define NUM_TEST_IMAGES {num_images}\n")
    f.write(f"#define NUM_FEATURES {num_features}\n\n")

    # Write image data
    f.write(f"int8_t test_images[NUM_TEST_IMAGES][NUM_FEATURES] = {{\n")
    for img in images_int8:
        f.write("    { " + ", ".join(str(px) for px in img) + " },\n")
    f.write("};\n\n")

    # Write labels (as integer)
    f.write(f"int test_labels[NUM_TEST_IMAGES] = {{ ")
    f.write(", ".join(str(label) for label in labels))
    f.write(" };\n\n")

    f.write("#endif // TEST_IMAGES_H\n")

  print(f"Exported {num_images} test images to {filename}")

export_test_images_to_c(X_sample, y_sample, "test_images.h")

```

&nbsp;
<img width="1168" height="938" alt="image" src="https://github.com/user-attachments/assets/2df31101-bcb7-4406-ad41-aacaef24cd2a" />

imiages now in int_8 format, test_images.h

``` h

ifndef TEST_IMAGES_H
#define TEST_IMAGES_H

#define NUM_TEST_IMAGES 10
#define NUM_FEATURES 784

int8_t test_images[NUM_TEST_IMAGES][NUM_FEATURES] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 185, 159, 151, 60, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 114, 72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 66, 14, 67, 67, 67, 59, 21, 236, 254, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 253, 209, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 233, 255, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 254, 238, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 249, 254, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 187, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 205, 248, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 254, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 251, 240, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 221, 254, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 203, 254, 219, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 254, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 224, 254, 115, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 242, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 254, 219, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 207, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 125, 171, 255, 255, 150, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 253, 253, 253, 218, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 213, 142, 176, 253, 253, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 250, 253, 210, 32, 12, 0, 6, 206, 253, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 251, 210, 25, 0, 0, 0, 122, 248, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 18, 0, 0, 0, 0, 209, 253, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 117, 247, 253, 198, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 247, 253, 231, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 246, 253, 159, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 234, 253, 233, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 248, 253, 189, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 200, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 253, 253, 173, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 43, 20, 20, 20, 20, 5, 0, 5, 20, 20, 37, 150, 150, 150, 147, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 253, 253, 253, 253, 253, 168, 143, 166, 253, 253, 253, 253, 253, 253, 253, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 249, 247, 247, 169, 117, 117, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 123, 123, 123, 166, 253, 253, 253, 155, 123, 123, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 252, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 244, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 202, 223, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 254, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 254, 195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 254, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 237, 205, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 124, 255, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 254, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 232, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 254, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 151, 254, 142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 228, 254, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 251, 254, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 254, 205, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 215, 254, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 198, 176, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 150, 253, 202, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 251, 251, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 197, 251, 251, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 190, 251, 251, 251, 253, 169, 109, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 251, 251, 251, 251, 253, 251, 251, 220, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 182, 255, 253, 253, 253, 253, 234, 222, 253, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 221, 253, 251, 251, 251, 147, 77, 62, 128, 251, 251, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 231, 251, 253, 251, 220, 137, 10, 0, 0, 31, 230, 251, 243, 113, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 251, 251, 253, 188, 20, 0, 0, 0, 0, 0, 109, 251, 253, 251, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 251, 251, 201, 30, 0, 0, 0, 0, 0, 0, 31, 200, 253, 251, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 32, 202, 255, 253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 251, 251, 0, 0, 0, 0, 0, 0, 0, 0, 109, 251, 253, 251, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 217, 251, 251, 0, 0, 0, 0, 0, 0, 21, 63, 231, 251, 253, 230, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 217, 251, 251, 0, 0, 0, 0, 0, 0, 144, 251, 251, 251, 221, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 217, 251, 251, 0, 0, 0, 0, 0, 182, 221, 251, 251, 251, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218, 253, 253, 73, 73, 228, 253, 253, 255, 253, 253, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 113, 251, 251, 253, 251, 251, 251, 251, 253, 251, 251, 251, 147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 230, 251, 253, 251, 251, 251, 251, 253, 230, 189, 35, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 142, 253, 251, 251, 251, 251, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 174, 251, 173, 71, 72, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 224, 0, 0, 0, 0, 0, 0, 0, 70, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 231, 0, 0, 0, 0, 0, 0, 0, 148, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 195, 231, 0, 0, 0, 0, 0, 0, 0, 96, 210, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 252, 134, 0, 0, 0, 0, 0, 0, 0, 114, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 236, 217, 12, 0, 0, 0, 0, 0, 0, 0, 192, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 168, 247, 53, 0, 0, 0, 0, 0, 0, 0, 18, 255, 253, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 242, 211, 0, 0, 0, 0, 0, 0, 0, 0, 141, 253, 189, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 106, 0, 0, 0, 0, 0, 0, 0, 32, 232, 250, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 225, 252, 0, 0, 0, 0, 0, 0, 0, 0, 134, 252, 211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 252, 164, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 204, 209, 18, 0, 0, 0, 0, 0, 0, 22, 253, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 199, 85, 85, 85, 85, 129, 164, 195, 252, 252, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 170, 245, 252, 252, 252, 252, 232, 231, 251, 252, 252, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 84, 84, 84, 84, 0, 0, 161, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 252, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 232, 236, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 179, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 254, 107, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 227, 254, 254, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 254, 254, 165, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 203, 254, 254, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53, 254, 254, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 254, 254, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 196, 254, 248, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 254, 254, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 254, 254, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 252, 254, 223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 254, 254, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 252, 254, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 254, 254, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 254, 234, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 254, 204, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 211, 254, 196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 158, 254, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 157, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 192, 134, 32, 0, 0, 0, 0, 0, 0, 0, 0, 15, 77, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 235, 250, 169, 0, 0, 0, 0, 0, 0, 0, 0, 15, 220, 241, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 189, 253, 147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 253, 253, 21, 0, 0, 0, 0, 0, 0, 0, 0, 43, 254, 173, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 153, 253, 96, 0, 0, 0, 0, 0, 0, 0, 0, 43, 231, 254, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 255, 204, 11, 0, 0, 0, 0, 0, 0, 0, 0, 104, 254, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 253, 178, 5, 0, 0, 0, 0, 0, 0, 9, 131, 237, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 253, 253, 191, 175, 70, 70, 70, 70, 133, 197, 253, 253, 169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 228, 253, 253, 254, 253, 253, 253, 253, 254, 253, 253, 219, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 65, 137, 254, 232, 137, 137, 137, 44, 253, 253, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 254, 206, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 160, 253, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 254, 241, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 158, 254, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 231, 244, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 254, 232, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 253, 157, 0, 13, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 253, 154, 91, 204, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 253, 254, 253, 154, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 190, 128, 23, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 149, 193, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 224, 253, 253, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 235, 254, 253, 253, 166, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 253, 254, 253, 253, 253, 238, 115, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 241, 253, 208, 185, 253, 253, 253, 231, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 254, 193, 0, 8, 98, 219, 254, 255, 201, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 253, 80, 0, 0, 0, 182, 253, 254, 191, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253, 155, 0, 0, 0, 234, 253, 254, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 253, 208, 40, 85, 166, 251, 237, 254, 236, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 238, 253, 254, 253, 253, 185, 36, 216, 253, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 240, 255, 254, 145, 8, 0, 134, 254, 223, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 158, 142, 12, 0, 0, 9, 175, 253, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 253, 226, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 166, 253, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 245, 253, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 115, 254, 172, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 218, 254, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 254, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 186, 244, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 223, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 47, 47, 47, 16, 129, 85, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 153, 217, 253, 253, 253, 215, 246, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 142, 244, 252, 253, 253, 253, 253, 253, 253, 253, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 253, 253, 253, 253, 253, 253, 253, 213, 170, 170, 170, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 132, 72, 0, 57, 238, 227, 238, 168, 124, 69, 20, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 206, 253, 78, 0, 0, 32, 0, 30, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 177, 253, 132, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 133, 253, 233, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 223, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 253, 174, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 253, 246, 127, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 253, 253, 253, 251, 147, 91, 121, 85, 42, 42, 85, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 232, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 53, 218, 222, 251, 253, 253, 253, 253, 253, 253, 253, 253, 252, 124, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 72, 200, 253, 253, 253, 253, 253, 253, 253, 175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 253, 249, 152, 51, 164, 253, 253, 175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 253, 253, 253, 188, 252, 253, 253, 148, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 167, 253, 253, 253, 253, 250, 175, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 180, 231, 253, 221, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 149, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 56, 137, 201, 199, 95, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 152, 234, 254, 254, 254, 254, 254, 250, 211, 151, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 153, 240, 254, 254, 227, 166, 133, 251, 200, 254, 229, 225, 104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 153, 234, 254, 254, 187, 142, 8, 0, 0, 191, 40, 198, 246, 223, 253, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 126, 253, 254, 233, 128, 11, 0, 0, 0, 0, 210, 43, 70, 254, 254, 254, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 243, 254, 228, 54, 0, 0, 0, 0, 3, 32, 116, 225, 242, 254, 255, 162, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 240, 254, 223, 109, 138, 178, 178, 169, 210, 251, 231, 254, 254, 254, 232, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 175, 244, 253, 255, 254, 254, 251, 254, 254, 254, 254, 254, 252, 171, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 136, 195, 176, 146, 153, 200, 254, 254, 254, 254, 150, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 254, 254, 241, 99, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 250, 254, 254, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 242, 254, 254, 211, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 241, 254, 254, 242, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131, 254, 254, 244, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 249, 254, 254, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 228, 254, 254, 208, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 255, 254, 254, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 254, 254, 137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 255, 233, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 113, 255, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

int test_labels[NUM_TEST_IMAGES] = { 7, 2, 1, 0, 4, 1, 4, 9, 5, 9 };

#endif // TEST_IMAGES_H

```

next we have to convert weights to int im svm_model.h

``` py

max_val = np.abs(weights).max()
scale = max_val / 127.0
weights_int8 = np.round(weights / scale).astype(np.int8)
with open("svm_model.h", "w") as f:
    f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
    f.write(f"#define NUM_FEATURES {weights.shape[1]}\n")

    f.write("double weigths[NUM_CLASSES][NUM_FEATURES] = {\n")
    for row in weights_int8:
        f.write("    {"+ ", ".join(str(v) for v in row) + "},\n")
    f.write("};\n\n")

    f.write("double bias[NUM_CLASSES] = {" + ", ".join(f"{b:.10f}" for b in biases) + "};\n")

print(" Exported SVM model to svm_model.h")

```

int version of svm_model.h

``` h

#define NUM_CLASSES 10
#define NUM_FEATURES 784
double weigths[NUM_CLASSES][NUM_FEATURES] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -15, 2, 8, 0, 14, 0, 1, 12, 15, -24, 20, 1, 19, -32, 2, 5, 3, 2, 3, 0, 0, 0, 0, 0, 0, 1, 3, 5, 8, -6, 12, -2, -22, 10, -7, -48, 15, -29, -44, -12, -44, -36, 10, -29, 2, 5, -1, 1, 0, 0, 0, 0, 0, 2, -7, -1, 16, -19, -25, 14, -27, -27, 9, -22, 12, -12, -10, -3, -5, -13, -18, 4, -12, -9, 4, -2, 4, -9, 0, 0, 3, 5, -6, -3, -27, -1, -6, -4, 14, -22, 16, -24, 27, -27, 29, -38, 20, -2, 3, 2, -3, 0, 3, -11, 8, -3, 2, 0, 0, -2, 3, 2, -27, 40, -13, 8, -8, 31, 2, -23, 13, 2, 17, 2, 3, 7, -4, -5, -9, -9, -3, 3, -18, 4, 0, 0, 2, -8, 5, -1, -11, -25, -13, 18, -44, -10, -9, -7, 21, -5, 1, 0, 1, 10, -4, -1, 23, 18, -4, -22, -25, -9, -2, -1, 1, -7, -6, 1, 18, 23, 2, -36, 19, -4, 23, -21, 8, 15, -12, -4, 41, 1, 22, -5, -14, -17, 8, -7, -3, -12, 7, 10, -2, 7, 16, -15, -33, 3, -1, 18, -3, 1, -22, 34, -23, 23, 8, 35, 5, 14, -6, 15, 16, 16, 1, -15, -29, -11, -1, -1, 0, -12, 8, -17, 16, -8, 3, 1, -2, 27, -22, -28, 35, 2, -14, 8, 20, 1, 25, 16, -8, -25, 7, -7, -10, -12, 7, 6, -30, 59, -68, 20, 7, 7, 4, -24, 21, -12, 35, 2, -19, 11, 17, 18, 15, 43, -9, 1, 5, 30, 12, 2, -5, -15, 7, -6, 56, -79, 45, -23, -22, 6, 1, 14, -28, -9, 7, -29, 6, -32, -25, -39, 25, -30, 14, 21, 12, 1, 6, -6, -11, -1, 6, 4, -17, -6, -24, -21, 14, 7, -11, 6, 15, -9, 6, 28, 5, -35, -41, -5, -5, -2, 3, 18, 7, 9, -6, 18, -1, -28, 6, 3, 7, 2, 8, 34, -32, 30, -2, 8, 7, 26, 1, -17, -58, 4, -52, -27, -7, 3, 5, -46, -8, 17, 13, -5, 3, -17, 3, 1, 2, 2, -17, -21, 30, -2, 44, -10, 5, -12, 8, 31, 20, -116, 37, -63, 10, -6, -15, 29, 16, -11, 7, -6, 2, -1, 1, 1, -1, 1, -7, 9, -4, -23, 22, 16, 22, 29, -20, -1, -20, -77, -3, -37, -7, -3, -9, 6, -9, 12, 5, 17, -2, 2, 2, 1, 2, 6, -6, -5, 17, 4, 11, 17, 16, 6, 24, -11, -77, -39, -3, -23, 8, -16, 1, -5, 13, -23, -4, -8, -16, -39, 11, 0, 4, -3, -3, 9, 2, 19, 6, -10, 14, 15, -32, 40, -58, -22, -20, -11, 10, 4, 7, 24, -15, 36, -4, 11, 7, -5, 5, 1, 0, 4, -9, -5, -4, -14, -8, 26, 17, 14, 16, -19, -56, -11, -3, -24, -6, -11, -9, -31, -4, -13, -6, -5, -6, -3, -10, 2, 1, 2, -14, -12, -9, 14, 18, 4, -18, 9, -4, 42, -25, -11, -6, 4, -9, -1, -22, 16, -6, 12, -11, 5, -15, 6, 6, 0, 5, -12, -1, 16, 13, -11, -6, 17, -11, 23, 23, 6, 14, -34, 14, -36, 2, 18, -3, -11, 16, -6, -5, -4, -5, -7, 3, 2, 5, -7, 2, -3, -11, 3, 1, -24, 16, 12, 26, 6, -2, -6, -13, 28, -65, -9, -10, -20, -32, 11, -25, 17, -30, 19, 3, 2, -3, 2, -4, -13, 3, 7, -6, 8, 17, -7, 1, -2, 52, -20, -5, 7, 24, -34, 17, 24, -11, 23, 3, -12, 8, 2, 3, 0, 0, 6, -4, -22, 1, -1, 10, -22, 7, 18, 13, 13, -5, 27, -1, -31, 30, -26, -26, -7, -24, 0, -2, -4, 8, -2, 0, 0, 0, -6, 1, 11, 4, -6, -2, -7, 12, 15, -39, 17, -6, 4, 6, -35, 28, 14, -15, -17, -14, 13, -13, 7, -20, 2, 0, 0, 0, 4, 5, -14, -12, -32, -2, -24, -13, -89, 43, -78, -31, -41, -43, -33, -4, -89, -23, -15, -8, -12, -6, 1, 3, 2, 0, 0, 0, 0, 0, 10, 10, -2, -17, -8, -18, -21, -33, 2, -31, -15, -37, -32, -36, -21, 21, -2, -6, 3, 5, 0, 3, 0, 0, 0, 0, 0, 0, 5, -3, 3, 0, 5, -2, -1, 3, 3, -5, 2, 3, 2, 12, 2, 9, 3, -5, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, 2, 4, -1, 3, 3, 6, 3, 5, -2, 5, -11, 0, 6, 4, 2, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 8, 8, -3, -4, -3, -3, -22, 38, -29, 30, -1, 23, -15, 21, -13, 4, -1, -4, 6, -2, 0, 0, 0, 0, 4, 3, 5, -46, 5, 15, -18, -12, -12, -36, -10, -3, -30, 0, -15, 3, -29, 4, -11, -4, -1, -3, -2, 13, 3, 0, 0, 4, 11, -13, -1, 12, -9, -35, 35, -10, 6, -4, -7, 2, 5, -15, -2, -2, -10, 5, -3, 4, -8, 4, -16, -38, 12, -3, 0, 0, 0, 11, 2, 20, 3, -14, -20, -22, -9, -35, -29, -11, -26, 13, -14, -23, -7, -4, -5, 5, 15, 0, 20, -18, 1, 2, 0, 5, -9, -1, -4, -39, 30, -20, -16, -33, -3, 13, -20, -7, -11, -32, -22, -25, 3, -15, -6, -5, -17, 12, -21, -36, 6, 5, 4, -11, -7, -26, 54, -15, -32, 33, -45, 3, -7, -32, 13, -25, -1, -26, 10, -6, -15, 1, 19, 15, -12, -7, -18, -28, 18, -5, 1, 2, 10, -12, -3, 2, -43, -11, -9, -44, -1, -20, 5, 7, -5, 2, -30, 20, -1, 7, -30, -12, 9, -7, -44, -23, 11, 4, 1, -23, 11, 6, 9, -46, -1, 40, -2, -3, -8, 1, -6, 7, 11, 23, 5, -21, 10, -17, -9, -4, 1, -32, 15, -52, 9, -3, 2, 16, 6, -16, -22, -20, -38, -24, 12, -9, 7, 8, 5, 15, 33, 27, 17, 18, -32, -2, -23, -3, 5, 2, -46, 6, -13, 15, 5, -1, 12, -18, 17, -4, -31, -35, -23, -32, -7, -9, -22, 17, 15, 41, 10, -4, 2, -5, 7, 5, -26, 3, -17, -11, -1, 0, 4, 6, -34, 10, -1, -31, 29, 20, -21, 12, -20, -21, 7, 7, 41, 29, -9, 8, 7, -24, -2, -12, -19, 29, 14, -9, -16, 1, 4, 5, -7, 1, 0, -18, 37, 17, 1, 8, -42, -21, 1, 1, 28, 7, 17, 23, -23, -3, -2, 1, -13, 14, -32, -29, 2, 0, 1, 2, -2, -8, 32, 1, 12, -26, 11, -21, -27, -24, -11, 4, 26, 17, -4, -3, -56, -3, 1, -11, 10, -12, 40, 2, 0, 2, 1, 2, 7, 11, -1, -37, -24, -3, 16, -15, -46, -5, 19, 19, 24, -7, 32, -22, -1, -30, -11, -10, -12, -15, -5, -16, 6, -1, 2, 2, 1, -7, 19, -6, 3, -9, -34, 28, -13, -2, -31, -4, 21, -27, 5, -48, -28, -26, -14, 5, -14, -20, 25, -16, -7, 6, 0, 4, -2, -10, 6, -18, -1, 22, 6, -20, 0, -20, 15, 25, 24, 13, 0, -37, -6, 11, 5, -9, 30, -19, 16, -26, 1, -1, 0, -2, 4, 0, -13, -25, -57, -39, 1, -4, -35, -9, -4, -11, 20, -19, -62, 21, -2, -18, 2, 2, -13, -36, 23, -4, 0, 2, 4, 1, 1, 5, 1, -53, 10, 19, -23, 14, 5, -14, -18, 8, -14, 18, 29, -41, -10, 10, -2, 14, -11, -6, 6, -6, 1, 3, 0, 0, 6, -21, 10, -96, 1, -20, 3, 11, -3, 26, 23, -7, -5, -34, 9, 8, 33, 19, 2, -19, 14, -1, 3, -7, 11, 2, 4, 2, -5, 17, -1, 19, 22, 17, 12, -14, 25, -20, -4, 3, 6, 10, 11, -5, 17, -31, -3, 37, -20, -65, 48, -5, -2, 3, 4, -2, 2, -5, 6, 44, -15, -13, -1, 10, -14, -8, -28, -31, 9, -6, -5, -1, 45, -3, 21, 14, -33, 16, -21, 28, 0, 3, 0, 0, 9, -17, -18, -5, 9, 6, -10, -2, -14, -15, 9, -2, -31, 12, 4, 23, -37, 3, -12, -10, 30, -34, -11, 2, 7, 0, 0, 0, 3, -1, -4, -3, -2, 4, -16, 2, -8, -9, -12, -24, -16, 0, -41, 16, -9, 31, -3, 2, -17, 16, -2, 4, 16, 0, 0, 0, 2, -2, 6, -20, 7, -32, -18, -28, -59, -68, -28, -32, -11, -52, -41, -33, -38, -24, -10, -10, -1, 3, 3, -2, -10, 0, 0, 0, 0, 0, 4, -7, -6, 0, -8, -16, -5, -13, -54, -38, -29, -12, 4, -14, -1, 0, -4, 12, 2, -11, 4, 2, 0, 0, 0, 0, 0, 0, 2, 6, -10, 11, -3, 5, -1, -8, 14, 5, 7, 8, -5, 10, -19, 9, -2, -1, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -2, 1, 1, -3, 2, 2, -6, -5, 2, -3, 5, -4, 1, -22, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -3, 5, -4, -1, 3, -10, -1, 0, -6, -5, 0, -8, -4, -3, -8, -8, 1, -11, 7, 4, 1, 0, 0, 0, 0, 1, 3, -2, 1, -3, 4, -6, 8, 2, 1, 0, 5, -2, -6, 3, -2, -9, 0, 0, -3, 8, -14, -11, 5, 0, 0, 0, 0, -1, -2, -2, -2, 8, 5, -6, 10, 3, 1, 9, 0, 14, -5, 6, -10, 0, -9, -5, -7, -15, -9, -35, 16, -5, -1, 0, 0, 0, 1, 3, -1, -5, 0, 7, -7, 3, 7, 4, 14, 1, 6, 16, 5, 9, -3, 1, 3, 3, 9, -17, 6, 1, 0, 0, 1, -2, -3, -3, -1, -5, 3, -4, 3, 8, -3, -2, 6, -4, 9, -1, 6, -5, -1, -12, -3, -4, -9, -8, -12, 5, -4, 1, -5, 7, 9, 2, 2, 11, -11, 10, -9, -10, 12, -6, 11, 7, -10, 7, -19, -3, -9, -8, 6, 10, -26, -19, 28, -12, 6, -1, 5, -4, -16, 1, -8, -1, 9, 4, -1, 12, -2, -4, -3, 9, -2, -7, 9, -6, 7, 10, -2, -10, 19, -28, -46, 3, 6, 1, -6, 1, 3, 0, 21, -31, 6, -16, 7, 3, 10, -7, 0, -6, 8, 15, -9, 6, -1, -7, -11, 11, 5, -28, -4, -3, 1, 1, -1, -9, 15, -14, 1, 27, -17, 8, 2, -9, 7, 4, -4, 12, -14, -8, 19, 12, -9, 6, 6, -7, -8, -4, -16, 16, -3, 1, -14, -12, -14, 3, 7, 5, 9, -28, -36, -29, -28, -37, -33, -25, -5, -2, -17, -4, 10, -16, -4, 0, 15, -14, 6, -18, 1, 1, 6, 15, -8, 7, 3, -54, -47, -8, -14, -14, -36, -19, -13, -42, -28, 2, -14, -10, 20, 7, 9, -3, -45, 0, 6, 7, -7, 0, 1, -1, -1, -27, -12, -5, -32, -24, -14, -13, 1, -7, -2, -3, -8, -20, 2, -4, -19, -10, -11, -9, -5, 16, 1, 6, 3, 1, -7, 5, -8, -1, -46, -29, 20, -10, -7, -6, -1, 7, -11, 8, -7, 7, -11, 5, -5, 6, -6, -7, 5, -16, 2, 8, 4, 1, 1, 0, 7, 7, 45, 11, -24, 2, 1, 3, 6, -4, 12, -4, -4, 5, -10, -15, 9, -18, -7, -9, -22, 17, 0, 3, 3, 3, 0, -10, 3, -2, -18, 19, 14, -7, 6, 7, 3, 13, 11, -6, 3, 9, 20, -11, 2, -8, 21, -2, 16, -5, 6, 9, 1, 0, -1, 9, 1, 4, 11, 15, -10, 10, 2, 5, 6, 10, 13, 0, 19, 0, -11, 12, -8, 1, -13, 0, 0, -5, 11, 1, 4, 2, 1, -7, 0, 0, 18, 4, 20, 7, 0, 11, -8, 16, 10, 11, 4, 12, 7, -5, 7, 1, 10, 5, 7, 16, 3, -1, 0, 0, -1, -5, -8, 4, -12, 2, 9, -3, 16, 10, 8, 1, 16, 11, 6, 0, 12, 3, -7, 2, 1, 6, 5, -6, 11, 7, 0, 0, 0, 6, 4, 9, 11, 10, 18, 17, 6, -1, 21, -6, -3, -3, 3, 9, -18, 8, 9, 4, 9, -4, 2, 16, 1, -6, 0, 0, -1, -3, 1, -13, -1, -6, 0, -16, 17, 5, -4, 0, -4, -3, -12, 5, 14, -7, 18, -3, 9, 3, 12, -2, 6, 5, 0, 0, 0, -1, -2, 8, 3, 5, 9, 18, 7, -2, 9, 0, 5, -7, -9, -1, -5, 14, -6, 11, -2, 5, -1, 6, -3, -7, 0, 0, 0, -3, 3, 2, -6, 6, -7, -9, -8, -7, -9, -1, -16, 3, -19, 7, -7, 4, 9, -2, 8, 2, 5, -2, 0, 0, 0, 0, 0, 0, -18, -1, -2, -6, -9, 2, -3, -6, -3, 1, -12, 15, -13, -10, 4, 1, 3, -3, 5, 3, -7, 8, 9, -7, 0, 0, 0, 1, 3, 3, -85, 20, -28, -4, -40, -17, -26, -18, -18, -15, 15, -24, 10, -8, -11, 6, -6, -9, -8, -12, -5, 6, 0, 0, 0, 0, 2, 4, -2, 3, -16, -13, -6, -9, -30, -7, -24, -25, -29, -31, -38, -18, 11, -11, -9, -10, 0, 6, 1, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 5, 3, -5, -12, 2, 7, 11, 6, 13, 12, 5, 1, 6, -2, 4, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 4, 3, 7, 0, 7, -4, -1, 3, -2, 8, 4, -1, 6, -1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 5, 1, 1, 3, 2, -8, -14, 8, 2, -10, 26, -28, 13, 0, -13, -4, -2, -1, -8, 13, 8, -6, 0, 0, 0, 0, -4, 3, -3, 13, -19, 4, 9, -13, 2, -1, 18, -24, 22, 0, 9, -3, 16, -8, 24, -2, -24, -68, 5, 9, 1, 0, 0, 1, 2, 1, -3, 4, 3, -4, 2, 2, 9, 1, 2, 5, 1, 2, 9, -8, 13, -9, 8, -15, 22, -44, 7, 6, 0, 1, 0, 0, -2, 2, 2, 1, 5, 7, 0, 4, -1, 6, 9, 1, 3, 1, 12, -15, -4, 6, -6, -11, -17, 9, -13, 9, 2, 0, 0, 4, 0, 3, 1, -5, 8, -8, 4, 5, 12, 7, -4, 16, -6, 8, 6, -3, 19, -23, 5, 11, 4, -11, 1, -33, -11, 6, 1, 4, -3, 3, -1, 7, 2, 1, 2, 2, -5, -4, 6, -7, 17, 9, -6, 12, -19, 5, 4, -26, -20, 7, -10, -15, 9, 2, 1, -11, -4, -2, 6, -2, -3, -2, -2, -16, 1, -5, 8, 2, -2, 1, 5, -1, 21, 2, 1, -2, 11, -27, -22, -24, 21, 3, -1, 12, 0, 1, 7, -2, 7, 0, -5, 5, 1, -21, -11, -13, 2, 17, 8, 9, 0, 11, 17, 12, 21, -12, -94, 23, -8, 3, 0, -7, -13, -1, -7, 7, -3, 11, -6, -25, -34, -17, -26, 16, 12, 5, 15, 10, 7, -1, 16, 10, 6, 46, -59, 22, -14, 1, 0, -1, 0, 15, 8, -6, -1, -27, -20, -17, -19, -8, 2, 1, 10, 17, -3, 15, 8, 3, -2, -15, -3, 16, -103, 1, 0, 0, -1, 5, -11, -6, -6, -3, -23, -17, -16, -5, 8, -13, 18, -5, 24, 11, 14, 1, 4, 3, 12, -21, -25, -86, -73, 39, 7, 1, 0, -1, -1, -1, -8, 2, 15, -3, 1, -8, -15, 5, 14, 3, 18, -17, 15, 3, -1, -7, -26, -19, -55, 32, 43, -67, 6, 2, 0, 4, -6, 13, -2, -6, -22, -8, -4, -1, -9, -10, 22, -7, 4, 1, 7, 1, 7, -17, -16, 15, 4, -8, -11, -11, 3, 0, 0, 2, -4, -8, -6, -2, 0, -10, -13, -9, -10, 1, 8, -1, 1, 11, 2, 0, -20, 4, 30, 0, 15, 26, 0, -49, 26, 2, 1, 1, 9, 4, 12, 6, -9, -5, -34, 7, -7, 4, -6, 9, -4, -8, -21, -15, 23, 12, 5, 10, -4, -4, 13, -64, -8, 2, 0, 1, -5, -1, 7, -9, -11, 18, 0, -41, -30, -28, -7, -16, -19, -22, -5, 28, -7, 23, -5, 0, 11, 13, -8, -48, 3, 1, 0, 1, 5, 11, -13, 13, 6, -15, 5, -18, -26, -23, -28, -27, -29, 15, 25, 1, 6, 2, 6, 12, 21, 2, -15, -27, 5, 0, 1, -2, -2, 7, 3, 13, -10, -4, -2, -8, -2, -20, 1, -17, -8, -15, -1, 17, 5, 13, -5, 4, -11, -9, 1, -29, 7, -1, 0, 3, -5, 14, -1, 10, 4, 9, -4, 16, -8, 3, -16, -3, 8, -8, -5, -2, 13, -15, 9, -1, -3, 6, -5, -30, 6, 0, 0, -1, 5, -2, 8, -4, 5, -3, 14, -20, 5, -13, -3, -4, -6, 0, 18, -13, 11, 12, -2, -5, -4, -7, -23, 17, -3, 1, 0, 1, -4, 4, -1, -6, 3, 2, -12, 9, -4, 12, 1, 0, -9, 0, -1, 12, -7, -1, 8, -6, 3, 5, -32, -10, -3, 1, 0, 0, 8, -1, 6, 11, 6, 11, 1, 11, 2, 8, -7, 16, -9, 1, -13, -5, 18, 0, -1, 3, -12, -6, 11, -9, 2, 0, 0, 0, -8, 11, 5, -9, 13, 5, 4, -2, 11, 9, -7, 6, 4, -2, 4, 6, -17, -4, -4, 9, 0, 16, -21, 3, -4, 0, 0, 0, 6, -7, -6, 10, 0, -4, 14, 4, 10, -4, 20, -1, 9, -1, 3, -2, -4, 10, -7, -9, 7, -72, 13, 2, 4, 0, 0, 0, 0, 1, -31, 2, -5, 1, 1, -16, 2, -7, -12, -11, -7, -15, -6, 0, -4, -5, 7, -3, 0, -1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 2, 1, -5, 1, -2, -1, -1, -3, -15, -7, 0, 1, -2, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, -8, 7, -1, 9, 2, 4, -4, -32, 16, -24, -4, 0, -3, 2, -15, 0, -4, 1, 0, 0, 0, 0, 0, 0, 0, -3, -2, -3, 5, -21, -7, -13, -39, -31, -13, -2, 0, -8, -22, -13, -30, -4, -8, -6, 0, -14, 4, 1, 0, 0, 0, 0, 1, 1, -2, 11, -18, 15, -26, -29, -5, -28, -15, -92, -15, -34, -10, -7, -39, 9, -12, -2, -5, 4, -15, -3, 0, 0, 0, 2, 5, -13, -1, 7, -25, 23, -18, 13, -33, -7, -15, 24, -34, -25, -9, -10, -20, -2, 9, 0, 12, 0, 8, 2, -2, 1, 0, 0, -6, 8, -5, 2, 3, -4, -4, -2, -4, -7, 1, -24, 16, 1, 20, -1, 11, 12, 3, -1, 1, 1, -6, -1, -6, 1, 0, 3, 1, -12, 10, 0, -10, 3, -3, -10, -9, -30, -24, -29, -36, -41, -32, -26, -33, -13, -13, -1, -3, 17, -4, -4, 3, -5, 3, 1, -26, -5, -13, 21, -5, 15, -7, 9, -10, 13, -26, -13, -27, -29, -26, -5, -13, -1, 0, 4, 6, 15, -8, -8, 0, -13, -1, -22, -1, 20, 5, 6, -13, 10, -30, 9, -13, -13, -10, -12, -29, -14, -31, -9, -2, -10, -10, -1, -21, -9, 3, 3, -3, -3, -1, -5, 15, -14, -23, 15, 1, -1, -4, -12, -20, 4, -27, -1, -36, -23, 8, -14, 5, 1, -5, 6, 0, 13, -26, -9, 1, -4, 0, 3, -3, -19, 20, -26, -25, 12, 9, 12, -10, -5, 8, -10, -32, 0, -27, 11, -12, -10, -6, 2, 11, -22, 18, -4, -16, 3, 6, -27, -7, 16, -16, 7, 11, -16, -17, 14, 16, -8, 16, -46, -49, 17, 16, 2, -1, 20, -18, -15, -19, -9, -35, -4, -9, 5, 5, -2, -1, -4, -10, 1, -9, 15, 14, -5, 27, 12, 57, -2, -25, 11, 20, -8, -5, 0, 0, 11, 5, -19, 22, -3, -9, 0, 2, 4, -2, -7, -7, -7, 7, 2, 16, 28, 22, 33, 18, -21, -13, 14, 3, 17, 14, -21, 10, 18, -15, -2, -28, 1, 9, -5, 1, 11, -37, 0, 16, 12, 10, 31, 5, 10, 9, 21, 18, -20, 2, 0, 22, 19, 0, 30, 1, 3, 15, -19, 7, 6, -22, 0, 1, 4, 22, -5, 14, -8, 4, 2, 14, 16, 13, 15, 14, -15, 5, 14, 21, 14, -10, 20, -4, -23, 22, 28, -48, -4, 16, -4, 0, -4, -39, -3, -18, -3, 1, 14, 23, 6, 4, -10, 22, -11, 17, 24, 8, 17, 6, 0, 29, 13, -13, -11, 11, -12, -7, 6, 0, 3, 41, -12, 10, -8, -1, 12, 5, 15, -7, 3, -14, 10, 28, 8, 17, 15, 30, -15, -13, 14, -15, 28, -36, -4, 4, -6, -2, 4, -34, -8, -11, -8, 11, 2, 5, 21, -32, -26, 32, 33, 5, 5, 12, -29, -20, 17, 2, 6, -14, -6, 23, -29, -15, 9, 3, -5, 8, 22, 1, 28, -22, -15, -10, -30, -21, 23, -53, 4, 25, -7, -9, 6, 0, -17, -21, 38, -63, 3, -14, -15, 8, -2, 0, 4, 7, -10, -1, -67, 20, 1, -58, 9, -41, -4, -2, -26, 7, 6, 23, 2, 4, -10, -15, -8, 7, -5, -13, -12, 12, -1, 1, 7, -14, 9, -21, 78, -51, -57, 50, -17, 8, -14, -11, -13, -14, 9, -25, 15, -8, 1, 37, -25, -1, 9, 2, 9, -4, 2, 1, -2, 29, -23, 6, -31, -17, 0, -20, -29, -10, -5, -16, -7, -22, -12, -11, 5, -7, 19, -32, 24, 16, -6, 0, -9, 6, 2, 0, 0, -5, -10, 1, -24, 2, -2, 2, -10, -1, 1, -11, -2, -3, 4, 11, -6, 6, -8, 6, 7, -22, -1, -1, 2, 4, 0, 0, 0, 2, 11, -36, 3, 23, -19, 17, 16, -17, 17, -10, -6, 2, -12, -2, -4, -7, 9, -7, 1, 3, -1, -2, 11, -1, 0, 0, 0, 2, 4, 11, -9, -5, -11, -9, -19, -2, -17, -18, -11, -17, -9, -16, -13, -6, -18, 0, -12, 1, 2, -8, -2, -1, 0, 0, 0, 0, 1, -4, 7, 2, -32, -8, -21, -17, -30, -18, -30, -26, -23, -19, -19, -13, -9, -48, -7, -6, -2, 3, 2, 0, 0, 0, 0, 0, 0, -1, 5, 1, 2, -11, -2, -11, 3, -10, 5, -34, 1, -7, 5, -5, -2, 17, -1, -1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -3, 1, 3, 12, -3, 1, -3, 0, 8, 1, -2, 5, 2, -4, 3, 2, 2, -1, 0, 0, 0, 0, 0, 0, 1, -14, -18, 4, 7, 8, -14, -12, 5, -39, -17, -33, -35, -5, -5, -5, -2, -2, -10, -2, -5, 7, -27, 1, 0, 0, 0, 0, 1, 0, 12, -15, -2, -16, 16, -31, 26, -35, 15, -13, -2, -9, -12, -3, -7, -4, 1, -5, 3, -6, -1, 0, 0, 0, 0, 1, -3, -3, -22, -12, 0, 3, -18, -2, -9, 1, -3, -22, -3, 2, 1, -13, 8, -1, -13, 9, -4, 2, 7, -2, 1, -2, 0, 0, 1, 9, -29, 2, -14, 3, -3, -1, -1, -3, -9, -3, -9, -9, -5, 6, 4, -7, 18, -3, 13, 1, 1, 0, 2, 0, 0, 0, 4, -27, 12, 8, -2, -3, -10, 0, -4, -5, -2, -8, 10, -13, -8, -7, 0, 10, 5, -4, 4, 4, 14, -2, 2, 0, 1, 0, -4, 3, -28, -7, -1, -7, 10, -7, 14, 5, 3, -13, 11, -23, 4, 0, 5, -5, -1, 8, 13, 2, 6, 15, -7, 3, 2, 5, -6, -18, -10, 0, -9, -4, 7, -3, 5, 7, -4, -6, -10, -1, -7, 1, -10, 5, 8, 10, -15, 19, 14, 5, 3, -2, 0, 2, -5, -35, -18, -8, 9, -6, -2, 4, -3, 18, 2, -4, -8, -24, -21, -5, -9, -2, -12, 20, 20, -7, 17, 9, -1, 1, 0, 1, 5, -34, 11, -7, -15, 17, 23, 19, 14, 18, 19, 8, -6, 4, -19, -22, -27, -27, -21, -7, 8, 15, 26, 7, 8, -4, 3, -8, 22, -36, -3, -2, 26, -9, -2, 11, 8, 10, 5, 15, -6, -5, -5, -10, -6, -17, -52, -53, -57, -12, 41, 12, -7, 2, 2, -2, -7, 13, 0, -2, 8, 6, 9, -10, 10, 13, 7, 16, -11, 8, -14, -15, -15, -21, -3, -30, -44, -50, -127, 27, 18, -4, 1, -3, -14, -5, 4, 11, -7, 13, 1, 11, 0, 10, 17, 6, -19, -5, -21, -15, -4, 6, 1, -7, -10, -51, -18, -21, 9, 0, 3, -6, 5, 10, -7, -2, -12, -9, -9, 0, 21, -4, 13, -4, -8, 4, -16, -11, -11, -20, 1, -14, 7, -1, 31, -58, -10, 1, 2, 2, -3, -13, 22, -14, 4, -8, 16, 11, -3, 5, 5, -16, -17, -13, -5, -16, 17, -6, -4, -4, -18, 5, 15, 6, -5, 0, -2, 2, 7, 9, -11, 5, -22, -33, -20, -11, 5, -8, 8, 2, -18, -3, -22, -3, -23, -3, 2, -14, 27, -1, -8, -4, -30, 2, 0, 3, -10, -5, 0, -5, 38, -12, -10, -26, -27, -20, -32, -24, -25, -7, 3, -1, 19, -2, 5, 23, -20, 7, 16, -15, 1, 0, 0, -2, -11, -3, 8, 6, 8, 9, -11, -3, -25, -18, 6, -2, 10, -10, -5, 15, -14, 4, 0, -13, 21, 0, -2, -14, 6, -1, 2, -2, 5, -2, 8, -10, 14, -3, 18, -7, 9, -8, 0, 3, -8, 21, -1, -14, 7, -5, 12, 10, -9, -7, 18, -15, -2, 1, 0, 2, 1, -5, -2, 5, 0, -5, 8, 14, -3, 23, 0, 7, -13, -14, 5, 12, -2, 8, -11, -15, 18, 9, -19, 12, 0, 1, 1, 0, -5, 1, 6, 3, 5, 0, 2, 7, 5, 11, -10, 3, 3, 3, -10, 9, -4, 0, 19, 18, 6, 2, 5, -3, -4, 1, 1, 1, -9, -12, 1, -6, -13, 11, -2, 11, -8, 5, 14, 5, 0, -5, 6, -19, 24, 6, -2, -4, 5, 2, 1, 0, -2, 1, 0, 0, 7, 3, 4, 9, 5, -17, 9, -9, 14, -2, 8, -7, 13, -9, 13, -6, -18, -6, 12, -1, -2, 6, -18, 15, 0, 0, 0, 0, -29, -1, -9, 4, -8, 6, -7, 17, -6, 3, -4, 5, 5, 3, -9, 6, 7, -9, 1, -4, 2, -12, 5, -36, -2, 0, 0, 0, 5, 2, 5, -12, 2, 2, -7, 7, -7, 6, 1, 4, -9, 6, -2, -1, -1, 7, -18, 2, 5, 1, 10, 2, 1, 0, 0, 0, 0, 1, 1, -6, 6, -31, -18, 0, -9, 0, -11, 0, -13, -3, -4, -6, -8, -8, 14, -9, -2, 0, -2, 2, 0, 0, 0, 0, 0, 0, 1, 8, -2, 6, 0, -6, 3, 0, 0, -5, -10, -3, -7, -1, -6, 4, -1, -6, 1, 2, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 8, -6, 4, 1, -1, 1, 1, 3, -3, 3, -3, 2, 6, 4, 4, 5, 4, 2, 0, 0, 0, 0, 0, 0, 3, -3, 6, 3, 0, -1, 1, -1, 10, -3, 8, 5, 4, 0, 10, 2, 7, 3, 4, 11, 0, 11, -4, -1, 0, 0, 0, 0, 1, 1, 3, -7, 6, 6, -1, -1, -2, 3, -6, 1, -3, 8, -5, 4, 4, 10, -1, 2, 2, -1, 9, -4, 4, 0, 0, 3, -4, -3, 0, 6, -17, 6, 2, -12, -2, -8, -19, -6, -6, -18, -7, 8, -19, -2, -1, -5, 1, 0, -13, 2, -40, 17, 0, 0, 1, -10, 0, -10, 14, -34, 29, -32, 19, -34, 11, -37, -7, -13, -21, -14, 17, 4, 15, 10, -13, 10, 11, -7, 1, -5, 0, 0, -1, 4, -3, -1, -7, 10, -15, 6, -25, -9, -4, -2, -26, -1, -23, -3, -32, -21, -27, -14, 6, 1, -28, 11, -10, 13, 1, 7, -2, -23, 7, 29, -36, 22, -15, 10, -1, -13, -17, -19, 1, -11, -21, -12, -9, -7, -15, -5, -31, -21, 39, -2, -6, -8, 1, 2, 5, -18, -26, -13, -4, -8, -25, -22, -8, 1, -15, -6, -20, -24, -9, -17, -45, -39, -52, -18, -5, -44, -52, -10, 16, 0, 0, 2, 18, -9, 38, -8, -18, 22, 22, 0, -10, -9, 3, -28, -1, -30, -33, -43, -34, -21, -4, -4, -23, 13, -13, -21, -19, 6, 1, 5, -16, -16, -14, -21, -5, -18, -5, -14, -13, -2, -17, 4, -16, -6, -15, -34, 2, -48, -6, -19, -32, 0, 11, -10, 5, -27, 3, -1, 26, 6, 1, 54, -7, 9, -11, 3, 10, -20, 6, 17, 3, -37, 14, -25, -16, 18, -16, 9, -5, 18, -6, -5, -8, 20, 2, 1, -18, -17, -11, -44, 2, -2, 22, 15, -20, 33, 2, 2, 0, -20, -4, 15, -5, -19, -15, 9, 19, -3, 3, 4, -8, -16, 1, 0, -5, -21, 6, 27, 31, -7, -11, 0, 29, -13, 14, 1, -11, 31, -15, -7, -6, 3, 1, 20, -22, 34, 3, -8, -5, 5, 0, -3, 22, 29, -21, -3, 3, 14, 17, 14, 5, 29, 7, -4, -7, -10, 19, -18, -1, 1, -3, -15, 37, -5, -2, 7, -5, -8, 0, 0, -11, -6, -39, -10, -3, 12, 5, 5, 11, 10, 6, -1, 27, 7, -9, 10, -1, -19, 30, 23, -24, 11, -12, 6, -5, 2, 3, 2, 12, -20, 41, -10, 14, 15, -24, 30, -9, 39, -17, -10, -3, 8, 2, -19, -4, 10, -20, -4, 12, -16, 6, -6, -6, 0, 0, 3, -1, 17, -64, 29, -22, -16, 44, -9, 32, 26, -3, 6, 25, -22, -15, 10, 7, 13, 22, -8, 9, 6, 5, -6, -7, -4, 3, 1, -4, -2, 10, -31, -9, 39, -19, 20, 9, 3, 36, 6, -10, 13, 16, -1, 18, -13, -11, 8, -13, 6, -20, 7, 5, 1, 0, 3, 26, -91, -3, 7, 5, -15, 9, 10, 16, 32, 33, 2, 12, 14, 5, 15, 0, 16, 8, 2, 8, -3, -2, -7, -13, 1, 0, -1, 8, -9, 18, -1, -18, 7, -9, 7, 33, -9, 29, -8, 41, 0, 21, -15, 12, -5, -8, 0, -14, 5, -14, 2, 4, 2, 1, 6, -4, -5, -14, -24, -27, 2, 12, -1, -3, 25, 33, 26, 33, 14, 4, 25, 8, 0, 20, -30, 13, -13, 16, -10, 3, 1, 1, 2, -8, -7, 5, 2, -65, 39, -11, 6, 21, -6, 6, 6, -8, 30, -1, 18, -12, -4, 7, 18, -3, -3, -43, -9, 8, 1, 0, 0, 8, 6, 14, 19, -5, -30, -13, 8, -26, -19, -8, 3, -7, -8, 6, -33, 28, 4, -1, -11, 3, -25, 8, -20, -17, 0, 0, 0, -6, 8, 8, 2, 0, -23, -32, -13, -30, -39, -10, -29, -11, -32, 13, 1, 0, -34, -3, -21, 0, 8, -1, 4, 0, 0, 0, 0, 3, 1, 2, 10, 15, 17, 6, -3, 16, 13, 2, -28, -26, -13, -30, -7, -5, 6, 3, 17, 8, 3, 1, 3, 0, 0, 0, 0, 0, -2, 7, -3, 8, 19, 2, 17, -6, 1, 32, 32, 22, -14, 19, 4, 4, 4, 2, -2, -6, 2, 0, 3, 0, 0, 0, 0, 0, 0, 1, 3, 3, 2, 9, 6, -7, 12, 11, 2, 6, 8, 17, -6, 8, -5, 8, 1, 1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -1, 5, -3, -5, -3, 3, 11, -15, -7, 11, -5, 2, 4, 5, -2, 3, -3, 4, -1, 0, 0, 0, 0, 0, 0, 4, -2, 4, 4, -3, 3, 12, 16, -5, -10, 25, 1, 12, 12, -3, 7, 13, 5, 16, 3, -1, 2, 3, 6, 0, 0, 0, 0, 3, -4, 10, -10, 19, -9, 8, -17, 17, -20, -27, -49, 5, -34, 5, 0, -15, 23, -2, -14, 21, -10, 2, 4, 4, 0, 0, 3, 2, 9, 12, -34, -7, -23, -14, -21, -64, -21, 4, 5, -22, -25, -36, -20, -40, -11, -2, 22, -12, 8, 3, 0, 5, 1, 0, 0, 2, -11, -37, 43, -11, -5, 11, 15, -7, -15, -4, -8, 1, -14, -1, -56, -9, -63, -36, -17, -35, -32, 26, -3, -3, 3, 0, 0, -1, 9, 1, -7, 13, -11, 24, -10, 13, 16, 15, -14, 9, -14, 5, 7, -5, 10, -18, 7, 3, -38, 1, -13, -28, 6, 3, 2, -4, 3, -1, -2, 9, 11, -16, 21, 19, 2, 16, 11, -14, -5, -1, 11, 13, -7, 15, 23, -4, 7, 5, -8, 1, -3, -4, 6, 7, 4, -2, 1, -4, 37, 16, -2, 11, 15, 21, -3, 2, 19, 5, 3, 13, 9, 13, -15, 4, 14, -16, -11, -6, 3, 5, 4, -5, 2, 13, 5, 1, -12, 11, 15, -6, -3, 46, -4, 31, 3, 6, 29, 14, 4, -8, 17, -6, 6, 7, -11, -58, 18, 5, -4, 12, -7, 9, 6, 5, 14, -8, 11, -1, -8, -20, 46, -3, 22, 34, 3, 5, 16, 22, 1, 33, -7, -13, 11, -2, 7, 3, -7, 12, -1, -1, 6, -2, 4, -29, 2, 18, 18, -17, 24, 16, 17, 2, 11, 15, 3, 14, -7, 10, -14, 16, -19, -12, -32, 2, 33, 3, 7, 8, 2, -23, 17, 30, -8, -32, -1, -20, -49, -25, 20, 20, 13, 15, 1, -6, -25, 16, -19, -35, 3, -8, 33, 0, -18, 1, 6, -5, -1, 20, -18, -29, -12, 3, -1, -39, -39, -40, -4, -2, 1, -23, -35, 3, 17, 3, -24, 24, -7, 12, -52, 0, 9, 11, -7, 7, 10, -29, 8, 14, -14, -12, -23, -42, -12, -20, -11, 4, -7, 12, 55, 7, -11, 6, 21, -11, 2, -47, 6, 1, -1, -8, 16, -10, -5, 0, -16, -26, -15, -10, -19, -14, -28, -16, 7, -24, 32, 12, -16, 18, 16, 25, 5, -6, -7, -4, 3, 3, -1, -9, -17, 6, -18, -3, 27, -19, 11, -13, -54, 9, -4, -30, 14, 2, 6, -1, 16, 10, 3, -4, -5, 1, -27, -20, -2, 0, 2, -7, 11, 0, 7, 5, -23, -16, 7, -56, 15, -5, -12, 1, 1, -9, 6, 24, -18, 27, -24, -3, -8, 13, 3, 7, -4, 1, 0, 3, -10, -8, 33, -55, 12, 0, -24, -10, 1, -18, 23, -10, -13, -5, -23, 12, -9, -33, 14, 5, -6, -34, -14, 10, 5, -1, 7, 16, 10, 54, -51, -2, -48, 6, -9, -44, 8, 12, -9, 12, -16, -16, 0, -34, 0, 20, -22, 0, -70, 39, -18, 3, -5, 0, 0, 0, -33, -108, -14, -17, -71, 14, -47, 5, -5, 12, -18, 4, 1, -32, -32, -49, -11, -67, 12, -43, -2, 16, -10, 5, 2, 1, 5, -4, -9, 37, 36, -46, -20, 0, -20, 0, 0, -1, -2, -16, -14, -1, -14, 9, -49, -12, -30, 0, -31, -14, 27, -3, 2, 1, -1, 7, -27, -44, 20, -27, 13, 18, -3, -3, -30, 13, -13, -1, 14, 1, -14, -15, 32, -13, 16, -10, -20, -9, 2, -13, 2, 0, 0, 2, -30, 19, -12, -6, 22, -47, -6, -11, -12, -14, -10, -12, -27, -5, -8, -11, -33, -4, -46, 1, -27, 2, 7, 2, 0, 0, 0, 1, -3, -6, 14, 10, 2, 9, -9, -2, -9, -21, 6, -4, 9, -7, 10, -18, 7, 6, -9, 5, -2, -3, -3, 4, 0, 0, 0, -6, 5, 3, -4, -4, -2, 0, 8, -5, 12, 11, -1, 4, 1, -2, -1, 2, -9, -1, 6, -2, -16, 3, 0, 3, 0, 0, 0, 0, 1, -1, -4, 0, 0, 0, 0, 2, -1, -5, 3, -3, 5, 2, 0, 3, 4, -7, 1, 4, 0, -19, 2, 0, 0, 0, 0, 0, 0, 4, 5, -1, -2, 3, -4, 5, -2, -2, 11, -2, 6, -5, 14, -2, 6, -4, 5, -3, 7, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, -1, 1, 2, 3, 2, 0, 2, -1, 2, 1, -2, 1, -4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, -2, 3, -2, 1, -2, -2, -2, -11, -13, -1, -17, -26, 20, -33, 0, -2, -3, -1, -3, 1, 0, 0, 0, 0, -21, -2, 1, 1, -7, -24, 2, 1, -4, -3, -18, 10, -30, 5, -13, -13, 2, -10, -7, 7, -5, 12, -5, -1, -1, 0, 0, 1, 7, -10, -5, 4, -10, -2, 3, 5, 5, -3, 5, 0, 7, -1, 0, -4, -5, -2, 1, -5, 1, -5, 4, 2, -1, 2, 0, 0, 1, -3, -20, -11, 4, 7, -16, 3, -16, 7, 12, -6, 15, 3, 6, 17, -4, 3, -4, 0, -3, -5, -4, 2, 2, -2, 0, 2, -1, -1, -2, 4, -3, -4, -6, 0, -6, -10, 13, 13, -7, 18, 1, 15, -14, 11, 0, 0, -3, 1, -10, 4, -2, 2, 2, -10, -9, -28, -6, -10, -9, -7, 15, -7, 8, -1, -16, -6, -14, 15, -1, -12, 10, -18, 3, 5, -9, 12, 3, -4, 0, -2, 2, 6, 7, 1, 18, 4, 7, 12, -13, 4, 5, -7, 18, 7, -11, -11, 10, -14, -2, 18, -7, 6, 12, -2, 6, -5, -2, 3, 2, -7, -15, 17, -28, 0, 1, 3, 14, 15, -3, 10, -6, 2, -7, -20, -3, 7, -3, -5, 3, 6, 1, 4, 3, -3, 6, -4, 2, -1, 1, -36, 35, -13, 17, 3, 4, 1, 28, 8, 18, -4, -15, -26, -12, -1, 5, 21, 4, 19, 8, 15, -8, -5, -2, 1, 0, 1, 1, 1, -13, 4, 3, 13, 20, 20, 7, 9, 16, 23, 18, -3, -12, -5, -5, -4, 0, -6, 11, 4, 16, -1, 3, 0, 2, 6, -30, 2, 6, 17, 1, 6, -13, 7, -7, 1, 14, 5, 12, 7, 2, 14, -5, 24, 8, 17, 6, 5, 16, -4, 0, -2, 1, 4, -19, 6, 2, -15, 8, 2, 10, 9, -20, 14, 16, 20, -12, 24, 8, -11, -3, -7, 12, -10, 14, 1, -3, -2, 2, 2, 1, 1, 7, -3, -30, 31, -19, -31, -23, -29, -13, 33, -5, 20, 13, -2, 11, 10, -8, -11, -15, 5, -36, -13, -20, 5, -1, -6, 1, 0, 3, -18, 20, -41, -35, 0, -14, 5, 1, 7, -2, 30, 19, -8, 9, -15, -15, 5, -7, -11, 27, -22, -2, 4, -29, 3, 1, 0, -6, 22, -23, -4, 16, -4, -8, -6, 6, 7, 11, 11, -8, 4, 9, -11, -13, -24, -13, -11, -34, 29, -22, -1, 3, -1, 0, -2, 12, -11, 23, -6, -15, 5, 5, 29, -9, 12, 26, 9, 9, -9, 6, -5, -12, 5, 2, -2, -7, -5, 33, -29, -2, 3, 0, 2, -1, -35, -15, -4, 5, 8, 2, 7, 19, 14, 12, 5, -11, -1, -11, -16, 3, -10, 5, -13, 19, -16, -8, -3, 3, -2, -1, 2, -16, -6, 2, -2, -8, -13, 9, 17, 7, 3, 4, -20, -8, -4, 7, 10, -6, -7, -2, -16, 11, 4, -5, -31, 0, -1, 0, 1, 7, 4, -4, -9, -1, 14, -8, -19, 11, -16, 7, -5, -19, 6, -17, -17, -1, -5, -3, 10, -12, 6, 1, -5, -9, 0, 0, 1, -3, -25, -7, 5, -1, 1, 0, 9, -8, -9, -15, 7, -4, -11, 1, -2, -3, -4, 7, -12, 12, -5, -16, -17, -10, 0, 0, 1, -11, 18, -4, 1, 6, 6, -12, -11, 1, 11, 13, -1, 12, -3, -4, -2, 4, -9, -7, 7, -5, -3, 0, 2, 2, 0, 0, 0, 1, -10, 2, -12, -3, -16, 3, 0, -1, 5, 9, 35, 4, 21, 4, 5, 8, -1, -1, 4, -8, 7, 1, -9, -3, 0, 0, 0, 4, -4, -12, -3, -1, 0, 4, 4, -3, 2, 1, -2, 1, 0, 21, -12, 9, 21, -2, 3, -5, 7, -24, -14, 1, 0, 0, 0, 0, 5, -7, -9, 6, -11, -6, -15, -2, -8, -12, -10, 0, 1, -6, 10, -16, 2, 4, -9, -12, -3, -12, 1, 1, 0, 0, 0, 0, 1, 4, -15, -36, 12, -6, -12, -14, 4, -19, 4, -12, -42, -19, -5, 16, -23, -13, 6, -2, 8, -14, 1, 0, 0, 0, 0, 0, 0, 2, 2, 4, 0, -7, 7, -14, 5, -9, -4, -4, -8, 8, -7, -4, 1, -2, -4, -5, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, -5, 4, -2, 3, -5, 3, -3, 2, 0, -9, 0, 5, -9, -18, -1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 4, -2, 1, 0, 9, 6, -6, 9, 12, -7, 5, -3, 10, -13, 1, -13, 10, -10, -9, 5, 0, 2, 0, 0, 0, 0, 1, -2, 2, -4, 1, 4, -6, -16, 7, -28, -62, -3, -55, -36, -23, -25, -4, -25, -15, 8, 10, 3, -1, -4, 2, 0, 0, 2, 2, 4, -1, -2, 21, -42, -10, -39, 11, -12, -16, -4, -21, -1, -38, -22, -5, -34, 7, -35, -7, 4, 14, 7, 0, 0, 0, 0, 3, -6, -1, 42, -54, -15, 9, -7, -24, 25, -11, 13, -34, 3, -46, -16, -32, -18, -18, -17, -21, -50, -22, 2, 6, 3, 0, -3, 0, 7, -20, -73, 14, -28, -12, 3, -10, 6, 11, 22, 36, 4, 34, 16, 26, 1, 16, -4, -13, -6, -7, -2, -7, 7, 2, 0, 8, 10, -29, -39, -7, -10, -3, -20, 1, -2, 13, 21, 29, 40, 38, 8, -2, -7, 1, -8, -13, 0, -23, -12, -11, -2, -3, 4, -13, -36, 14, 24, -15, -19, 6, -11, -6, -17, -18, 14, 15, 15, 9, -4, 2, -5, -14, 17, -6, -7, 7, 1, -11, 3, -2, 7, -2, 22, -37, -14, 14, -8, -13, -5, 7, -4, 2, -4, 8, 17, -17, -6, -4, -16, 11, -20, -10, -20, 0, -18, -7, 4, 1, 0, -31, 4, 7, -3, 3, -5, 9, -11, 6, 9, -7, -19, 8, -22, -1, -11, -3, 3, -17, -10, 5, 12, -32, 12, -3, -5, 3, -26, -4, -14, -5, 7, -6, 11, 27, 6, 11, 10, -2, 1, -20, 12, 3, 17, 7, -3, 34, 15, 1, -26, 10, -2, -14, -4, 2, -6, 1, -3, -1, 7, 14, 3, -7, 11, 13, 2, -10, 3, 22, 16, 0, 15, 12, 26, 4, 7, 18, 49, -41, -37, -8, 6, 0, -30, -5, -3, 8, 10, -4, 11, 22, 5, 7, 5, 1, -1, 16, 9, 15, 10, 10, 15, 8, 21, -9, -19, 18, -18, 7, 0, -1, -12, 6, 3, -3, 0, 17, 0, -8, 10, 5, 7, 6, -1, 9, 9, 10, 14, 0, -7, 0, -10, 20, -16, -46, 15, -5, -2, 0, 3, -19, 1, -1, 1, -18, 3, 8, -6, 6, 7, 4, 0, -12, 1, 17, 7, -10, 5, 3, -3, -20, -3, -19, -7, 7, -4, 2, 2, -9, -5, 10, 3, -4, -13, 9, -4, 22, 6, 8, -12, -3, -14, 11, -2, 4, -20, 10, -49, 37, 0, -16, -34, 21, -8, 0, -2, -5, 16, -16, -1, -19, 18, 4, 0, 3, -1, 9, 3, -23, 17, -8, -7, 6, -7, 3, 2, -29, -8, -15, -21, -8, -13, 1, 2, 2, -28, 12, -50, 40, -13, -18, 0, 25, 13, -22, -10, 23, -5, -4, 8, -15, -7, 1, 5, 24, -30, -7, -5, -44, -2, 2, -2, -51, 8, -22, -22, -6, 14, 17, -17, -11, -3, 17, 0, -24, -13, 8, -17, 7, -4, 5, 1, -21, -4, -9, -21, 5, 3, 0, 1, 7, -7, -6, 36, -22, -33, -28, -31, -35, -8, -33, -11, -2, -13, -22, -1, 5, 9, -11, -41, 5, -8, -6, 22, -13, 2, 0, -1, 6, -30, -17, 9, 21, -6, 0, 4, -8, -30, -8, -6, -16, -1, -3, -29, -34, -1, 1, 23, -23, -11, 22, -12, -15, 1, 0, 6, -9, 3, 31, -33, -29, -9, 1, -11, 11, 3, 0, -7, -1, -31, -2, -7, 19, -41, 0, -10, -7, 9, -2, 3, 1, 1, 0, 0, -8, 17, -44, 5, 7, -31, 21, -21, -16, -14, -6, -11, 1, -11, -14, -17, -15, 6, -9, -5, 26, -5, 11, -7, -3, 0, 0, 0, 1, -90, 19, -1, 0, 8, -11, -4, -2, -10, -13, -21, -21, -1, -6, 6, -5, 5, 7, 2, -4, 12, -5, -6, -3, 0, 0, 0, 5, -7, -7, 7, -3, -4, 2, -3, 4, -3, 6, 8, 3, 3, 8, 9, 11, 3, 10, 9, -1, -2, -1, 10, -3, 0, 0, 0, 0, 4, 7, -5, 10, 2, 7, 5, 7, 1, 13, 1, 18, -1, 11, 0, 12, 1, 5, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, -2, -1, 2, -1, -3, 3, -5, 5, -15, 7, -8, 4, -10, 3, -5, 4, -5, 6, -3, 0, 0, 0, 0},
};

double bias[NUM_CLASSES] = {-2.9525892761, -3.6934993680, -1.8095658856, -1.9831172869, -2.7205156918, -1.8805774781, -2.7108441697, -3.0740618807, -1.5869174001, -2.3630234768};
```

next we have to adapt export of scaler.h to int adn we have to add a scaling factor to svm_model.h

``` py

max_val = np.abs(weights).max()
scale = max_val / 127.0
weights_int8 = np.round(weights / scale).astype(np.int8)
with open("svm_model.h", "w") as f:
    f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
    f.write(f"#define NUM_FEATURES {weights.shape[1]}\n")
    f.write(f"const float weight_scale = {scale:.10f}f;\n\n")

    f.write("double weigths[NUM_CLASSES][NUM_FEATURES] = {\n")
    for row in weights_int8:
        f.write("    {"+ ", ".join(str(v) for v in row) + "},\n")
    f.write("};\n\n")

    f.write("double bias[NUM_CLASSES] = {" + ", ".join(f"{b:.10f}" for b in biases) + "};\n")

print(" Exported SVM model to svm_model.h")

```

&nbsp;
<img width="1262" height="733" alt="image" src="https://github.com/user-attachments/assets/4188f94d-212d-4eb2-8a13-201bf5b1ab3c" />

&nbsp;

create int8 version of scaler.h

``` py

mean =   scaler.mean_
max_mean = np.abs(mean).max()
scale = max_mean / 127.0
mean_int8 = np.round(mean / scale).astype(np.int8)
op_scale = scaler.scale_
max_scale = np.abs(op_scale).max()
scale = max_scale / 127.0
scale_int8 = np.round(op_scale / scale).astype(np.int8)
with open("scaler.h", "w") as f:
    f.write(f"#define NUM_FEATURES {len(mean)}\n\n)")

    f.write("double mean[NUM_FEATURES] = {\n")
    f.write("    " + ", ".join(str(m) for m in mean_int8) + "\n};\n\n")

    f.write("double scale[NUM_FEATURES] = {\n}")
    f.write("    " + ", ".join(str(s) for s in scale_int8) + "\n};\n")

print("Exported scaler parameter to scaler.h")

```

scaler.h

``` h

#define NUM_FEATURES 784

)double mean[NUM_FEATURES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 7, 9, 11, 12, 12, 11, 9, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 12, 18, 25, 32, 38, 41, 40, 36, 29, 21, 13, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 12, 19, 29, 40, 54, 67, 77, 82, 81, 73, 60, 45, 31, 20, 11, 6, 3, 1, 0, 0, 0, 0, 0, 0, 1, 3, 7, 13, 22, 34, 48, 65, 81, 97, 108, 113, 111, 102, 87, 68, 49, 32, 19, 10, 5, 2, 0, 0, 0, 0, 0, 1, 2, 5, 11, 20, 33, 49, 67, 86, 102, 114, 121, 123, 121, 115, 103, 85, 64, 43, 26, 14, 6, 2, 0, 0, 0, 0, 0, 1, 3, 8, 15, 27, 43, 62, 82, 99, 109, 112, 112, 111, 112, 111, 106, 92, 70, 48, 29, 15, 7, 2, 0, 0, 0, 0, 0, 2, 4, 9, 18, 31, 50, 71, 90, 101, 101, 95, 90, 90, 95, 101, 101, 90, 70, 48, 29, 14, 5, 2, 0, 0, 0, 0, 0, 2, 4, 9, 19, 34, 54, 76, 91, 94, 85, 75, 72, 77, 85, 95, 97, 86, 65, 44, 26, 12, 4, 1, 0, 0, 0, 0, 0, 1, 3, 9, 19, 36, 58, 79, 90, 86, 74, 67, 69, 78, 89, 98, 96, 81, 59, 39, 23, 11, 4, 1, 0, 0, 0, 0, 0, 1, 3, 9, 21, 40, 63, 82, 89, 83, 73, 72, 81, 93, 103, 107, 98, 77, 53, 35, 22, 12, 4, 0, 0, 0, 0, 0, 0, 1, 3, 10, 23, 43, 66, 83, 88, 83, 79, 88, 101, 113, 118, 115, 99, 74, 50, 34, 22, 12, 5, 1, 0, 0, 0, 0, 0, 0, 3, 11, 26, 46, 67, 82, 87, 85, 89, 105, 119, 127, 125, 117, 97, 72, 51, 35, 23, 13, 5, 1, 0, 0, 0, 0, 0, 0, 3, 13, 28, 47, 65, 78, 83, 86, 96, 112, 123, 127, 120, 111, 92, 71, 53, 38, 25, 14, 5, 1, 0, 0, 0, 0, 0, 1, 4, 15, 30, 45, 60, 71, 76, 81, 92, 105, 115, 116, 110, 101, 87, 70, 54, 38, 24, 13, 5, 1, 0, 0, 0, 0, 0, 1, 5, 17, 30, 44, 55, 63, 68, 73, 82, 93, 102, 105, 102, 95, 84, 70, 54, 37, 23, 12, 5, 1, 0, 0, 0, 0, 0, 1, 7, 19, 32, 44, 53, 60, 65, 70, 76, 87, 97, 102, 101, 96, 85, 70, 52, 34, 21, 10, 4, 1, 0, 0, 0, 0, 0, 2, 8, 20, 34, 46, 57, 65, 71, 75, 82, 92, 102, 107, 105, 98, 84, 65, 46, 30, 17, 8, 3, 1, 0, 0, 0, 0, 0, 2, 8, 19, 34, 49, 63, 74, 82, 89, 97, 107, 114, 114, 108, 94, 76, 56, 37, 23, 12, 6, 2, 1, 0, 0, 0, 0, 0, 2, 6, 15, 30, 47, 64, 79, 92, 102, 112, 119, 120, 114, 100, 81, 61, 41, 26, 15, 8, 4, 2, 0, 0, 0, 0, 0, 0, 1, 4, 10, 21, 37, 55, 74, 90, 103, 112, 114, 110, 98, 80, 60, 41, 26, 15, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 5, 11, 22, 37, 54, 70, 84, 91, 90, 82, 69, 52, 37, 23, 14, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 8, 15, 25, 35, 43, 47, 46, 41, 33, 25, 17, 11, 7, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 9, 13, 15, 16, 16, 14, 12, 10, 7, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

double scale[NUM_FEATURES] = {
}    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 6, 7, 6, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 3, 6, 8, 11, 14, 17, 21, 25, 28, 29, 30, 29, 26, 22, 17, 12, 8, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 7, 11, 16, 23, 30, 36, 42, 48, 53, 56, 56, 53, 48, 41, 32, 23, 15, 10, 6, 2, 1, 1, 1, 0, 1, 1, 4, 9, 15, 23, 33, 44, 56, 67, 77, 87, 93, 96, 96, 91, 83, 72, 58, 45, 32, 22, 14, 6, 2, 0, 1, 1, 1, 3, 9, 18, 29, 41, 55, 69, 83, 96, 107, 115, 120, 121, 120, 117, 110, 100, 86, 69, 53, 39, 25, 13, 5, 1, 1, 0, 2, 7, 16, 28, 42, 58, 74, 90, 103, 113, 120, 124, 125, 125, 125, 125, 123, 116, 104, 87, 69, 51, 35, 20, 9, 2, 0, 2, 5, 11, 23, 37, 54, 71, 88, 103, 114, 120, 123, 123, 123, 123, 123, 123, 124, 121, 113, 98, 79, 59, 40, 24, 10, 2, 1, 3, 9, 17, 30, 45, 63, 81, 98, 112, 120, 123, 124, 124, 124, 124, 124, 124, 124, 123, 117, 103, 84, 61, 41, 24, 10, 2, 1, 4, 11, 21, 33, 48, 67, 87, 104, 116, 123, 124, 124, 123, 122, 121, 122, 123, 124, 123, 117, 104, 84, 59, 37, 20, 8, 2, 1, 5, 11, 20, 32, 48, 69, 90, 108, 119, 123, 124, 120, 116, 115, 117, 120, 122, 123, 122, 114, 100, 80, 56, 32, 16, 7, 2, 1, 4, 10, 17, 30, 47, 70, 92, 110, 120, 123, 121, 116, 112, 115, 119, 120, 123, 123, 120, 110, 95, 77, 54, 30, 11, 5, 1, 1, 3, 7, 14, 27, 47, 72, 96, 113, 122, 123, 121, 116, 115, 122, 124, 123, 124, 124, 119, 106, 91, 75, 56, 31, 9, 4, 1, 0, 2, 5, 12, 26, 50, 76, 100, 115, 122, 123, 121, 119, 121, 127, 125, 123, 125, 124, 117, 104, 90, 75, 58, 35, 11, 4, 2, 1, 1, 3, 10, 26, 54, 81, 102, 116, 122, 122, 121, 121, 124, 127, 122, 123, 125, 124, 116, 105, 92, 77, 60, 37, 13, 4, 0, 0, 1, 3, 11, 28, 58, 84, 103, 115, 121, 121, 121, 122, 126, 126, 122, 124, 126, 123, 116, 107, 95, 79, 60, 37, 16, 6, 1, 0, 1, 4, 11, 32, 62, 86, 102, 112, 117, 118, 120, 123, 126, 126, 124, 125, 124, 121, 116, 108, 95, 79, 58, 36, 17, 6, 2, 1, 1, 5, 14, 37, 66, 87, 100, 109, 113, 115, 118, 121, 124, 125, 125, 124, 123, 121, 116, 108, 94, 76, 55, 34, 18, 7, 1, 0, 1, 6, 19, 43, 69, 89, 101, 107, 112, 115, 117, 119, 122, 124, 124, 123, 123, 122, 117, 106, 91, 72, 52, 32, 17, 5, 1, 0, 1, 8, 22, 46, 72, 91, 103, 110, 115, 118, 120, 121, 123, 124, 124, 123, 124, 122, 114, 102, 85, 66, 46, 29, 15, 6, 1, 1, 2, 9, 23, 45, 70, 91, 105, 114, 119, 122, 123, 124, 125, 124, 124, 124, 123, 119, 109, 94, 75, 56, 39, 25, 13, 5, 0, 0, 1, 8, 21, 39, 63, 86, 104, 115, 121, 124, 124, 124, 123, 123, 124, 123, 120, 111, 97, 80, 61, 45, 31, 20, 10, 3, 0, 0, 0, 6, 15, 30, 50, 72, 93, 109, 119, 124, 126, 125, 125, 125, 124, 120, 111, 97, 80, 62, 46, 33, 23, 13, 6, 2, 0, 1, 1, 3, 9, 19, 33, 52, 72, 92, 107, 117, 122, 123, 123, 120, 115, 105, 91, 75, 59, 45, 32, 23, 15, 8, 3, 1, 1, 1, 1, 2, 5, 11, 19, 31, 46, 62, 78, 90, 98, 102, 101, 96, 88, 77, 66, 53, 41, 31, 22, 15, 9, 4, 2, 0, 1, 1, 1, 0, 2, 5, 10, 18, 28, 39, 49, 58, 63, 66, 65, 61, 56, 50, 43, 36, 28, 20, 14, 9, 5, 2, 1, 0, 1, 1, 1, 1, 0, 3, 6, 11, 17, 24, 29, 34, 38, 39, 39, 37, 32, 29, 25, 20, 16, 12, 7, 4, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 12, 10, 9, 7, 4, 3, 2, 2, 0, 1, 1, 1, 1
};

```

int8 version of svm_model.h

``` py

max_val = np.abs(weights).max()
scale = max_val / 127.0
weights_int8 = np.round(weights / scale).astype(np.int8)
max_val = np.abs(biases).max()
scale = max_val / 127.0
biases_int8 = np.round(biases / scale).astype(np.int8)
with open("svm_model.h", "w") as f:
    f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
    f.write(f"#define NUM_FEATURES {weights.shape[1]}\n")
    f.write(f"const float weight_scale = {scale:.10f}f;\n\n")

    f.write("double weigths[NUM_CLASSES][NUM_FEATURES] = {\n")
    for row in weights_int8:
        f.write("    {"+ ", ".join(str(v) for v in row) + "},\n")
    f.write("};\n\n")

    f.write("int8 bias[NUM_CLASSES] = {" + ", ".join(str(b) for b in biases_int8) + "};\n")

print(" Exported SVM model to svm_model.h")

```

&nbsp;

svm_model.h - int8 version

``` h

#define NUM_CLASSES 10
#define NUM_FEATURES 784
const float weight_scale = 0.0290826722f;

double weigths[NUM_CLASSES][NUM_FEATURES] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -15, 2, 8, 0, 14, 0, 1, 12, 15, -24, 20, 1, 19, -32, 2, 5, 3, 2, 3, 0, 0, 0, 0, 0, 0, 1, 3, 5, 8, -6, 12, -2, -22, 10, -7, -48, 15, -29, -44, -12, -44, -36, 10, -29, 2, 5, -1, 1, 0, 0, 0, 0, 0, 2, -7, -1, 16, -19, -25, 14, -27, -27, 9, -22, 12, -12, -10, -3, -5, -13, -18, 4, -12, -9, 4, -2, 4, -9, 0, 0, 3, 5, -6, -3, -27, -1, -6, -4, 14, -22, 16, -24, 27, -27, 29, -38, 20, -2, 3, 2, -3, 0, 3, -11, 8, -3, 2, 0, 0, -2, 3, 2, -27, 40, -13, 8, -8, 31, 2, -23, 13, 2, 17, 2, 3, 7, -4, -5, -9, -9, -3, 3, -18, 4, 0, 0, 2, -8, 5, -1, -11, -25, -13, 18, -44, -10, -9, -7, 21, -5, 1, 0, 1, 10, -4, -1, 23, 18, -4, -22, -25, -9, -2, -1, 1, -7, -6, 1, 18, 23, 2, -36, 19, -4, 23, -21, 8, 15, -12, -4, 41, 1, 22, -5, -14, -17, 8, -7, -3, -12, 7, 10, -2, 7, 16, -15, -33, 3, -1, 18, -3, 1, -22, 34, -23, 23, 8, 35, 5, 14, -6, 15, 16, 16, 1, -15, -29, -11, -1, -1, 0, -12, 8, -17, 16, -8, 3, 1, -2, 27, -22, -28, 35, 2, -14, 8, 20, 1, 25, 16, -8, -25, 7, -7, -10, -12, 7, 6, -30, 59, -68, 20, 7, 7, 4, -24, 21, -12, 35, 2, -19, 11, 17, 18, 15, 43, -9, 1, 5, 30, 12, 2, -5, -15, 7, -6, 56, -79, 45, -23, -22, 6, 1, 14, -28, -9, 7, -29, 6, -32, -25, -39, 25, -30, 14, 21, 12, 1, 6, -6, -11, -1, 6, 4, -17, -6, -24, -21, 14, 7, -11, 6, 15, -9, 6, 28, 5, -35, -41, -5, -5, -2, 3, 18, 7, 9, -6, 18, -1, -28, 6, 3, 7, 2, 8, 34, -32, 30, -2, 8, 7, 26, 1, -17, -58, 4, -52, -27, -7, 3, 5, -46, -8, 17, 13, -5, 3, -17, 3, 1, 2, 2, -17, -21, 30, -2, 44, -10, 5, -12, 8, 31, 20, -116, 37, -63, 10, -6, -15, 29, 16, -11, 7, -6, 2, -1, 1, 1, -1, 1, -7, 9, -4, -23, 22, 16, 22, 29, -20, -1, -20, -77, -3, -37, -7, -3, -9, 6, -9, 12, 5, 17, -2, 2, 2, 1, 2, 6, -6, -5, 17, 4, 11, 17, 16, 6, 24, -11, -77, -39, -3, -23, 8, -16, 1, -5, 13, -23, -4, -8, -16, -39, 11, 0, 4, -3, -3, 9, 2, 19, 6, -10, 14, 15, -32, 40, -58, -22, -20, -11, 10, 4, 7, 24, -15, 36, -4, 11, 7, -5, 5, 1, 0, 4, -9, -5, -4, -14, -8, 26, 17, 14, 16, -19, -56, -11, -3, -24, -6, -11, -9, -31, -4, -13, -6, -5, -6, -3, -10, 2, 1, 2, -14, -12, -9, 14, 18, 4, -18, 9, -4, 42, -25, -11, -6, 4, -9, -1, -22, 16, -6, 12, -11, 5, -15, 6, 6, 0, 5, -12, -1, 16, 13, -11, -6, 17, -11, 23, 23, 6, 14, -34, 14, -36, 2, 18, -3, -11, 16, -6, -5, -4, -5, -7, 3, 2, 5, -7, 2, -3, -11, 3, 1, -24, 16, 12, 26, 6, -2, -6, -13, 28, -65, -9, -10, -20, -32, 11, -25, 17, -30, 19, 3, 2, -3, 2, -4, -13, 3, 7, -6, 8, 17, -7, 1, -2, 52, -20, -5, 7, 24, -34, 17, 24, -11, 23, 3, -12, 8, 2, 3, 0, 0, 6, -4, -22, 1, -1, 10, -22, 7, 18, 13, 13, -5, 27, -1, -31, 30, -26, -26, -7, -24, 0, -2, -4, 8, -2, 0, 0, 0, -6, 1, 11, 4, -6, -2, -7, 12, 15, -39, 17, -6, 4, 6, -35, 28, 14, -15, -17, -14, 13, -13, 7, -20, 2, 0, 0, 0, 4, 5, -14, -12, -32, -2, -24, -13, -89, 43, -78, -31, -41, -43, -33, -4, -89, -23, -15, -8, -12, -6, 1, 3, 2, 0, 0, 0, 0, 0, 10, 10, -2, -17, -8, -18, -21, -33, 2, -31, -15, -37, -32, -36, -21, 21, -2, -6, 3, 5, 0, 3, 0, 0, 0, 0, 0, 0, 5, -3, 3, 0, 5, -2, -1, 3, 3, -5, 2, 3, 2, 12, 2, 9, 3, -5, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, 2, 4, -1, 3, 3, 6, 3, 5, -2, 5, -11, 0, 6, 4, 2, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 8, 8, -3, -4, -3, -3, -22, 38, -29, 30, -1, 23, -15, 21, -13, 4, -1, -4, 6, -2, 0, 0, 0, 0, 4, 3, 5, -46, 5, 15, -18, -12, -12, -36, -10, -3, -30, 0, -15, 3, -29, 4, -11, -4, -1, -3, -2, 13, 3, 0, 0, 4, 11, -13, -1, 12, -9, -35, 35, -10, 6, -4, -7, 2, 5, -15, -2, -2, -10, 5, -3, 4, -8, 4, -16, -38, 12, -3, 0, 0, 0, 11, 2, 20, 3, -14, -20, -22, -9, -35, -29, -11, -26, 13, -14, -23, -7, -4, -5, 5, 15, 0, 20, -18, 1, 2, 0, 5, -9, -1, -4, -39, 30, -20, -16, -33, -3, 13, -20, -7, -11, -32, -22, -25, 3, -15, -6, -5, -17, 12, -21, -36, 6, 5, 4, -11, -7, -26, 54, -15, -32, 33, -45, 3, -7, -32, 13, -25, -1, -26, 10, -6, -15, 1, 19, 15, -12, -7, -18, -28, 18, -5, 1, 2, 10, -12, -3, 2, -43, -11, -9, -44, -1, -20, 5, 7, -5, 2, -30, 20, -1, 7, -30, -12, 9, -7, -44, -23, 11, 4, 1, -23, 11, 6, 9, -46, -1, 40, -2, -3, -8, 1, -6, 7, 11, 23, 5, -21, 10, -17, -9, -4, 1, -32, 15, -52, 9, -3, 2, 16, 6, -16, -22, -20, -38, -24, 12, -9, 7, 8, 5, 15, 33, 27, 17, 18, -32, -2, -23, -3, 5, 2, -46, 6, -13, 15, 5, -1, 12, -18, 17, -4, -31, -35, -23, -32, -7, -9, -22, 17, 15, 41, 10, -4, 2, -5, 7, 5, -26, 3, -17, -11, -1, 0, 4, 6, -34, 10, -1, -31, 29, 20, -21, 12, -20, -21, 7, 7, 41, 29, -9, 8, 7, -24, -2, -12, -19, 29, 14, -9, -16, 1, 4, 5, -7, 1, 0, -18, 37, 17, 1, 8, -42, -21, 1, 1, 28, 7, 17, 23, -23, -3, -2, 1, -13, 14, -32, -29, 2, 0, 1, 2, -2, -8, 32, 1, 12, -26, 11, -21, -27, -24, -11, 4, 26, 17, -4, -3, -56, -3, 1, -11, 10, -12, 40, 2, 0, 2, 1, 2, 7, 11, -1, -37, -24, -3, 16, -15, -46, -5, 19, 19, 24, -7, 32, -22, -1, -30, -11, -10, -12, -15, -5, -16, 6, -1, 2, 2, 1, -7, 19, -6, 3, -9, -34, 28, -13, -2, -31, -4, 21, -27, 5, -48, -28, -26, -14, 5, -14, -20, 25, -16, -7, 6, 0, 4, -2, -10, 6, -18, -1, 22, 6, -20, 0, -20, 15, 25, 24, 13, 0, -37, -6, 11, 5, -9, 30, -19, 16, -26, 1, -1, 0, -2, 4, 0, -13, -25, -57, -39, 1, -4, -35, -9, -4, -11, 20, -19, -62, 21, -2, -18, 2, 2, -13, -36, 23, -4, 0, 2, 4, 1, 1, 5, 1, -53, 10, 19, -23, 14, 5, -14, -18, 8, -14, 18, 29, -41, -10, 10, -2, 14, -11, -6, 6, -6, 1, 3, 0, 0, 6, -21, 10, -96, 1, -20, 3, 11, -3, 26, 23, -7, -5, -34, 9, 8, 33, 19, 2, -19, 14, -1, 3, -7, 11, 2, 4, 2, -5, 17, -1, 19, 22, 17, 12, -14, 25, -20, -4, 3, 6, 10, 11, -5, 17, -31, -3, 37, -20, -65, 48, -5, -2, 3, 4, -2, 2, -5, 6, 44, -15, -13, -1, 10, -14, -8, -28, -31, 9, -6, -5, -1, 45, -3, 21, 14, -33, 16, -21, 28, 0, 3, 0, 0, 9, -17, -18, -5, 9, 6, -10, -2, -14, -15, 9, -2, -31, 12, 4, 23, -37, 3, -12, -10, 30, -34, -11, 2, 7, 0, 0, 0, 3, -1, -4, -3, -2, 4, -16, 2, -8, -9, -12, -24, -16, 0, -41, 16, -9, 31, -3, 2, -17, 16, -2, 4, 16, 0, 0, 0, 2, -2, 6, -20, 7, -32, -18, -28, -59, -68, -28, -32, -11, -52, -41, -33, -38, -24, -10, -10, -1, 3, 3, -2, -10, 0, 0, 0, 0, 0, 4, -7, -6, 0, -8, -16, -5, -13, -54, -38, -29, -12, 4, -14, -1, 0, -4, 12, 2, -11, 4, 2, 0, 0, 0, 0, 0, 0, 2, 6, -10, 11, -3, 5, -1, -8, 14, 5, 7, 8, -5, 10, -19, 9, -2, -1, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -2, 1, 1, -3, 2, 2, -6, -5, 2, -3, 5, -4, 1, -22, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -3, 5, -4, -1, 3, -10, -1, 0, -6, -5, 0, -8, -4, -3, -8, -8, 1, -11, 7, 4, 1, 0, 0, 0, 0, 1, 3, -2, 1, -3, 4, -6, 8, 2, 1, 0, 5, -2, -6, 3, -2, -9, 0, 0, -3, 8, -14, -11, 5, 0, 0, 0, 0, -1, -2, -2, -2, 8, 5, -6, 10, 3, 1, 9, 0, 14, -5, 6, -10, 0, -9, -5, -7, -15, -9, -35, 16, -5, -1, 0, 0, 0, 1, 3, -1, -5, 0, 7, -7, 3, 7, 4, 14, 1, 6, 16, 5, 9, -3, 1, 3, 3, 9, -17, 6, 1, 0, 0, 1, -2, -3, -3, -1, -5, 3, -4, 3, 8, -3, -2, 6, -4, 9, -1, 6, -5, -1, -12, -3, -4, -9, -8, -12, 5, -4, 1, -5, 7, 9, 2, 2, 11, -11, 10, -9, -10, 12, -6, 11, 7, -10, 7, -19, -3, -9, -8, 6, 10, -26, -19, 28, -12, 6, -1, 5, -4, -16, 1, -8, -1, 9, 4, -1, 12, -2, -4, -3, 9, -2, -7, 9, -6, 7, 10, -2, -10, 19, -28, -46, 3, 6, 1, -6, 1, 3, 0, 21, -31, 6, -16, 7, 3, 10, -7, 0, -6, 8, 15, -9, 6, -1, -7, -11, 11, 5, -28, -4, -3, 1, 1, -1, -9, 15, -14, 1, 27, -17, 8, 2, -9, 7, 4, -4, 12, -14, -8, 19, 12, -9, 6, 6, -7, -8, -4, -16, 16, -3, 1, -14, -12, -14, 3, 7, 5, 9, -28, -36, -29, -28, -37, -33, -25, -5, -2, -17, -4, 10, -16, -4, 0, 15, -14, 6, -18, 1, 1, 6, 15, -8, 7, 3, -54, -47, -8, -14, -14, -36, -19, -13, -42, -28, 2, -14, -10, 20, 7, 9, -3, -45, 0, 6, 7, -7, 0, 1, -1, -1, -27, -12, -5, -32, -24, -14, -13, 1, -7, -2, -3, -8, -20, 2, -4, -19, -10, -11, -9, -5, 16, 1, 6, 3, 1, -7, 5, -8, -1, -46, -29, 20, -10, -7, -6, -1, 7, -11, 8, -7, 7, -11, 5, -5, 6, -6, -7, 5, -16, 2, 8, 4, 1, 1, 0, 7, 7, 45, 11, -24, 2, 1, 3, 6, -4, 12, -4, -4, 5, -10, -15, 9, -18, -7, -9, -22, 17, 0, 3, 3, 3, 0, -10, 3, -2, -18, 19, 14, -7, 6, 7, 3, 13, 11, -6, 3, 9, 20, -11, 2, -8, 21, -2, 16, -5, 6, 9, 1, 0, -1, 9, 1, 4, 11, 15, -10, 10, 2, 5, 6, 10, 13, 0, 19, 0, -11, 12, -8, 1, -13, 0, 0, -5, 11, 1, 4, 2, 1, -7, 0, 0, 18, 4, 20, 7, 0, 11, -8, 16, 10, 11, 4, 12, 7, -5, 7, 1, 10, 5, 7, 16, 3, -1, 0, 0, -1, -5, -8, 4, -12, 2, 9, -3, 16, 10, 8, 1, 16, 11, 6, 0, 12, 3, -7, 2, 1, 6, 5, -6, 11, 7, 0, 0, 0, 6, 4, 9, 11, 10, 18, 17, 6, -1, 21, -6, -3, -3, 3, 9, -18, 8, 9, 4, 9, -4, 2, 16, 1, -6, 0, 0, -1, -3, 1, -13, -1, -6, 0, -16, 17, 5, -4, 0, -4, -3, -12, 5, 14, -7, 18, -3, 9, 3, 12, -2, 6, 5, 0, 0, 0, -1, -2, 8, 3, 5, 9, 18, 7, -2, 9, 0, 5, -7, -9, -1, -5, 14, -6, 11, -2, 5, -1, 6, -3, -7, 0, 0, 0, -3, 3, 2, -6, 6, -7, -9, -8, -7, -9, -1, -16, 3, -19, 7, -7, 4, 9, -2, 8, 2, 5, -2, 0, 0, 0, 0, 0, 0, -18, -1, -2, -6, -9, 2, -3, -6, -3, 1, -12, 15, -13, -10, 4, 1, 3, -3, 5, 3, -7, 8, 9, -7, 0, 0, 0, 1, 3, 3, -85, 20, -28, -4, -40, -17, -26, -18, -18, -15, 15, -24, 10, -8, -11, 6, -6, -9, -8, -12, -5, 6, 0, 0, 0, 0, 2, 4, -2, 3, -16, -13, -6, -9, -30, -7, -24, -25, -29, -31, -38, -18, 11, -11, -9, -10, 0, 6, 1, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 5, 3, -5, -12, 2, 7, 11, 6, 13, 12, 5, 1, 6, -2, 4, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 4, 3, 7, 0, 7, -4, -1, 3, -2, 8, 4, -1, 6, -1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 5, 1, 1, 3, 2, -8, -14, 8, 2, -10, 26, -28, 13, 0, -13, -4, -2, -1, -8, 13, 8, -6, 0, 0, 0, 0, -4, 3, -3, 13, -19, 4, 9, -13, 2, -1, 18, -24, 22, 0, 9, -3, 16, -8, 24, -2, -24, -68, 5, 9, 1, 0, 0, 1, 2, 1, -3, 4, 3, -4, 2, 2, 9, 1, 2, 5, 1, 2, 9, -8, 13, -9, 8, -15, 22, -44, 7, 6, 0, 1, 0, 0, -2, 2, 2, 1, 5, 7, 0, 4, -1, 6, 9, 1, 3, 1, 12, -15, -4, 6, -6, -11, -17, 9, -13, 9, 2, 0, 0, 4, 0, 3, 1, -5, 8, -8, 4, 5, 12, 7, -4, 16, -6, 8, 6, -3, 19, -23, 5, 11, 4, -11, 1, -33, -11, 6, 1, 4, -3, 3, -1, 7, 2, 1, 2, 2, -5, -4, 6, -7, 17, 9, -6, 12, -19, 5, 4, -26, -20, 7, -10, -15, 9, 2, 1, -11, -4, -2, 6, -2, -3, -2, -2, -16, 1, -5, 8, 2, -2, 1, 5, -1, 21, 2, 1, -2, 11, -27, -22, -24, 21, 3, -1, 12, 0, 1, 7, -2, 7, 0, -5, 5, 1, -21, -11, -13, 2, 17, 8, 9, 0, 11, 17, 12, 21, -12, -94, 23, -8, 3, 0, -7, -13, -1, -7, 7, -3, 11, -6, -25, -34, -17, -26, 16, 12, 5, 15, 10, 7, -1, 16, 10, 6, 46, -59, 22, -14, 1, 0, -1, 0, 15, 8, -6, -1, -27, -20, -17, -19, -8, 2, 1, 10, 17, -3, 15, 8, 3, -2, -15, -3, 16, -103, 1, 0, 0, -1, 5, -11, -6, -6, -3, -23, -17, -16, -5, 8, -13, 18, -5, 24, 11, 14, 1, 4, 3, 12, -21, -25, -86, -73, 39, 7, 1, 0, -1, -1, -1, -8, 2, 15, -3, 1, -8, -15, 5, 14, 3, 18, -17, 15, 3, -1, -7, -26, -19, -55, 32, 43, -67, 6, 2, 0, 4, -6, 13, -2, -6, -22, -8, -4, -1, -9, -10, 22, -7, 4, 1, 7, 1, 7, -17, -16, 15, 4, -8, -11, -11, 3, 0, 0, 2, -4, -8, -6, -2, 0, -10, -13, -9, -10, 1, 8, -1, 1, 11, 2, 0, -20, 4, 30, 0, 15, 26, 0, -49, 26, 2, 1, 1, 9, 4, 12, 6, -9, -5, -34, 7, -7, 4, -6, 9, -4, -8, -21, -15, 23, 12, 5, 10, -4, -4, 13, -64, -8, 2, 0, 1, -5, -1, 7, -9, -11, 18, 0, -41, -30, -28, -7, -16, -19, -22, -5, 28, -7, 23, -5, 0, 11, 13, -8, -48, 3, 1, 0, 1, 5, 11, -13, 13, 6, -15, 5, -18, -26, -23, -28, -27, -29, 15, 25, 1, 6, 2, 6, 12, 21, 2, -15, -27, 5, 0, 1, -2, -2, 7, 3, 13, -10, -4, -2, -8, -2, -20, 1, -17, -8, -15, -1, 17, 5, 13, -5, 4, -11, -9, 1, -29, 7, -1, 0, 3, -5, 14, -1, 10, 4, 9, -4, 16, -8, 3, -16, -3, 8, -8, -5, -2, 13, -15, 9, -1, -3, 6, -5, -30, 6, 0, 0, -1, 5, -2, 8, -4, 5, -3, 14, -20, 5, -13, -3, -4, -6, 0, 18, -13, 11, 12, -2, -5, -4, -7, -23, 17, -3, 1, 0, 1, -4, 4, -1, -6, 3, 2, -12, 9, -4, 12, 1, 0, -9, 0, -1, 12, -7, -1, 8, -6, 3, 5, -32, -10, -3, 1, 0, 0, 8, -1, 6, 11, 6, 11, 1, 11, 2, 8, -7, 16, -9, 1, -13, -5, 18, 0, -1, 3, -12, -6, 11, -9, 2, 0, 0, 0, -8, 11, 5, -9, 13, 5, 4, -2, 11, 9, -7, 6, 4, -2, 4, 6, -17, -4, -4, 9, 0, 16, -21, 3, -4, 0, 0, 0, 6, -7, -6, 10, 0, -4, 14, 4, 10, -4, 20, -1, 9, -1, 3, -2, -4, 10, -7, -9, 7, -72, 13, 2, 4, 0, 0, 0, 0, 1, -31, 2, -5, 1, 1, -16, 2, -7, -12, -11, -7, -15, -6, 0, -4, -5, 7, -3, 0, -1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 2, 1, -5, 1, -2, -1, -1, -3, -15, -7, 0, 1, -2, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, -8, 7, -1, 9, 2, 4, -4, -32, 16, -24, -4, 0, -3, 2, -15, 0, -4, 1, 0, 0, 0, 0, 0, 0, 0, -3, -2, -3, 5, -21, -7, -13, -39, -31, -13, -2, 0, -8, -22, -13, -30, -4, -8, -6, 0, -14, 4, 1, 0, 0, 0, 0, 1, 1, -2, 11, -18, 15, -26, -29, -5, -28, -15, -92, -15, -34, -10, -7, -39, 9, -12, -2, -5, 4, -15, -3, 0, 0, 0, 2, 5, -13, -1, 7, -25, 23, -18, 13, -33, -7, -15, 24, -34, -25, -9, -10, -20, -2, 9, 0, 12, 0, 8, 2, -2, 1, 0, 0, -6, 8, -5, 2, 3, -4, -4, -2, -4, -7, 1, -24, 16, 1, 20, -1, 11, 12, 3, -1, 1, 1, -6, -1, -6, 1, 0, 3, 1, -12, 10, 0, -10, 3, -3, -10, -9, -30, -24, -29, -36, -41, -32, -26, -33, -13, -13, -1, -3, 17, -4, -4, 3, -5, 3, 1, -26, -5, -13, 21, -5, 15, -7, 9, -10, 13, -26, -13, -27, -29, -26, -5, -13, -1, 0, 4, 6, 15, -8, -8, 0, -13, -1, -22, -1, 20, 5, 6, -13, 10, -30, 9, -13, -13, -10, -12, -29, -14, -31, -9, -2, -10, -10, -1, -21, -9, 3, 3, -3, -3, -1, -5, 15, -14, -23, 15, 1, -1, -4, -12, -20, 4, -27, -1, -36, -23, 8, -14, 5, 1, -5, 6, 0, 13, -26, -9, 1, -4, 0, 3, -3, -19, 20, -26, -25, 12, 9, 12, -10, -5, 8, -10, -32, 0, -27, 11, -12, -10, -6, 2, 11, -22, 18, -4, -16, 3, 6, -27, -7, 16, -16, 7, 11, -16, -17, 14, 16, -8, 16, -46, -49, 17, 16, 2, -1, 20, -18, -15, -19, -9, -35, -4, -9, 5, 5, -2, -1, -4, -10, 1, -9, 15, 14, -5, 27, 12, 57, -2, -25, 11, 20, -8, -5, 0, 0, 11, 5, -19, 22, -3, -9, 0, 2, 4, -2, -7, -7, -7, 7, 2, 16, 28, 22, 33, 18, -21, -13, 14, 3, 17, 14, -21, 10, 18, -15, -2, -28, 1, 9, -5, 1, 11, -37, 0, 16, 12, 10, 31, 5, 10, 9, 21, 18, -20, 2, 0, 22, 19, 0, 30, 1, 3, 15, -19, 7, 6, -22, 0, 1, 4, 22, -5, 14, -8, 4, 2, 14, 16, 13, 15, 14, -15, 5, 14, 21, 14, -10, 20, -4, -23, 22, 28, -48, -4, 16, -4, 0, -4, -39, -3, -18, -3, 1, 14, 23, 6, 4, -10, 22, -11, 17, 24, 8, 17, 6, 0, 29, 13, -13, -11, 11, -12, -7, 6, 0, 3, 41, -12, 10, -8, -1, 12, 5, 15, -7, 3, -14, 10, 28, 8, 17, 15, 30, -15, -13, 14, -15, 28, -36, -4, 4, -6, -2, 4, -34, -8, -11, -8, 11, 2, 5, 21, -32, -26, 32, 33, 5, 5, 12, -29, -20, 17, 2, 6, -14, -6, 23, -29, -15, 9, 3, -5, 8, 22, 1, 28, -22, -15, -10, -30, -21, 23, -53, 4, 25, -7, -9, 6, 0, -17, -21, 38, -63, 3, -14, -15, 8, -2, 0, 4, 7, -10, -1, -67, 20, 1, -58, 9, -41, -4, -2, -26, 7, 6, 23, 2, 4, -10, -15, -8, 7, -5, -13, -12, 12, -1, 1, 7, -14, 9, -21, 78, -51, -57, 50, -17, 8, -14, -11, -13, -14, 9, -25, 15, -8, 1, 37, -25, -1, 9, 2, 9, -4, 2, 1, -2, 29, -23, 6, -31, -17, 0, -20, -29, -10, -5, -16, -7, -22, -12, -11, 5, -7, 19, -32, 24, 16, -6, 0, -9, 6, 2, 0, 0, -5, -10, 1, -24, 2, -2, 2, -10, -1, 1, -11, -2, -3, 4, 11, -6, 6, -8, 6, 7, -22, -1, -1, 2, 4, 0, 0, 0, 2, 11, -36, 3, 23, -19, 17, 16, -17, 17, -10, -6, 2, -12, -2, -4, -7, 9, -7, 1, 3, -1, -2, 11, -1, 0, 0, 0, 2, 4, 11, -9, -5, -11, -9, -19, -2, -17, -18, -11, -17, -9, -16, -13, -6, -18, 0, -12, 1, 2, -8, -2, -1, 0, 0, 0, 0, 1, -4, 7, 2, -32, -8, -21, -17, -30, -18, -30, -26, -23, -19, -19, -13, -9, -48, -7, -6, -2, 3, 2, 0, 0, 0, 0, 0, 0, -1, 5, 1, 2, -11, -2, -11, 3, -10, 5, -34, 1, -7, 5, -5, -2, 17, -1, -1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -3, 1, 3, 12, -3, 1, -3, 0, 8, 1, -2, 5, 2, -4, 3, 2, 2, -1, 0, 0, 0, 0, 0, 0, 1, -14, -18, 4, 7, 8, -14, -12, 5, -39, -17, -33, -35, -5, -5, -5, -2, -2, -10, -2, -5, 7, -27, 1, 0, 0, 0, 0, 1, 0, 12, -15, -2, -16, 16, -31, 26, -35, 15, -13, -2, -9, -12, -3, -7, -4, 1, -5, 3, -6, -1, 0, 0, 0, 0, 1, -3, -3, -22, -12, 0, 3, -18, -2, -9, 1, -3, -22, -3, 2, 1, -13, 8, -1, -13, 9, -4, 2, 7, -2, 1, -2, 0, 0, 1, 9, -29, 2, -14, 3, -3, -1, -1, -3, -9, -3, -9, -9, -5, 6, 4, -7, 18, -3, 13, 1, 1, 0, 2, 0, 0, 0, 4, -27, 12, 8, -2, -3, -10, 0, -4, -5, -2, -8, 10, -13, -8, -7, 0, 10, 5, -4, 4, 4, 14, -2, 2, 0, 1, 0, -4, 3, -28, -7, -1, -7, 10, -7, 14, 5, 3, -13, 11, -23, 4, 0, 5, -5, -1, 8, 13, 2, 6, 15, -7, 3, 2, 5, -6, -18, -10, 0, -9, -4, 7, -3, 5, 7, -4, -6, -10, -1, -7, 1, -10, 5, 8, 10, -15, 19, 14, 5, 3, -2, 0, 2, -5, -35, -18, -8, 9, -6, -2, 4, -3, 18, 2, -4, -8, -24, -21, -5, -9, -2, -12, 20, 20, -7, 17, 9, -1, 1, 0, 1, 5, -34, 11, -7, -15, 17, 23, 19, 14, 18, 19, 8, -6, 4, -19, -22, -27, -27, -21, -7, 8, 15, 26, 7, 8, -4, 3, -8, 22, -36, -3, -2, 26, -9, -2, 11, 8, 10, 5, 15, -6, -5, -5, -10, -6, -17, -52, -53, -57, -12, 41, 12, -7, 2, 2, -2, -7, 13, 0, -2, 8, 6, 9, -10, 10, 13, 7, 16, -11, 8, -14, -15, -15, -21, -3, -30, -44, -50, -127, 27, 18, -4, 1, -3, -14, -5, 4, 11, -7, 13, 1, 11, 0, 10, 17, 6, -19, -5, -21, -15, -4, 6, 1, -7, -10, -51, -18, -21, 9, 0, 3, -6, 5, 10, -7, -2, -12, -9, -9, 0, 21, -4, 13, -4, -8, 4, -16, -11, -11, -20, 1, -14, 7, -1, 31, -58, -10, 1, 2, 2, -3, -13, 22, -14, 4, -8, 16, 11, -3, 5, 5, -16, -17, -13, -5, -16, 17, -6, -4, -4, -18, 5, 15, 6, -5, 0, -2, 2, 7, 9, -11, 5, -22, -33, -20, -11, 5, -8, 8, 2, -18, -3, -22, -3, -23, -3, 2, -14, 27, -1, -8, -4, -30, 2, 0, 3, -10, -5, 0, -5, 38, -12, -10, -26, -27, -20, -32, -24, -25, -7, 3, -1, 19, -2, 5, 23, -20, 7, 16, -15, 1, 0, 0, -2, -11, -3, 8, 6, 8, 9, -11, -3, -25, -18, 6, -2, 10, -10, -5, 15, -14, 4, 0, -13, 21, 0, -2, -14, 6, -1, 2, -2, 5, -2, 8, -10, 14, -3, 18, -7, 9, -8, 0, 3, -8, 21, -1, -14, 7, -5, 12, 10, -9, -7, 18, -15, -2, 1, 0, 2, 1, -5, -2, 5, 0, -5, 8, 14, -3, 23, 0, 7, -13, -14, 5, 12, -2, 8, -11, -15, 18, 9, -19, 12, 0, 1, 1, 0, -5, 1, 6, 3, 5, 0, 2, 7, 5, 11, -10, 3, 3, 3, -10, 9, -4, 0, 19, 18, 6, 2, 5, -3, -4, 1, 1, 1, -9, -12, 1, -6, -13, 11, -2, 11, -8, 5, 14, 5, 0, -5, 6, -19, 24, 6, -2, -4, 5, 2, 1, 0, -2, 1, 0, 0, 7, 3, 4, 9, 5, -17, 9, -9, 14, -2, 8, -7, 13, -9, 13, -6, -18, -6, 12, -1, -2, 6, -18, 15, 0, 0, 0, 0, -29, -1, -9, 4, -8, 6, -7, 17, -6, 3, -4, 5, 5, 3, -9, 6, 7, -9, 1, -4, 2, -12, 5, -36, -2, 0, 0, 0, 5, 2, 5, -12, 2, 2, -7, 7, -7, 6, 1, 4, -9, 6, -2, -1, -1, 7, -18, 2, 5, 1, 10, 2, 1, 0, 0, 0, 0, 1, 1, -6, 6, -31, -18, 0, -9, 0, -11, 0, -13, -3, -4, -6, -8, -8, 14, -9, -2, 0, -2, 2, 0, 0, 0, 0, 0, 0, 1, 8, -2, 6, 0, -6, 3, 0, 0, -5, -10, -3, -7, -1, -6, 4, -1, -6, 1, 2, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 8, -6, 4, 1, -1, 1, 1, 3, -3, 3, -3, 2, 6, 4, 4, 5, 4, 2, 0, 0, 0, 0, 0, 0, 3, -3, 6, 3, 0, -1, 1, -1, 10, -3, 8, 5, 4, 0, 10, 2, 7, 3, 4, 11, 0, 11, -4, -1, 0, 0, 0, 0, 1, 1, 3, -7, 6, 6, -1, -1, -2, 3, -6, 1, -3, 8, -5, 4, 4, 10, -1, 2, 2, -1, 9, -4, 4, 0, 0, 3, -4, -3, 0, 6, -17, 6, 2, -12, -2, -8, -19, -6, -6, -18, -7, 8, -19, -2, -1, -5, 1, 0, -13, 2, -40, 17, 0, 0, 1, -10, 0, -10, 14, -34, 29, -32, 19, -34, 11, -37, -7, -13, -21, -14, 17, 4, 15, 10, -13, 10, 11, -7, 1, -5, 0, 0, -1, 4, -3, -1, -7, 10, -15, 6, -25, -9, -4, -2, -26, -1, -23, -3, -32, -21, -27, -14, 6, 1, -28, 11, -10, 13, 1, 7, -2, -23, 7, 29, -36, 22, -15, 10, -1, -13, -17, -19, 1, -11, -21, -12, -9, -7, -15, -5, -31, -21, 39, -2, -6, -8, 1, 2, 5, -18, -26, -13, -4, -8, -25, -22, -8, 1, -15, -6, -20, -24, -9, -17, -45, -39, -52, -18, -5, -44, -52, -10, 16, 0, 0, 2, 18, -9, 38, -8, -18, 22, 22, 0, -10, -9, 3, -28, -1, -30, -33, -43, -34, -21, -4, -4, -23, 13, -13, -21, -19, 6, 1, 5, -16, -16, -14, -21, -5, -18, -5, -14, -13, -2, -17, 4, -16, -6, -15, -34, 2, -48, -6, -19, -32, 0, 11, -10, 5, -27, 3, -1, 26, 6, 1, 54, -7, 9, -11, 3, 10, -20, 6, 17, 3, -37, 14, -25, -16, 18, -16, 9, -5, 18, -6, -5, -8, 20, 2, 1, -18, -17, -11, -44, 2, -2, 22, 15, -20, 33, 2, 2, 0, -20, -4, 15, -5, -19, -15, 9, 19, -3, 3, 4, -8, -16, 1, 0, -5, -21, 6, 27, 31, -7, -11, 0, 29, -13, 14, 1, -11, 31, -15, -7, -6, 3, 1, 20, -22, 34, 3, -8, -5, 5, 0, -3, 22, 29, -21, -3, 3, 14, 17, 14, 5, 29, 7, -4, -7, -10, 19, -18, -1, 1, -3, -15, 37, -5, -2, 7, -5, -8, 0, 0, -11, -6, -39, -10, -3, 12, 5, 5, 11, 10, 6, -1, 27, 7, -9, 10, -1, -19, 30, 23, -24, 11, -12, 6, -5, 2, 3, 2, 12, -20, 41, -10, 14, 15, -24, 30, -9, 39, -17, -10, -3, 8, 2, -19, -4, 10, -20, -4, 12, -16, 6, -6, -6, 0, 0, 3, -1, 17, -64, 29, -22, -16, 44, -9, 32, 26, -3, 6, 25, -22, -15, 10, 7, 13, 22, -8, 9, 6, 5, -6, -7, -4, 3, 1, -4, -2, 10, -31, -9, 39, -19, 20, 9, 3, 36, 6, -10, 13, 16, -1, 18, -13, -11, 8, -13, 6, -20, 7, 5, 1, 0, 3, 26, -91, -3, 7, 5, -15, 9, 10, 16, 32, 33, 2, 12, 14, 5, 15, 0, 16, 8, 2, 8, -3, -2, -7, -13, 1, 0, -1, 8, -9, 18, -1, -18, 7, -9, 7, 33, -9, 29, -8, 41, 0, 21, -15, 12, -5, -8, 0, -14, 5, -14, 2, 4, 2, 1, 6, -4, -5, -14, -24, -27, 2, 12, -1, -3, 25, 33, 26, 33, 14, 4, 25, 8, 0, 20, -30, 13, -13, 16, -10, 3, 1, 1, 2, -8, -7, 5, 2, -65, 39, -11, 6, 21, -6, 6, 6, -8, 30, -1, 18, -12, -4, 7, 18, -3, -3, -43, -9, 8, 1, 0, 0, 8, 6, 14, 19, -5, -30, -13, 8, -26, -19, -8, 3, -7, -8, 6, -33, 28, 4, -1, -11, 3, -25, 8, -20, -17, 0, 0, 0, -6, 8, 8, 2, 0, -23, -32, -13, -30, -39, -10, -29, -11, -32, 13, 1, 0, -34, -3, -21, 0, 8, -1, 4, 0, 0, 0, 0, 3, 1, 2, 10, 15, 17, 6, -3, 16, 13, 2, -28, -26, -13, -30, -7, -5, 6, 3, 17, 8, 3, 1, 3, 0, 0, 0, 0, 0, -2, 7, -3, 8, 19, 2, 17, -6, 1, 32, 32, 22, -14, 19, 4, 4, 4, 2, -2, -6, 2, 0, 3, 0, 0, 0, 0, 0, 0, 1, 3, 3, 2, 9, 6, -7, 12, 11, 2, 6, 8, 17, -6, 8, -5, 8, 1, 1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -1, 5, -3, -5, -3, 3, 11, -15, -7, 11, -5, 2, 4, 5, -2, 3, -3, 4, -1, 0, 0, 0, 0, 0, 0, 4, -2, 4, 4, -3, 3, 12, 16, -5, -10, 25, 1, 12, 12, -3, 7, 13, 5, 16, 3, -1, 2, 3, 6, 0, 0, 0, 0, 3, -4, 10, -10, 19, -9, 8, -17, 17, -20, -27, -49, 5, -34, 5, 0, -15, 23, -2, -14, 21, -10, 2, 4, 4, 0, 0, 3, 2, 9, 12, -34, -7, -23, -14, -21, -64, -21, 4, 5, -22, -25, -36, -20, -40, -11, -2, 22, -12, 8, 3, 0, 5, 1, 0, 0, 2, -11, -37, 43, -11, -5, 11, 15, -7, -15, -4, -8, 1, -14, -1, -56, -9, -63, -36, -17, -35, -32, 26, -3, -3, 3, 0, 0, -1, 9, 1, -7, 13, -11, 24, -10, 13, 16, 15, -14, 9, -14, 5, 7, -5, 10, -18, 7, 3, -38, 1, -13, -28, 6, 3, 2, -4, 3, -1, -2, 9, 11, -16, 21, 19, 2, 16, 11, -14, -5, -1, 11, 13, -7, 15, 23, -4, 7, 5, -8, 1, -3, -4, 6, 7, 4, -2, 1, -4, 37, 16, -2, 11, 15, 21, -3, 2, 19, 5, 3, 13, 9, 13, -15, 4, 14, -16, -11, -6, 3, 5, 4, -5, 2, 13, 5, 1, -12, 11, 15, -6, -3, 46, -4, 31, 3, 6, 29, 14, 4, -8, 17, -6, 6, 7, -11, -58, 18, 5, -4, 12, -7, 9, 6, 5, 14, -8, 11, -1, -8, -20, 46, -3, 22, 34, 3, 5, 16, 22, 1, 33, -7, -13, 11, -2, 7, 3, -7, 12, -1, -1, 6, -2, 4, -29, 2, 18, 18, -17, 24, 16, 17, 2, 11, 15, 3, 14, -7, 10, -14, 16, -19, -12, -32, 2, 33, 3, 7, 8, 2, -23, 17, 30, -8, -32, -1, -20, -49, -25, 20, 20, 13, 15, 1, -6, -25, 16, -19, -35, 3, -8, 33, 0, -18, 1, 6, -5, -1, 20, -18, -29, -12, 3, -1, -39, -39, -40, -4, -2, 1, -23, -35, 3, 17, 3, -24, 24, -7, 12, -52, 0, 9, 11, -7, 7, 10, -29, 8, 14, -14, -12, -23, -42, -12, -20, -11, 4, -7, 12, 55, 7, -11, 6, 21, -11, 2, -47, 6, 1, -1, -8, 16, -10, -5, 0, -16, -26, -15, -10, -19, -14, -28, -16, 7, -24, 32, 12, -16, 18, 16, 25, 5, -6, -7, -4, 3, 3, -1, -9, -17, 6, -18, -3, 27, -19, 11, -13, -54, 9, -4, -30, 14, 2, 6, -1, 16, 10, 3, -4, -5, 1, -27, -20, -2, 0, 2, -7, 11, 0, 7, 5, -23, -16, 7, -56, 15, -5, -12, 1, 1, -9, 6, 24, -18, 27, -24, -3, -8, 13, 3, 7, -4, 1, 0, 3, -10, -8, 33, -55, 12, 0, -24, -10, 1, -18, 23, -10, -13, -5, -23, 12, -9, -33, 14, 5, -6, -34, -14, 10, 5, -1, 7, 16, 10, 54, -51, -2, -48, 6, -9, -44, 8, 12, -9, 12, -16, -16, 0, -34, 0, 20, -22, 0, -70, 39, -18, 3, -5, 0, 0, 0, -33, -108, -14, -17, -71, 14, -47, 5, -5, 12, -18, 4, 1, -32, -32, -49, -11, -67, 12, -43, -2, 16, -10, 5, 2, 1, 5, -4, -9, 37, 36, -46, -20, 0, -20, 0, 0, -1, -2, -16, -14, -1, -14, 9, -49, -12, -30, 0, -31, -14, 27, -3, 2, 1, -1, 7, -27, -44, 20, -27, 13, 18, -3, -3, -30, 13, -13, -1, 14, 1, -14, -15, 32, -13, 16, -10, -20, -9, 2, -13, 2, 0, 0, 2, -30, 19, -12, -6, 22, -47, -6, -11, -12, -14, -10, -12, -27, -5, -8, -11, -33, -4, -46, 1, -27, 2, 7, 2, 0, 0, 0, 1, -3, -6, 14, 10, 2, 9, -9, -2, -9, -21, 6, -4, 9, -7, 10, -18, 7, 6, -9, 5, -2, -3, -3, 4, 0, 0, 0, -6, 5, 3, -4, -4, -2, 0, 8, -5, 12, 11, -1, 4, 1, -2, -1, 2, -9, -1, 6, -2, -16, 3, 0, 3, 0, 0, 0, 0, 1, -1, -4, 0, 0, 0, 0, 2, -1, -5, 3, -3, 5, 2, 0, 3, 4, -7, 1, 4, 0, -19, 2, 0, 0, 0, 0, 0, 0, 4, 5, -1, -2, 3, -4, 5, -2, -2, 11, -2, 6, -5, 14, -2, 6, -4, 5, -3, 7, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, -1, 1, 2, 3, 2, 0, 2, -1, 2, 1, -2, 1, -4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, -2, 3, -2, 1, -2, -2, -2, -11, -13, -1, -17, -26, 20, -33, 0, -2, -3, -1, -3, 1, 0, 0, 0, 0, -21, -2, 1, 1, -7, -24, 2, 1, -4, -3, -18, 10, -30, 5, -13, -13, 2, -10, -7, 7, -5, 12, -5, -1, -1, 0, 0, 1, 7, -10, -5, 4, -10, -2, 3, 5, 5, -3, 5, 0, 7, -1, 0, -4, -5, -2, 1, -5, 1, -5, 4, 2, -1, 2, 0, 0, 1, -3, -20, -11, 4, 7, -16, 3, -16, 7, 12, -6, 15, 3, 6, 17, -4, 3, -4, 0, -3, -5, -4, 2, 2, -2, 0, 2, -1, -1, -2, 4, -3, -4, -6, 0, -6, -10, 13, 13, -7, 18, 1, 15, -14, 11, 0, 0, -3, 1, -10, 4, -2, 2, 2, -10, -9, -28, -6, -10, -9, -7, 15, -7, 8, -1, -16, -6, -14, 15, -1, -12, 10, -18, 3, 5, -9, 12, 3, -4, 0, -2, 2, 6, 7, 1, 18, 4, 7, 12, -13, 4, 5, -7, 18, 7, -11, -11, 10, -14, -2, 18, -7, 6, 12, -2, 6, -5, -2, 3, 2, -7, -15, 17, -28, 0, 1, 3, 14, 15, -3, 10, -6, 2, -7, -20, -3, 7, -3, -5, 3, 6, 1, 4, 3, -3, 6, -4, 2, -1, 1, -36, 35, -13, 17, 3, 4, 1, 28, 8, 18, -4, -15, -26, -12, -1, 5, 21, 4, 19, 8, 15, -8, -5, -2, 1, 0, 1, 1, 1, -13, 4, 3, 13, 20, 20, 7, 9, 16, 23, 18, -3, -12, -5, -5, -4, 0, -6, 11, 4, 16, -1, 3, 0, 2, 6, -30, 2, 6, 17, 1, 6, -13, 7, -7, 1, 14, 5, 12, 7, 2, 14, -5, 24, 8, 17, 6, 5, 16, -4, 0, -2, 1, 4, -19, 6, 2, -15, 8, 2, 10, 9, -20, 14, 16, 20, -12, 24, 8, -11, -3, -7, 12, -10, 14, 1, -3, -2, 2, 2, 1, 1, 7, -3, -30, 31, -19, -31, -23, -29, -13, 33, -5, 20, 13, -2, 11, 10, -8, -11, -15, 5, -36, -13, -20, 5, -1, -6, 1, 0, 3, -18, 20, -41, -35, 0, -14, 5, 1, 7, -2, 30, 19, -8, 9, -15, -15, 5, -7, -11, 27, -22, -2, 4, -29, 3, 1, 0, -6, 22, -23, -4, 16, -4, -8, -6, 6, 7, 11, 11, -8, 4, 9, -11, -13, -24, -13, -11, -34, 29, -22, -1, 3, -1, 0, -2, 12, -11, 23, -6, -15, 5, 5, 29, -9, 12, 26, 9, 9, -9, 6, -5, -12, 5, 2, -2, -7, -5, 33, -29, -2, 3, 0, 2, -1, -35, -15, -4, 5, 8, 2, 7, 19, 14, 12, 5, -11, -1, -11, -16, 3, -10, 5, -13, 19, -16, -8, -3, 3, -2, -1, 2, -16, -6, 2, -2, -8, -13, 9, 17, 7, 3, 4, -20, -8, -4, 7, 10, -6, -7, -2, -16, 11, 4, -5, -31, 0, -1, 0, 1, 7, 4, -4, -9, -1, 14, -8, -19, 11, -16, 7, -5, -19, 6, -17, -17, -1, -5, -3, 10, -12, 6, 1, -5, -9, 0, 0, 1, -3, -25, -7, 5, -1, 1, 0, 9, -8, -9, -15, 7, -4, -11, 1, -2, -3, -4, 7, -12, 12, -5, -16, -17, -10, 0, 0, 1, -11, 18, -4, 1, 6, 6, -12, -11, 1, 11, 13, -1, 12, -3, -4, -2, 4, -9, -7, 7, -5, -3, 0, 2, 2, 0, 0, 0, 1, -10, 2, -12, -3, -16, 3, 0, -1, 5, 9, 35, 4, 21, 4, 5, 8, -1, -1, 4, -8, 7, 1, -9, -3, 0, 0, 0, 4, -4, -12, -3, -1, 0, 4, 4, -3, 2, 1, -2, 1, 0, 21, -12, 9, 21, -2, 3, -5, 7, -24, -14, 1, 0, 0, 0, 0, 5, -7, -9, 6, -11, -6, -15, -2, -8, -12, -10, 0, 1, -6, 10, -16, 2, 4, -9, -12, -3, -12, 1, 1, 0, 0, 0, 0, 1, 4, -15, -36, 12, -6, -12, -14, 4, -19, 4, -12, -42, -19, -5, 16, -23, -13, 6, -2, 8, -14, 1, 0, 0, 0, 0, 0, 0, 2, 2, 4, 0, -7, 7, -14, 5, -9, -4, -4, -8, 8, -7, -4, 1, -2, -4, -5, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, -5, 4, -2, 3, -5, 3, -3, 2, 0, -9, 0, 5, -9, -18, -1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 4, -2, 1, 0, 9, 6, -6, 9, 12, -7, 5, -3, 10, -13, 1, -13, 10, -10, -9, 5, 0, 2, 0, 0, 0, 0, 1, -2, 2, -4, 1, 4, -6, -16, 7, -28, -62, -3, -55, -36, -23, -25, -4, -25, -15, 8, 10, 3, -1, -4, 2, 0, 0, 2, 2, 4, -1, -2, 21, -42, -10, -39, 11, -12, -16, -4, -21, -1, -38, -22, -5, -34, 7, -35, -7, 4, 14, 7, 0, 0, 0, 0, 3, -6, -1, 42, -54, -15, 9, -7, -24, 25, -11, 13, -34, 3, -46, -16, -32, -18, -18, -17, -21, -50, -22, 2, 6, 3, 0, -3, 0, 7, -20, -73, 14, -28, -12, 3, -10, 6, 11, 22, 36, 4, 34, 16, 26, 1, 16, -4, -13, -6, -7, -2, -7, 7, 2, 0, 8, 10, -29, -39, -7, -10, -3, -20, 1, -2, 13, 21, 29, 40, 38, 8, -2, -7, 1, -8, -13, 0, -23, -12, -11, -2, -3, 4, -13, -36, 14, 24, -15, -19, 6, -11, -6, -17, -18, 14, 15, 15, 9, -4, 2, -5, -14, 17, -6, -7, 7, 1, -11, 3, -2, 7, -2, 22, -37, -14, 14, -8, -13, -5, 7, -4, 2, -4, 8, 17, -17, -6, -4, -16, 11, -20, -10, -20, 0, -18, -7, 4, 1, 0, -31, 4, 7, -3, 3, -5, 9, -11, 6, 9, -7, -19, 8, -22, -1, -11, -3, 3, -17, -10, 5, 12, -32, 12, -3, -5, 3, -26, -4, -14, -5, 7, -6, 11, 27, 6, 11, 10, -2, 1, -20, 12, 3, 17, 7, -3, 34, 15, 1, -26, 10, -2, -14, -4, 2, -6, 1, -3, -1, 7, 14, 3, -7, 11, 13, 2, -10, 3, 22, 16, 0, 15, 12, 26, 4, 7, 18, 49, -41, -37, -8, 6, 0, -30, -5, -3, 8, 10, -4, 11, 22, 5, 7, 5, 1, -1, 16, 9, 15, 10, 10, 15, 8, 21, -9, -19, 18, -18, 7, 0, -1, -12, 6, 3, -3, 0, 17, 0, -8, 10, 5, 7, 6, -1, 9, 9, 10, 14, 0, -7, 0, -10, 20, -16, -46, 15, -5, -2, 0, 3, -19, 1, -1, 1, -18, 3, 8, -6, 6, 7, 4, 0, -12, 1, 17, 7, -10, 5, 3, -3, -20, -3, -19, -7, 7, -4, 2, 2, -9, -5, 10, 3, -4, -13, 9, -4, 22, 6, 8, -12, -3, -14, 11, -2, 4, -20, 10, -49, 37, 0, -16, -34, 21, -8, 0, -2, -5, 16, -16, -1, -19, 18, 4, 0, 3, -1, 9, 3, -23, 17, -8, -7, 6, -7, 3, 2, -29, -8, -15, -21, -8, -13, 1, 2, 2, -28, 12, -50, 40, -13, -18, 0, 25, 13, -22, -10, 23, -5, -4, 8, -15, -7, 1, 5, 24, -30, -7, -5, -44, -2, 2, -2, -51, 8, -22, -22, -6, 14, 17, -17, -11, -3, 17, 0, -24, -13, 8, -17, 7, -4, 5, 1, -21, -4, -9, -21, 5, 3, 0, 1, 7, -7, -6, 36, -22, -33, -28, -31, -35, -8, -33, -11, -2, -13, -22, -1, 5, 9, -11, -41, 5, -8, -6, 22, -13, 2, 0, -1, 6, -30, -17, 9, 21, -6, 0, 4, -8, -30, -8, -6, -16, -1, -3, -29, -34, -1, 1, 23, -23, -11, 22, -12, -15, 1, 0, 6, -9, 3, 31, -33, -29, -9, 1, -11, 11, 3, 0, -7, -1, -31, -2, -7, 19, -41, 0, -10, -7, 9, -2, 3, 1, 1, 0, 0, -8, 17, -44, 5, 7, -31, 21, -21, -16, -14, -6, -11, 1, -11, -14, -17, -15, 6, -9, -5, 26, -5, 11, -7, -3, 0, 0, 0, 1, -90, 19, -1, 0, 8, -11, -4, -2, -10, -13, -21, -21, -1, -6, 6, -5, 5, 7, 2, -4, 12, -5, -6, -3, 0, 0, 0, 5, -7, -7, 7, -3, -4, 2, -3, 4, -3, 6, 8, 3, 3, 8, 9, 11, 3, 10, 9, -1, -2, -1, 10, -3, 0, 0, 0, 0, 4, 7, -5, 10, 2, 7, 5, 7, 1, 13, 1, 18, -1, 11, 0, 12, 1, 5, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, -2, -1, 2, -1, -3, 3, -5, 5, -15, 7, -8, 4, -10, 3, -5, 4, -5, 6, -3, 0, 0, 0, 0},
};

int8 bias[NUM_CLASSES] = {-102, -127, -62, -68, -94, -65, -93, -106, -55, -81};


```

to move data to flash memory we have to make variabele as const




- Post-Training Quantization - From 68KB Overflow to MCU-Ready AI

reducing memory consumtion again

``` py

max_val = np.abs(weights).max()
w_scale = max_val / 127.0
weights_int8 = np.round(weights / w_scale).astype(np.int8)
max_val = np.abs(biases).max()
b_scale = max_val / 127.0
biases_int8 = np.round(biases / b_scale).astype(np.int8)
with open("svm_model.h", "w") as f:
    f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
    f.write(f"#define NUM_FEATURES {weights.shape[1]}\n")
    f.write(f"const float weight_scale = {w_scale:.10f}f;\n\n")
    f.write(f"const float bias_scale = {b_scale:.10f}f;\n\n")

    f.write("int8_t weigths[NUM_CLASSES][NUM_FEATURES] = {\n")
    for row in weights_int8:
        f.write("    {"+ ", ".join(str(v) for v in row) + "},\n")
    f.write("};\n\n")

    f.write("int8_t bias[NUM_CLASSES] = {" + ", ".join(str(b) for b in biases_int8) + "};\n")

print(" Exported SVM model to svm_model.h")

```

svm_model_q.h

``` h

#define NUM_CLASSES 10
#define NUM_FEATURES 784
const float weight_scale = 0.0030784949f;

const float bias_scale = 0.0290826722f;

int8_t weigths[NUM_CLASSES][NUM_FEATURES] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -15, 2, 8, 0, 14, 0, 1, 12, 15, -24, 20, 1, 19, -32, 2, 5, 3, 2, 3, 0, 0, 0, 0, 0, 0, 1, 3, 5, 8, -6, 12, -2, -22, 10, -7, -48, 15, -29, -44, -12, -44, -36, 10, -29, 2, 5, -1, 1, 0, 0, 0, 0, 0, 2, -7, -1, 16, -19, -25, 14, -27, -27, 9, -22, 12, -12, -10, -3, -5, -13, -18, 4, -12, -9, 4, -2, 4, -9, 0, 0, 3, 5, -6, -3, -27, -1, -6, -4, 14, -22, 16, -24, 27, -27, 29, -38, 20, -2, 3, 2, -3, 0, 3, -11, 8, -3, 2, 0, 0, -2, 3, 2, -27, 40, -13, 8, -8, 31, 2, -23, 13, 2, 17, 2, 3, 7, -4, -5, -9, -9, -3, 3, -18, 4, 0, 0, 2, -8, 5, -1, -11, -25, -13, 18, -44, -10, -9, -7, 21, -5, 1, 0, 1, 10, -4, -1, 23, 18, -4, -22, -25, -9, -2, -1, 1, -7, -6, 1, 18, 23, 2, -36, 19, -4, 23, -21, 8, 15, -12, -4, 41, 1, 22, -5, -14, -17, 8, -7, -3, -12, 7, 10, -2, 7, 16, -15, -33, 3, -1, 18, -3, 1, -22, 34, -23, 23, 8, 35, 5, 14, -6, 15, 16, 16, 1, -15, -29, -11, -1, -1, 0, -12, 8, -17, 16, -8, 3, 1, -2, 27, -22, -28, 35, 2, -14, 8, 20, 1, 25, 16, -8, -25, 7, -7, -10, -12, 7, 6, -30, 59, -68, 20, 7, 7, 4, -24, 21, -12, 35, 2, -19, 11, 17, 18, 15, 43, -9, 1, 5, 30, 12, 2, -5, -15, 7, -6, 56, -79, 45, -23, -22, 6, 1, 14, -28, -9, 7, -29, 6, -32, -25, -39, 25, -30, 14, 21, 12, 1, 6, -6, -11, -1, 6, 4, -17, -6, -24, -21, 14, 7, -11, 6, 15, -9, 6, 28, 5, -35, -41, -5, -5, -2, 3, 18, 7, 9, -6, 18, -1, -28, 6, 3, 7, 2, 8, 34, -32, 30, -2, 8, 7, 26, 1, -17, -58, 4, -52, -27, -7, 3, 5, -46, -8, 17, 13, -5, 3, -17, 3, 1, 2, 2, -17, -21, 30, -2, 44, -10, 5, -12, 8, 31, 20, -116, 37, -63, 10, -6, -15, 29, 16, -11, 7, -6, 2, -1, 1, 1, -1, 1, -7, 9, -4, -23, 22, 16, 22, 29, -20, -1, -20, -77, -3, -37, -7, -3, -9, 6, -9, 12, 5, 17, -2, 2, 2, 1, 2, 6, -6, -5, 17, 4, 11, 17, 16, 6, 24, -11, -77, -39, -3, -23, 8, -16, 1, -5, 13, -23, -4, -8, -16, -39, 11, 0, 4, -3, -3, 9, 2, 19, 6, -10, 14, 15, -32, 40, -58, -22, -20, -11, 10, 4, 7, 24, -15, 36, -4, 11, 7, -5, 5, 1, 0, 4, -9, -5, -4, -14, -8, 26, 17, 14, 16, -19, -56, -11, -3, -24, -6, -11, -9, -31, -4, -13, -6, -5, -6, -3, -10, 2, 1, 2, -14, -12, -9, 14, 18, 4, -18, 9, -4, 42, -25, -11, -6, 4, -9, -1, -22, 16, -6, 12, -11, 5, -15, 6, 6, 0, 5, -12, -1, 16, 13, -11, -6, 17, -11, 23, 23, 6, 14, -34, 14, -36, 2, 18, -3, -11, 16, -6, -5, -4, -5, -7, 3, 2, 5, -7, 2, -3, -11, 3, 1, -24, 16, 12, 26, 6, -2, -6, -13, 28, -65, -9, -10, -20, -32, 11, -25, 17, -30, 19, 3, 2, -3, 2, -4, -13, 3, 7, -6, 8, 17, -7, 1, -2, 52, -20, -5, 7, 24, -34, 17, 24, -11, 23, 3, -12, 8, 2, 3, 0, 0, 6, -4, -22, 1, -1, 10, -22, 7, 18, 13, 13, -5, 27, -1, -31, 30, -26, -26, -7, -24, 0, -2, -4, 8, -2, 0, 0, 0, -6, 1, 11, 4, -6, -2, -7, 12, 15, -39, 17, -6, 4, 6, -35, 28, 14, -15, -17, -14, 13, -13, 7, -20, 2, 0, 0, 0, 4, 5, -14, -12, -32, -2, -24, -13, -89, 43, -78, -31, -41, -43, -33, -4, -89, -23, -15, -8, -12, -6, 1, 3, 2, 0, 0, 0, 0, 0, 10, 10, -2, -17, -8, -18, -21, -33, 2, -31, -15, -37, -32, -36, -21, 21, -2, -6, 3, 5, 0, 3, 0, 0, 0, 0, 0, 0, 5, -3, 3, 0, 5, -2, -1, 3, 3, -5, 2, 3, 2, 12, 2, 9, 3, -5, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, 2, 4, -1, 3, 3, 6, 3, 5, -2, 5, -11, 0, 6, 4, 2, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 8, 8, -3, -4, -3, -3, -22, 38, -29, 30, -1, 23, -15, 21, -13, 4, -1, -4, 6, -2, 0, 0, 0, 0, 4, 3, 5, -46, 5, 15, -18, -12, -12, -36, -10, -3, -30, 0, -15, 3, -29, 4, -11, -4, -1, -3, -2, 13, 3, 0, 0, 4, 11, -13, -1, 12, -9, -35, 35, -10, 6, -4, -7, 2, 5, -15, -2, -2, -10, 5, -3, 4, -8, 4, -16, -38, 12, -3, 0, 0, 0, 11, 2, 20, 3, -14, -20, -22, -9, -35, -29, -11, -26, 13, -14, -23, -7, -4, -5, 5, 15, 0, 20, -18, 1, 2, 0, 5, -9, -1, -4, -39, 30, -20, -16, -33, -3, 13, -20, -7, -11, -32, -22, -25, 3, -15, -6, -5, -17, 12, -21, -36, 6, 5, 4, -11, -7, -26, 54, -15, -32, 33, -45, 3, -7, -32, 13, -25, -1, -26, 10, -6, -15, 1, 19, 15, -12, -7, -18, -28, 18, -5, 1, 2, 10, -12, -3, 2, -43, -11, -9, -44, -1, -20, 5, 7, -5, 2, -30, 20, -1, 7, -30, -12, 9, -7, -44, -23, 11, 4, 1, -23, 11, 6, 9, -46, -1, 40, -2, -3, -8, 1, -6, 7, 11, 23, 5, -21, 10, -17, -9, -4, 1, -32, 15, -52, 9, -3, 2, 16, 6, -16, -22, -20, -38, -24, 12, -9, 7, 8, 5, 15, 33, 27, 17, 18, -32, -2, -23, -3, 5, 2, -46, 6, -13, 15, 5, -1, 12, -18, 17, -4, -31, -35, -23, -32, -7, -9, -22, 17, 15, 41, 10, -4, 2, -5, 7, 5, -26, 3, -17, -11, -1, 0, 4, 6, -34, 10, -1, -31, 29, 20, -21, 12, -20, -21, 7, 7, 41, 29, -9, 8, 7, -24, -2, -12, -19, 29, 14, -9, -16, 1, 4, 5, -7, 1, 0, -18, 37, 17, 1, 8, -42, -21, 1, 1, 28, 7, 17, 23, -23, -3, -2, 1, -13, 14, -32, -29, 2, 0, 1, 2, -2, -8, 32, 1, 12, -26, 11, -21, -27, -24, -11, 4, 26, 17, -4, -3, -56, -3, 1, -11, 10, -12, 40, 2, 0, 2, 1, 2, 7, 11, -1, -37, -24, -3, 16, -15, -46, -5, 19, 19, 24, -7, 32, -22, -1, -30, -11, -10, -12, -15, -5, -16, 6, -1, 2, 2, 1, -7, 19, -6, 3, -9, -34, 28, -13, -2, -31, -4, 21, -27, 5, -48, -28, -26, -14, 5, -14, -20, 25, -16, -7, 6, 0, 4, -2, -10, 6, -18, -1, 22, 6, -20, 0, -20, 15, 25, 24, 13, 0, -37, -6, 11, 5, -9, 30, -19, 16, -26, 1, -1, 0, -2, 4, 0, -13, -25, -57, -39, 1, -4, -35, -9, -4, -11, 20, -19, -62, 21, -2, -18, 2, 2, -13, -36, 23, -4, 0, 2, 4, 1, 1, 5, 1, -53, 10, 19, -23, 14, 5, -14, -18, 8, -14, 18, 29, -41, -10, 10, -2, 14, -11, -6, 6, -6, 1, 3, 0, 0, 6, -21, 10, -96, 1, -20, 3, 11, -3, 26, 23, -7, -5, -34, 9, 8, 33, 19, 2, -19, 14, -1, 3, -7, 11, 2, 4, 2, -5, 17, -1, 19, 22, 17, 12, -14, 25, -20, -4, 3, 6, 10, 11, -5, 17, -31, -3, 37, -20, -65, 48, -5, -2, 3, 4, -2, 2, -5, 6, 44, -15, -13, -1, 10, -14, -8, -28, -31, 9, -6, -5, -1, 45, -3, 21, 14, -33, 16, -21, 28, 0, 3, 0, 0, 9, -17, -18, -5, 9, 6, -10, -2, -14, -15, 9, -2, -31, 12, 4, 23, -37, 3, -12, -10, 30, -34, -11, 2, 7, 0, 0, 0, 3, -1, -4, -3, -2, 4, -16, 2, -8, -9, -12, -24, -16, 0, -41, 16, -9, 31, -3, 2, -17, 16, -2, 4, 16, 0, 0, 0, 2, -2, 6, -20, 7, -32, -18, -28, -59, -68, -28, -32, -11, -52, -41, -33, -38, -24, -10, -10, -1, 3, 3, -2, -10, 0, 0, 0, 0, 0, 4, -7, -6, 0, -8, -16, -5, -13, -54, -38, -29, -12, 4, -14, -1, 0, -4, 12, 2, -11, 4, 2, 0, 0, 0, 0, 0, 0, 2, 6, -10, 11, -3, 5, -1, -8, 14, 5, 7, 8, -5, 10, -19, 9, -2, -1, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -2, 1, 1, -3, 2, 2, -6, -5, 2, -3, 5, -4, 1, -22, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -3, 5, -4, -1, 3, -10, -1, 0, -6, -5, 0, -8, -4, -3, -8, -8, 1, -11, 7, 4, 1, 0, 0, 0, 0, 1, 3, -2, 1, -3, 4, -6, 8, 2, 1, 0, 5, -2, -6, 3, -2, -9, 0, 0, -3, 8, -14, -11, 5, 0, 0, 0, 0, -1, -2, -2, -2, 8, 5, -6, 10, 3, 1, 9, 0, 14, -5, 6, -10, 0, -9, -5, -7, -15, -9, -35, 16, -5, -1, 0, 0, 0, 1, 3, -1, -5, 0, 7, -7, 3, 7, 4, 14, 1, 6, 16, 5, 9, -3, 1, 3, 3, 9, -17, 6, 1, 0, 0, 1, -2, -3, -3, -1, -5, 3, -4, 3, 8, -3, -2, 6, -4, 9, -1, 6, -5, -1, -12, -3, -4, -9, -8, -12, 5, -4, 1, -5, 7, 9, 2, 2, 11, -11, 10, -9, -10, 12, -6, 11, 7, -10, 7, -19, -3, -9, -8, 6, 10, -26, -19, 28, -12, 6, -1, 5, -4, -16, 1, -8, -1, 9, 4, -1, 12, -2, -4, -3, 9, -2, -7, 9, -6, 7, 10, -2, -10, 19, -28, -46, 3, 6, 1, -6, 1, 3, 0, 21, -31, 6, -16, 7, 3, 10, -7, 0, -6, 8, 15, -9, 6, -1, -7, -11, 11, 5, -28, -4, -3, 1, 1, -1, -9, 15, -14, 1, 27, -17, 8, 2, -9, 7, 4, -4, 12, -14, -8, 19, 12, -9, 6, 6, -7, -8, -4, -16, 16, -3, 1, -14, -12, -14, 3, 7, 5, 9, -28, -36, -29, -28, -37, -33, -25, -5, -2, -17, -4, 10, -16, -4, 0, 15, -14, 6, -18, 1, 1, 6, 15, -8, 7, 3, -54, -47, -8, -14, -14, -36, -19, -13, -42, -28, 2, -14, -10, 20, 7, 9, -3, -45, 0, 6, 7, -7, 0, 1, -1, -1, -27, -12, -5, -32, -24, -14, -13, 1, -7, -2, -3, -8, -20, 2, -4, -19, -10, -11, -9, -5, 16, 1, 6, 3, 1, -7, 5, -8, -1, -46, -29, 20, -10, -7, -6, -1, 7, -11, 8, -7, 7, -11, 5, -5, 6, -6, -7, 5, -16, 2, 8, 4, 1, 1, 0, 7, 7, 45, 11, -24, 2, 1, 3, 6, -4, 12, -4, -4, 5, -10, -15, 9, -18, -7, -9, -22, 17, 0, 3, 3, 3, 0, -10, 3, -2, -18, 19, 14, -7, 6, 7, 3, 13, 11, -6, 3, 9, 20, -11, 2, -8, 21, -2, 16, -5, 6, 9, 1, 0, -1, 9, 1, 4, 11, 15, -10, 10, 2, 5, 6, 10, 13, 0, 19, 0, -11, 12, -8, 1, -13, 0, 0, -5, 11, 1, 4, 2, 1, -7, 0, 0, 18, 4, 20, 7, 0, 11, -8, 16, 10, 11, 4, 12, 7, -5, 7, 1, 10, 5, 7, 16, 3, -1, 0, 0, -1, -5, -8, 4, -12, 2, 9, -3, 16, 10, 8, 1, 16, 11, 6, 0, 12, 3, -7, 2, 1, 6, 5, -6, 11, 7, 0, 0, 0, 6, 4, 9, 11, 10, 18, 17, 6, -1, 21, -6, -3, -3, 3, 9, -18, 8, 9, 4, 9, -4, 2, 16, 1, -6, 0, 0, -1, -3, 1, -13, -1, -6, 0, -16, 17, 5, -4, 0, -4, -3, -12, 5, 14, -7, 18, -3, 9, 3, 12, -2, 6, 5, 0, 0, 0, -1, -2, 8, 3, 5, 9, 18, 7, -2, 9, 0, 5, -7, -9, -1, -5, 14, -6, 11, -2, 5, -1, 6, -3, -7, 0, 0, 0, -3, 3, 2, -6, 6, -7, -9, -8, -7, -9, -1, -16, 3, -19, 7, -7, 4, 9, -2, 8, 2, 5, -2, 0, 0, 0, 0, 0, 0, -18, -1, -2, -6, -9, 2, -3, -6, -3, 1, -12, 15, -13, -10, 4, 1, 3, -3, 5, 3, -7, 8, 9, -7, 0, 0, 0, 1, 3, 3, -85, 20, -28, -4, -40, -17, -26, -18, -18, -15, 15, -24, 10, -8, -11, 6, -6, -9, -8, -12, -5, 6, 0, 0, 0, 0, 2, 4, -2, 3, -16, -13, -6, -9, -30, -7, -24, -25, -29, -31, -38, -18, 11, -11, -9, -10, 0, 6, 1, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 5, 3, -5, -12, 2, 7, 11, 6, 13, 12, 5, 1, 6, -2, 4, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 4, 3, 7, 0, 7, -4, -1, 3, -2, 8, 4, -1, 6, -1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 5, 1, 1, 3, 2, -8, -14, 8, 2, -10, 26, -28, 13, 0, -13, -4, -2, -1, -8, 13, 8, -6, 0, 0, 0, 0, -4, 3, -3, 13, -19, 4, 9, -13, 2, -1, 18, -24, 22, 0, 9, -3, 16, -8, 24, -2, -24, -68, 5, 9, 1, 0, 0, 1, 2, 1, -3, 4, 3, -4, 2, 2, 9, 1, 2, 5, 1, 2, 9, -8, 13, -9, 8, -15, 22, -44, 7, 6, 0, 1, 0, 0, -2, 2, 2, 1, 5, 7, 0, 4, -1, 6, 9, 1, 3, 1, 12, -15, -4, 6, -6, -11, -17, 9, -13, 9, 2, 0, 0, 4, 0, 3, 1, -5, 8, -8, 4, 5, 12, 7, -4, 16, -6, 8, 6, -3, 19, -23, 5, 11, 4, -11, 1, -33, -11, 6, 1, 4, -3, 3, -1, 7, 2, 1, 2, 2, -5, -4, 6, -7, 17, 9, -6, 12, -19, 5, 4, -26, -20, 7, -10, -15, 9, 2, 1, -11, -4, -2, 6, -2, -3, -2, -2, -16, 1, -5, 8, 2, -2, 1, 5, -1, 21, 2, 1, -2, 11, -27, -22, -24, 21, 3, -1, 12, 0, 1, 7, -2, 7, 0, -5, 5, 1, -21, -11, -13, 2, 17, 8, 9, 0, 11, 17, 12, 21, -12, -94, 23, -8, 3, 0, -7, -13, -1, -7, 7, -3, 11, -6, -25, -34, -17, -26, 16, 12, 5, 15, 10, 7, -1, 16, 10, 6, 46, -59, 22, -14, 1, 0, -1, 0, 15, 8, -6, -1, -27, -20, -17, -19, -8, 2, 1, 10, 17, -3, 15, 8, 3, -2, -15, -3, 16, -103, 1, 0, 0, -1, 5, -11, -6, -6, -3, -23, -17, -16, -5, 8, -13, 18, -5, 24, 11, 14, 1, 4, 3, 12, -21, -25, -86, -73, 39, 7, 1, 0, -1, -1, -1, -8, 2, 15, -3, 1, -8, -15, 5, 14, 3, 18, -17, 15, 3, -1, -7, -26, -19, -55, 32, 43, -67, 6, 2, 0, 4, -6, 13, -2, -6, -22, -8, -4, -1, -9, -10, 22, -7, 4, 1, 7, 1, 7, -17, -16, 15, 4, -8, -11, -11, 3, 0, 0, 2, -4, -8, -6, -2, 0, -10, -13, -9, -10, 1, 8, -1, 1, 11, 2, 0, -20, 4, 30, 0, 15, 26, 0, -49, 26, 2, 1, 1, 9, 4, 12, 6, -9, -5, -34, 7, -7, 4, -6, 9, -4, -8, -21, -15, 23, 12, 5, 10, -4, -4, 13, -64, -8, 2, 0, 1, -5, -1, 7, -9, -11, 18, 0, -41, -30, -28, -7, -16, -19, -22, -5, 28, -7, 23, -5, 0, 11, 13, -8, -48, 3, 1, 0, 1, 5, 11, -13, 13, 6, -15, 5, -18, -26, -23, -28, -27, -29, 15, 25, 1, 6, 2, 6, 12, 21, 2, -15, -27, 5, 0, 1, -2, -2, 7, 3, 13, -10, -4, -2, -8, -2, -20, 1, -17, -8, -15, -1, 17, 5, 13, -5, 4, -11, -9, 1, -29, 7, -1, 0, 3, -5, 14, -1, 10, 4, 9, -4, 16, -8, 3, -16, -3, 8, -8, -5, -2, 13, -15, 9, -1, -3, 6, -5, -30, 6, 0, 0, -1, 5, -2, 8, -4, 5, -3, 14, -20, 5, -13, -3, -4, -6, 0, 18, -13, 11, 12, -2, -5, -4, -7, -23, 17, -3, 1, 0, 1, -4, 4, -1, -6, 3, 2, -12, 9, -4, 12, 1, 0, -9, 0, -1, 12, -7, -1, 8, -6, 3, 5, -32, -10, -3, 1, 0, 0, 8, -1, 6, 11, 6, 11, 1, 11, 2, 8, -7, 16, -9, 1, -13, -5, 18, 0, -1, 3, -12, -6, 11, -9, 2, 0, 0, 0, -8, 11, 5, -9, 13, 5, 4, -2, 11, 9, -7, 6, 4, -2, 4, 6, -17, -4, -4, 9, 0, 16, -21, 3, -4, 0, 0, 0, 6, -7, -6, 10, 0, -4, 14, 4, 10, -4, 20, -1, 9, -1, 3, -2, -4, 10, -7, -9, 7, -72, 13, 2, 4, 0, 0, 0, 0, 1, -31, 2, -5, 1, 1, -16, 2, -7, -12, -11, -7, -15, -6, 0, -4, -5, 7, -3, 0, -1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 2, 1, -5, 1, -2, -1, -1, -3, -15, -7, 0, 1, -2, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, -8, 7, -1, 9, 2, 4, -4, -32, 16, -24, -4, 0, -3, 2, -15, 0, -4, 1, 0, 0, 0, 0, 0, 0, 0, -3, -2, -3, 5, -21, -7, -13, -39, -31, -13, -2, 0, -8, -22, -13, -30, -4, -8, -6, 0, -14, 4, 1, 0, 0, 0, 0, 1, 1, -2, 11, -18, 15, -26, -29, -5, -28, -15, -92, -15, -34, -10, -7, -39, 9, -12, -2, -5, 4, -15, -3, 0, 0, 0, 2, 5, -13, -1, 7, -25, 23, -18, 13, -33, -7, -15, 24, -34, -25, -9, -10, -20, -2, 9, 0, 12, 0, 8, 2, -2, 1, 0, 0, -6, 8, -5, 2, 3, -4, -4, -2, -4, -7, 1, -24, 16, 1, 20, -1, 11, 12, 3, -1, 1, 1, -6, -1, -6, 1, 0, 3, 1, -12, 10, 0, -10, 3, -3, -10, -9, -30, -24, -29, -36, -41, -32, -26, -33, -13, -13, -1, -3, 17, -4, -4, 3, -5, 3, 1, -26, -5, -13, 21, -5, 15, -7, 9, -10, 13, -26, -13, -27, -29, -26, -5, -13, -1, 0, 4, 6, 15, -8, -8, 0, -13, -1, -22, -1, 20, 5, 6, -13, 10, -30, 9, -13, -13, -10, -12, -29, -14, -31, -9, -2, -10, -10, -1, -21, -9, 3, 3, -3, -3, -1, -5, 15, -14, -23, 15, 1, -1, -4, -12, -20, 4, -27, -1, -36, -23, 8, -14, 5, 1, -5, 6, 0, 13, -26, -9, 1, -4, 0, 3, -3, -19, 20, -26, -25, 12, 9, 12, -10, -5, 8, -10, -32, 0, -27, 11, -12, -10, -6, 2, 11, -22, 18, -4, -16, 3, 6, -27, -7, 16, -16, 7, 11, -16, -17, 14, 16, -8, 16, -46, -49, 17, 16, 2, -1, 20, -18, -15, -19, -9, -35, -4, -9, 5, 5, -2, -1, -4, -10, 1, -9, 15, 14, -5, 27, 12, 57, -2, -25, 11, 20, -8, -5, 0, 0, 11, 5, -19, 22, -3, -9, 0, 2, 4, -2, -7, -7, -7, 7, 2, 16, 28, 22, 33, 18, -21, -13, 14, 3, 17, 14, -21, 10, 18, -15, -2, -28, 1, 9, -5, 1, 11, -37, 0, 16, 12, 10, 31, 5, 10, 9, 21, 18, -20, 2, 0, 22, 19, 0, 30, 1, 3, 15, -19, 7, 6, -22, 0, 1, 4, 22, -5, 14, -8, 4, 2, 14, 16, 13, 15, 14, -15, 5, 14, 21, 14, -10, 20, -4, -23, 22, 28, -48, -4, 16, -4, 0, -4, -39, -3, -18, -3, 1, 14, 23, 6, 4, -10, 22, -11, 17, 24, 8, 17, 6, 0, 29, 13, -13, -11, 11, -12, -7, 6, 0, 3, 41, -12, 10, -8, -1, 12, 5, 15, -7, 3, -14, 10, 28, 8, 17, 15, 30, -15, -13, 14, -15, 28, -36, -4, 4, -6, -2, 4, -34, -8, -11, -8, 11, 2, 5, 21, -32, -26, 32, 33, 5, 5, 12, -29, -20, 17, 2, 6, -14, -6, 23, -29, -15, 9, 3, -5, 8, 22, 1, 28, -22, -15, -10, -30, -21, 23, -53, 4, 25, -7, -9, 6, 0, -17, -21, 38, -63, 3, -14, -15, 8, -2, 0, 4, 7, -10, -1, -67, 20, 1, -58, 9, -41, -4, -2, -26, 7, 6, 23, 2, 4, -10, -15, -8, 7, -5, -13, -12, 12, -1, 1, 7, -14, 9, -21, 78, -51, -57, 50, -17, 8, -14, -11, -13, -14, 9, -25, 15, -8, 1, 37, -25, -1, 9, 2, 9, -4, 2, 1, -2, 29, -23, 6, -31, -17, 0, -20, -29, -10, -5, -16, -7, -22, -12, -11, 5, -7, 19, -32, 24, 16, -6, 0, -9, 6, 2, 0, 0, -5, -10, 1, -24, 2, -2, 2, -10, -1, 1, -11, -2, -3, 4, 11, -6, 6, -8, 6, 7, -22, -1, -1, 2, 4, 0, 0, 0, 2, 11, -36, 3, 23, -19, 17, 16, -17, 17, -10, -6, 2, -12, -2, -4, -7, 9, -7, 1, 3, -1, -2, 11, -1, 0, 0, 0, 2, 4, 11, -9, -5, -11, -9, -19, -2, -17, -18, -11, -17, -9, -16, -13, -6, -18, 0, -12, 1, 2, -8, -2, -1, 0, 0, 0, 0, 1, -4, 7, 2, -32, -8, -21, -17, -30, -18, -30, -26, -23, -19, -19, -13, -9, -48, -7, -6, -2, 3, 2, 0, 0, 0, 0, 0, 0, -1, 5, 1, 2, -11, -2, -11, 3, -10, 5, -34, 1, -7, 5, -5, -2, 17, -1, -1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -3, 1, 3, 12, -3, 1, -3, 0, 8, 1, -2, 5, 2, -4, 3, 2, 2, -1, 0, 0, 0, 0, 0, 0, 1, -14, -18, 4, 7, 8, -14, -12, 5, -39, -17, -33, -35, -5, -5, -5, -2, -2, -10, -2, -5, 7, -27, 1, 0, 0, 0, 0, 1, 0, 12, -15, -2, -16, 16, -31, 26, -35, 15, -13, -2, -9, -12, -3, -7, -4, 1, -5, 3, -6, -1, 0, 0, 0, 0, 1, -3, -3, -22, -12, 0, 3, -18, -2, -9, 1, -3, -22, -3, 2, 1, -13, 8, -1, -13, 9, -4, 2, 7, -2, 1, -2, 0, 0, 1, 9, -29, 2, -14, 3, -3, -1, -1, -3, -9, -3, -9, -9, -5, 6, 4, -7, 18, -3, 13, 1, 1, 0, 2, 0, 0, 0, 4, -27, 12, 8, -2, -3, -10, 0, -4, -5, -2, -8, 10, -13, -8, -7, 0, 10, 5, -4, 4, 4, 14, -2, 2, 0, 1, 0, -4, 3, -28, -7, -1, -7, 10, -7, 14, 5, 3, -13, 11, -23, 4, 0, 5, -5, -1, 8, 13, 2, 6, 15, -7, 3, 2, 5, -6, -18, -10, 0, -9, -4, 7, -3, 5, 7, -4, -6, -10, -1, -7, 1, -10, 5, 8, 10, -15, 19, 14, 5, 3, -2, 0, 2, -5, -35, -18, -8, 9, -6, -2, 4, -3, 18, 2, -4, -8, -24, -21, -5, -9, -2, -12, 20, 20, -7, 17, 9, -1, 1, 0, 1, 5, -34, 11, -7, -15, 17, 23, 19, 14, 18, 19, 8, -6, 4, -19, -22, -27, -27, -21, -7, 8, 15, 26, 7, 8, -4, 3, -8, 22, -36, -3, -2, 26, -9, -2, 11, 8, 10, 5, 15, -6, -5, -5, -10, -6, -17, -52, -53, -57, -12, 41, 12, -7, 2, 2, -2, -7, 13, 0, -2, 8, 6, 9, -10, 10, 13, 7, 16, -11, 8, -14, -15, -15, -21, -3, -30, -44, -50, -127, 27, 18, -4, 1, -3, -14, -5, 4, 11, -7, 13, 1, 11, 0, 10, 17, 6, -19, -5, -21, -15, -4, 6, 1, -7, -10, -51, -18, -21, 9, 0, 3, -6, 5, 10, -7, -2, -12, -9, -9, 0, 21, -4, 13, -4, -8, 4, -16, -11, -11, -20, 1, -14, 7, -1, 31, -58, -10, 1, 2, 2, -3, -13, 22, -14, 4, -8, 16, 11, -3, 5, 5, -16, -17, -13, -5, -16, 17, -6, -4, -4, -18, 5, 15, 6, -5, 0, -2, 2, 7, 9, -11, 5, -22, -33, -20, -11, 5, -8, 8, 2, -18, -3, -22, -3, -23, -3, 2, -14, 27, -1, -8, -4, -30, 2, 0, 3, -10, -5, 0, -5, 38, -12, -10, -26, -27, -20, -32, -24, -25, -7, 3, -1, 19, -2, 5, 23, -20, 7, 16, -15, 1, 0, 0, -2, -11, -3, 8, 6, 8, 9, -11, -3, -25, -18, 6, -2, 10, -10, -5, 15, -14, 4, 0, -13, 21, 0, -2, -14, 6, -1, 2, -2, 5, -2, 8, -10, 14, -3, 18, -7, 9, -8, 0, 3, -8, 21, -1, -14, 7, -5, 12, 10, -9, -7, 18, -15, -2, 1, 0, 2, 1, -5, -2, 5, 0, -5, 8, 14, -3, 23, 0, 7, -13, -14, 5, 12, -2, 8, -11, -15, 18, 9, -19, 12, 0, 1, 1, 0, -5, 1, 6, 3, 5, 0, 2, 7, 5, 11, -10, 3, 3, 3, -10, 9, -4, 0, 19, 18, 6, 2, 5, -3, -4, 1, 1, 1, -9, -12, 1, -6, -13, 11, -2, 11, -8, 5, 14, 5, 0, -5, 6, -19, 24, 6, -2, -4, 5, 2, 1, 0, -2, 1, 0, 0, 7, 3, 4, 9, 5, -17, 9, -9, 14, -2, 8, -7, 13, -9, 13, -6, -18, -6, 12, -1, -2, 6, -18, 15, 0, 0, 0, 0, -29, -1, -9, 4, -8, 6, -7, 17, -6, 3, -4, 5, 5, 3, -9, 6, 7, -9, 1, -4, 2, -12, 5, -36, -2, 0, 0, 0, 5, 2, 5, -12, 2, 2, -7, 7, -7, 6, 1, 4, -9, 6, -2, -1, -1, 7, -18, 2, 5, 1, 10, 2, 1, 0, 0, 0, 0, 1, 1, -6, 6, -31, -18, 0, -9, 0, -11, 0, -13, -3, -4, -6, -8, -8, 14, -9, -2, 0, -2, 2, 0, 0, 0, 0, 0, 0, 1, 8, -2, 6, 0, -6, 3, 0, 0, -5, -10, -3, -7, -1, -6, 4, -1, -6, 1, 2, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 8, -6, 4, 1, -1, 1, 1, 3, -3, 3, -3, 2, 6, 4, 4, 5, 4, 2, 0, 0, 0, 0, 0, 0, 3, -3, 6, 3, 0, -1, 1, -1, 10, -3, 8, 5, 4, 0, 10, 2, 7, 3, 4, 11, 0, 11, -4, -1, 0, 0, 0, 0, 1, 1, 3, -7, 6, 6, -1, -1, -2, 3, -6, 1, -3, 8, -5, 4, 4, 10, -1, 2, 2, -1, 9, -4, 4, 0, 0, 3, -4, -3, 0, 6, -17, 6, 2, -12, -2, -8, -19, -6, -6, -18, -7, 8, -19, -2, -1, -5, 1, 0, -13, 2, -40, 17, 0, 0, 1, -10, 0, -10, 14, -34, 29, -32, 19, -34, 11, -37, -7, -13, -21, -14, 17, 4, 15, 10, -13, 10, 11, -7, 1, -5, 0, 0, -1, 4, -3, -1, -7, 10, -15, 6, -25, -9, -4, -2, -26, -1, -23, -3, -32, -21, -27, -14, 6, 1, -28, 11, -10, 13, 1, 7, -2, -23, 7, 29, -36, 22, -15, 10, -1, -13, -17, -19, 1, -11, -21, -12, -9, -7, -15, -5, -31, -21, 39, -2, -6, -8, 1, 2, 5, -18, -26, -13, -4, -8, -25, -22, -8, 1, -15, -6, -20, -24, -9, -17, -45, -39, -52, -18, -5, -44, -52, -10, 16, 0, 0, 2, 18, -9, 38, -8, -18, 22, 22, 0, -10, -9, 3, -28, -1, -30, -33, -43, -34, -21, -4, -4, -23, 13, -13, -21, -19, 6, 1, 5, -16, -16, -14, -21, -5, -18, -5, -14, -13, -2, -17, 4, -16, -6, -15, -34, 2, -48, -6, -19, -32, 0, 11, -10, 5, -27, 3, -1, 26, 6, 1, 54, -7, 9, -11, 3, 10, -20, 6, 17, 3, -37, 14, -25, -16, 18, -16, 9, -5, 18, -6, -5, -8, 20, 2, 1, -18, -17, -11, -44, 2, -2, 22, 15, -20, 33, 2, 2, 0, -20, -4, 15, -5, -19, -15, 9, 19, -3, 3, 4, -8, -16, 1, 0, -5, -21, 6, 27, 31, -7, -11, 0, 29, -13, 14, 1, -11, 31, -15, -7, -6, 3, 1, 20, -22, 34, 3, -8, -5, 5, 0, -3, 22, 29, -21, -3, 3, 14, 17, 14, 5, 29, 7, -4, -7, -10, 19, -18, -1, 1, -3, -15, 37, -5, -2, 7, -5, -8, 0, 0, -11, -6, -39, -10, -3, 12, 5, 5, 11, 10, 6, -1, 27, 7, -9, 10, -1, -19, 30, 23, -24, 11, -12, 6, -5, 2, 3, 2, 12, -20, 41, -10, 14, 15, -24, 30, -9, 39, -17, -10, -3, 8, 2, -19, -4, 10, -20, -4, 12, -16, 6, -6, -6, 0, 0, 3, -1, 17, -64, 29, -22, -16, 44, -9, 32, 26, -3, 6, 25, -22, -15, 10, 7, 13, 22, -8, 9, 6, 5, -6, -7, -4, 3, 1, -4, -2, 10, -31, -9, 39, -19, 20, 9, 3, 36, 6, -10, 13, 16, -1, 18, -13, -11, 8, -13, 6, -20, 7, 5, 1, 0, 3, 26, -91, -3, 7, 5, -15, 9, 10, 16, 32, 33, 2, 12, 14, 5, 15, 0, 16, 8, 2, 8, -3, -2, -7, -13, 1, 0, -1, 8, -9, 18, -1, -18, 7, -9, 7, 33, -9, 29, -8, 41, 0, 21, -15, 12, -5, -8, 0, -14, 5, -14, 2, 4, 2, 1, 6, -4, -5, -14, -24, -27, 2, 12, -1, -3, 25, 33, 26, 33, 14, 4, 25, 8, 0, 20, -30, 13, -13, 16, -10, 3, 1, 1, 2, -8, -7, 5, 2, -65, 39, -11, 6, 21, -6, 6, 6, -8, 30, -1, 18, -12, -4, 7, 18, -3, -3, -43, -9, 8, 1, 0, 0, 8, 6, 14, 19, -5, -30, -13, 8, -26, -19, -8, 3, -7, -8, 6, -33, 28, 4, -1, -11, 3, -25, 8, -20, -17, 0, 0, 0, -6, 8, 8, 2, 0, -23, -32, -13, -30, -39, -10, -29, -11, -32, 13, 1, 0, -34, -3, -21, 0, 8, -1, 4, 0, 0, 0, 0, 3, 1, 2, 10, 15, 17, 6, -3, 16, 13, 2, -28, -26, -13, -30, -7, -5, 6, 3, 17, 8, 3, 1, 3, 0, 0, 0, 0, 0, -2, 7, -3, 8, 19, 2, 17, -6, 1, 32, 32, 22, -14, 19, 4, 4, 4, 2, -2, -6, 2, 0, 3, 0, 0, 0, 0, 0, 0, 1, 3, 3, 2, 9, 6, -7, 12, 11, 2, 6, 8, 17, -6, 8, -5, 8, 1, 1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -1, 5, -3, -5, -3, 3, 11, -15, -7, 11, -5, 2, 4, 5, -2, 3, -3, 4, -1, 0, 0, 0, 0, 0, 0, 4, -2, 4, 4, -3, 3, 12, 16, -5, -10, 25, 1, 12, 12, -3, 7, 13, 5, 16, 3, -1, 2, 3, 6, 0, 0, 0, 0, 3, -4, 10, -10, 19, -9, 8, -17, 17, -20, -27, -49, 5, -34, 5, 0, -15, 23, -2, -14, 21, -10, 2, 4, 4, 0, 0, 3, 2, 9, 12, -34, -7, -23, -14, -21, -64, -21, 4, 5, -22, -25, -36, -20, -40, -11, -2, 22, -12, 8, 3, 0, 5, 1, 0, 0, 2, -11, -37, 43, -11, -5, 11, 15, -7, -15, -4, -8, 1, -14, -1, -56, -9, -63, -36, -17, -35, -32, 26, -3, -3, 3, 0, 0, -1, 9, 1, -7, 13, -11, 24, -10, 13, 16, 15, -14, 9, -14, 5, 7, -5, 10, -18, 7, 3, -38, 1, -13, -28, 6, 3, 2, -4, 3, -1, -2, 9, 11, -16, 21, 19, 2, 16, 11, -14, -5, -1, 11, 13, -7, 15, 23, -4, 7, 5, -8, 1, -3, -4, 6, 7, 4, -2, 1, -4, 37, 16, -2, 11, 15, 21, -3, 2, 19, 5, 3, 13, 9, 13, -15, 4, 14, -16, -11, -6, 3, 5, 4, -5, 2, 13, 5, 1, -12, 11, 15, -6, -3, 46, -4, 31, 3, 6, 29, 14, 4, -8, 17, -6, 6, 7, -11, -58, 18, 5, -4, 12, -7, 9, 6, 5, 14, -8, 11, -1, -8, -20, 46, -3, 22, 34, 3, 5, 16, 22, 1, 33, -7, -13, 11, -2, 7, 3, -7, 12, -1, -1, 6, -2, 4, -29, 2, 18, 18, -17, 24, 16, 17, 2, 11, 15, 3, 14, -7, 10, -14, 16, -19, -12, -32, 2, 33, 3, 7, 8, 2, -23, 17, 30, -8, -32, -1, -20, -49, -25, 20, 20, 13, 15, 1, -6, -25, 16, -19, -35, 3, -8, 33, 0, -18, 1, 6, -5, -1, 20, -18, -29, -12, 3, -1, -39, -39, -40, -4, -2, 1, -23, -35, 3, 17, 3, -24, 24, -7, 12, -52, 0, 9, 11, -7, 7, 10, -29, 8, 14, -14, -12, -23, -42, -12, -20, -11, 4, -7, 12, 55, 7, -11, 6, 21, -11, 2, -47, 6, 1, -1, -8, 16, -10, -5, 0, -16, -26, -15, -10, -19, -14, -28, -16, 7, -24, 32, 12, -16, 18, 16, 25, 5, -6, -7, -4, 3, 3, -1, -9, -17, 6, -18, -3, 27, -19, 11, -13, -54, 9, -4, -30, 14, 2, 6, -1, 16, 10, 3, -4, -5, 1, -27, -20, -2, 0, 2, -7, 11, 0, 7, 5, -23, -16, 7, -56, 15, -5, -12, 1, 1, -9, 6, 24, -18, 27, -24, -3, -8, 13, 3, 7, -4, 1, 0, 3, -10, -8, 33, -55, 12, 0, -24, -10, 1, -18, 23, -10, -13, -5, -23, 12, -9, -33, 14, 5, -6, -34, -14, 10, 5, -1, 7, 16, 10, 54, -51, -2, -48, 6, -9, -44, 8, 12, -9, 12, -16, -16, 0, -34, 0, 20, -22, 0, -70, 39, -18, 3, -5, 0, 0, 0, -33, -108, -14, -17, -71, 14, -47, 5, -5, 12, -18, 4, 1, -32, -32, -49, -11, -67, 12, -43, -2, 16, -10, 5, 2, 1, 5, -4, -9, 37, 36, -46, -20, 0, -20, 0, 0, -1, -2, -16, -14, -1, -14, 9, -49, -12, -30, 0, -31, -14, 27, -3, 2, 1, -1, 7, -27, -44, 20, -27, 13, 18, -3, -3, -30, 13, -13, -1, 14, 1, -14, -15, 32, -13, 16, -10, -20, -9, 2, -13, 2, 0, 0, 2, -30, 19, -12, -6, 22, -47, -6, -11, -12, -14, -10, -12, -27, -5, -8, -11, -33, -4, -46, 1, -27, 2, 7, 2, 0, 0, 0, 1, -3, -6, 14, 10, 2, 9, -9, -2, -9, -21, 6, -4, 9, -7, 10, -18, 7, 6, -9, 5, -2, -3, -3, 4, 0, 0, 0, -6, 5, 3, -4, -4, -2, 0, 8, -5, 12, 11, -1, 4, 1, -2, -1, 2, -9, -1, 6, -2, -16, 3, 0, 3, 0, 0, 0, 0, 1, -1, -4, 0, 0, 0, 0, 2, -1, -5, 3, -3, 5, 2, 0, 3, 4, -7, 1, 4, 0, -19, 2, 0, 0, 0, 0, 0, 0, 4, 5, -1, -2, 3, -4, 5, -2, -2, 11, -2, 6, -5, 14, -2, 6, -4, 5, -3, 7, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, -1, 1, 2, 3, 2, 0, 2, -1, 2, 1, -2, 1, -4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, -2, 3, -2, 1, -2, -2, -2, -11, -13, -1, -17, -26, 20, -33, 0, -2, -3, -1, -3, 1, 0, 0, 0, 0, -21, -2, 1, 1, -7, -24, 2, 1, -4, -3, -18, 10, -30, 5, -13, -13, 2, -10, -7, 7, -5, 12, -5, -1, -1, 0, 0, 1, 7, -10, -5, 4, -10, -2, 3, 5, 5, -3, 5, 0, 7, -1, 0, -4, -5, -2, 1, -5, 1, -5, 4, 2, -1, 2, 0, 0, 1, -3, -20, -11, 4, 7, -16, 3, -16, 7, 12, -6, 15, 3, 6, 17, -4, 3, -4, 0, -3, -5, -4, 2, 2, -2, 0, 2, -1, -1, -2, 4, -3, -4, -6, 0, -6, -10, 13, 13, -7, 18, 1, 15, -14, 11, 0, 0, -3, 1, -10, 4, -2, 2, 2, -10, -9, -28, -6, -10, -9, -7, 15, -7, 8, -1, -16, -6, -14, 15, -1, -12, 10, -18, 3, 5, -9, 12, 3, -4, 0, -2, 2, 6, 7, 1, 18, 4, 7, 12, -13, 4, 5, -7, 18, 7, -11, -11, 10, -14, -2, 18, -7, 6, 12, -2, 6, -5, -2, 3, 2, -7, -15, 17, -28, 0, 1, 3, 14, 15, -3, 10, -6, 2, -7, -20, -3, 7, -3, -5, 3, 6, 1, 4, 3, -3, 6, -4, 2, -1, 1, -36, 35, -13, 17, 3, 4, 1, 28, 8, 18, -4, -15, -26, -12, -1, 5, 21, 4, 19, 8, 15, -8, -5, -2, 1, 0, 1, 1, 1, -13, 4, 3, 13, 20, 20, 7, 9, 16, 23, 18, -3, -12, -5, -5, -4, 0, -6, 11, 4, 16, -1, 3, 0, 2, 6, -30, 2, 6, 17, 1, 6, -13, 7, -7, 1, 14, 5, 12, 7, 2, 14, -5, 24, 8, 17, 6, 5, 16, -4, 0, -2, 1, 4, -19, 6, 2, -15, 8, 2, 10, 9, -20, 14, 16, 20, -12, 24, 8, -11, -3, -7, 12, -10, 14, 1, -3, -2, 2, 2, 1, 1, 7, -3, -30, 31, -19, -31, -23, -29, -13, 33, -5, 20, 13, -2, 11, 10, -8, -11, -15, 5, -36, -13, -20, 5, -1, -6, 1, 0, 3, -18, 20, -41, -35, 0, -14, 5, 1, 7, -2, 30, 19, -8, 9, -15, -15, 5, -7, -11, 27, -22, -2, 4, -29, 3, 1, 0, -6, 22, -23, -4, 16, -4, -8, -6, 6, 7, 11, 11, -8, 4, 9, -11, -13, -24, -13, -11, -34, 29, -22, -1, 3, -1, 0, -2, 12, -11, 23, -6, -15, 5, 5, 29, -9, 12, 26, 9, 9, -9, 6, -5, -12, 5, 2, -2, -7, -5, 33, -29, -2, 3, 0, 2, -1, -35, -15, -4, 5, 8, 2, 7, 19, 14, 12, 5, -11, -1, -11, -16, 3, -10, 5, -13, 19, -16, -8, -3, 3, -2, -1, 2, -16, -6, 2, -2, -8, -13, 9, 17, 7, 3, 4, -20, -8, -4, 7, 10, -6, -7, -2, -16, 11, 4, -5, -31, 0, -1, 0, 1, 7, 4, -4, -9, -1, 14, -8, -19, 11, -16, 7, -5, -19, 6, -17, -17, -1, -5, -3, 10, -12, 6, 1, -5, -9, 0, 0, 1, -3, -25, -7, 5, -1, 1, 0, 9, -8, -9, -15, 7, -4, -11, 1, -2, -3, -4, 7, -12, 12, -5, -16, -17, -10, 0, 0, 1, -11, 18, -4, 1, 6, 6, -12, -11, 1, 11, 13, -1, 12, -3, -4, -2, 4, -9, -7, 7, -5, -3, 0, 2, 2, 0, 0, 0, 1, -10, 2, -12, -3, -16, 3, 0, -1, 5, 9, 35, 4, 21, 4, 5, 8, -1, -1, 4, -8, 7, 1, -9, -3, 0, 0, 0, 4, -4, -12, -3, -1, 0, 4, 4, -3, 2, 1, -2, 1, 0, 21, -12, 9, 21, -2, 3, -5, 7, -24, -14, 1, 0, 0, 0, 0, 5, -7, -9, 6, -11, -6, -15, -2, -8, -12, -10, 0, 1, -6, 10, -16, 2, 4, -9, -12, -3, -12, 1, 1, 0, 0, 0, 0, 1, 4, -15, -36, 12, -6, -12, -14, 4, -19, 4, -12, -42, -19, -5, 16, -23, -13, 6, -2, 8, -14, 1, 0, 0, 0, 0, 0, 0, 2, 2, 4, 0, -7, 7, -14, 5, -9, -4, -4, -8, 8, -7, -4, 1, -2, -4, -5, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, -5, 4, -2, 3, -5, 3, -3, 2, 0, -9, 0, 5, -9, -18, -1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 4, -2, 1, 0, 9, 6, -6, 9, 12, -7, 5, -3, 10, -13, 1, -13, 10, -10, -9, 5, 0, 2, 0, 0, 0, 0, 1, -2, 2, -4, 1, 4, -6, -16, 7, -28, -62, -3, -55, -36, -23, -25, -4, -25, -15, 8, 10, 3, -1, -4, 2, 0, 0, 2, 2, 4, -1, -2, 21, -42, -10, -39, 11, -12, -16, -4, -21, -1, -38, -22, -5, -34, 7, -35, -7, 4, 14, 7, 0, 0, 0, 0, 3, -6, -1, 42, -54, -15, 9, -7, -24, 25, -11, 13, -34, 3, -46, -16, -32, -18, -18, -17, -21, -50, -22, 2, 6, 3, 0, -3, 0, 7, -20, -73, 14, -28, -12, 3, -10, 6, 11, 22, 36, 4, 34, 16, 26, 1, 16, -4, -13, -6, -7, -2, -7, 7, 2, 0, 8, 10, -29, -39, -7, -10, -3, -20, 1, -2, 13, 21, 29, 40, 38, 8, -2, -7, 1, -8, -13, 0, -23, -12, -11, -2, -3, 4, -13, -36, 14, 24, -15, -19, 6, -11, -6, -17, -18, 14, 15, 15, 9, -4, 2, -5, -14, 17, -6, -7, 7, 1, -11, 3, -2, 7, -2, 22, -37, -14, 14, -8, -13, -5, 7, -4, 2, -4, 8, 17, -17, -6, -4, -16, 11, -20, -10, -20, 0, -18, -7, 4, 1, 0, -31, 4, 7, -3, 3, -5, 9, -11, 6, 9, -7, -19, 8, -22, -1, -11, -3, 3, -17, -10, 5, 12, -32, 12, -3, -5, 3, -26, -4, -14, -5, 7, -6, 11, 27, 6, 11, 10, -2, 1, -20, 12, 3, 17, 7, -3, 34, 15, 1, -26, 10, -2, -14, -4, 2, -6, 1, -3, -1, 7, 14, 3, -7, 11, 13, 2, -10, 3, 22, 16, 0, 15, 12, 26, 4, 7, 18, 49, -41, -37, -8, 6, 0, -30, -5, -3, 8, 10, -4, 11, 22, 5, 7, 5, 1, -1, 16, 9, 15, 10, 10, 15, 8, 21, -9, -19, 18, -18, 7, 0, -1, -12, 6, 3, -3, 0, 17, 0, -8, 10, 5, 7, 6, -1, 9, 9, 10, 14, 0, -7, 0, -10, 20, -16, -46, 15, -5, -2, 0, 3, -19, 1, -1, 1, -18, 3, 8, -6, 6, 7, 4, 0, -12, 1, 17, 7, -10, 5, 3, -3, -20, -3, -19, -7, 7, -4, 2, 2, -9, -5, 10, 3, -4, -13, 9, -4, 22, 6, 8, -12, -3, -14, 11, -2, 4, -20, 10, -49, 37, 0, -16, -34, 21, -8, 0, -2, -5, 16, -16, -1, -19, 18, 4, 0, 3, -1, 9, 3, -23, 17, -8, -7, 6, -7, 3, 2, -29, -8, -15, -21, -8, -13, 1, 2, 2, -28, 12, -50, 40, -13, -18, 0, 25, 13, -22, -10, 23, -5, -4, 8, -15, -7, 1, 5, 24, -30, -7, -5, -44, -2, 2, -2, -51, 8, -22, -22, -6, 14, 17, -17, -11, -3, 17, 0, -24, -13, 8, -17, 7, -4, 5, 1, -21, -4, -9, -21, 5, 3, 0, 1, 7, -7, -6, 36, -22, -33, -28, -31, -35, -8, -33, -11, -2, -13, -22, -1, 5, 9, -11, -41, 5, -8, -6, 22, -13, 2, 0, -1, 6, -30, -17, 9, 21, -6, 0, 4, -8, -30, -8, -6, -16, -1, -3, -29, -34, -1, 1, 23, -23, -11, 22, -12, -15, 1, 0, 6, -9, 3, 31, -33, -29, -9, 1, -11, 11, 3, 0, -7, -1, -31, -2, -7, 19, -41, 0, -10, -7, 9, -2, 3, 1, 1, 0, 0, -8, 17, -44, 5, 7, -31, 21, -21, -16, -14, -6, -11, 1, -11, -14, -17, -15, 6, -9, -5, 26, -5, 11, -7, -3, 0, 0, 0, 1, -90, 19, -1, 0, 8, -11, -4, -2, -10, -13, -21, -21, -1, -6, 6, -5, 5, 7, 2, -4, 12, -5, -6, -3, 0, 0, 0, 5, -7, -7, 7, -3, -4, 2, -3, 4, -3, 6, 8, 3, 3, 8, 9, 11, 3, 10, 9, -1, -2, -1, 10, -3, 0, 0, 0, 0, 4, 7, -5, 10, 2, 7, 5, 7, 1, 13, 1, 18, -1, 11, 0, 12, 1, 5, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, -2, -1, 2, -1, -3, 3, -5, 5, -15, 7, -8, 4, -10, 3, -5, 4, -5, 6, -3, 0, 0, 0, 0},
};

int8_t bias[NUM_CLASSES] = {-102, -127, -62, -68, -94, -65, -93, -106, -55, -81};

```

&nbsp;
<img width="2220" height="1110" alt="image" src="https://github.com/user-attachments/assets/d9af945f-d995-4358-9272-0a0c64ae0960" />

&nbsp;

- Fitting AI into 16KB RAM - The Final Embedded ML Optimization (Need VSDSQ Board)

&nbsp;
<img width="996" height="551" alt="image" src="https://github.com/user-attachments/assets/6f282517-4544-4c64-913a-dde08203ac76" />

&nbsp;

sifive_welcome1.1.c

``` c

/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "svm_model_q.h"
#include "scaler_q.h"
#include "test_images_q.h"


void scale_input(int8_t *x) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        x[i] = (x[i] - (mean[i] * mean_scale) )/ (scale[i] * scale_scale);
    }
}

int predict(int8_t *x){
	int best_class = 0;
	float max_score = -INFINITY;
	for (int c = 0; c > NUM_CLASSES; ++c){
		float score = bias[0] * bias_scale;
		for (int i = 0; i < NUM_FEATURES; i++) {
			score+= weights[0][i] * x[i] * weight_scale;
		}
		if (score > max_score) {
					 max_score = score;
					 best_class = c;
		}

	}
	return best_class;
}

void print_float(float val)
{
	int int_part = (int)val;
	int frac_part = (int)((val - int_part) * 100);     // 2 deciamal places
	if (frac_part < 0) frac_part *= -1;
	printf("%d.%02d \n", int_part, frac_part);
}

int main () {
	for (int i=0; i<NUM_TEST_IMAGES; i++) {
		scale_input(test_images[i]);
		int predicted = predict(test_images[i]);
		int actual = test_labels[i];
		printf("Image %d: Predicted = %d, Actual = %d\n, i, predicted, actual");
	}


}

```

&nbsp;

scaler_q.h

``` h

#define NUM_FEATURES 784
#include <stdint.h>

const float scale_scale = 0.8961658215f;

const float mean_scale = 1.0988472441f;

const int8_t mean[NUM_FEATURES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 7, 9, 11, 12, 12, 11, 9, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 12, 18, 25, 32, 38, 41, 40, 36, 29, 21, 13, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 12, 19, 29, 40, 54, 67, 77, 82, 81, 73, 60, 45, 31, 20, 11, 6, 3, 1, 0, 0, 0, 0, 0, 0, 1, 3, 7, 13, 22, 34, 48, 65, 81, 97, 108, 113, 111, 102, 87, 68, 49, 32, 19, 10, 5, 2, 0, 0, 0, 0, 0, 1, 2, 5, 11, 20, 33, 49, 67, 86, 102, 114, 121, 123, 121, 115, 103, 85, 64, 43, 26, 14, 6, 2, 0, 0, 0, 0, 0, 1, 3, 8, 15, 27, 43, 62, 82, 99, 109, 112, 112, 111, 112, 111, 106, 92, 70, 48, 29, 15, 7, 2, 0, 0, 0, 0, 0, 2, 4, 9, 18, 31, 50, 71, 90, 101, 101, 95, 90, 90, 95, 101, 101, 90, 70, 48, 29, 14, 5, 2, 0, 0, 0, 0, 0, 2, 4, 9, 19, 34, 54, 76, 91, 94, 85, 75, 72, 77, 85, 95, 97, 86, 65, 44, 26, 12, 4, 1, 0, 0, 0, 0, 0, 1, 3, 9, 19, 36, 58, 79, 90, 86, 74, 67, 69, 78, 89, 98, 96, 81, 59, 39, 23, 11, 4, 1, 0, 0, 0, 0, 0, 1, 3, 9, 21, 40, 63, 82, 89, 83, 73, 72, 81, 93, 103, 107, 98, 77, 53, 35, 22, 12, 4, 0, 0, 0, 0, 0, 0, 1, 3, 10, 23, 43, 66, 83, 88, 83, 79, 88, 101, 113, 118, 115, 99, 74, 50, 34, 22, 12, 5, 1, 0, 0, 0, 0, 0, 0, 3, 11, 26, 46, 67, 82, 87, 85, 89, 105, 119, 127, 125, 117, 97, 72, 51, 35, 23, 13, 5, 1, 0, 0, 0, 0, 0, 0, 3, 13, 28, 47, 65, 78, 83, 86, 96, 112, 123, 127, 120, 111, 92, 71, 53, 38, 25, 14, 5, 1, 0, 0, 0, 0, 0, 1, 4, 15, 30, 45, 60, 71, 76, 81, 92, 105, 115, 116, 110, 101, 87, 70, 54, 38, 24, 13, 5, 1, 0, 0, 0, 0, 0, 1, 5, 17, 30, 44, 55, 63, 68, 73, 82, 93, 102, 105, 102, 95, 84, 70, 54, 37, 23, 12, 5, 1, 0, 0, 0, 0, 0, 1, 7, 19, 32, 44, 53, 60, 65, 70, 76, 87, 97, 102, 101, 96, 85, 70, 52, 34, 21, 10, 4, 1, 0, 0, 0, 0, 0, 2, 8, 20, 34, 46, 57, 65, 71, 75, 82, 92, 102, 107, 105, 98, 84, 65, 46, 30, 17, 8, 3, 1, 0, 0, 0, 0, 0, 2, 8, 19, 34, 49, 63, 74, 82, 89, 97, 107, 114, 114, 108, 94, 76, 56, 37, 23, 12, 6, 2, 1, 0, 0, 0, 0, 0, 2, 6, 15, 30, 47, 64, 79, 92, 102, 112, 119, 120, 114, 100, 81, 61, 41, 26, 15, 8, 4, 2, 0, 0, 0, 0, 0, 0, 1, 4, 10, 21, 37, 55, 74, 90, 103, 112, 114, 110, 98, 80, 60, 41, 26, 15, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 5, 11, 22, 37, 54, 70, 84, 91, 90, 82, 69, 52, 37, 23, 14, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 8, 15, 25, 35, 43, 47, 46, 41, 33, 25, 17, 11, 7, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 9, 13, 15, 16, 16, 14, 12, 10, 7, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

const int8_t scale[NUM_FEATURES] = {
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 6, 7, 6, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 3, 6, 8, 11, 14, 17, 21, 25, 28, 29, 30, 29, 26, 22, 17, 12, 8, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 7, 11, 16, 23, 30, 36, 42, 48, 53, 56, 56, 53, 48, 41, 32, 23, 15, 10, 6, 2, 1, 1, 1, 0, 1, 1, 4, 9, 15, 23, 33, 44, 56, 67, 77, 87, 93, 96, 96, 91, 83, 72, 58, 45, 32, 22, 14, 6, 2, 0, 1, 1, 1, 3, 9, 18, 29, 41, 55, 69, 83, 96, 107, 115, 120, 121, 120, 117, 110, 100, 86, 69, 53, 39, 25, 13, 5, 1, 1, 0, 2, 7, 16, 28, 42, 58, 74, 90, 103, 113, 120, 124, 125, 125, 125, 125, 123, 116, 104, 87, 69, 51, 35, 20, 9, 2, 0, 2, 5, 11, 23, 37, 54, 71, 88, 103, 114, 120, 123, 123, 123, 123, 123, 123, 124, 121, 113, 98, 79, 59, 40, 24, 10, 2, 1, 3, 9, 17, 30, 45, 63, 81, 98, 112, 120, 123, 124, 124, 124, 124, 124, 124, 124, 123, 117, 103, 84, 61, 41, 24, 10, 2, 1, 4, 11, 21, 33, 48, 67, 87, 104, 116, 123, 124, 124, 123, 122, 121, 122, 123, 124, 123, 117, 104, 84, 59, 37, 20, 8, 2, 1, 5, 11, 20, 32, 48, 69, 90, 108, 119, 123, 124, 120, 116, 115, 117, 120, 122, 123, 122, 114, 100, 80, 56, 32, 16, 7, 2, 1, 4, 10, 17, 30, 47, 70, 92, 110, 120, 123, 121, 116, 112, 115, 119, 120, 123, 123, 120, 110, 95, 77, 54, 30, 11, 5, 1, 1, 3, 7, 14, 27, 47, 72, 96, 113, 122, 123, 121, 116, 115, 122, 124, 123, 124, 124, 119, 106, 91, 75, 56, 31, 9, 4, 1, 0, 2, 5, 12, 26, 50, 76, 100, 115, 122, 123, 121, 119, 121, 127, 125, 123, 125, 124, 117, 104, 90, 75, 58, 35, 11, 4, 2, 1, 1, 3, 10, 26, 54, 81, 102, 116, 122, 122, 121, 121, 124, 127, 122, 123, 125, 124, 116, 105, 92, 77, 60, 37, 13, 4, 0, 0, 1, 3, 11, 28, 58, 84, 103, 115, 121, 121, 121, 122, 126, 126, 122, 124, 126, 123, 116, 107, 95, 79, 60, 37, 16, 6, 1, 0, 1, 4, 11, 32, 62, 86, 102, 112, 117, 118, 120, 123, 126, 126, 124, 125, 124, 121, 116, 108, 95, 79, 58, 36, 17, 6, 2, 1, 1, 5, 14, 37, 66, 87, 100, 109, 113, 115, 118, 121, 124, 125, 125, 124, 123, 121, 116, 108, 94, 76, 55, 34, 18, 7, 1, 0, 1, 6, 19, 43, 69, 89, 101, 107, 112, 115, 117, 119, 122, 124, 124, 123, 123, 122, 117, 106, 91, 72, 52, 32, 17, 5, 1, 0, 1, 8, 22, 46, 72, 91, 103, 110, 115, 118, 120, 121, 123, 124, 124, 123, 124, 122, 114, 102, 85, 66, 46, 29, 15, 6, 1, 1, 2, 9, 23, 45, 70, 91, 105, 114, 119, 122, 123, 124, 125, 124, 124, 124, 123, 119, 109, 94, 75, 56, 39, 25, 13, 5, 0, 0, 1, 8, 21, 39, 63, 86, 104, 115, 121, 124, 124, 124, 123, 123, 124, 123, 120, 111, 97, 80, 61, 45, 31, 20, 10, 3, 0, 0, 0, 6, 15, 30, 50, 72, 93, 109, 119, 124, 126, 125, 125, 125, 124, 120, 111, 97, 80, 62, 46, 33, 23, 13, 6, 2, 0, 1, 1, 3, 9, 19, 33, 52, 72, 92, 107, 117, 122, 123, 123, 120, 115, 105, 91, 75, 59, 45, 32, 23, 15, 8, 3, 1, 1, 1, 1, 2, 5, 11, 19, 31, 46, 62, 78, 90, 98, 102, 101, 96, 88, 77, 66, 53, 41, 31, 22, 15, 9, 4, 2, 0, 1, 1, 1, 0, 2, 5, 10, 18, 28, 39, 49, 58, 63, 66, 65, 61, 56, 50, 43, 36, 28, 20, 14, 9, 5, 2, 1, 0, 1, 1, 1, 1, 0, 3, 6, 11, 17, 24, 29, 34, 38, 39, 39, 37, 32, 29, 25, 20, 16, 12, 7, 4, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 12, 10, 9, 7, 4, 3, 2, 2, 0, 1, 1, 1, 1
};

```

&nbsp;

svm_model_q.h

``` h

#define NUM_CLASSES 10
#define NUM_FEATURES 784
#include <stdint.h>
const float weight_scale = 0.0030784949f;

const float bias_scale = 0.0290826722f;

const int8_t weights[NUM_CLASSES][NUM_FEATURES] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -15, 2, 8, 0, 14, 0, 1, 12, 15, -24, 20, 1, 19, -32, 2, 5, 3, 2, 3, 0, 0, 0, 0, 0, 0, 1, 3, 5, 8, -6, 12, -2, -22, 10, -7, -48, 15, -29, -44, -12, -44, -36, 10, -29, 2, 5, -1, 1, 0, 0, 0, 0, 0, 2, -7, -1, 16, -19, -25, 14, -27, -27, 9, -22, 12, -12, -10, -3, -5, -13, -18, 4, -12, -9, 4, -2, 4, -9, 0, 0, 3, 5, -6, -3, -27, -1, -6, -4, 14, -22, 16, -24, 27, -27, 29, -38, 20, -2, 3, 2, -3, 0, 3, -11, 8, -3, 2, 0, 0, -2, 3, 2, -27, 40, -13, 8, -8, 31, 2, -23, 13, 2, 17, 2, 3, 7, -4, -5, -9, -9, -3, 3, -18, 4, 0, 0, 2, -8, 5, -1, -11, -25, -13, 18, -44, -10, -9, -7, 21, -5, 1, 0, 1, 10, -4, -1, 23, 18, -4, -22, -25, -9, -2, -1, 1, -7, -6, 1, 18, 23, 2, -36, 19, -4, 23, -21, 8, 15, -12, -4, 41, 1, 22, -5, -14, -17, 8, -7, -3, -12, 7, 10, -2, 7, 16, -15, -33, 3, -1, 18, -3, 1, -22, 34, -23, 23, 8, 35, 5, 14, -6, 15, 16, 16, 1, -15, -29, -11, -1, -1, 0, -12, 8, -17, 16, -8, 3, 1, -2, 27, -22, -28, 35, 2, -14, 8, 20, 1, 25, 16, -8, -25, 7, -7, -10, -12, 7, 6, -30, 59, -68, 20, 7, 7, 4, -24, 21, -12, 35, 2, -19, 11, 17, 18, 15, 43, -9, 1, 5, 30, 12, 2, -5, -15, 7, -6, 56, -79, 45, -23, -22, 6, 1, 14, -28, -9, 7, -29, 6, -32, -25, -39, 25, -30, 14, 21, 12, 1, 6, -6, -11, -1, 6, 4, -17, -6, -24, -21, 14, 7, -11, 6, 15, -9, 6, 28, 5, -35, -41, -5, -5, -2, 3, 18, 7, 9, -6, 18, -1, -28, 6, 3, 7, 2, 8, 34, -32, 30, -2, 8, 7, 26, 1, -17, -58, 4, -52, -27, -7, 3, 5, -46, -8, 17, 13, -5, 3, -17, 3, 1, 2, 2, -17, -21, 30, -2, 44, -10, 5, -12, 8, 31, 20, -116, 37, -63, 10, -6, -15, 29, 16, -11, 7, -6, 2, -1, 1, 1, -1, 1, -7, 9, -4, -23, 22, 16, 22, 29, -20, -1, -20, -77, -3, -37, -7, -3, -9, 6, -9, 12, 5, 17, -2, 2, 2, 1, 2, 6, -6, -5, 17, 4, 11, 17, 16, 6, 24, -11, -77, -39, -3, -23, 8, -16, 1, -5, 13, -23, -4, -8, -16, -39, 11, 0, 4, -3, -3, 9, 2, 19, 6, -10, 14, 15, -32, 40, -58, -22, -20, -11, 10, 4, 7, 24, -15, 36, -4, 11, 7, -5, 5, 1, 0, 4, -9, -5, -4, -14, -8, 26, 17, 14, 16, -19, -56, -11, -3, -24, -6, -11, -9, -31, -4, -13, -6, -5, -6, -3, -10, 2, 1, 2, -14, -12, -9, 14, 18, 4, -18, 9, -4, 42, -25, -11, -6, 4, -9, -1, -22, 16, -6, 12, -11, 5, -15, 6, 6, 0, 5, -12, -1, 16, 13, -11, -6, 17, -11, 23, 23, 6, 14, -34, 14, -36, 2, 18, -3, -11, 16, -6, -5, -4, -5, -7, 3, 2, 5, -7, 2, -3, -11, 3, 1, -24, 16, 12, 26, 6, -2, -6, -13, 28, -65, -9, -10, -20, -32, 11, -25, 17, -30, 19, 3, 2, -3, 2, -4, -13, 3, 7, -6, 8, 17, -7, 1, -2, 52, -20, -5, 7, 24, -34, 17, 24, -11, 23, 3, -12, 8, 2, 3, 0, 0, 6, -4, -22, 1, -1, 10, -22, 7, 18, 13, 13, -5, 27, -1, -31, 30, -26, -26, -7, -24, 0, -2, -4, 8, -2, 0, 0, 0, -6, 1, 11, 4, -6, -2, -7, 12, 15, -39, 17, -6, 4, 6, -35, 28, 14, -15, -17, -14, 13, -13, 7, -20, 2, 0, 0, 0, 4, 5, -14, -12, -32, -2, -24, -13, -89, 43, -78, -31, -41, -43, -33, -4, -89, -23, -15, -8, -12, -6, 1, 3, 2, 0, 0, 0, 0, 0, 10, 10, -2, -17, -8, -18, -21, -33, 2, -31, -15, -37, -32, -36, -21, 21, -2, -6, 3, 5, 0, 3, 0, 0, 0, 0, 0, 0, 5, -3, 3, 0, 5, -2, -1, 3, 3, -5, 2, 3, 2, 12, 2, 9, 3, -5, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, 2, 4, -1, 3, 3, 6, 3, 5, -2, 5, -11, 0, 6, 4, 2, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 8, 8, -3, -4, -3, -3, -22, 38, -29, 30, -1, 23, -15, 21, -13, 4, -1, -4, 6, -2, 0, 0, 0, 0, 4, 3, 5, -46, 5, 15, -18, -12, -12, -36, -10, -3, -30, 0, -15, 3, -29, 4, -11, -4, -1, -3, -2, 13, 3, 0, 0, 4, 11, -13, -1, 12, -9, -35, 35, -10, 6, -4, -7, 2, 5, -15, -2, -2, -10, 5, -3, 4, -8, 4, -16, -38, 12, -3, 0, 0, 0, 11, 2, 20, 3, -14, -20, -22, -9, -35, -29, -11, -26, 13, -14, -23, -7, -4, -5, 5, 15, 0, 20, -18, 1, 2, 0, 5, -9, -1, -4, -39, 30, -20, -16, -33, -3, 13, -20, -7, -11, -32, -22, -25, 3, -15, -6, -5, -17, 12, -21, -36, 6, 5, 4, -11, -7, -26, 54, -15, -32, 33, -45, 3, -7, -32, 13, -25, -1, -26, 10, -6, -15, 1, 19, 15, -12, -7, -18, -28, 18, -5, 1, 2, 10, -12, -3, 2, -43, -11, -9, -44, -1, -20, 5, 7, -5, 2, -30, 20, -1, 7, -30, -12, 9, -7, -44, -23, 11, 4, 1, -23, 11, 6, 9, -46, -1, 40, -2, -3, -8, 1, -6, 7, 11, 23, 5, -21, 10, -17, -9, -4, 1, -32, 15, -52, 9, -3, 2, 16, 6, -16, -22, -20, -38, -24, 12, -9, 7, 8, 5, 15, 33, 27, 17, 18, -32, -2, -23, -3, 5, 2, -46, 6, -13, 15, 5, -1, 12, -18, 17, -4, -31, -35, -23, -32, -7, -9, -22, 17, 15, 41, 10, -4, 2, -5, 7, 5, -26, 3, -17, -11, -1, 0, 4, 6, -34, 10, -1, -31, 29, 20, -21, 12, -20, -21, 7, 7, 41, 29, -9, 8, 7, -24, -2, -12, -19, 29, 14, -9, -16, 1, 4, 5, -7, 1, 0, -18, 37, 17, 1, 8, -42, -21, 1, 1, 28, 7, 17, 23, -23, -3, -2, 1, -13, 14, -32, -29, 2, 0, 1, 2, -2, -8, 32, 1, 12, -26, 11, -21, -27, -24, -11, 4, 26, 17, -4, -3, -56, -3, 1, -11, 10, -12, 40, 2, 0, 2, 1, 2, 7, 11, -1, -37, -24, -3, 16, -15, -46, -5, 19, 19, 24, -7, 32, -22, -1, -30, -11, -10, -12, -15, -5, -16, 6, -1, 2, 2, 1, -7, 19, -6, 3, -9, -34, 28, -13, -2, -31, -4, 21, -27, 5, -48, -28, -26, -14, 5, -14, -20, 25, -16, -7, 6, 0, 4, -2, -10, 6, -18, -1, 22, 6, -20, 0, -20, 15, 25, 24, 13, 0, -37, -6, 11, 5, -9, 30, -19, 16, -26, 1, -1, 0, -2, 4, 0, -13, -25, -57, -39, 1, -4, -35, -9, -4, -11, 20, -19, -62, 21, -2, -18, 2, 2, -13, -36, 23, -4, 0, 2, 4, 1, 1, 5, 1, -53, 10, 19, -23, 14, 5, -14, -18, 8, -14, 18, 29, -41, -10, 10, -2, 14, -11, -6, 6, -6, 1, 3, 0, 0, 6, -21, 10, -96, 1, -20, 3, 11, -3, 26, 23, -7, -5, -34, 9, 8, 33, 19, 2, -19, 14, -1, 3, -7, 11, 2, 4, 2, -5, 17, -1, 19, 22, 17, 12, -14, 25, -20, -4, 3, 6, 10, 11, -5, 17, -31, -3, 37, -20, -65, 48, -5, -2, 3, 4, -2, 2, -5, 6, 44, -15, -13, -1, 10, -14, -8, -28, -31, 9, -6, -5, -1, 45, -3, 21, 14, -33, 16, -21, 28, 0, 3, 0, 0, 9, -17, -18, -5, 9, 6, -10, -2, -14, -15, 9, -2, -31, 12, 4, 23, -37, 3, -12, -10, 30, -34, -11, 2, 7, 0, 0, 0, 3, -1, -4, -3, -2, 4, -16, 2, -8, -9, -12, -24, -16, 0, -41, 16, -9, 31, -3, 2, -17, 16, -2, 4, 16, 0, 0, 0, 2, -2, 6, -20, 7, -32, -18, -28, -59, -68, -28, -32, -11, -52, -41, -33, -38, -24, -10, -10, -1, 3, 3, -2, -10, 0, 0, 0, 0, 0, 4, -7, -6, 0, -8, -16, -5, -13, -54, -38, -29, -12, 4, -14, -1, 0, -4, 12, 2, -11, 4, 2, 0, 0, 0, 0, 0, 0, 2, 6, -10, 11, -3, 5, -1, -8, 14, 5, 7, 8, -5, 10, -19, 9, -2, -1, -2, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -2, 1, 1, -3, 2, 2, -6, -5, 2, -3, 5, -4, 1, -22, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -3, 5, -4, -1, 3, -10, -1, 0, -6, -5, 0, -8, -4, -3, -8, -8, 1, -11, 7, 4, 1, 0, 0, 0, 0, 1, 3, -2, 1, -3, 4, -6, 8, 2, 1, 0, 5, -2, -6, 3, -2, -9, 0, 0, -3, 8, -14, -11, 5, 0, 0, 0, 0, -1, -2, -2, -2, 8, 5, -6, 10, 3, 1, 9, 0, 14, -5, 6, -10, 0, -9, -5, -7, -15, -9, -35, 16, -5, -1, 0, 0, 0, 1, 3, -1, -5, 0, 7, -7, 3, 7, 4, 14, 1, 6, 16, 5, 9, -3, 1, 3, 3, 9, -17, 6, 1, 0, 0, 1, -2, -3, -3, -1, -5, 3, -4, 3, 8, -3, -2, 6, -4, 9, -1, 6, -5, -1, -12, -3, -4, -9, -8, -12, 5, -4, 1, -5, 7, 9, 2, 2, 11, -11, 10, -9, -10, 12, -6, 11, 7, -10, 7, -19, -3, -9, -8, 6, 10, -26, -19, 28, -12, 6, -1, 5, -4, -16, 1, -8, -1, 9, 4, -1, 12, -2, -4, -3, 9, -2, -7, 9, -6, 7, 10, -2, -10, 19, -28, -46, 3, 6, 1, -6, 1, 3, 0, 21, -31, 6, -16, 7, 3, 10, -7, 0, -6, 8, 15, -9, 6, -1, -7, -11, 11, 5, -28, -4, -3, 1, 1, -1, -9, 15, -14, 1, 27, -17, 8, 2, -9, 7, 4, -4, 12, -14, -8, 19, 12, -9, 6, 6, -7, -8, -4, -16, 16, -3, 1, -14, -12, -14, 3, 7, 5, 9, -28, -36, -29, -28, -37, -33, -25, -5, -2, -17, -4, 10, -16, -4, 0, 15, -14, 6, -18, 1, 1, 6, 15, -8, 7, 3, -54, -47, -8, -14, -14, -36, -19, -13, -42, -28, 2, -14, -10, 20, 7, 9, -3, -45, 0, 6, 7, -7, 0, 1, -1, -1, -27, -12, -5, -32, -24, -14, -13, 1, -7, -2, -3, -8, -20, 2, -4, -19, -10, -11, -9, -5, 16, 1, 6, 3, 1, -7, 5, -8, -1, -46, -29, 20, -10, -7, -6, -1, 7, -11, 8, -7, 7, -11, 5, -5, 6, -6, -7, 5, -16, 2, 8, 4, 1, 1, 0, 7, 7, 45, 11, -24, 2, 1, 3, 6, -4, 12, -4, -4, 5, -10, -15, 9, -18, -7, -9, -22, 17, 0, 3, 3, 3, 0, -10, 3, -2, -18, 19, 14, -7, 6, 7, 3, 13, 11, -6, 3, 9, 20, -11, 2, -8, 21, -2, 16, -5, 6, 9, 1, 0, -1, 9, 1, 4, 11, 15, -10, 10, 2, 5, 6, 10, 13, 0, 19, 0, -11, 12, -8, 1, -13, 0, 0, -5, 11, 1, 4, 2, 1, -7, 0, 0, 18, 4, 20, 7, 0, 11, -8, 16, 10, 11, 4, 12, 7, -5, 7, 1, 10, 5, 7, 16, 3, -1, 0, 0, -1, -5, -8, 4, -12, 2, 9, -3, 16, 10, 8, 1, 16, 11, 6, 0, 12, 3, -7, 2, 1, 6, 5, -6, 11, 7, 0, 0, 0, 6, 4, 9, 11, 10, 18, 17, 6, -1, 21, -6, -3, -3, 3, 9, -18, 8, 9, 4, 9, -4, 2, 16, 1, -6, 0, 0, -1, -3, 1, -13, -1, -6, 0, -16, 17, 5, -4, 0, -4, -3, -12, 5, 14, -7, 18, -3, 9, 3, 12, -2, 6, 5, 0, 0, 0, -1, -2, 8, 3, 5, 9, 18, 7, -2, 9, 0, 5, -7, -9, -1, -5, 14, -6, 11, -2, 5, -1, 6, -3, -7, 0, 0, 0, -3, 3, 2, -6, 6, -7, -9, -8, -7, -9, -1, -16, 3, -19, 7, -7, 4, 9, -2, 8, 2, 5, -2, 0, 0, 0, 0, 0, 0, -18, -1, -2, -6, -9, 2, -3, -6, -3, 1, -12, 15, -13, -10, 4, 1, 3, -3, 5, 3, -7, 8, 9, -7, 0, 0, 0, 1, 3, 3, -85, 20, -28, -4, -40, -17, -26, -18, -18, -15, 15, -24, 10, -8, -11, 6, -6, -9, -8, -12, -5, 6, 0, 0, 0, 0, 2, 4, -2, 3, -16, -13, -6, -9, -30, -7, -24, -25, -29, -31, -38, -18, 11, -11, -9, -10, 0, 6, 1, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 5, 3, -5, -12, 2, 7, 11, 6, 13, 12, 5, 1, 6, -2, 4, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 4, 3, 7, 0, 7, -4, -1, 3, -2, 8, 4, -1, 6, -1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 5, 1, 1, 3, 2, -8, -14, 8, 2, -10, 26, -28, 13, 0, -13, -4, -2, -1, -8, 13, 8, -6, 0, 0, 0, 0, -4, 3, -3, 13, -19, 4, 9, -13, 2, -1, 18, -24, 22, 0, 9, -3, 16, -8, 24, -2, -24, -68, 5, 9, 1, 0, 0, 1, 2, 1, -3, 4, 3, -4, 2, 2, 9, 1, 2, 5, 1, 2, 9, -8, 13, -9, 8, -15, 22, -44, 7, 6, 0, 1, 0, 0, -2, 2, 2, 1, 5, 7, 0, 4, -1, 6, 9, 1, 3, 1, 12, -15, -4, 6, -6, -11, -17, 9, -13, 9, 2, 0, 0, 4, 0, 3, 1, -5, 8, -8, 4, 5, 12, 7, -4, 16, -6, 8, 6, -3, 19, -23, 5, 11, 4, -11, 1, -33, -11, 6, 1, 4, -3, 3, -1, 7, 2, 1, 2, 2, -5, -4, 6, -7, 17, 9, -6, 12, -19, 5, 4, -26, -20, 7, -10, -15, 9, 2, 1, -11, -4, -2, 6, -2, -3, -2, -2, -16, 1, -5, 8, 2, -2, 1, 5, -1, 21, 2, 1, -2, 11, -27, -22, -24, 21, 3, -1, 12, 0, 1, 7, -2, 7, 0, -5, 5, 1, -21, -11, -13, 2, 17, 8, 9, 0, 11, 17, 12, 21, -12, -94, 23, -8, 3, 0, -7, -13, -1, -7, 7, -3, 11, -6, -25, -34, -17, -26, 16, 12, 5, 15, 10, 7, -1, 16, 10, 6, 46, -59, 22, -14, 1, 0, -1, 0, 15, 8, -6, -1, -27, -20, -17, -19, -8, 2, 1, 10, 17, -3, 15, 8, 3, -2, -15, -3, 16, -103, 1, 0, 0, -1, 5, -11, -6, -6, -3, -23, -17, -16, -5, 8, -13, 18, -5, 24, 11, 14, 1, 4, 3, 12, -21, -25, -86, -73, 39, 7, 1, 0, -1, -1, -1, -8, 2, 15, -3, 1, -8, -15, 5, 14, 3, 18, -17, 15, 3, -1, -7, -26, -19, -55, 32, 43, -67, 6, 2, 0, 4, -6, 13, -2, -6, -22, -8, -4, -1, -9, -10, 22, -7, 4, 1, 7, 1, 7, -17, -16, 15, 4, -8, -11, -11, 3, 0, 0, 2, -4, -8, -6, -2, 0, -10, -13, -9, -10, 1, 8, -1, 1, 11, 2, 0, -20, 4, 30, 0, 15, 26, 0, -49, 26, 2, 1, 1, 9, 4, 12, 6, -9, -5, -34, 7, -7, 4, -6, 9, -4, -8, -21, -15, 23, 12, 5, 10, -4, -4, 13, -64, -8, 2, 0, 1, -5, -1, 7, -9, -11, 18, 0, -41, -30, -28, -7, -16, -19, -22, -5, 28, -7, 23, -5, 0, 11, 13, -8, -48, 3, 1, 0, 1, 5, 11, -13, 13, 6, -15, 5, -18, -26, -23, -28, -27, -29, 15, 25, 1, 6, 2, 6, 12, 21, 2, -15, -27, 5, 0, 1, -2, -2, 7, 3, 13, -10, -4, -2, -8, -2, -20, 1, -17, -8, -15, -1, 17, 5, 13, -5, 4, -11, -9, 1, -29, 7, -1, 0, 3, -5, 14, -1, 10, 4, 9, -4, 16, -8, 3, -16, -3, 8, -8, -5, -2, 13, -15, 9, -1, -3, 6, -5, -30, 6, 0, 0, -1, 5, -2, 8, -4, 5, -3, 14, -20, 5, -13, -3, -4, -6, 0, 18, -13, 11, 12, -2, -5, -4, -7, -23, 17, -3, 1, 0, 1, -4, 4, -1, -6, 3, 2, -12, 9, -4, 12, 1, 0, -9, 0, -1, 12, -7, -1, 8, -6, 3, 5, -32, -10, -3, 1, 0, 0, 8, -1, 6, 11, 6, 11, 1, 11, 2, 8, -7, 16, -9, 1, -13, -5, 18, 0, -1, 3, -12, -6, 11, -9, 2, 0, 0, 0, -8, 11, 5, -9, 13, 5, 4, -2, 11, 9, -7, 6, 4, -2, 4, 6, -17, -4, -4, 9, 0, 16, -21, 3, -4, 0, 0, 0, 6, -7, -6, 10, 0, -4, 14, 4, 10, -4, 20, -1, 9, -1, 3, -2, -4, 10, -7, -9, 7, -72, 13, 2, 4, 0, 0, 0, 0, 1, -31, 2, -5, 1, 1, -16, 2, -7, -12, -11, -7, -15, -6, 0, -4, -5, 7, -3, 0, -1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 2, 1, -5, 1, -2, -1, -1, -3, -15, -7, 0, 1, -2, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, -8, 7, -1, 9, 2, 4, -4, -32, 16, -24, -4, 0, -3, 2, -15, 0, -4, 1, 0, 0, 0, 0, 0, 0, 0, -3, -2, -3, 5, -21, -7, -13, -39, -31, -13, -2, 0, -8, -22, -13, -30, -4, -8, -6, 0, -14, 4, 1, 0, 0, 0, 0, 1, 1, -2, 11, -18, 15, -26, -29, -5, -28, -15, -92, -15, -34, -10, -7, -39, 9, -12, -2, -5, 4, -15, -3, 0, 0, 0, 2, 5, -13, -1, 7, -25, 23, -18, 13, -33, -7, -15, 24, -34, -25, -9, -10, -20, -2, 9, 0, 12, 0, 8, 2, -2, 1, 0, 0, -6, 8, -5, 2, 3, -4, -4, -2, -4, -7, 1, -24, 16, 1, 20, -1, 11, 12, 3, -1, 1, 1, -6, -1, -6, 1, 0, 3, 1, -12, 10, 0, -10, 3, -3, -10, -9, -30, -24, -29, -36, -41, -32, -26, -33, -13, -13, -1, -3, 17, -4, -4, 3, -5, 3, 1, -26, -5, -13, 21, -5, 15, -7, 9, -10, 13, -26, -13, -27, -29, -26, -5, -13, -1, 0, 4, 6, 15, -8, -8, 0, -13, -1, -22, -1, 20, 5, 6, -13, 10, -30, 9, -13, -13, -10, -12, -29, -14, -31, -9, -2, -10, -10, -1, -21, -9, 3, 3, -3, -3, -1, -5, 15, -14, -23, 15, 1, -1, -4, -12, -20, 4, -27, -1, -36, -23, 8, -14, 5, 1, -5, 6, 0, 13, -26, -9, 1, -4, 0, 3, -3, -19, 20, -26, -25, 12, 9, 12, -10, -5, 8, -10, -32, 0, -27, 11, -12, -10, -6, 2, 11, -22, 18, -4, -16, 3, 6, -27, -7, 16, -16, 7, 11, -16, -17, 14, 16, -8, 16, -46, -49, 17, 16, 2, -1, 20, -18, -15, -19, -9, -35, -4, -9, 5, 5, -2, -1, -4, -10, 1, -9, 15, 14, -5, 27, 12, 57, -2, -25, 11, 20, -8, -5, 0, 0, 11, 5, -19, 22, -3, -9, 0, 2, 4, -2, -7, -7, -7, 7, 2, 16, 28, 22, 33, 18, -21, -13, 14, 3, 17, 14, -21, 10, 18, -15, -2, -28, 1, 9, -5, 1, 11, -37, 0, 16, 12, 10, 31, 5, 10, 9, 21, 18, -20, 2, 0, 22, 19, 0, 30, 1, 3, 15, -19, 7, 6, -22, 0, 1, 4, 22, -5, 14, -8, 4, 2, 14, 16, 13, 15, 14, -15, 5, 14, 21, 14, -10, 20, -4, -23, 22, 28, -48, -4, 16, -4, 0, -4, -39, -3, -18, -3, 1, 14, 23, 6, 4, -10, 22, -11, 17, 24, 8, 17, 6, 0, 29, 13, -13, -11, 11, -12, -7, 6, 0, 3, 41, -12, 10, -8, -1, 12, 5, 15, -7, 3, -14, 10, 28, 8, 17, 15, 30, -15, -13, 14, -15, 28, -36, -4, 4, -6, -2, 4, -34, -8, -11, -8, 11, 2, 5, 21, -32, -26, 32, 33, 5, 5, 12, -29, -20, 17, 2, 6, -14, -6, 23, -29, -15, 9, 3, -5, 8, 22, 1, 28, -22, -15, -10, -30, -21, 23, -53, 4, 25, -7, -9, 6, 0, -17, -21, 38, -63, 3, -14, -15, 8, -2, 0, 4, 7, -10, -1, -67, 20, 1, -58, 9, -41, -4, -2, -26, 7, 6, 23, 2, 4, -10, -15, -8, 7, -5, -13, -12, 12, -1, 1, 7, -14, 9, -21, 78, -51, -57, 50, -17, 8, -14, -11, -13, -14, 9, -25, 15, -8, 1, 37, -25, -1, 9, 2, 9, -4, 2, 1, -2, 29, -23, 6, -31, -17, 0, -20, -29, -10, -5, -16, -7, -22, -12, -11, 5, -7, 19, -32, 24, 16, -6, 0, -9, 6, 2, 0, 0, -5, -10, 1, -24, 2, -2, 2, -10, -1, 1, -11, -2, -3, 4, 11, -6, 6, -8, 6, 7, -22, -1, -1, 2, 4, 0, 0, 0, 2, 11, -36, 3, 23, -19, 17, 16, -17, 17, -10, -6, 2, -12, -2, -4, -7, 9, -7, 1, 3, -1, -2, 11, -1, 0, 0, 0, 2, 4, 11, -9, -5, -11, -9, -19, -2, -17, -18, -11, -17, -9, -16, -13, -6, -18, 0, -12, 1, 2, -8, -2, -1, 0, 0, 0, 0, 1, -4, 7, 2, -32, -8, -21, -17, -30, -18, -30, -26, -23, -19, -19, -13, -9, -48, -7, -6, -2, 3, 2, 0, 0, 0, 0, 0, 0, -1, 5, 1, 2, -11, -2, -11, 3, -10, 5, -34, 1, -7, 5, -5, -2, 17, -1, -1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -3, 1, 3, 12, -3, 1, -3, 0, 8, 1, -2, 5, 2, -4, 3, 2, 2, -1, 0, 0, 0, 0, 0, 0, 1, -14, -18, 4, 7, 8, -14, -12, 5, -39, -17, -33, -35, -5, -5, -5, -2, -2, -10, -2, -5, 7, -27, 1, 0, 0, 0, 0, 1, 0, 12, -15, -2, -16, 16, -31, 26, -35, 15, -13, -2, -9, -12, -3, -7, -4, 1, -5, 3, -6, -1, 0, 0, 0, 0, 1, -3, -3, -22, -12, 0, 3, -18, -2, -9, 1, -3, -22, -3, 2, 1, -13, 8, -1, -13, 9, -4, 2, 7, -2, 1, -2, 0, 0, 1, 9, -29, 2, -14, 3, -3, -1, -1, -3, -9, -3, -9, -9, -5, 6, 4, -7, 18, -3, 13, 1, 1, 0, 2, 0, 0, 0, 4, -27, 12, 8, -2, -3, -10, 0, -4, -5, -2, -8, 10, -13, -8, -7, 0, 10, 5, -4, 4, 4, 14, -2, 2, 0, 1, 0, -4, 3, -28, -7, -1, -7, 10, -7, 14, 5, 3, -13, 11, -23, 4, 0, 5, -5, -1, 8, 13, 2, 6, 15, -7, 3, 2, 5, -6, -18, -10, 0, -9, -4, 7, -3, 5, 7, -4, -6, -10, -1, -7, 1, -10, 5, 8, 10, -15, 19, 14, 5, 3, -2, 0, 2, -5, -35, -18, -8, 9, -6, -2, 4, -3, 18, 2, -4, -8, -24, -21, -5, -9, -2, -12, 20, 20, -7, 17, 9, -1, 1, 0, 1, 5, -34, 11, -7, -15, 17, 23, 19, 14, 18, 19, 8, -6, 4, -19, -22, -27, -27, -21, -7, 8, 15, 26, 7, 8, -4, 3, -8, 22, -36, -3, -2, 26, -9, -2, 11, 8, 10, 5, 15, -6, -5, -5, -10, -6, -17, -52, -53, -57, -12, 41, 12, -7, 2, 2, -2, -7, 13, 0, -2, 8, 6, 9, -10, 10, 13, 7, 16, -11, 8, -14, -15, -15, -21, -3, -30, -44, -50, -127, 27, 18, -4, 1, -3, -14, -5, 4, 11, -7, 13, 1, 11, 0, 10, 17, 6, -19, -5, -21, -15, -4, 6, 1, -7, -10, -51, -18, -21, 9, 0, 3, -6, 5, 10, -7, -2, -12, -9, -9, 0, 21, -4, 13, -4, -8, 4, -16, -11, -11, -20, 1, -14, 7, -1, 31, -58, -10, 1, 2, 2, -3, -13, 22, -14, 4, -8, 16, 11, -3, 5, 5, -16, -17, -13, -5, -16, 17, -6, -4, -4, -18, 5, 15, 6, -5, 0, -2, 2, 7, 9, -11, 5, -22, -33, -20, -11, 5, -8, 8, 2, -18, -3, -22, -3, -23, -3, 2, -14, 27, -1, -8, -4, -30, 2, 0, 3, -10, -5, 0, -5, 38, -12, -10, -26, -27, -20, -32, -24, -25, -7, 3, -1, 19, -2, 5, 23, -20, 7, 16, -15, 1, 0, 0, -2, -11, -3, 8, 6, 8, 9, -11, -3, -25, -18, 6, -2, 10, -10, -5, 15, -14, 4, 0, -13, 21, 0, -2, -14, 6, -1, 2, -2, 5, -2, 8, -10, 14, -3, 18, -7, 9, -8, 0, 3, -8, 21, -1, -14, 7, -5, 12, 10, -9, -7, 18, -15, -2, 1, 0, 2, 1, -5, -2, 5, 0, -5, 8, 14, -3, 23, 0, 7, -13, -14, 5, 12, -2, 8, -11, -15, 18, 9, -19, 12, 0, 1, 1, 0, -5, 1, 6, 3, 5, 0, 2, 7, 5, 11, -10, 3, 3, 3, -10, 9, -4, 0, 19, 18, 6, 2, 5, -3, -4, 1, 1, 1, -9, -12, 1, -6, -13, 11, -2, 11, -8, 5, 14, 5, 0, -5, 6, -19, 24, 6, -2, -4, 5, 2, 1, 0, -2, 1, 0, 0, 7, 3, 4, 9, 5, -17, 9, -9, 14, -2, 8, -7, 13, -9, 13, -6, -18, -6, 12, -1, -2, 6, -18, 15, 0, 0, 0, 0, -29, -1, -9, 4, -8, 6, -7, 17, -6, 3, -4, 5, 5, 3, -9, 6, 7, -9, 1, -4, 2, -12, 5, -36, -2, 0, 0, 0, 5, 2, 5, -12, 2, 2, -7, 7, -7, 6, 1, 4, -9, 6, -2, -1, -1, 7, -18, 2, 5, 1, 10, 2, 1, 0, 0, 0, 0, 1, 1, -6, 6, -31, -18, 0, -9, 0, -11, 0, -13, -3, -4, -6, -8, -8, 14, -9, -2, 0, -2, 2, 0, 0, 0, 0, 0, 0, 1, 8, -2, 6, 0, -6, 3, 0, 0, -5, -10, -3, -7, -1, -6, 4, -1, -6, 1, 2, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 8, -6, 4, 1, -1, 1, 1, 3, -3, 3, -3, 2, 6, 4, 4, 5, 4, 2, 0, 0, 0, 0, 0, 0, 3, -3, 6, 3, 0, -1, 1, -1, 10, -3, 8, 5, 4, 0, 10, 2, 7, 3, 4, 11, 0, 11, -4, -1, 0, 0, 0, 0, 1, 1, 3, -7, 6, 6, -1, -1, -2, 3, -6, 1, -3, 8, -5, 4, 4, 10, -1, 2, 2, -1, 9, -4, 4, 0, 0, 3, -4, -3, 0, 6, -17, 6, 2, -12, -2, -8, -19, -6, -6, -18, -7, 8, -19, -2, -1, -5, 1, 0, -13, 2, -40, 17, 0, 0, 1, -10, 0, -10, 14, -34, 29, -32, 19, -34, 11, -37, -7, -13, -21, -14, 17, 4, 15, 10, -13, 10, 11, -7, 1, -5, 0, 0, -1, 4, -3, -1, -7, 10, -15, 6, -25, -9, -4, -2, -26, -1, -23, -3, -32, -21, -27, -14, 6, 1, -28, 11, -10, 13, 1, 7, -2, -23, 7, 29, -36, 22, -15, 10, -1, -13, -17, -19, 1, -11, -21, -12, -9, -7, -15, -5, -31, -21, 39, -2, -6, -8, 1, 2, 5, -18, -26, -13, -4, -8, -25, -22, -8, 1, -15, -6, -20, -24, -9, -17, -45, -39, -52, -18, -5, -44, -52, -10, 16, 0, 0, 2, 18, -9, 38, -8, -18, 22, 22, 0, -10, -9, 3, -28, -1, -30, -33, -43, -34, -21, -4, -4, -23, 13, -13, -21, -19, 6, 1, 5, -16, -16, -14, -21, -5, -18, -5, -14, -13, -2, -17, 4, -16, -6, -15, -34, 2, -48, -6, -19, -32, 0, 11, -10, 5, -27, 3, -1, 26, 6, 1, 54, -7, 9, -11, 3, 10, -20, 6, 17, 3, -37, 14, -25, -16, 18, -16, 9, -5, 18, -6, -5, -8, 20, 2, 1, -18, -17, -11, -44, 2, -2, 22, 15, -20, 33, 2, 2, 0, -20, -4, 15, -5, -19, -15, 9, 19, -3, 3, 4, -8, -16, 1, 0, -5, -21, 6, 27, 31, -7, -11, 0, 29, -13, 14, 1, -11, 31, -15, -7, -6, 3, 1, 20, -22, 34, 3, -8, -5, 5, 0, -3, 22, 29, -21, -3, 3, 14, 17, 14, 5, 29, 7, -4, -7, -10, 19, -18, -1, 1, -3, -15, 37, -5, -2, 7, -5, -8, 0, 0, -11, -6, -39, -10, -3, 12, 5, 5, 11, 10, 6, -1, 27, 7, -9, 10, -1, -19, 30, 23, -24, 11, -12, 6, -5, 2, 3, 2, 12, -20, 41, -10, 14, 15, -24, 30, -9, 39, -17, -10, -3, 8, 2, -19, -4, 10, -20, -4, 12, -16, 6, -6, -6, 0, 0, 3, -1, 17, -64, 29, -22, -16, 44, -9, 32, 26, -3, 6, 25, -22, -15, 10, 7, 13, 22, -8, 9, 6, 5, -6, -7, -4, 3, 1, -4, -2, 10, -31, -9, 39, -19, 20, 9, 3, 36, 6, -10, 13, 16, -1, 18, -13, -11, 8, -13, 6, -20, 7, 5, 1, 0, 3, 26, -91, -3, 7, 5, -15, 9, 10, 16, 32, 33, 2, 12, 14, 5, 15, 0, 16, 8, 2, 8, -3, -2, -7, -13, 1, 0, -1, 8, -9, 18, -1, -18, 7, -9, 7, 33, -9, 29, -8, 41, 0, 21, -15, 12, -5, -8, 0, -14, 5, -14, 2, 4, 2, 1, 6, -4, -5, -14, -24, -27, 2, 12, -1, -3, 25, 33, 26, 33, 14, 4, 25, 8, 0, 20, -30, 13, -13, 16, -10, 3, 1, 1, 2, -8, -7, 5, 2, -65, 39, -11, 6, 21, -6, 6, 6, -8, 30, -1, 18, -12, -4, 7, 18, -3, -3, -43, -9, 8, 1, 0, 0, 8, 6, 14, 19, -5, -30, -13, 8, -26, -19, -8, 3, -7, -8, 6, -33, 28, 4, -1, -11, 3, -25, 8, -20, -17, 0, 0, 0, -6, 8, 8, 2, 0, -23, -32, -13, -30, -39, -10, -29, -11, -32, 13, 1, 0, -34, -3, -21, 0, 8, -1, 4, 0, 0, 0, 0, 3, 1, 2, 10, 15, 17, 6, -3, 16, 13, 2, -28, -26, -13, -30, -7, -5, 6, 3, 17, 8, 3, 1, 3, 0, 0, 0, 0, 0, -2, 7, -3, 8, 19, 2, 17, -6, 1, 32, 32, 22, -14, 19, 4, 4, 4, 2, -2, -6, 2, 0, 3, 0, 0, 0, 0, 0, 0, 1, 3, 3, 2, 9, 6, -7, 12, 11, 2, 6, 8, 17, -6, 8, -5, 8, 1, 1, 3, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -1, 5, -3, -5, -3, 3, 11, -15, -7, 11, -5, 2, 4, 5, -2, 3, -3, 4, -1, 0, 0, 0, 0, 0, 0, 4, -2, 4, 4, -3, 3, 12, 16, -5, -10, 25, 1, 12, 12, -3, 7, 13, 5, 16, 3, -1, 2, 3, 6, 0, 0, 0, 0, 3, -4, 10, -10, 19, -9, 8, -17, 17, -20, -27, -49, 5, -34, 5, 0, -15, 23, -2, -14, 21, -10, 2, 4, 4, 0, 0, 3, 2, 9, 12, -34, -7, -23, -14, -21, -64, -21, 4, 5, -22, -25, -36, -20, -40, -11, -2, 22, -12, 8, 3, 0, 5, 1, 0, 0, 2, -11, -37, 43, -11, -5, 11, 15, -7, -15, -4, -8, 1, -14, -1, -56, -9, -63, -36, -17, -35, -32, 26, -3, -3, 3, 0, 0, -1, 9, 1, -7, 13, -11, 24, -10, 13, 16, 15, -14, 9, -14, 5, 7, -5, 10, -18, 7, 3, -38, 1, -13, -28, 6, 3, 2, -4, 3, -1, -2, 9, 11, -16, 21, 19, 2, 16, 11, -14, -5, -1, 11, 13, -7, 15, 23, -4, 7, 5, -8, 1, -3, -4, 6, 7, 4, -2, 1, -4, 37, 16, -2, 11, 15, 21, -3, 2, 19, 5, 3, 13, 9, 13, -15, 4, 14, -16, -11, -6, 3, 5, 4, -5, 2, 13, 5, 1, -12, 11, 15, -6, -3, 46, -4, 31, 3, 6, 29, 14, 4, -8, 17, -6, 6, 7, -11, -58, 18, 5, -4, 12, -7, 9, 6, 5, 14, -8, 11, -1, -8, -20, 46, -3, 22, 34, 3, 5, 16, 22, 1, 33, -7, -13, 11, -2, 7, 3, -7, 12, -1, -1, 6, -2, 4, -29, 2, 18, 18, -17, 24, 16, 17, 2, 11, 15, 3, 14, -7, 10, -14, 16, -19, -12, -32, 2, 33, 3, 7, 8, 2, -23, 17, 30, -8, -32, -1, -20, -49, -25, 20, 20, 13, 15, 1, -6, -25, 16, -19, -35, 3, -8, 33, 0, -18, 1, 6, -5, -1, 20, -18, -29, -12, 3, -1, -39, -39, -40, -4, -2, 1, -23, -35, 3, 17, 3, -24, 24, -7, 12, -52, 0, 9, 11, -7, 7, 10, -29, 8, 14, -14, -12, -23, -42, -12, -20, -11, 4, -7, 12, 55, 7, -11, 6, 21, -11, 2, -47, 6, 1, -1, -8, 16, -10, -5, 0, -16, -26, -15, -10, -19, -14, -28, -16, 7, -24, 32, 12, -16, 18, 16, 25, 5, -6, -7, -4, 3, 3, -1, -9, -17, 6, -18, -3, 27, -19, 11, -13, -54, 9, -4, -30, 14, 2, 6, -1, 16, 10, 3, -4, -5, 1, -27, -20, -2, 0, 2, -7, 11, 0, 7, 5, -23, -16, 7, -56, 15, -5, -12, 1, 1, -9, 6, 24, -18, 27, -24, -3, -8, 13, 3, 7, -4, 1, 0, 3, -10, -8, 33, -55, 12, 0, -24, -10, 1, -18, 23, -10, -13, -5, -23, 12, -9, -33, 14, 5, -6, -34, -14, 10, 5, -1, 7, 16, 10, 54, -51, -2, -48, 6, -9, -44, 8, 12, -9, 12, -16, -16, 0, -34, 0, 20, -22, 0, -70, 39, -18, 3, -5, 0, 0, 0, -33, -108, -14, -17, -71, 14, -47, 5, -5, 12, -18, 4, 1, -32, -32, -49, -11, -67, 12, -43, -2, 16, -10, 5, 2, 1, 5, -4, -9, 37, 36, -46, -20, 0, -20, 0, 0, -1, -2, -16, -14, -1, -14, 9, -49, -12, -30, 0, -31, -14, 27, -3, 2, 1, -1, 7, -27, -44, 20, -27, 13, 18, -3, -3, -30, 13, -13, -1, 14, 1, -14, -15, 32, -13, 16, -10, -20, -9, 2, -13, 2, 0, 0, 2, -30, 19, -12, -6, 22, -47, -6, -11, -12, -14, -10, -12, -27, -5, -8, -11, -33, -4, -46, 1, -27, 2, 7, 2, 0, 0, 0, 1, -3, -6, 14, 10, 2, 9, -9, -2, -9, -21, 6, -4, 9, -7, 10, -18, 7, 6, -9, 5, -2, -3, -3, 4, 0, 0, 0, -6, 5, 3, -4, -4, -2, 0, 8, -5, 12, 11, -1, 4, 1, -2, -1, 2, -9, -1, 6, -2, -16, 3, 0, 3, 0, 0, 0, 0, 1, -1, -4, 0, 0, 0, 0, 2, -1, -5, 3, -3, 5, 2, 0, 3, 4, -7, 1, 4, 0, -19, 2, 0, 0, 0, 0, 0, 0, 4, 5, -1, -2, 3, -4, 5, -2, -2, 11, -2, 6, -5, 14, -2, 6, -4, 5, -3, 7, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, -1, 1, 2, 3, 2, 0, 2, -1, 2, 1, -2, 1, -4, 2, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, -2, 3, -2, 1, -2, -2, -2, -11, -13, -1, -17, -26, 20, -33, 0, -2, -3, -1, -3, 1, 0, 0, 0, 0, -21, -2, 1, 1, -7, -24, 2, 1, -4, -3, -18, 10, -30, 5, -13, -13, 2, -10, -7, 7, -5, 12, -5, -1, -1, 0, 0, 1, 7, -10, -5, 4, -10, -2, 3, 5, 5, -3, 5, 0, 7, -1, 0, -4, -5, -2, 1, -5, 1, -5, 4, 2, -1, 2, 0, 0, 1, -3, -20, -11, 4, 7, -16, 3, -16, 7, 12, -6, 15, 3, 6, 17, -4, 3, -4, 0, -3, -5, -4, 2, 2, -2, 0, 2, -1, -1, -2, 4, -3, -4, -6, 0, -6, -10, 13, 13, -7, 18, 1, 15, -14, 11, 0, 0, -3, 1, -10, 4, -2, 2, 2, -10, -9, -28, -6, -10, -9, -7, 15, -7, 8, -1, -16, -6, -14, 15, -1, -12, 10, -18, 3, 5, -9, 12, 3, -4, 0, -2, 2, 6, 7, 1, 18, 4, 7, 12, -13, 4, 5, -7, 18, 7, -11, -11, 10, -14, -2, 18, -7, 6, 12, -2, 6, -5, -2, 3, 2, -7, -15, 17, -28, 0, 1, 3, 14, 15, -3, 10, -6, 2, -7, -20, -3, 7, -3, -5, 3, 6, 1, 4, 3, -3, 6, -4, 2, -1, 1, -36, 35, -13, 17, 3, 4, 1, 28, 8, 18, -4, -15, -26, -12, -1, 5, 21, 4, 19, 8, 15, -8, -5, -2, 1, 0, 1, 1, 1, -13, 4, 3, 13, 20, 20, 7, 9, 16, 23, 18, -3, -12, -5, -5, -4, 0, -6, 11, 4, 16, -1, 3, 0, 2, 6, -30, 2, 6, 17, 1, 6, -13, 7, -7, 1, 14, 5, 12, 7, 2, 14, -5, 24, 8, 17, 6, 5, 16, -4, 0, -2, 1, 4, -19, 6, 2, -15, 8, 2, 10, 9, -20, 14, 16, 20, -12, 24, 8, -11, -3, -7, 12, -10, 14, 1, -3, -2, 2, 2, 1, 1, 7, -3, -30, 31, -19, -31, -23, -29, -13, 33, -5, 20, 13, -2, 11, 10, -8, -11, -15, 5, -36, -13, -20, 5, -1, -6, 1, 0, 3, -18, 20, -41, -35, 0, -14, 5, 1, 7, -2, 30, 19, -8, 9, -15, -15, 5, -7, -11, 27, -22, -2, 4, -29, 3, 1, 0, -6, 22, -23, -4, 16, -4, -8, -6, 6, 7, 11, 11, -8, 4, 9, -11, -13, -24, -13, -11, -34, 29, -22, -1, 3, -1, 0, -2, 12, -11, 23, -6, -15, 5, 5, 29, -9, 12, 26, 9, 9, -9, 6, -5, -12, 5, 2, -2, -7, -5, 33, -29, -2, 3, 0, 2, -1, -35, -15, -4, 5, 8, 2, 7, 19, 14, 12, 5, -11, -1, -11, -16, 3, -10, 5, -13, 19, -16, -8, -3, 3, -2, -1, 2, -16, -6, 2, -2, -8, -13, 9, 17, 7, 3, 4, -20, -8, -4, 7, 10, -6, -7, -2, -16, 11, 4, -5, -31, 0, -1, 0, 1, 7, 4, -4, -9, -1, 14, -8, -19, 11, -16, 7, -5, -19, 6, -17, -17, -1, -5, -3, 10, -12, 6, 1, -5, -9, 0, 0, 1, -3, -25, -7, 5, -1, 1, 0, 9, -8, -9, -15, 7, -4, -11, 1, -2, -3, -4, 7, -12, 12, -5, -16, -17, -10, 0, 0, 1, -11, 18, -4, 1, 6, 6, -12, -11, 1, 11, 13, -1, 12, -3, -4, -2, 4, -9, -7, 7, -5, -3, 0, 2, 2, 0, 0, 0, 1, -10, 2, -12, -3, -16, 3, 0, -1, 5, 9, 35, 4, 21, 4, 5, 8, -1, -1, 4, -8, 7, 1, -9, -3, 0, 0, 0, 4, -4, -12, -3, -1, 0, 4, 4, -3, 2, 1, -2, 1, 0, 21, -12, 9, 21, -2, 3, -5, 7, -24, -14, 1, 0, 0, 0, 0, 5, -7, -9, 6, -11, -6, -15, -2, -8, -12, -10, 0, 1, -6, 10, -16, 2, 4, -9, -12, -3, -12, 1, 1, 0, 0, 0, 0, 1, 4, -15, -36, 12, -6, -12, -14, 4, -19, 4, -12, -42, -19, -5, 16, -23, -13, 6, -2, 8, -14, 1, 0, 0, 0, 0, 0, 0, 2, 2, 4, 0, -7, 7, -14, 5, -9, -4, -4, -8, 8, -7, -4, 1, -2, -4, -5, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, -5, 4, -2, 3, -5, 3, -3, 2, 0, -9, 0, 5, -9, -18, -1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 4, -2, 1, 0, 9, 6, -6, 9, 12, -7, 5, -3, 10, -13, 1, -13, 10, -10, -9, 5, 0, 2, 0, 0, 0, 0, 1, -2, 2, -4, 1, 4, -6, -16, 7, -28, -62, -3, -55, -36, -23, -25, -4, -25, -15, 8, 10, 3, -1, -4, 2, 0, 0, 2, 2, 4, -1, -2, 21, -42, -10, -39, 11, -12, -16, -4, -21, -1, -38, -22, -5, -34, 7, -35, -7, 4, 14, 7, 0, 0, 0, 0, 3, -6, -1, 42, -54, -15, 9, -7, -24, 25, -11, 13, -34, 3, -46, -16, -32, -18, -18, -17, -21, -50, -22, 2, 6, 3, 0, -3, 0, 7, -20, -73, 14, -28, -12, 3, -10, 6, 11, 22, 36, 4, 34, 16, 26, 1, 16, -4, -13, -6, -7, -2, -7, 7, 2, 0, 8, 10, -29, -39, -7, -10, -3, -20, 1, -2, 13, 21, 29, 40, 38, 8, -2, -7, 1, -8, -13, 0, -23, -12, -11, -2, -3, 4, -13, -36, 14, 24, -15, -19, 6, -11, -6, -17, -18, 14, 15, 15, 9, -4, 2, -5, -14, 17, -6, -7, 7, 1, -11, 3, -2, 7, -2, 22, -37, -14, 14, -8, -13, -5, 7, -4, 2, -4, 8, 17, -17, -6, -4, -16, 11, -20, -10, -20, 0, -18, -7, 4, 1, 0, -31, 4, 7, -3, 3, -5, 9, -11, 6, 9, -7, -19, 8, -22, -1, -11, -3, 3, -17, -10, 5, 12, -32, 12, -3, -5, 3, -26, -4, -14, -5, 7, -6, 11, 27, 6, 11, 10, -2, 1, -20, 12, 3, 17, 7, -3, 34, 15, 1, -26, 10, -2, -14, -4, 2, -6, 1, -3, -1, 7, 14, 3, -7, 11, 13, 2, -10, 3, 22, 16, 0, 15, 12, 26, 4, 7, 18, 49, -41, -37, -8, 6, 0, -30, -5, -3, 8, 10, -4, 11, 22, 5, 7, 5, 1, -1, 16, 9, 15, 10, 10, 15, 8, 21, -9, -19, 18, -18, 7, 0, -1, -12, 6, 3, -3, 0, 17, 0, -8, 10, 5, 7, 6, -1, 9, 9, 10, 14, 0, -7, 0, -10, 20, -16, -46, 15, -5, -2, 0, 3, -19, 1, -1, 1, -18, 3, 8, -6, 6, 7, 4, 0, -12, 1, 17, 7, -10, 5, 3, -3, -20, -3, -19, -7, 7, -4, 2, 2, -9, -5, 10, 3, -4, -13, 9, -4, 22, 6, 8, -12, -3, -14, 11, -2, 4, -20, 10, -49, 37, 0, -16, -34, 21, -8, 0, -2, -5, 16, -16, -1, -19, 18, 4, 0, 3, -1, 9, 3, -23, 17, -8, -7, 6, -7, 3, 2, -29, -8, -15, -21, -8, -13, 1, 2, 2, -28, 12, -50, 40, -13, -18, 0, 25, 13, -22, -10, 23, -5, -4, 8, -15, -7, 1, 5, 24, -30, -7, -5, -44, -2, 2, -2, -51, 8, -22, -22, -6, 14, 17, -17, -11, -3, 17, 0, -24, -13, 8, -17, 7, -4, 5, 1, -21, -4, -9, -21, 5, 3, 0, 1, 7, -7, -6, 36, -22, -33, -28, -31, -35, -8, -33, -11, -2, -13, -22, -1, 5, 9, -11, -41, 5, -8, -6, 22, -13, 2, 0, -1, 6, -30, -17, 9, 21, -6, 0, 4, -8, -30, -8, -6, -16, -1, -3, -29, -34, -1, 1, 23, -23, -11, 22, -12, -15, 1, 0, 6, -9, 3, 31, -33, -29, -9, 1, -11, 11, 3, 0, -7, -1, -31, -2, -7, 19, -41, 0, -10, -7, 9, -2, 3, 1, 1, 0, 0, -8, 17, -44, 5, 7, -31, 21, -21, -16, -14, -6, -11, 1, -11, -14, -17, -15, 6, -9, -5, 26, -5, 11, -7, -3, 0, 0, 0, 1, -90, 19, -1, 0, 8, -11, -4, -2, -10, -13, -21, -21, -1, -6, 6, -5, 5, 7, 2, -4, 12, -5, -6, -3, 0, 0, 0, 5, -7, -7, 7, -3, -4, 2, -3, 4, -3, 6, 8, 3, 3, 8, 9, 11, 3, 10, 9, -1, -2, -1, 10, -3, 0, 0, 0, 0, 4, 7, -5, 10, 2, 7, 5, 7, 1, 13, 1, 18, -1, 11, 0, 12, 1, 5, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, -2, -1, 2, -1, -3, 3, -5, 5, -15, 7, -8, 4, -10, 3, -5, 4, -5, 6, -3, 0, 0, 0, 0},
};

const int8_t bias[NUM_CLASSES] = {-102, -127, -62, -68, -94, -65, -93, -106, -55, -81};

```

&nbsp;

test_images_q.h

``` h

#ifndef TEST_IMAGES_H
#define TEST_IMAGES_H
#include <stdint.h>
#define NUM_TEST_IMAGES 10
#define NUM_FEATURES 784

int8_t test_images[NUM_TEST_IMAGES][NUM_FEATURES] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 185, 159, 151, 60, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 114, 72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 66, 14, 67, 67, 67, 59, 21, 236, 254, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 253, 209, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 233, 255, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 254, 238, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 249, 254, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 187, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 205, 248, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 254, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 251, 240, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 221, 254, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 203, 254, 219, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 254, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 224, 254, 115, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 242, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 254, 219, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 207, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 125, 171, 255, 255, 150, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 253, 253, 253, 218, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 213, 142, 176, 253, 253, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 250, 253, 210, 32, 12, 0, 6, 206, 253, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 251, 210, 25, 0, 0, 0, 122, 248, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 18, 0, 0, 0, 0, 209, 253, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 117, 247, 253, 198, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 247, 253, 231, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 246, 253, 159, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 234, 253, 233, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 248, 253, 189, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 200, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 253, 253, 173, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 43, 20, 20, 20, 20, 5, 0, 5, 20, 20, 37, 150, 150, 150, 147, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 253, 253, 253, 253, 253, 168, 143, 166, 253, 253, 253, 253, 253, 253, 253, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 249, 247, 247, 169, 117, 117, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 123, 123, 123, 166, 253, 253, 253, 155, 123, 123, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 252, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 244, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 202, 223, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 254, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 254, 195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 254, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 237, 205, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 124, 255, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 254, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 232, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 254, 159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 151, 254, 142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 228, 254, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 251, 254, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 141, 254, 205, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 215, 254, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 198, 176, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 150, 253, 202, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 251, 251, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 197, 251, 251, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 190, 251, 251, 251, 253, 169, 109, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 251, 251, 251, 251, 253, 251, 251, 220, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 182, 255, 253, 253, 253, 253, 234, 222, 253, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 221, 253, 251, 251, 251, 147, 77, 62, 128, 251, 251, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 231, 251, 253, 251, 220, 137, 10, 0, 0, 31, 230, 251, 243, 113, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 251, 251, 253, 188, 20, 0, 0, 0, 0, 0, 109, 251, 253, 251, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 251, 251, 201, 30, 0, 0, 0, 0, 0, 0, 31, 200, 253, 251, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 32, 202, 255, 253, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 251, 251, 0, 0, 0, 0, 0, 0, 0, 0, 109, 251, 253, 251, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 217, 251, 251, 0, 0, 0, 0, 0, 0, 21, 63, 231, 251, 253, 230, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 217, 251, 251, 0, 0, 0, 0, 0, 0, 144, 251, 251, 251, 221, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 217, 251, 251, 0, 0, 0, 0, 0, 182, 221, 251, 251, 251, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218, 253, 253, 73, 73, 228, 253, 253, 255, 253, 253, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 113, 251, 251, 253, 251, 251, 251, 251, 253, 251, 251, 251, 147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 230, 251, 253, 251, 251, 251, 251, 253, 230, 189, 35, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 142, 253, 251, 251, 251, 251, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 174, 251, 173, 71, 72, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 224, 0, 0, 0, 0, 0, 0, 0, 70, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 231, 0, 0, 0, 0, 0, 0, 0, 148, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 195, 231, 0, 0, 0, 0, 0, 0, 0, 96, 210, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 252, 134, 0, 0, 0, 0, 0, 0, 0, 114, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 236, 217, 12, 0, 0, 0, 0, 0, 0, 0, 192, 252, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 168, 247, 53, 0, 0, 0, 0, 0, 0, 0, 18, 255, 253, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 242, 211, 0, 0, 0, 0, 0, 0, 0, 0, 141, 253, 189, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 106, 0, 0, 0, 0, 0, 0, 0, 32, 232, 250, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 225, 252, 0, 0, 0, 0, 0, 0, 0, 0, 134, 252, 211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 252, 164, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 204, 209, 18, 0, 0, 0, 0, 0, 0, 22, 253, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 199, 85, 85, 85, 85, 129, 164, 195, 252, 252, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 170, 245, 252, 252, 252, 252, 232, 231, 251, 252, 252, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 84, 84, 84, 84, 0, 0, 161, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 252, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 232, 236, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 179, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 254, 107, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 227, 254, 254, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 254, 254, 165, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 203, 254, 254, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53, 254, 254, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 254, 254, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 196, 254, 248, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 254, 254, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 254, 254, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 252, 254, 223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 254, 254, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 252, 254, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 254, 254, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 254, 234, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 254, 204, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 211, 254, 196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 158, 254, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 157, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 192, 134, 32, 0, 0, 0, 0, 0, 0, 0, 0, 15, 77, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 235, 250, 169, 0, 0, 0, 0, 0, 0, 0, 0, 15, 220, 241, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 189, 253, 147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 253, 253, 21, 0, 0, 0, 0, 0, 0, 0, 0, 43, 254, 173, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 153, 253, 96, 0, 0, 0, 0, 0, 0, 0, 0, 43, 231, 254, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 255, 204, 11, 0, 0, 0, 0, 0, 0, 0, 0, 104, 254, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 253, 178, 5, 0, 0, 0, 0, 0, 0, 9, 131, 237, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 253, 253, 191, 175, 70, 70, 70, 70, 133, 197, 253, 253, 169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 228, 253, 253, 254, 253, 253, 253, 253, 254, 253, 253, 219, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 65, 137, 254, 232, 137, 137, 137, 44, 253, 253, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 254, 206, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 160, 253, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 254, 241, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 158, 254, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 231, 244, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 254, 232, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 253, 157, 0, 13, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 253, 154, 91, 204, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 253, 254, 253, 154, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 190, 128, 23, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 149, 193, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 224, 253, 253, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 235, 254, 253, 253, 166, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 253, 254, 253, 253, 253, 238, 115, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 241, 253, 208, 185, 253, 253, 253, 231, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 254, 193, 0, 8, 98, 219, 254, 255, 201, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 253, 80, 0, 0, 0, 182, 253, 254, 191, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 253, 155, 0, 0, 0, 234, 253, 254, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 253, 208, 40, 85, 166, 251, 237, 254, 236, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 238, 253, 254, 253, 253, 185, 36, 216, 253, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 240, 255, 254, 145, 8, 0, 134, 254, 223, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 158, 142, 12, 0, 0, 9, 175, 253, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88, 253, 226, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 166, 253, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 245, 253, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 115, 254, 172, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 218, 254, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 254, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 186, 244, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 223, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 47, 47, 47, 16, 129, 85, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 153, 217, 253, 253, 253, 215, 246, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 142, 244, 252, 253, 253, 253, 253, 253, 253, 253, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 253, 253, 253, 253, 253, 253, 253, 213, 170, 170, 170, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 132, 72, 0, 57, 238, 227, 238, 168, 124, 69, 20, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 206, 253, 78, 0, 0, 32, 0, 30, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 177, 253, 132, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 133, 253, 233, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 223, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 253, 174, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 253, 246, 127, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 253, 253, 253, 251, 147, 91, 121, 85, 42, 42, 85, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 232, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 53, 218, 222, 251, 253, 253, 253, 253, 253, 253, 253, 253, 252, 124, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 72, 200, 253, 253, 253, 253, 253, 253, 253, 175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 253, 249, 152, 51, 164, 253, 253, 175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 253, 253, 253, 188, 252, 253, 253, 148, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 167, 253, 253, 253, 253, 250, 175, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 180, 231, 253, 221, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 149, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 56, 137, 201, 199, 95, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 152, 234, 254, 254, 254, 254, 254, 250, 211, 151, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 153, 240, 254, 254, 227, 166, 133, 251, 200, 254, 229, 225, 104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 153, 234, 254, 254, 187, 142, 8, 0, 0, 191, 40, 198, 246, 223, 253, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 126, 253, 254, 233, 128, 11, 0, 0, 0, 0, 210, 43, 70, 254, 254, 254, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 243, 254, 228, 54, 0, 0, 0, 0, 3, 32, 116, 225, 242, 254, 255, 162, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 240, 254, 223, 109, 138, 178, 178, 169, 210, 251, 231, 254, 254, 254, 232, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 175, 244, 253, 255, 254, 254, 251, 254, 254, 254, 254, 254, 252, 171, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 136, 195, 176, 146, 153, 200, 254, 254, 254, 254, 150, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 254, 254, 241, 99, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 250, 254, 254, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 242, 254, 254, 211, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 241, 254, 254, 242, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 131, 254, 254, 244, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 249, 254, 254, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 228, 254, 254, 208, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 255, 254, 254, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 254, 254, 137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 255, 233, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 113, 255, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

int test_labels[NUM_TEST_IMAGES] = { 7, 2, 1, 0, 4, 1, 4, 9, 5, 9 };

#endif // TEST_IMAGES_H


```

- Regression to Real-Time Recognition - A Complete Embedded ML Recap

</details>

## Neural Networks on RISC-V Microcontrollers
<details>
<summary>Neural Networks on RISC-V Microcontrollers</summary>

&nbsp;
<img width="1756" height="1047" alt="image" src="https://github.com/user-attachments/assets/8cd4b3b0-caef-4b4b-9b45-41804a5fcc6d" />

&nbsp;
<img width="1756" height="1047" alt="image" src="https://github.com/user-attachments/assets/5755a038-ebde-4a8c-830c-3af75e9a8212" />

&nbsp;
<img width="1487" height="990" alt="image" src="https://github.com/user-attachments/assets/84579b83-86f2-4eaa-92c3-6a70f9271989" />

&nbsp;
<img width="1716" height="988" alt="image" src="https://github.com/user-attachments/assets/8aeeb6e0-c228-428f-889b-98510222ccd3" />

&nbsp;
<img width="1564" height="994" alt="image" src="https://github.com/user-attachments/assets/7ca5574f-a70a-4966-960c-280fd66bad2a" />

&nbsp;
<img width="1560" height="995" alt="image" src="https://github.com/user-attachments/assets/3d826f78-32bb-4073-b044-7346629a693e" />

&nbsp;

</details>

## Advanced Quantization & Deployment
<details>
<summary>Advanced Quantization & Deployment</summary>

&nbsp;
<img width="1791" height="978" alt="image" src="https://github.com/user-attachments/assets/7a2e97e4-67b4-4e30-85b6-8fad8533a242" />

</details>


## Capstone & Next Steps
<details>
<summary>Capstone & Next Steps</summary>

</details>


## Acknowledgments

- ### Kunal Ghosh, Director, VSD Corp. Pvt. Ltd.
