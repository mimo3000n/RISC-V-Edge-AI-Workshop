
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
