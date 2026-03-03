# 🧠 ANN Project — `experiments.ipynb` Explanation (Hinglish)

Yeh notebook ek **Artificial Neural Network (ANN)** project ka **data preprocessing** part hai. Isme humne ek bank ke customer data ko clean, encode, aur scale kiya hai taaki baad mein ek deep learning model train kar sakein jo predict kare ki **customer bank chhod dega ya nahi (Churn Prediction)**.

---

## 📦 Cell 1 — Libraries Import karna

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
```

### 🔍 Explanation:

- **`pandas`** — Yeh library hai data ko table (DataFrame) form mein load aur manipulate karne ke liye. Jaise Excel mein data hota hai, waise hi yahan `pd.DataFrame` mein hota hai.
- **`train_test_split`** — Yeh function data ko **training set** aur **testing set** mein divide karta hai. Training pe model seekhega, testing pe humne check kiya ki model kitna accha seekha.
- **`StandardScaler`** — Yeh feature values ko **normalize** (standard scale) karta hai. Matlab har feature ka mean 0 aur standard deviation 1 ho jata hai. Yeh isliye zaroori hai kyunki neural networks bahut sensitive hote hain feature ki range ke prati. Agar ek feature 0–1 ke beech hai aur doosra 10000–200000 ke beech, toh model confuse ho jayega.
- **`LabelEncoder`** — Yeh **categorical text data** (jaise "Male", "Female") ko **numbers** (0, 1) mein convert karta hai, kyunki machine learning models sirf numbers samajhte hain.
- **`pickle`** — Yeh Python ka built-in module hai jo objects ko **file mein save** karne ke kaam aata hai. Isse hum trained encoder aur scaler ko save kar sakte hain taaki jab naya prediction karna ho toh wahi encoder/scaler use kar sakein.

---

## 📊 Cell 2 — Data Load karna aur dekhna

```python
data = pd.read_csv("Churn_Modelling.csv")
data.head(5)
```

### 🔍 Explanation:

- **`pd.read_csv()`** — Yeh CSV file (`Churn_Modelling.csv`) ko read karke ek DataFrame bana deta hai.
- **`data.head(5)`** — Yeh pehle **5 rows** dikhata hai, taaki humein idea mile ki data kaisa dikh raha hai.

**Data mein yeh columns hain:**

| Column | Matlab |
|---|---|
| `RowNumber` | Row ka number (useless) |
| `CustomerId` | Customer ka unique ID (useless for prediction) |
| `Surname` | Customer ka surname (useless) |
| `CreditScore` | Customer ka credit score |
| `Geography` | Kis desh se hai (France, Spain, Germany) |
| `Gender` | Male ya Female |
| `Age` | Umar |
| `Tenure` | Kitne saal se bank mein hai |
| `Balance` | Account balance |
| `NumOfProducts` | Kitne products use kar raha hai |
| `HasCrCard` | Credit card hai ya nahi (0/1) |
| `IsActiveMember` | Active member hai ya nahi (0/1) |
| `EstimatedSalary` | Estimated salary |
| **`Exited`** | **🎯 TARGET — 1 = chhod diya, 0 = nahi chhoda** |

---

## 🗑️ Cell 3 — Irrelevant Columns Drop karna

```python
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```

### 🔍 Explanation:

- **`RowNumber`**, **`CustomerId`**, aur **`Surname`** — Yeh columns prediction ke liye bilkul **kaam ke nahi hain**. Inhe row number ya unique ID zyada koi pattern nahi dega model ko.
- **`axis=1`** ka matlab hai hum **columns** drop kar rahe hain (agar `axis=0` hota toh rows drop hoti).
- 🧠 **Concept:** Machine learning mein **feature selection** bahut important hai. Jo columns prediction mein help nahi karte, unhe hata dena chahiye warna model unnecessarily complex ho jayega aur overfitting ho sakti hai.

---

## 🏷️ Cell 4 — Gender Column ko Label Encode karna

```python
label_encode_gender = LabelEncoder()
data['Gender'] = label_encode_gender.fit_transform(data['Gender'])
```

### 🔍 Explanation:

- **`LabelEncoder()`** — Ek object create kiya jisse text ko number mein convert karenge.
- **`fit_transform()`** — Pehle `fit` karta hai (seekhta hai ki kitni unique values hain — "Female", "Male"), phir `transform` karta hai (convert karta hai numbers mein).
  - `Female` → **0**
  - `Male` → **1**
- 🧠 **Concept:** Gender sirf **2 categories** hai (binary), toh `LabelEncoder` kaafi hai. Agar 3+ categories hoti jahan koi ordering nahi hoti, toh LabelEncoder galat results de sakta hai kyunki model sochega ki 2 > 1 > 0 (ordering hai), isliye wahan One-Hot Encoding better hai.

---

## 👀 Cell 5 — Data Check karna

```python
data
```

### 🔍 Explanation:

- Bas poora DataFrame display kar rahe hain taaki confirm ho jaye ki Gender column successfully encode ho gaya hai (ab 0 aur 1 dikh raha hai "Female"/"Male" ki jagah).

---

## 🌍 Cell 6 — Geography Column ko One-Hot Encode karna

```python
from sklearn.preprocessing import OneHotEncoder
ohe_geo = OneHotEncoder()
geo_encoder = ohe_geo.fit_transform(data[['Geography']])
geo_encoder
```

### 🔍 Explanation:

- **`OneHotEncoder()`** — Yeh categorical variable ko **multiple binary columns** mein convert karta hai.
- **`data[['Geography']]`** — Double brackets isliye hain kyunki `OneHotEncoder` ko 2D input chahiye (DataFrame format).
- **Result:** Geography mein 3 unique values hain: France, Germany, Spain. Toh yeh 3 nayi columns banayega:
  - `Geography_France` → [1, 0, 0]
  - `Geography_Germany` → [0, 1, 0]
  - `Geography_Spain` → [0, 0, 1]
- Output ek **Sparse Matrix** hai (memory efficient storage), isliye print karne pe `<Compressed Sparse Row sparse matrix>` dikhta hai.

> 🧠 **Concept — LabelEncoder vs OneHotEncoder:**
> Geography mein 3 categories hain jahan koi **natural ordering** nahi hai (France > Germany? Nahi!). Agar LabelEncoder lagate toh 0, 1, 2 assign hota aur model galat samajhta ki inme ranking hai. OneHotEncoder yeh problem solve karta hai — har category ke liye alag binary column bana deta hai.

---

## 📋 Cell 7 — Encoded Feature Names dekhna

```python
ohe_geo.get_feature_names_out(['Geography'])
```

### 🔍 Explanation:

- Yeh dikhata hai ki OneHotEncoder ne kaun-kaun si nayi columns banaayi hain:
  - `Geography_France`
  - `Geography_Germany`
  - `Geography_Spain`
- Yeh names baad mein DataFrame banane ke kaam aayenge.

---

## 📊 Cell 8 — Sparse Matrix ko DataFrame mein Convert karna

```python
geo_en_df = pd.DataFrame(geo_encoder.toarray(), columns=ohe_geo.get_feature_names_out(['Geography']))
```

### 🔍 Explanation:

- **`.toarray()`** — Sparse matrix ko normal numpy array mein convert karta hai.
- Phir usse ek proper DataFrame bana diya with correct column names.
- Ab humre paas ek alag DataFrame hai jisme 3 columns hain: `Geography_France`, `Geography_Germany`, `Geography_Spain`.

---

## 🔗 Cell 9 — Original Data mein Encoded Columns Merge karna

```python
data = pd.concat([data.drop('Geography', axis=1), geo_en_df], axis=1)
```

### 🔍 Explanation:

- **`data.drop('Geography', axis=1)`** — Pehle original `Geography` column hata diya (kyunki ab humne usse encode kar diya hai).
- **`pd.concat([..., geo_en_df], axis=1)`** — Phir encoded columns (`Geography_France`, `Geography_Germany`, `Geography_Spain`) ko original data ke saath **horizontally (side by side)** jod diya.
- `axis=1` matlab columns ke saath concat karo (left to right), `axis=0` hota toh rows ke saath (top to bottom).

---

## 👀 Cell 10 — Final Preprocessed Data dekhna

```python
data
```

### 🔍 Explanation:

- Ab data mein 13 columns hain — sab numerical hain, koi text column nahi bacha. Yeh neural network ke liye ready hai.

---

## 💾 Cell 11 — Encoders ko Save karna (Pickle)

```python
with open('label_encode_gender.pkl', 'wb') as file:
    pickle.dump(label_encode_gender, file)

with open('geo_en_df.pkl', 'wb') as file:
    pickle.dump(geo_en_df, file)
```

### 🔍 Explanation:

- **`pickle.dump()`** — Object ko file mein save karta hai binary format mein.
- **`'wb'`** — Write + Binary mode.
- Humne **gender encoder** aur **geography encoded DataFrame** ko save kiya.

> 🧠 **Concept — Kyun save kar rahe hain?**
> Jab hum baad mein naye customer ka data prediction ke liye denge, toh uss naye data ko bhi **exactly waise hi encode** karna padega jaise training data ko kiya tha. Isliye hum encoder objects save kar lete hain taaki baad mein `pickle.load()` se load karke use kar sakein. Agar naye encoder se encode karenge toh mapping alag ho sakti hai aur prediction galat aayega!

---

## ✂️ Cell 12 — Feature-Target Split, Train-Test Split, aur Scaling

```python
X = data.drop('Exited', axis=1)
Y = data['Exited']

# Split in training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 🔍 Explanation:

**Step 1 — Feature-Target Split:**
- **`X`** = Saare input features (12 columns — Gender, CreditScore, Age, etc.)
- **`Y`** = Target variable — `Exited` (0 ya 1 — customer chhoda ya nahi)

**Step 2 — Train-Test Split:**
- **`train_test_split(X, Y, test_size=0.2, random_state=42)`**
  - **80%** data → Training ke liye (`X_train`, `Y_train`)
  - **20%** data → Testing ke liye (`X_test`, `Y_test`)
  - **`random_state=42`** — Yeh ek seed number hai. Isse ensure hota hai ki har baar same split hoga (reproducibility ke liye). 42 koi magic number nahi hai, kuch bhi de sakte ho — bas har baar consistent results chahiye isliye fix karte hain.

**Step 3 — Feature Scaling:**
- **`StandardScaler()`** — Har feature ko standardize karta hai: `z = (x - mean) / std`
  - Matlab har feature ka **mean 0** aur **standard deviation 1** ho jayega.
- **`scaler.fit_transform(X_train)`** — Training data pe **fit** (mean/std calculate) + **transform** (apply) dono kar raha hai.
- **`scaler.transform(X_test)`** — Test data pe sirf **transform** kar raha hai (fit nahi!). ⚠️ Yeh bahut important hai!

> 🧠 **Concept — Data Leakage:**
> Agar hum test data pe bhi `fit_transform` karenge, toh test data ki information (mean/std) model ke training process mein leak ho jayegi, jisse model ki real-world performance ka galat estimate milega. Isliye test data pe hamesha sirf `transform` karte hain.

---

## 💾 Cell 13 — Scaler ko Save karna

```python
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
```

### 🔍 Explanation:

- **StandardScaler** ko bhi save kar rahe hain — kyunki jab naya data aayega prediction ke liye, toh usse bhi same scale (same mean aur std values) se transform karna hoga.

---

## 🎯 Summary — Overall Flow

```
CSV Data Load
    ↓
Drop useless columns (RowNumber, CustomerId, Surname)
    ↓
Label Encode: Gender (Male/Female → 1/0)
    ↓
One-Hot Encode: Geography (France/Germany/Spain → 3 binary columns)
    ↓
Merge encoded columns back
    ↓
Save encoders (pickle)
    ↓
Feature-Target Split (X, Y)
    ↓
Train-Test Split (80-20)
    ↓
Feature Scaling (StandardScaler)
    ↓
Save scaler (pickle)
    ↓
✅ Data ready for ANN training!
```

---

---

# 🧠 ANN Implementation — Cell by Cell Explanation

Ab data preprocessing complete ho chuka hai. Ab hum **Artificial Neural Network (ANN)** build, train, aur save karenge!

---

## 🔧 Cell 14 (Markdown) — Section Header

```markdown
# ANN Implementation
```

### 🔍 Explanation:

Yeh sirf ek **heading cell** hai jo notebook mein visually separate karta hai ki ab se ANN (Artificial Neural Network) ka implementation shuru ho raha hai.

---

## 📦 Cell 15 — Deep Learning Libraries Import karna

```python
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
```

### 🔍 Explanation:

- **`tensorflow`** — Yeh Google ka **deep learning framework** hai. Isse hum neural networks banate, train karte, aur deploy karte hain. Yeh GPU acceleration bhi support karta hai.
- **`Sequential`** — Yeh ek model type hai jisme layers **ek ke baad ek (sequentially)** lagti hain — jaise ek pipe mein paani ek direction mein behta hai. Simple feedforward networks ke liye best hai.
- **`Dense`** — Yeh **fully connected layer** hai. Iska matlab hai ki is layer ka har neuron pichli layer ke har neuron se connected hai. Yeh ANN ki building block hai.
- **`EarlyStopping`** — Yeh ek **callback** hai. Agar training ke dauran model improve hona band ho jaye (val_loss barhne lage), toh yeh automatically training rok deta hai. Isse **overfitting** se bachte hain.
- **`TensorBoard`** — Yeh ek **visualization tool** hai jo training ke dauran loss, accuracy, weights, biases wagairah ke graphs dikhata hai. Real-time monitoring ke liye use hota hai.
- **`datetime`** — Yeh Python ka built-in module hai jo current date-time deta hai. Isko TensorBoard ke log folder ka unique naam banane ke liye use kiya hai.

---

## 🏗️ Cell 16 — ANN Model Build karna

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # HL 1
    Dense(32, activation='relu'), # HL 2
    Dense(1, activation='sigmoid') # o/p layer
])
```

### 🔍 Explanation:

Yahan hum **3 layers** ka ek neural network bana rahe hain:

**Layer 1 — Hidden Layer 1:**
- `Dense(64)` — **64 neurons** hain is layer mein.
- `activation='relu'` — **ReLU (Rectified Linear Unit)** activation function use ho raha hai: `f(x) = max(0, x)`. Matlab agar input negative hai toh output 0, agar positive hai toh wahi value. Yeh deep learning mein sabse popular activation hai kyunki yeh **vanishing gradient problem** solve karta hai.
- `input_shape=(X_train.shape[1],)` — Pehli layer mein batana padta hai ki input mein kitne features hain. `X_train.shape[1]` = **12 features** (humne preprocessing mein 12 columns rakhe the).

**Layer 2 — Hidden Layer 2:**
- `Dense(32)` — **32 neurons**. Layer 1 se choti hai — yeh ek common practice hai ki layers progressively chhoti hoti jaayein (64 → 32 → 1). Isse model **abstract features** seekhna shuru karta hai.

> 🧠 **"Abstract Features Seekhna" ka Matlab:**
>
> **Layer-by-Layer kya hota hai:**
>
> | Layer | Neurons | Kya seekhta hai? |
> |-------|---------|-------------------|
> | **Layer 1 (64)** | Zyada neurons | **Low-level / basic features** — jaise raw data ke simple patterns (e.g., "age zyada hai", "salary kam hai") |
> | **Layer 2 (32)** | Kam neurons | **Abstract / high-level features** — pehle layer ke basic patterns ko **combine** karke complex relationships banata hai (e.g., "age zyada hai AUR salary bhi zyada hai toh churn kam hoga") |
> | **Layer 3 (1)** | Ek neuron | **Final decision** — sab abstract features ko milake ek final prediction deta hai (0 ya 1) |
>
> **"Abstract" ka simple meaning:** Abstract = wo cheez jo **directly data mein dikhti nahi**, lekin model usse khud seekh leta hai.
>
> **Example:** Socho tumhare paas customer ka data hai — `age`, `salary`, `credit_score`, `balance`.
> - **Basic feature:** "Credit score 300 se kam hai" ← yeh directly data mein dikh raha hai
> - **Abstract feature:** "Yeh customer financially unstable hai" ← yeh kisi ek column mein nahi likha, lekin model ne `credit_score` + `balance` + `salary` ko **combine** karke yeh pattern khud seekh liya
>
> **Layers progressively chhoti kyun hoti hain?**
> ```
> 64 neurons → bahut saare basic patterns detect karo
> 32 neurons → un patterns ko compress/combine karke abstract meaning nikalo
>  1 neuron  → final answer do
> ```
> Yeh ek **funnel (깔때기) jaisa** kaam karta hai — pehle bahut saari information collect hoti hai, phir woh **summarize** hoti jaati hai, aur end mein ek **single decision** aata hai.

- `activation='relu'` — Same ReLU activation.

**Layer 3 — Output Layer:**
- `Dense(1)` — Sirf **1 neuron** kyunki yeh **binary classification** hai (Exited: 0 ya 1).
- `activation='sigmoid'` — **Sigmoid function** output ko 0 se 1 ke beech laata hai. Yeh probability deta hai — jaise 0.85 matlab 85% chance hai ki customer chhod dega. Agar output > 0.5, toh class 1 (Exited), warna class 0 (Not Exited).

> 🧠 **Architecture Diagram:**
> ```
> Input (12 features) → [Dense 64, ReLU] → [Dense 32, ReLU] → [Dense 1, Sigmoid] → Output (0 or 1)
> ```

---

## 📊 Cell 17 — Model Summary dekhna

```python
model.summary()
```

### 🔍 Explanation:

Yeh model ki poori structure dikhata hai:

| Layer | Output Shape | Parameters |
|---|---|---|
| dense_3 (Dense) | (None, 64) | **832** |
| dense_4 (Dense) | (None, 32) | **2080** |
| dense_5 (Dense) | (None, 1) | **33** |
| **Total** | | **2,945** |

**Parameters kaise calculate hote hain?**
- Layer 1: `(12 inputs × 64 neurons) + 64 biases = 768 + 64 = 832`
- Layer 2: `(64 inputs × 32 neurons) + 32 biases = 2048 + 32 = 2080`
- Layer 3: `(32 inputs × 1 neuron) + 1 bias = 32 + 1 = 33`

> 🧠 **Formula:** `Parameters = (inputs × neurons) + biases`
> Har neuron ka apna ek bias hota hai — yeh intercept jaisa kaam karta hai (jaise y = mx + **b** mein b).

---

## ⚙️ Cell 18 — Optimizer aur Loss Function Define karna

```python
opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.01)
loss = tensorflow.keras.losses.BinaryCrossentropy()
loss
```

### 🔍 Explanation:

- **`Adam` Optimizer** — Yeh sabse popular optimizer hai. Yeh **learning rate ko automatically adjust** karta hai training ke dauran. `SGD` se bahut better aur faster converge karta hai.
  - `learning_rate = 0.01` — Model kitne bade steps mein seekhega. Bahut bada hoga toh optimal point miss ho jayega, bahut chhota hoga toh bahut slow seekhega. 0.01 ek reasonable starting point hai.

- **`BinaryCrossentropy`** — Yeh binary classification ke liye standard **loss function** hai. Yeh measure karta hai ki predicted probability actual label se kitni door hai.
  - Formula: `Loss = -[y·log(p) + (1-y)·log(1-p)]`
  - Jahan `y` = actual label (0 ya 1), `p` = predicted probability.
  - Agar model sahi predict karta hai toh loss kam, galat karta hai toh loss zyada.

---

## 🔨 Cell 19 — Model Compile karna

```python
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])
```

### 🔍 Explanation:

- **`compile()`** — Model ko training ke liye ready karta hai. Teen cheezein specify karni padti hain:
  1. **`optimizer`** — Weights kaise update hone chahiye (Adam use kar rahe hain)
  2. **`loss`** — Galti kaise measure karni hai (Binary Crossentropy)
  3. **`metrics`** — Training ke dauran kya track karna hai — yahan **accuracy** track ho rahi hai

> 🧠 **Analogy:** Compile karna aise hai jaise exam se pehle decide karna ki — konsi pen se likhenge (optimizer), marks kaise calculate honge (loss), aur result kaise dekhenge (metrics).

---

## 📈 Cell 20 — TensorBoard Setup karna

```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d - %H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
```

### 🔍 Explanation:

- **`log_dir`** — Ek unique folder bana rahe hain jisme training logs store honge. `datetime.datetime.now().strftime(...)` current timestamp deta hai (e.g., `20260301 - 204530`), taaki har run ki logs alag folder mein jaayein.
- **`TensorBoard(log_dir=log_dir, histogram_freq=1)`** — TensorBoard callback create kiya:
  - `histogram_freq=1` — Har epoch ke baad weights aur biases ke **histograms** bhi log honge. Yeh visualize karna helpful hai ki training ke dauran model ke parameters kaise change ho rahe hain.

> 🧠 **TensorBoard kya hai?** Yeh ek web-based dashboard hai (localhost:6006 pe) jahan aap training curves dekh sakte ho — loss kaise gir raha hai, accuracy kaise badh rahi hai, etc. Debugging ke liye bahut powerful tool hai.

---

## 🛑 Cell 21 — Early Stopping Setup karna

```python
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

### 🔍 Explanation:

- **`monitor='val_loss'`** — **Validation loss** ko monitor kar raha hai. Hum chahte hain ki model unseen data pe bhi accha perform kare, sirf training data pe nahi.
- **`patience=10`** — Agar **10 consecutive epochs** tak val_loss improve nahi hua (kam nahi hua), toh training ruk jayegi. Yeh overfitting rokta hai.
- **`restore_best_weights=True`** — Jab training ruke, tab model ke weights ko **sabse acche epoch** ke weights pe reset kar dega. Matlab last epoch ke weights nahi, balki jis epoch pe val_loss sabse kam tha woh weights use honge.

> 🧠 **Concept — Overfitting kya hai?**
> Jab model training data ko toh bahut accha seekh le (ratta laga le) lekin naye data pe galat predict kare — isko overfitting kehte hain. Early Stopping isse rokta hai by stopping training at the right time.

---

## 🏋️ Cell 22 — Model Training karna

```python
history = model.fit(
    X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100,
    callbacks = [tensorflow_callback, early_stopping_callback]
)
```

### 🔍 Explanation:

- **`model.fit()`** — Yeh actual mein model ko **train** karta hai.
- **`X_train, Y_train`** — Training data pe seekhega.
- **`validation_data = (X_test, Y_test)`** — Har epoch ke baad test data pe evaluate karega (taaki overfitting check ho sake).
- **`epochs = 100`** — Maximum 100 baar poora training data dekhega. Lekin Early Stopping lagayi hai toh zaruri nahi ki 100 tak jaye.
- **`callbacks`** — Dono callbacks pass kiye — TensorBoard (logging ke liye) + EarlyStopping (overfitting rokne ke liye).
- **`history`** — Training ki history return hoti hai — loss, accuracy, val_loss, val_accuracy har epoch ki. Yeh baad mein plotting ke liye use ho sakti hai.

**Training Output Results:**
- Model **15 epochs** mein ruk gaya (Early Stopping ne rok diya).
- **Final Training Accuracy: ~87%**
- **Final Validation Accuracy: ~86%**
- Training aur validation accuracy close hain — matlab model ne **na overfitting ki, na underfitting** — ek accha generalized model ban gaya! ✅

---

## 💾 Cell 23 — Trained Model ko Save karna

```python
model.save('model.h5')
```

### 🔍 Explanation:

- **`.save('model.h5')`** — Poore trained model ko (architecture + weights + optimizer state) ek **HDF5 file** mein save kar diya.
- `.h5` format purana (legacy) hai — naye versions mein `.keras` format recommended hai. Lekin `.h5` bhi perfectly kaam karta hai.
- Baad mein `tensorflow.keras.models.load_model('model.h5')` se load karke direct predictions le sakte hain bina dobara training kiye.

> 🧠 **Kyun save karte hain?** Training mein time aur compute lagta hai. Ek baar train karke save karo, phir jab chahiye tab load karke use karo — jaise ek trained employee ko hire karna vs. naye ko train karna!

---

## 📊 Cell 24 — TensorBoard Extension Load karna

```python
%reload_ext tensorboard
```

### 🔍 Explanation:

- **`%reload_ext`** — Yeh ek **Jupyter magic command** hai. Yeh TensorBoard extension ko Jupyter notebook ke andar reload karta hai.
- Isse TensorBoard notebook ke andar hi render ho sakta hai, alag browser tab kholne ki zarurat nahi.

---

## 📊 Cell 25 — TensorBoard Launch karna

```python
%tensorboard --logdir logs/fit
```

### 🔍 Explanation:

- **`%tensorboard`** — Yeh directly notebook mein TensorBoard dashboard embed kar deta hai.
- **`--logdir logs/fit`** — Batata hai ki logs kahan se padhne hain. Humne training ke dauran logs `logs/fit/` folder mein save kiye the.
- Yeh **localhost:6006** pe TensorBoard start karta hai jahan aap dekh sakte ho:
  - **Loss curves** — Training loss vs Validation loss
  - **Accuracy curves** — Training accuracy vs Validation accuracy
  - **Histograms** — Weights aur biases kaise change ho rahe hain har epoch mein
  - **Distributions** — Layers ke parameters ka distribution

---

## 🎯 Complete ANN Pipeline Summary

```
📦 Import Libraries (TensorFlow, Keras)
    ↓
🏗️ Build Model (Sequential: 64 → 32 → 1 neurons)
    ↓
📊 Check Model Summary (2,945 trainable parameters)
    ↓
⚙️ Define Optimizer (Adam, lr=0.01) & Loss (Binary Crossentropy)
    ↓
🔨 Compile Model
    ↓
📈 Setup TensorBoard (logging)
    ↓
🛑 Setup Early Stopping (patience=10)
    ↓
🏋️ Train Model (stopped at epoch 15, ~86% accuracy)
    ↓
💾 Save Model (model.h5)
    ↓
📊 Visualize in TensorBoard
    ↓
✅ ANN Model Ready for Predictions!
```

---


---

---

# 🔮 Prediction Notebook — `prediction.ipynb` Explanation

Ab humara model train ho chuka hai aur save bhi ho gaya hai. Ab hum ek **naye customer ka data** deke predict karenge ki woh bank chhod dega ya nahi. Yeh notebook exactly wahi karta hai!

---

```python
import tensorflow
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
```

Libraries import ki:

- **`tensorflow`** — Google ka deep learning framework.
- **`load_model`** — Saved trained model (`.h5` file) ko load karta hai. Dobara training ki zarurat nahi!
- **`pickle`** — Saved encoders aur scaler ko load karne ke liye (`.pkl` files).
- **`pandas`** — Data ko DataFrame format mein handle karne ke liye.
- **`numpy`** — Numerical operations ke liye (prediction output numpy array mein aata hai).

---

```python
model = load_model('model.h5')

with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('label_encode_gender.pkl', 'rb') as file:
    label_encode_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
```

Yahan 4 cheezein load ho rahi hain:

- **`model.h5`** — Trained ANN model load kiya (architecture + weights + optimizer sab stored hai).
- **`ohe_geo.pkl`** — Geography ka **OneHotEncoder** load kiya (France/Germany/Spain → 3 binary columns).
- **`label_encode_gender.pkl`** — Gender ka **LabelEncoder** load kiya (Male → 1, Female → 0).
- **`scaler.pkl`** — **StandardScaler** load kiya (features ko normalize karne ke liye).

> 🧠 **Important:** Yeh wahi objects hain jo training mein save kiye the. Naye nahi bana rahe — kyunki naye banayenge toh mapping alag ho jayegi aur prediction galat aayega!

---

```python
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
```

Ek naye customer ka data dictionary mein define kiya — yeh woh customer hai jiske liye hum predict karenge ki woh bank chhod dega ya nahi.

---

```python
geo_encoded = ohe_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
geo_encoded_df
```

`'France'` ko OneHotEncoder se encode kiya → `[1.0, 0.0, 0.0]`. Double brackets `[[...]]` isliye kyunki encoder ko 2D input chahiye. `.toarray()` sparse matrix ko normal array banata hai. Phir DataFrame bana diya column names ke saath (`Geography_France`, `Geography_Germany`, `Geography_Spain`).

---

```python
input_df = pd.DataFrame([input_data])
input_df
```

Dictionary ko DataFrame mein convert kiya — kyunki encoding aur scaling DataFrame pe kaam karte hain.

---

```python
input_df['Gender'] = label_encode_gender.transform(input_df['Gender'])
input_df
```

Gender ko LabelEncoder se encode kiya — `Male` → `1`, `Female` → `0`. Sirf `transform` use kiya, `fit_transform` nahi — kyunki encoder pehle se trained hai.

---

```python
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
input_df
```

Original `Geography` text column hataya aur one-hot encoded columns (`Geography_France`, `Geography_Germany`, `Geography_Spain`) jod diye. Ab DataFrame mein **12 numerical columns** hain — exactly training data jaisi.

---

```python
input_scaled = scaler.transform(input_df)
input_scaled
```

Saved StandardScaler se data normalize kiya. Saari values ab `-1 se +1` ke around aa gayi. Sirf `transform` kiya — training ke mean/std se scale ho raha hai.

### 📏 `scaler.transform()` — Detail Explanation:

**StandardScaler ka formula:**

```
z = (x - mean) / std
```

- `x` = original value
- `mean` = us feature ka average (training data se calculate hua tha)
- `std` = us feature ka standard deviation (training data se calculate hua tha)
- `z` = scaled/normalized value

**`transform` vs `fit_transform` — Yeh kyun important hai?**

Training ke time (`experiments.ipynb` mein):
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit + transform
```
- **`fit`** → Training data ka har column ka `mean` aur `std` calculate karke **yaad kar liya**
- **`transform`** → Phir uss mean/std se values ko scale kar diya

Prediction ke time (`prediction.ipynb` mein):
```python
input_scaled = scaler.transform(input_df)  # sirf transform
```
- **Sirf `transform`** — Wahi **purane mean/std** (jo training data se aaye the) use karke naye data ko scale kar raha hai
- **`fit` nahi kar rahe** — Kyunki agar naye data ka apna mean/std calculate karenge toh scale alag ho jayega aur model confuse ho jayega

**Example:** Maan lo training data mein `Balance` column ka mean = 76,000 aur std = 62,000. Naye customer ka `Balance = 60,000`:

```
z = (60,000 - 76,000) / 62,000 = -16,000 / 62,000 = -0.258
```

Yahi `-0.258` output mein dikh raha hai Balance ke liye!

**Actual output — har feature ka scaled value:**

| Feature | Original Value | Scaled Value | Matlab |
|---|---|---|---|
| CreditScore | 600 | -0.536 | Average se thoda neeche |
| Gender | 1 (Male) | 0.913 | Average se upar |
| Age | 40 | 0.105 | Average ke aas-paas |
| Tenure | 3 | -0.695 | Average se neeche |
| Balance | 60,000 | -0.258 | Average se thoda kam |
| NumOfProducts | 2 | 0.808 | Average se upar |
| HasCrCard | 1 | 0.649 | Average se upar |
| IsActiveMember | 1 | 0.975 | Average se kaafi upar |
| EstimatedSalary | 50,000 | -0.877 | Average se kaafi neeche |
| Geo_France | 1.0 | 1.002 | France hai |
| Geo_Germany | 0.0 | -0.579 | Germany nahi |
| Geo_Spain | 0.0 | -0.576 | Spain nahi |

**Scaling kyun zaroori hai?** Bina scaling ke `Balance = 60,000` aur `HasCrCard = 1` — model ko lagega Balance zyada important hai kyunki value badi hai. Scaling ke baad sab features **same scale** pe aa jaate hain, toh model fair comparison kar pata hai.

---

```python
prediction = model.predict(input_scaled)
prediction
```

Scaled data ko model mein daala. Sigmoid output layer ne probability di: **`0.0267`** (2.67%) — matlab is customer ke churn chances bahut kam hain.

> 🧠 **Yaad karo:** Sigmoid output = probability. Jitna 1 ke paas, utna zyada chance ki customer chhod dega. Jitna 0 ke paas, utna kam chance.

---

```python
prediction_proba = prediction[0][0]
prediction_proba
```

`prediction` ek 2D array hai `[[0.02668187]]`. `[0][0]` se single number nikala = `0.026681866`.

---

```python
if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')
```

Probability `0.5` se compare ki. `0.0267` < `0.5`, toh output: **"The customer is not likely to churn."** ✅

> 🧠 **0.5 threshold kyun?** Yeh binary classification ka standard threshold hai. Lekin real-world mein business requirement ke hisaab se change ho sakta hai — jaise bank conservative hona chahta hai toh 0.3 bhi rakh sakte hain.

---

## 🎯 Complete Prediction Pipeline Summary

```
📂 Load Model + Encoders + Scaler
    ↓
📝 Naye customer ka data define karo
    ↓
🌍 Geography One-Hot Encode karo (France → [1,0,0])
    ↓
📊 DataFrame mein convert karo
    ↓
🏷️ Gender Label Encode karo (Male → 1)
    ↓
🔗 Encoded columns merge karo (12 features ready)
    ↓
📏 Scale karo (StandardScaler se normalize)
    ↓
🔮 Model.predict() se probability nikalo
    ↓
✅ Probability > 0.5? → Churn! | ≤ 0.5? → Not Churn!
```

**Is customer ka result:** Probability = 2.67% → **Not likely to churn** 🟢

---


