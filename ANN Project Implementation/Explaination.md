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

> Koi bhi cell ya concept clearly nahi samajh aaya toh poocho, main aur detail mein samjha dunga! 😊
