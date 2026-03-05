# 📘 Embedding.ipynb — Complete Explanation (Hinglish)

Yeh notebook **Word Embedding** aur **One-Hot Encoding** ka concept sikhata hai. Isme hum dekhenge ki NLP mein text ko numbers mein kaise convert karte hain taaki machine learning models samajh sake.

---

## Cell 1 — Import One-Hot Function

```python
from tensorflow.keras.preprocessing.text import one_hot
```

**Explanation:**
- Yahan hum `one_hot` function import kar rahe hain Keras se.
- **One-Hot Encoding** ek technique hai jisme har word ko ek unique number (integer) mein convert kiya jaata hai ek fixed vocabulary size ke andar.
- Yeh number actually ek hashing trick use karta hai — word ko hash karta hai aur vocabulary size ke andar ek index assign karta hai.

> ⚠️ **Note:** Yeh traditional one-hot encoding nahi hai (jahan ek long vector hota hai with 0s and 1s). Yeh actually **hashing-based integer encoding** hai — har word ko ek random integer milta hai vocabulary size ke range mein.

---

## Cell 2 — Define Sample Sentences

```python
### sentences
sent = [
    'the glass of milk',
    'the glass of juice',
    'the cup of tea',
    'I am a good boy',
    'I am a good developer',
    'understand the meaning of words',
    'your videos are good',
]
```

**Explanation:**
- Yahan humne 7 simple English sentences define kiye hain ek list mein.
- Yeh hamare **corpus** (text data) ka kaam karega.
- In sentences ko hum aage one-hot encode karenge aur fir embedding layer mein pass karenge.

---

## Cell 3 — Display Sentences

```python
sent
```

**Output:**
```
['the glass of milk',
 'the glass of juice',
 'the cup of tea',
 'I am a good boy',
 'I am a good developer',
 'understand the meaning of words',
 'your videos are good']
```

**Explanation:**
- Bas sentences ko print/display kar rahe hain confirm karne ke liye ki data sahi hai.

---

## Cell 4 — Set Vocabulary Size

```python
voc_size = 100000
```

**Explanation:**
- Yahan hum **vocabulary size** set kar rahe hain `100000`.
- Iska matlab hai ki one-hot encoding mein har word ko `0` se `99999` ke beech mein ek integer assign hoga.
- Bada vocabulary size rakhne se **hash collision** (do alag words ko same number milna) kam hota hai.
- Real projects mein vocabulary size dataset ke unique words ke hisaab se set karte hain.

---

## Cell 5 — One-Hot Representation

```python
# One Hot Representation

one_hot_repr = [one_hot(word, voc_size) for word in sent]
one_hot_repr
```

**Output:**
```
[[8310, 66459, 38152, 16661],
 [8310, 66459, 38152, 82490],
 [8310, 62026, 38152, 4914],
 [84431, 40976, 40760, 1594, 63505],
 [84431, 40976, 40760, 1594, 43673],
 [82339, 8310, 5221, 38152, 46676],
 [85869, 37489, 91034, 1594]]
```

**Explanation:**
- Har sentence ke har word ko ek integer assign ho gaya hai using `one_hot()` function.
- `one_hot(sentence, vocabulary_size)` → sentence ke har word ko ek hashed integer deta hai `voc_size` ke range mein.
- **Example:** `'the glass of milk'` → `[8310, 66459, 38152, 16661]`
  - `the` → `8310`
  - `glass` → `66459`
  - `of` → `38152`
  - `milk` → `16661`
- Notice karo ki same words ko same number mila hai (jaise `the` hamesha `8310` hai, `of` hamesha `38152` hai).
- Alag sentences ki alag length hai (4 words, 5 words) — yeh problem hai, kyunki neural networks ko **fixed-length input** chahiye. Isko aage **padding** se solve karenge.

---

## Cell 6 — Import Embedding & Padding Libraries

```python
# Word Embedding Representation

from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential

import numpy as np
```

**Explanation:**
- **`Embedding`** — Yeh Keras ka Embedding layer hai. Yeh har integer (word index) ko ek **dense vector** (fixed-size float array) mein convert karta hai. Yahi actual "word embedding" hai.
- **`pad_sequences`** — Yeh function sequences ko **equal length** banata hai by adding zeros (padding). Neural network ko same length input chahiye.
- **`Sequential`** — Keras ka Sequential model, jisme layers ek ke baad ek add karte hain.
- **`numpy`** — Numerical computation ke liye standard library.

---

## Cell 7 — Pad Sequences

```python
sent_length = 8
embedded_docs = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
embedded_docs
```

**Output:**
```
array([[    0,     0,     0,     0,  8310, 66459, 38152, 16661],
       [    0,     0,     0,     0,  8310, 66459, 38152, 82490],
       [    0,     0,     0,     0,  8310, 62026, 38152,  4914],
       [    0,     0,     0, 84431, 40976, 40760,  1594, 63505],
       [    0,     0,     0, 84431, 40976, 40760,  1594, 43673],
       [    0,     0,     0, 82339,  8310,  5221, 38152, 46676],
       [    0,     0,     0,     0, 85869, 37489, 91034,  1594]])
```

**Explanation:**
- Humne `maxlen=8` set kiya, matlab har sentence exactly 8 positions ka hoga.
- **`padding='pre'`** — Choti sentences ke **shuru mein** zeros add ho gaye (pre-padding).
  - Example: `[8310, 66459, 38152, 16661]` (4 words) → `[0, 0, 0, 0, 8310, 66459, 38152, 16661]` (8 positions)
- Agar `padding='post'` hota toh zeros end mein lagte.
- Ab sab sentences ki **same length** (8) hai — neural network ko input dene ke liye ready hai! ✅

---

## Cell 8 — Set Embedding Dimension

```python
## Feature Extraction

dim = 10
```

**Explanation:**
- **Embedding dimension** = `10` set kiya.
- Iska matlab hai ki har word ko ek **10-dimensional dense vector** mein represent kiya jaayega.
- Example: Word `"the"` (index `8310`) → `[0.027, 0.038, -0.007, ..., 0.031]` (10 float numbers)
- Yeh dimension hyperparameter hai — isko 32, 64, 128, 256 bhi rakh sakte hain depending on dataset size aur complexity.
- Zyada dimension = zyada information capture hoti hai, lekin zyada computation bhi lagta hai.

---

## Cell 9 — Build Embedding Model

```python
model = Sequential()
model.add(Embedding(voc_size, dim, input_length=sent_length))
model.compile('adam', 'mse')
```

**Explanation:**
- Ek **Sequential model** banaya aur usme ek **Embedding layer** add kiya.
- **`Embedding(voc_size, dim, input_length=sent_length)`**:
  - `voc_size = 100000` — Total vocabulary size (kitne unique words ho sakte hain)
  - `dim = 10` — Har word ka embedding vector 10 numbers ka hoga
  - `input_length = 8` — Har input sequence ki length 8 hai
- **Internally kya hota hai:**
  - Ek bada **weight matrix** banta hai of shape `(100000, 10)` — matlab 100000 words ke liye 10-dim vectors
  - Jab koi word index (e.g., `8310`) pass hota hai, toh us row ka 10-dim vector return hota hai
  - Yeh vectors initially random hote hain, training ke dauran learn hote hain
- **`model.compile('adam', 'mse')`** — Adam optimizer aur Mean Squared Error loss se compile kiya (yahan loss koi matter nahi karta kyunki hum sirf embedding dekhna chahte hain, training nahi kar rahe).

---

## Cell 10 — Model Summary

```python
model.summary()
```

**Output:**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 8, 10)             1000000   
=================================================================
Total params: 1000000 (3.81 MB)
Trainable params: 1000000 (3.81 MB)
Non-trainable params: 0 (0.00 Byte)
```

**Explanation:**
- **Output Shape: `(None, 8, 10)`**:
  - `None` = batch size (kitne bhi sentences ek saath de sakte hain)
  - `8` = sentence length (padded)
  - `10` = embedding dimension (har word ka vector size)
- **Parameters: `1,000,000`**:
  - `voc_size × dim = 100000 × 10 = 1,000,000`
  - Yeh sab **trainable** hain — training ke dauran yeh weights update hote hain taaki similar words ke similar vectors ban jayein.
- **Size: 3.81 MB** — Is weight matrix ka memory footprint.

---

## Cell 11 — Get Embedding Vectors (Prediction)

```python
model.predict(embedded_docs)
```

**Output:**
```
array([[[ 0.02745477,  0.03850808, -0.00771869, ...],   # padding (0)
        [ 0.02745477,  0.03850808, -0.00771869, ...],   # padding (0)
        [ 0.02745477,  0.03850808, -0.00771869, ...],   # padding (0)
        [ 0.02745477,  0.03850808, -0.00771869, ...],   # padding (0)
        [-0.03278833, -0.02300494, -0.02865194, ...],   # 'the' (8310)
        [-0.03393288, -0.04706965,  0.0279108 , ...],   # 'glass' (66459)
        [-0.02182406, -0.03444018,  0.01741042, ...],   # 'of' (38152)
        [ 0.04517697,  0.04107872,  0.02064132, ...]],  # 'milk' (16661)
       ...
```

**Explanation:**
- Humne `model.predict(embedded_docs)` call kiya — isse har word ka **embedding vector** mil gaya.
- Har word ke liye ek **10-dimensional vector** aaya hai (10 float numbers).
- **Padded positions (0)** ke vectors same hain — kyunki `0` index ka embedding vector same hai.
- Actual words ke alag-alag vectors hain.
- **Abhi yeh vectors random hain** (kyunki model train nahi hua hai). Jab kisi actual task (sentiment analysis, classification, etc.) pe train karenge, tab yeh vectors meaningful ban jaayenge — similar words ke vectors close aa jaayenge.

---

## Cell 12 — View First Sentence's Padded Sequence

```python
embedded_docs[0]
```

**Output:**
```
array([    0,     0,     0,     0,  8310, 66459, 38152, 16661])
```

**Explanation:**
- Yeh pehli sentence `'the glass of milk'` ka padded representation hai.
- Pehle 4 positions mein `0` (padding) hai, aur last 4 mein actual word indices hain.

---

## 🧠 Summary — Kya Seekha Is Notebook Se?

| Concept | Kya Hai? |
|---------|----------|
| **One-Hot Encoding** | Har word ko ek unique integer assign karna (hashing-based) |
| **Padding** | Sab sentences ko same length banana by adding zeros |
| **Word Embedding** | Har word integer ko ek dense float vector mein convert karna |
| **Embedding Layer** | Keras layer jo ek lookup table maintain karta hai (word index → vector) |
| **Trainable Vectors** | Embedding vectors training ke dauran meaningful ban jaate hain |

### Flow Diagram:
```
Raw Text → One-Hot Integers → Padding → Embedding Layer → Dense Vectors
"the glass of milk" → [8310, 66459, 38152, 16661] → [0,0,0,0,8310,...] → [[0.027,...], ...]
```

> 💡 **Key Takeaway:** Yeh notebook word embedding ka **foundation** hai. Real projects (jaise sentiment analysis with RNN/LSTM) mein isi Embedding layer ko model ke first layer ke roop mein use karte hain, aur training ke dauran yeh vectors automatically learn ho jaate hain!

---

---

# 🔄 `simplernn.ipynb` — IMDB Sentiment Analysis with Simple RNN (Hinglish)

## 🤔 RNN kya hai aur kyun chahiye?

ANN mein har input **independent** hota hai. Lekin **text/sentences** mein sequence matter karta hai — "not good" aur "good" mein ek word ka order poora meaning badal deta hai! **RNN (Recurrent Neural Network)** **sequential data** ke liye design hua hai — yeh pichle words ko "yaad" rakh ke current word process karta hai.

**Is notebook mein:** IMDB movie reviews se predict karenge ki review **positive** hai ya **negative** — yeh ek **Sentiment Analysis** problem hai.

---

## 📦 Cell 1 — Libraries Import karna

```python
import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
```

### 🔍 Explanation:

- **`numpy`** — Numerical arrays ke liye.
- **`imdb`** — Keras ka built-in **IMDB movie reviews dataset**. 50,000 reviews hain — already numbers mein converted.
- **`sequence`** — `pad_sequences` function ke liye — review lengths ko uniform banane ke liye.
- **`Embedding`** — 🔑 Har word ke integer ID ko ek **dense vector** mein convert karti hai. Jaise `"good"` → `[0.2, 0.8, -0.1, ...]`. Model training ke dauran seekhti hai ki kaunse words similar hain.
- **`SimpleRNN`** — Basic **Recurrent Neural Network** layer. Har word ko process karti hai aur **pichle words ki memory** rakhti hai.
- **`Dense`** — Fully connected output layer.

---

## 📊 Cell 2 — IMDB Dataset Load karna

```python
max_features = 10000  # vocabulary size

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
print(f'Testing Data shape: {y_test.shape}, Testing labels shape: {y_test.shape}')
```

### 🔍 Explanation:

- **`max_features = 10000`** — Sirf **top 10,000 most frequent words** use karenge. Rare words ko ignore karenge.
- **`imdb.load_data(num_words=max_features)`** — Dataset load hota hai — pehle se split hai:
  - **25,000 training reviews** + labels
  - **25,000 testing reviews** + labels
- **Labels:** `1` = Positive review, `0` = Negative review
- Dataset **pre-processed** hai — reviews text nahi, **integer sequences** hain.

### 🎯 Train-Test Split kya hai?

```
Total Data: 50,000 reviews
        ↓
   ┌──────────────┬───────────────┐
   │  Training    │   Testing     │
   │  25,000      │   25,000      │
   │  Model isse  │  Model ne yeh │
   │  SEEKHEGA    │  KABHI NAHI    │
   │  (padhai)    │  DEKHA (exam)  │
   └──────────────┴───────────────┘
```

- **Training Data** → Model isse patterns seekhega
- **Testing Data** → Unseen data pe check karenge ki model ne actually seekha ya nahi

---

## 👀 Cell 3 — Ek Sample Review dekhna

```python
X_train[0], y_train[0]
```

### 🔍 Explanation:

- Pehla review dikhata hai — integers ki list `[1, 14, 22, 16, 43, 530, ...]`
- Label = `1` (Positive review)
- Har number ek word ko represent karta hai

---

## 🔍 Cell 4 — Sample Review Inspect karna

```python
sample_review = X_train[0]
sample_label = y_train[0]
print(f'Sample review as integers: {sample_review}')
print(f'Sample label: {sample_label}')
```

### 🔍 Explanation:

Review ko variables mein store karke print kiya — confirm karne ke liye ki ek review integers ki list hai aur label 0 ya 1 hai.

---

## 📖 Cell 5 — Word Index Dictionary dekhna

```python
word_index = imdb.get_word_index()
word_index
```

### 🔍 Explanation:

- **`get_word_index()`** — Dictionary return karta hai: word → integer mapping
  - `'kids': 359`, `'much': 73`, `'the': 1`, etc.
  - Frequent word = chhota number
- Dictionary mein **88,000+ words** hain, lekin hum sirf top 10,000 use kar rahe hain.

---

## 🔄 Cell 6 — Reverse Word Index banana

```python
reverse_word_index = {value: key for key, value in word_index.items()}
reverse_word_index
```

### 🔍 Explanation:

- Original: `{'good': 42, ...}` (word → number)
- **Reverse:** `{42: 'good', ...}` (number → word)
- Integers se wapas **readable text** banane ke liye!

---

## 📝 Cell 7 — Review ko Readable Text mein Convert karna

```python
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in sample_review])
decoded_review
```

### 🔍 Explanation:

- **`i-3`** kyun? IMDB mein pehle 3 indices reserved hain: `0`=padding, `1`=start, `2`=unknown. Actual words index 3 se shuru hote hain.
- **Result:** *"this film was just brilliant casting location scenery story direction..."*
- Label `1` (positive) sahi confirm hua ✅

---

## 📏 Cell 8 — Sequences ko Pad karna

```python
max_len = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

X_train
```

### 🔍 Explanation:

- **Problem:** Har review ki length alag (koi 100, koi 800 words). Neural network ko **fixed size input** chahiye!
- **Solution:** `pad_sequences` — saari reviews ko **500 words** par set kiya:
  - Chhoti review → **shuruaat mein 0s add** (padding)
  - Badi review → **shuruaat ke words kaat diye** (truncation)

---

## 👀 Cell 9 — Test Data Check

```python
X_test
```

Test data bhi 500 length mein padded ho gaya — confirm kiya.

---

## 🏗️ Cell 10 — Simple RNN Model Build karna

```python
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### 🔍 Explanation:

**Layer 1 — Embedding:**
- `Embedding(10000, 128, input_length=500)` — har word ID ko **128-dim vector** mein convert karta hai. Similar words ke vectors close honge. Training mein seekhe jaate hain.

**Layer 2 — SimpleRNN:**
- `SimpleRNN(128, activation='relu')` — **128 units ki memory.** Har word pe:
  1. Current word ka embedding leta hai
  2. Previous hidden state (pichle words ki memory) leta hai
  3. New hidden state generate karta hai
- 500 words baad, final state = poore review ka **"summary"**

**Layer 3 — Output:**
- `Dense(1, activation='sigmoid')` — Positive (1) ya Negative (0)

> 🧠 **RNN ka Flow:**
> ```
> Word₁ → Embedding → RNN(state₁)
> Word₂ → Embedding → RNN(state₂ = f(state₁, word₂))
> ...
> Word₅₀₀ → Embedding → RNN(state₅₀₀) → Dense → Positive/Negative
> ```

---

## 📊 Cell 11 — Model Summary

```python
model.summary()
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 500, 128) | **1,280,000** |
| SimpleRNN | (None, 128) | **32,896** |
| Dense | (None, 1) | **129** |
| **Total** | | **1,313,025** |

**Parameters:** Embedding = `10000 × 128 = 1.28M`, SimpleRNN = `(128+128) × 128 + 128 = 32,896`, Dense = `128 + 1 = 129`

---

## 🔨 Cell 12 — Model Compile

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Adam optimizer + Binary Crossentropy loss (positive/negative binary classification).

---

## 🛑 Cell 13 — Early Stopping Setup

```python
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

- **`patience=5`** — 5 epochs improvement nahi toh training ruko.
- **`restore_best_weights=True`** — Best epoch ke weights restore karo.

---

## 🏋️ Cell 14 — Model Training

```python
history = model.fit(
    X_train, y_train, epochs=10, batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)
```

### 🔍 Training Results:

| Epoch | Loss | Accuracy | Val Accuracy |
|---|---|---|---|
| 1 | 51.6 | 61.58% | 65.66% |
| 2+ | **NaN** ❌ | ~50% | ~50% |

### ⚠️ NaN Loss Problem:

Model **explode** ho gaya! SimpleRNN mein **vanishing/exploding gradient** problem hota hai. Long sequences (500 words) mein gradients bahut bade ho jaate hain. **Solutions:** `tanh` activation, LSTM/GRU use karo, Gradient Clipping lagao.

---

## 🎯 Complete Pipeline Summary

```
📦 Import Libraries
    ↓
📊 Load IMDB Dataset (25K train + 25K test)
    ↓
📖 Word Index + Reverse Index (number ↔ word)
    ↓
📝 Decode Review (integers → text)
    ↓
📏 Pad Sequences (length = 500)
    ↓
🏗️ Build Model (Embedding → SimpleRNN → Dense)
    ↓
🔨 Compile + Early Stopping
    ↓
🏋️ Train (NaN issue ← gradient explosion!)
```

## 🧠 Key Concepts:

| Concept | Explanation |
|---|---|
| **RNN** | Sequential data ke liye — pichle inputs ki "memory" rakhta hai |
| **Embedding** | Words ko dense vectors mein convert karta hai |
| **Padding** | Sequences ko same length banana |
| **Vanishing/Exploding Gradient** | Long sequences mein SimpleRNN ki weakness |
| **LSTM/GRU** | SimpleRNN ke advanced versions — gradient problem solve karte hain |

> 💡 **Pro Tip:** Real projects mein **LSTM** ya **GRU** use karte hain — SimpleRNN sirf learning ke liye hai!
