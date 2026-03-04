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
