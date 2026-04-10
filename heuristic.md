# Confidence Calibration Heuristic for Reverse Prediction

## 1. Problem Overview

In our system, we use a **black-box model** to perform reverse prediction (masked reconstruction) on tabular data. The model returns:

- A predicted value per cell
- A single confidence score

However, these raw confidence scores are **not directly reliable or comparable** across different columns due to:

### 1.1 High Cardinality
Columns with many possible values (e.g., `merchant_id`) force the model to distribute probability mass across many classes, resulting in **lower confidence scores**, even when predictions are correct.

### 1.2 Lack of Uncertainty Information
The model provides only a **single confidence value**, not the full probability distribution. Therefore:
- We cannot compute entropy
- We cannot measure true uncertainty directly

### 1.3 Context Dependence
Predictions rely heavily on **context rows (historical data)**, but raw confidence does not reflect whether the predicted value is supported by this context.

---

## 2. Key Intuition

### 2.1 Probability Mass Spreading

The model's output probabilities must sum to 1:


p1 + p2 + ... + pK = 1


- When **K is small**, the model can assign high probability to a few values.
- When **K is large**, probability must be spread across many values.

This leads to **lower top-1 confidence** for high-cardinality columns.

---

### 2.2 Why Confidence is Not Comparable

Consider:

| Column | K | Confidence | Interpretation |
|--------|--|------------|----------------|
| Gender | 2 | 0.2 | Weak |
| Merchant ID | 10,000 | 0.2 | Strong |

Why?

Because of the **random baseline**:

baseline = 1 / K


- For K=2 → baseline = 0.5 → 0.2 is poor
- For K=10,000 → baseline = 0.0001 → 0.2 is extremely strong

 **Same confidence, different meaning**

---

### 2.3 Goal

We aim to estimate:

> A confidence score that reflects what the model would output if difficulty and uncertainty were properly accounted for.

---

## 3. Heuristic Definition

Let:

- `confidence` = model output
- `K` = column cardinality
- `K_ref` = reference cardinality for scaling
- `support` = frequency of predicted value in context rows (smoothed if 0)

### Final Heuristic

uncertainty = 1 - confidence × (log(K) / log(K_ref)) × (1 - support)
improved_confidence = 1 - uncertainty


### Simplified Form

improved_confidence =
confidence
× (log(K) / log(K_ref))
× (1 - support)


---

## 4. Heuristic Breakdown

### 4.1 Confidence

- Represents the model’s belief
- Serves as the **base signal**
- Not calibrated across columns

---

### 4.2 Difficulty Factor: `log(K) / log(K_ref)`

#### Why K?
- Larger K → more possible outcomes → harder problem

#### Why log(K)?
- Difficulty grows **logarithmically**, not linearly
- Based on information theory:
  - Identifying 1 out of K requires ~log(K) information

#### Why normalize?
- Prevents large values from dominating
- Provides **relative scaling**

 Interpretation:

difficulty_factor ≈ relative difficulty of the column


---

### 4.3 Context Support: `(1 - support)`

#### Definition

support = frequency of predicted value in context rows


#### Purpose
- Measures **agreement with historical data**
- Acts as a **contextual reliability signal**

#### Behavior
- High support → prediction is more plausible
- Low support → prediction is less reliable

#### Smoothing
If support = 0:
- assign a small value (e.g., 0.01)
- prevents over-penalizing rare but valid cases

---

## 5. Conceptual Interpretation

The heuristic can be interpreted as:

improved_confidence =
confidence × difficulty × context_adjustment


---

### Mapping to ML Concepts

| Concept | Component |
|--------|----------|
| Model belief | confidence |
| Problem difficulty | log(K) |
| Evidence | support |

---

### Bayesian Interpretation

Posterior ∝ Likelihood × Prior × Evidence


- Likelihood → model confidence
- Prior → difficulty (K)
- Evidence → context support

---

## 6. Why Logarithmic Scaling?

Difficulty does not grow linearly with K.

Example:

| K | Difficulty (log scale) |
|---|------------------------|
| 2 | Low |
| 100 | Moderate |
| 10,000 | High |

Doubling K does not double difficulty  
Log scale reflects **true growth of uncertainty**

---

## 7. Practical Behavior

### 7.1 Low Cardinality Columns

- Small K → low difficulty factor
- Confidence remains mostly unchanged

---

### 7.2 High Cardinality Columns

- Large K → higher difficulty factor
- Confidence is boosted to compensate for probability dilution

---

### 7.3 High Support

- Prediction appears frequently in context
- Confidence remains high

---

### 7.4 Low Support

- Prediction rarely appears in context
- Confidence is penalized

---

### 7.5 Zero Support

- Treated with smoothing
- Avoids complete collapse of confidence

---

## 8. Advantages

-  Works with black-box models (no training required)
-  Handles high-cardinality features effectively
-  Incorporates contextual evidence
-  Improves comparability across columns
-  Simple and deterministic
-  Easy to implement and maintain

---

## 9. Limitations

-  Not a true probabilistic calibration
-  Cannot model full uncertainty (no distribution available)
-  Depends on quality of context data
-  Heuristic approximation (not learned)

---

## 10. Design Philosophy

Given constraints:

- No access to model internals
- No probability distribution
- No ability to retrain

We approximate uncertainty using:

- model confidence
- column cardinality
- contextual support

---

## 11. Summary

Raw confidence from the model is **biased by problem difficulty and lack of context awareness**.

This heuristic:

- compensates for **probability dilution** in high-cardinality columns
- incorporates **contextual evidence**
- produces a more **interpretable and comparable confidence score**

---

### Final Takeaway

> High-cardinality forces the model to spread probability across many options, making raw confidence artificially low.  
> This heuristic corrects for that by scaling confidence according to problem difficulty and contextual support.
