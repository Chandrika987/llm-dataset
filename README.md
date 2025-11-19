
---

# 🌾 LLM Micro Challenge — **Challenge 1: Domain-Specific Dataset Creation**

This repository contains the deliverables for **Challenge 1** of the *LLM Micro Challenge Series*.
The goal of this challenge was to create a high-quality, domain-specific dataset that will be used to train and fine-tune an LLM in the upcoming stages of the series.

---

## 📌 **Chosen Domain: Agriculture**

Among the available domains (Health, Finance, Agriculture, etc.), we selected **Agriculture** because:

* It has rich, practical, real-world applications
* It intersects with food security, sustainability, and smart farming
* It aligns well with IoT/Smart Irrigation interests
* There is abundant publicly available, non-sensitive information

This dataset focuses on agricultural topics such as:

* Crop management
* Soil health
* Pest control
* Irrigation
* Climate impact
* Agricultural practices and best methods

---

## 🧩 **What Was Done in This Challenge**

### ✅ 1. **Data Collection**

Gathered text content from publicly available sources such as:

* Agricultural knowledge bases
* Research-driven articles
* Educational websites
* Government agricultural resources
* Open datasets

A total of **100+ entries** of agricultural information were compiled.

---

### ✅ 2. **Data Cleaning**

To ensure high-quality dataset preparation, the following cleaning steps were done:

* Removed noise, ads, symbols, formatting errors
* Standardized the text for consistency
* Removed duplicates
* Removed unwanted whitespace and line breaks
* Ensured valid UTF-8 characters
* Verified JSON structure and corrected any invalid entries

A custom Python script (`clean_jsonl.py`) was used to clean the dataset programmatically.

---

### ✅ 3. **Metadata Addition**

Each data entry was enriched with essential metadata fields such as:

* `title`
* `domain`
* `source`
* `content`

This improves future indexing, retrieval, and model training quality.

---

### ✅ 4. **JSONL Conversion**

The cleaned dataset was converted into **JSONL** format, which is ideal for:

* LLM fine-tuning
* Embeddings
* Data streaming
* High-volume processing

A Python script (`create_jsonl.py`) was created to transform cleaned text into structured JSONL.

---

## 📂 **Files in This Repository**

| File Name                    | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| **agri_dataset_100.jsonl**   | Original 100-entry agricultural dataset              |
| **agri_dataset_clean.jsonl** | Fully cleaned & processed dataset ready for training |
| **agri_dataset_pretty.json** | Pretty-printed JSON version for human reading        |
| **clean_jsonl.py**           | Script used to clean JSONL entries                   |
| **create_jsonl.py**          | Script used to convert data into JSONL format        |

---

## 🚀 **Outcome**

This challenge successfully produced a **clean, structured, metadata-enriched, domain-specific agricultural dataset** in JSONL format — fully ready for use in the next LLM development challenges.

---


