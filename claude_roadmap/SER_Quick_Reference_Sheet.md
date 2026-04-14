# Speech Emotion Recognition - Quick Reference Sheet

## DATASETS

### English Datasets

| Dataset | Size | Speakers | Emotions | Download | Typical Accuracy | Best For |
|---------|------|----------|----------|----------|-----------------|----------|
| **RAVDESS** | 1440 files | 24 actors (12M, 12F) | 7 emotions | https://zenodo.org/record/1188976 | 92-98% | Baseline, published benchmarks |
| **TESS** | 2800 files | 2 female (26, 64 yrs) | 7 emotions | https://tspace.library.utoronto.ca/handle/1807/24602 | 99-100% | Easy baseline (TOO easy) |
| **CREMA-D** | 7442 files | 91 actors (diverse) | 6 emotions | https://github.com/CheyneyComputerScience/CREMA-D | 85-95% | Robustness, diversity |
| **EmoDB** | 535 files | 10 German speakers | 7 emotions | https://www.audeering.com/research/emodb/ | 95-100% | Cross-linguistic testing |
| **SAVEE** | 480 files | 4 male British actors | 7 emotions | https://personal.surrey.ac.uk/p.jackson/SAVEE/Download.html | 90-98% | Gender variation |

**Recommended Combination:** RAVDESS + TESS + CREMA-D (11,242 files total)

### Hindi Datasets

| Dataset | Size | Speakers | Format | Source | Status | Use Case |
|---------|------|----------|--------|--------|--------|----------|
| **IITKGP-SEHSC** | ~1000 | 10 speakers | Simulated | https://ieeexplore.ieee.org/document/5738540/ | Public (request authors) | Hindi baseline |
| **EmoInHindi** | 44,247 utterances | Multiple | Conversational text | https://github.com/SinghGD/EmoInHindi | Public (GitHub) | Conversational Hindi |
| **Kaggle Hindi SER** | Variable | Variable | Mixed | https://www.kaggle.com/datasets/vishlb/speech-emotion-recognition-hindi | Community | Supplementary |

**Recommendation:** Start with IITKGP-SEHSC subset + create your own custom Hindi dataset

---

## KEY PAPERS (2024-2025)

### Must-Read Papers

1. **Barhoumi & BenAyed (2024)** 
   - Title: "Real-time speech emotion recognition using deep learning and data augmentation"
   - Journal: Artificial Intelligence Review (Springer)
   - **Results:** 100% TESS, 100% EmoDB, 98%+ RAVDESS
   - **Architecture:** CNN + BiLSTM
   - **Key Insight:** Data augmentation (noise, spectrogram shifting) critical for performance
   - **URL:** https://link.springer.com/article/10.1007/s10462-024-11065-x
   - **Citation:** Cite if using CNN+BiLSTM approach

2. **Chowdhury et al. (2025)**
   - Title: "Speech emotion recognition with light weight deep neural ensemble"
   - Journal: Scientific Reports (Nature)
   - **Results:** 100% TESS, 97.83% RAVDESS, 95.1% CREMA-D, 93.76% combined 5 datasets
   - **Architecture:** Ensemble CNN + BiLSTM with hand-crafted features
   - **Key Insight:** Hand-crafted features (MFCC, ZCR, Chroma, RMSE) outperform spectrogram-only
   - **URL:** https://www.nature.com/articles/s41598-025-95734-z
   - **Citation:** Excellent for ensemble & feature engineering sections

3. **Tiwari et al. (2025)**
   - Title: "Emotion Detection from Speech Using CNN-BiLSTM with Feature Rich Audio Inputs"
   - Journal: ICCK Transactions on Machine Intelligence
   - **Results:** 94% RAVDESS with real-time deployment
   - **Key Features:** MFCC, Chroma, Mel-spectrograms combined
   - **Bonus:** Includes web-based inference interface code
   - **URL:** https://www.icck.org/article/abs/TMI.2025.306750
   - **Citation:** Good for multi-feature extraction

4. **Kim & Kwak (2024)**
   - Title: "Speech Emotion Recognition Using Deep Learning Transfer Models and Explainable Techniques"
   - Journal: Applied Sciences (MDPI)
   - **Results:** 91-95% on multiple datasets
   - **Key Features:** Grad-CAM visualization + LIME explanations
   - **Novelty:** Shows which spectrogram parts matter for emotions
   - **URL:** https://www.mdpi.com/2076-3417/14/4/1553
   - **Citation:** ESSENTIAL if adding explainability

5. **Koolagudi et al. (2011)**
   - Title: "IITKGP-SEHSC: Hindi Speech Corpus for Emotion Analysis"
   - Journal: IEEE Conference (ICDECOM)
   - **Importance:** Foundational work for Hindi SER
   - **Dataset:** IITKGP-SEHSC (main Hindi dataset)
   - **URL:** https://ieeexplore.ieee.org/document/5738540/
   - **Citation:** MUST cite if using Hindi dataset

6. **Singh et al. (2022)**
   - Title: "EmoInHindi: A Multi-label Emotion and Intensity Dataset in Hindi for Emotion Recognition in Dialogues"
   - Journal: LREC (Language Resources and Evaluation)
   - **Dataset:** 44,247 Hindi utterances with multi-label emotions
   - **URL:** https://aclanthology.org/2022.lrec-1.627.pdf
   - **Citation:** For conversational Hindi emotions

### Cross-Lingual & Robustness Papers

7. **Poria et al. (2023)** - Noisy SER
   - Topic: Speech emotion recognition in noise
   - Key for real-world robustness

8. **Recent Transformer Papers** - Vision Transformer on Spectrograms
   - Better generalization across datasets
   - Zero-shot transfer capabilities

---

## GITHUB REPOSITORIES

### Complete Implementations (Well-Documented)

1. **Speech-Emotion-Recognition - Kanike Sai Prakash**
   - **URL:** https://github.com/KanikeSaiPrakash/Speech-Emotion-Recognition
   - **Datasets:** RAVDESS, TESS, EmoDB
   - **Methods:** CNN models with multiple architectures
   - **Quality:** Well-organized, beginner-friendly
   - **Stars:** 100+ (reliable)
   - **Use Case:** Learn feature extraction and baseline models

2. **Emotion-Recognition-in-Hindi-Speech - Anku**
   - **URL:** https://github.com/ankuPRK/Emotion-Recognition-in-Hindi-Speech
   - **Dataset:** IITKGP-SEHSC (Hindi)
   - **Methods:** SVM, classical ML + neural networks
   - **Results:** 89% male, 83% female dataset
   - **Quality:** Specific to Hindi, good reference
   - **Use Case:** Hindi emotion recognition baseline

3. **CNN-BiLSTM Real-Time Implementation**
   - Multiple repos available
   - Look for keyword: "CNN-BiLSTM" on GitHub
   - Search filter: Language=Python, Stars > 50

### Specialized Repos

4. **Data Augmentation for Speech**
   - Look for: "audio-augmentation-python"
   - Libraries: audiomentations, librosa

5. **Transformer-Based SER**
   - Wav2Vec 2.0 implementations
   - Vision Transformer on spectrograms
   - Search: "ViT spectrogram"

6. **Deployment Examples**
   - Flask + TensorFlow Lite
   - Real-time emotion detection web apps
   - Mobile app integrations

---

## ARCHITECTURE RECOMMENDATIONS

### For High Accuracy (95%+)

**Best Architecture:**
```
Input: Mel-Spectrogram (128, time_steps)
       ↓
Conv2D Layers (extract spatial patterns)
       ↓
BiLSTM (capture temporal dependencies)
       ↓
Attention Mechanism (optional but recommended)
       ↓
Dense Layers + Softmax
       ↓
Output: Emotion probabilities
```

**Code Template Available In:**
- Barhoumi & BenAyed (2024) - most cited
- Tiwari et al. (2025) - simplest implementation
- Search GitHub: "CNN LSTM emotion"

### For Explainability (95% + Interpretability)

**Add:**
- Grad-CAM for visualization
- LIME for local explanations
- Attention weight visualization

**Reference:** Kim & Kwak (2024)

### For Ensemble Approach (98% accuracy possible)

**Stack:**
- CNN model
- LSTM model
- GRU model
- Weighted voting/averaging

**Reference:** Chowdhury et al. (2025)

---

## FEATURE EXTRACTION SUMMARY

### Hand-Crafted Features (Classical ML)

```
MFCC (Mel-Frequency Cepstral Coefficients) - 13 coefficients
ZCR (Zero Crossing Rate) - 1 value
RMSE (Root Mean Square Energy) - 1 value
Chroma (Tonal Content) - 12 coefficients
Spectral Centroid - 1 value
Spectral Rolloff - 1 value
Spectral Contrast - 7 values

Total: ~50-80 features per audio
```

**Tool:** `librosa` library
```python
import librosa
y, sr = librosa.load('audio.wav')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```

### Deep Learning Input: Mel-Spectrogram

```
Mel-Spectrogram: (128, time_steps)
- 128: Mel-frequency bins
- time_steps: ~100-200 frames per 3-second audio

Advantages: Aligned with human hearing, suitable for CNN
```

---

## QUICK ACCURACY BENCHMARKS

| Method | RAVDESS | TESS | CREMA-D | EmoDB | Effort |
|--------|---------|------|---------|-------|--------|
| SVM (baseline) | 75% | 85% | 72% | 80% | Low |
| Random Forest | 78% | 87% | 75% | 82% | Low |
| CNN only | 85% | 94% | 80% | 90% | Medium |
| CNN+LSTM | 92-95% | 98-100% | 87-92% | 95-98% | Medium-High |
| Ensemble CNN+LSTM+GRU | 95-97% | 99-100% | 92-95% | 98-100% | High |
| Transformer/ViT | 93-96% | 97-99% | 90-94% | 96-99% | Very High |

---

## PROJECT MILESTONES & TIMELINE

| Week | Task | Expected Output | Accuracy |
|------|------|-----------------|----------|
| 1-2 | Classical ML baseline | SVM/RF models trained | 75-85% |
| 3-4 | CNN+BiLSTM model | State-of-art deep learning model | 92-97% |
| 5-6 | Advanced technique | Ensemble/Transformer/Transfer | 94-98% |
| 6-7 | Cross-dataset testing | Evaluate on unseen datasets | 85-92% |
| 7-8 | Hindi extension | Apply to Hindi data | 80-90% |
| 8-10 | Custom test dataset | Record & test emotions | 80-95% |

---

## CRITICAL SUCCESS FACTORS

✅ **Use Multi-Dataset Training** (not single dataset)
✅ **Implement Data Augmentation** (+3-5% accuracy boost)
✅ **Test on Unseen Datasets** (proves generalization)
✅ **Custom Test Set** (project requirement)
✅ **Report All Metrics** (accuracy, F1, precision, recall)
✅ **Visualize Confusion Matrices** (show per-emotion performance)
✅ **Include Error Analysis** (why does it fail?)
✅ **Add Explainability** (Grad-CAM, attention)
✅ **Compare with Published Results** (show competitive performance)
✅ **Document Everything** (code, methodology, results)

---

## QUICK START CHECKLIST

- [ ] Download RAVDESS dataset (Week 1)
- [ ] Extract MFCC features (Week 1)
- [ ] Train SVM baseline (Week 1-2)
- [ ] Build CNN+BiLSTM model (Week 3-4)
- [ ] Test on TESS/CREMA-D (Week 6)
- [ ] Download IITKGP-SEHSC Hindi data (Week 7)
- [ ] Record custom test set (Week 8)
- [ ] Create confusion matrices (Week 9)
- [ ] Write final report (Week 10)
- [ ] Push to GitHub (Week 10)

---

## PYTHON LIBRARIES CHEAT SHEET

```python
# Audio Processing
librosa              - Feature extraction & audio loading
soundfile            - Audio I/O
scipy.signal         - Signal processing

# ML Models
scikit-learn         - Classical ML (SVM, RF, etc.)
tensorflow.keras     - Deep learning (recommended)
pytorch              - Alternative DL framework

# Evaluation
sklearn.metrics      - Classification metrics
numpy, pandas        - Data manipulation
matplotlib, seaborn  - Visualization

# Augmentation
audiomentations      - Audio data augmentation
librosa.effects      - pitch_shift, time_stretch

# Quick Install:
# pip install librosa tensorflow scikit-learn numpy pandas matplotlib seaborn audiomentations
```

---

## HOW TO CITE PAPERS IN YOUR REPORT

### For CNN+BiLSTM Approach:
```
"Following Barhoumi & BenAyed (2024), we implemented a CNN+BiLSTM 
hybrid architecture that achieves state-of-the-art performance across 
multiple datasets..."
```

### For Hand-Crafted Features:
```
"Chowdhury et al. (2025) demonstrated that hand-crafted features 
(MFCC, ZCR, Chroma, RMSE) outperform spectrogram-only approaches 
in ensemble models, achieving 97.83% on RAVDESS..."
```

### For Hindi Emotion Recognition:
```
"Building on Koolagudi et al. (2011) who introduced the IITKGP-SEHSC 
Hindi emotion corpus, we extended speech emotion recognition to 
bilingual (English-Hindi) scenarios..."
```

### For Explainability:
```
"Following Kim & Kwak (2024), we incorporated Grad-CAM visualizations 
and LIME explanations to make our emotion recognition model interpretable..."
```

---

## FAILURE POINTS TO AVOID

❌ **Only using TESS dataset** (too easy, 100% accuracy = not impressive)
❌ **Not doing cross-dataset evaluation** (models overfit to one dataset)
❌ **Reporting only accuracy** (use F1, precision, recall, confusion matrix)
❌ **No data augmentation** (deep learning needs it)
❌ **Training on entire audio file** (emotions are in segments)
❌ **No validation set** (risk of overfitting)
❌ **Not recording custom test set** (project explicitly requires this)
❌ **Copying code without understanding** (show your knowledge!)
❌ **Poor documentation** (nobody can reproduce your results)

---

## ADVANCED ADDITIONS FOR EXTRA CREDIT

✨ **Real-time web deployment** (Flask/FastAPI app)
✨ **Mobile app integration** (TensorFlow Lite)
✨ **Noise-robust models** (test on noisy audio)
✨ **Cross-lingual transfer** (English→Hindi transfer learning)
✨ **Multi-speaker generalization** (test on different speakers)
✨ **Emotion intensity estimation** (not just which emotion, but how intense)
✨ **Multimodal fusion** (if adding video/facial expressions)
✨ **Publication submission** (arXiv, conferences)

---

## CONTACT & DISCUSSION

When stuck, check:
1. **GitHub Issues** in referenced repos
2. **Stack Overflow** (tag: speech-recognition, emotion)
3. **Papers' supplementary materials**
4. **Author GitHub profiles** (email authors if needed)
5. **Academic communities** (Reddit r/MachineLearning, r/SER)

---

**Last Updated:** April 2026
**Recommended for:** ML course project, research work, portfolio
**Estimated Project Duration:** 8-12 weeks
**Difficulty:** Intermediate to Advanced
