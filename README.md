# COVID-19 Chest X-Ray Classification

Deep learning classifier for COVID-19 detection from chest X-rays using transfer learning.

## Dataset
- **Source:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Classes:** COVID-19, Lung Opacity, Normal, Viral Pneumonia
- **Total Images:** 21,165 X-rays
- **Split:** 70% train, 15% val, 15% test (stratified)

## Model
- **Architecture:** ResNet18 (pretrained on ImageNet)
- **Training:** 20 epochs with class-weighted loss
- **Optimizer:** Adam (lr=0.001)

## Results
- **Test Accuracy:** 92.09%
- **Per-Class F1-Scores:**
  - COVID-19: 97%
  - Viral Pneumonia: 97%
  - Normal: 92%
  - Lung Opacity: 88%

## Requirements
```bash
pip install -r requirements.txt
```

## Training
```bash
jupyter notebook covid-xray-classifier.ipynb
```
## Dataset Citation

This project uses the COVID-19 CHEST X-RAY DATABASE:
- M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
- Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.
