# Дипломна робота
## 🇺🇦 Українська
###  Назва дипломної роботи
**Розробка інтелектуальної системи асистування водія для спостереження концентрації та безпеки під час керування транспортним засобом**
###  Структура проєкту
- `main.py` – Основний скрипт, що запускає вебкамеру та систему моніторингу втоми.
- `processor.py` – Основна логіка: сегментація обличчя, класифікація стану очей/рота, аналіз втоми.
- `face_parsing/` – Git-сабмодуль з нейромережею BiSeNet для сегментації обличчя.
- `fatigue_model_training/` – Папка з моделлю (`yawn_eye_model.pkl`) для класифікації втоми.
- `prototypes/` – Прототипи та дослідницький код:
  - `live_detection/` – Простий прототип виявлення втоми.
  - `live_detection_with_image_segmentation/` – Покращений варіант з нейромережевою сегментацією.
- `requirements.txt` – Список необхідних бібліотек.
###  Клонування репозиторію разом із сабмодулями
```bash
git clone --recurse-submodules <АДРЕСА_РЕПОЗИТОРІЮ>
```
Вже клонували без сабмодулів?
```bash
git submodule update --init --recursive
```
###  Створення та активація віртуального середовища (необов'язково)
```bash
python -m venv fatigue_detection_env
source fatigue_detection_env/bin/activate   # Для Linux/Mac
fatigue_detection_env\Scripts\activate.bat  # Для Windows
```
###  Встановлення залежностей
```bash
pip install -r requirements.txt
```
###  Додавання ваг моделей
- Ваги сегментації обличчя (`resnet34.pt`) ➜ `prototypes/live_detection_with_image_segmentation/weights/`
- Модель класифікації (`yawn_eye_model.pkl`) ➜ `fatigue_model_training/`
###  Запуск програми
```bash
python main.py --type simple
```
###  Керування
| Кнопка | Дія                                 |
|--------|--------------------------------------|
| `0`    | Показати звичайний відеопотік       |
| `1`    | Накласти маску сегментації обличчя  |
| `2`    | Показати області та статус втоми    |
| `q`    | Вийти з програми                    |
# Thesis
## 🇬🇧 English
### Thesis Title
**Development of an Intelligent Driver Assistance System to Monitor Concentration and Safety While Driving**
### Project Structure
- `main.py` – Main script for running real-time driver monitoring via webcam.
- `processor.py` – Core logic: facial region segmentation, eye/mouth state classification, fatigue analysis.
- `face_parsing/` – Git submodule containing BiSeNet for facial segmentation.
- `fatigue_model_training/` – Contains training logic and the final model (`yawn_eye_model.pkl`) for classification of eye and mouth states.
- `prototypes/` – Experimental code:
  - `live_detection/` – Simple prototype for detecting fatigue.
  - `live_detection_with_image_segmentation/` – Advanced prototype using semantic segmentation with helper scripts and notebooks.
- `requirements.txt` – Required Python packages.
### Clone the Repository with Submodules
```bash
git clone --recurse-submodules <REPO_URL>
```
Already cloned without submodules?
```bash
git submodule update --init --recursive
```
### Create and Activate a Virtual Environment (Optional)
```bash
python -m venv fatigue_detection_env
source fatigue_detection_env/bin/activate   # Linux/Mac
fatigue_detection_env\Scripts\activate.bat  # Windows
```
###  Install Dependencies
```bash
pip install -r requirements.txt
```
###  Place the Model Weights
- Face segmentation weights (`resnet34.pt`) ➜ `prototypes/live_detection_with_image_segmentation/weights/`
- Classifier model (`yawn_eye_model.pkl`) ➜ `fatigue_model_training/`
###  Run the Application
```bash
python main.py --type simple
```
###  Controls
| Key | Action                      |
|-----|-----------------------------|
| `0` | Raw camera feed             |
| `1` | Show face segmentation mask |
| `2` | Show ROIs and fatigue state |
| `q` | Quit                        |
