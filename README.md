# 🐱🐶 Cats vs Dogs Classifier API

FastAPI-based image classification API using EfficientNetB0 to distinguish between cats and dogs.


## ✨ Features

- 🚀 Fast batch processing (memory & disk modes)
- 🔒 API key authentication
- 📊 Confidence scores for predictions
- 📝 Auto-generated API documentation
- 🎯 High accuracy with EfficientNetB0

## 📦 Installation

```bash
git clone https://github.com/yourusername/cats-dogs-classifier.git
cd cats-dogs-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ⚙️ Setup

Create `.env` file (see `.env.example`):

```env
APP_NAME=Cats vs Dogs Classifier
VERSION=1.0.0
API_SECRET_KEY=your-secret-key-here
```

## 🚀 Usage

**Start server:**
```bash
python main.py
```

Server runs at `http://127.0.0.1:8001`

## 📡 API Endpoints

### 🏠 Home
```
GET /
```

### 🔮 Classify (Memory Mode)
```
POST /Classify-batches-memory
```
Fast processing for small batches

### 💾 Classify (Disk Mode)
```
POST /Classify-batches-disk
```
Efficient processing for large batches

## 💡 Example

**cURL:**
```bash
curl -X POST "http://127.0.0.1:8001/Classify-batches-memory" \
  -H "X-API-Key: your-secret-key" \
  -F "files=@cat.jpg" \
  -F "files=@dog.jpg"
```

**Python:**
```python
import requests

url = "http://127.0.0.1:8001/Classify-batches-memory"
headers = {"X-API-Key": "your-secret-key"}
files = [('files', open('cat.jpg', 'rb'))]

response = requests.post(url, headers=headers, files=files)
print(response.json())
```

**Response:**
```json
{
  "predictions": [
    {
      "base_name": "cat.jpg",
      "class_index": 0,
      "class_name": "cat",
      "confidence": 0.98
    }
  ]
}
```

## 📚 Documentation

Interactive API docs: `http://127.0.0.1:8001/docs`

## 🏗️ Project Structure

```
cats-dogs-classifier/
├── Src/
│   ├── config.py       # Config & model loading
│   ├── inference.py    # Classifier logic
│   ├── schemas.py      # Response models
│   └── utils.py        # Helper functions
├── artifacts/
│   ├── model_weights.h5
│   └── idx2label.joblib
├── main.py
├── .env
└── .env.example
```

## 🤖 Model

- **Architecture**: EfficientNetB0
- **Input**: 150x150 RGB images
- **Output**: Binary classification (cat/dog)
- **Activation**: Sigmoid

## 🔧 Troubleshooting

**Port in use:**
```bash
lsof -i :8001  # Find process
kill -9 <PID>  # Kill it
```

**Wrong API key:** Check `.env` matches request header

**Model not loading:** Ensure `artifacts/` contains weights file

## 📄 License

MIT License

---

**Author**: Zeyad Elgabbas  
**GitHub**: [@Zeyadelgabbas](https://github.com/Zeyadelgabbas)