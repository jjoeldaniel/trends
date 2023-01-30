# discord-keywords

[![.github/workflows/main.yml](https://github.com/jjoeldaniel/discord-keywords/actions/workflows/main.yml/badge.svg)](https://github.com/jjoeldaniel/discord-keywords/actions/workflows/main.yml)

> Analyze Discord message keywords

## **Running**

### **Create a virtual environment**

#### *Linux/Mac*

```bash
python3 -m venv venv
source venv/bin/activate
```

#### *Windows*

> Note: Dependency issues may occur when using Windows. If you are using Windows, it is recommended to use WSL.

```bash
python -m venv venv
venv\Scripts\activate.bat
```

### **Install dependencies**

```bash
pip install -r requirements.txt

# Highly recommend but not required to install the following
# for increased performance
pip install python-Levenshtein   
```

### **Run**

```bash
python3 trends.py
```
