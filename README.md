# stockGinie

A price prediction application using PyTorch LSTM, trained on live stock price data and deployed using Streamlit.

## Features
- Real-time stock price data fetching
- Live model training & prediction
- Interactive visualizations
- Model performance metrics

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Testing
```bash
pytest tests/
```

## Project Structure
```
stock_predictor/
├── src/                # Source code
│   ├── data/           # Data handling
│   ├── logs/           # Logs
│   ├── models/         # ML models
│   ├── utils/          # Utilities
│   └── config.py       # Configuration
├── tests/              # Unit tests
├── app.py              # Streamlit app
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Contributing

Contributions are welcome! Here's how you can help:
1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Future Tasks
- [x] Add progress bars (for model training)
- [ ] Add more data sources
- [ ] Try SOTA ML models
- [ ] Refine visualizations

---

Made with 💙 by Argish