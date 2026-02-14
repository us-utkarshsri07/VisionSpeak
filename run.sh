echo "Starting preprocessing..."
python -m src.preprocessing

echo "Extracting features..."
python -m src.feature_extraction

echo "Training model..."
python -m src.train

echo "Evaluating model..."
python -m src.evaluate

echo "Done."
