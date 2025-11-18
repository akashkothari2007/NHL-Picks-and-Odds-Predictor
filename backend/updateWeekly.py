from data.scrapeRawData import updateAll
from models.RandomForestClassifier import trainModel
if __name__ == "__main__":
    updateAll(current_season=2025)
    trainModel()