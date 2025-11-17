from appLogic import getGamesToday, buildSingleGameFeatures, predictSingleGame
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    try:
        hI, aI, hT, aT, times, hS, aS = getGamesToday()
        featureDF = buildSingleGameFeatures(hI, aI)
        predictions, probabilities = predictSingleGame(featureDF)
    
        games = []
        for i in range(len(hI)):
            if (predictions[i] == 1):
                predictions[i] = hT[i]
            else:
                predictions[i] = aT[i]
            games.append({
                "home_team": hT[i],
                "away_team": aT[i],
                "time": times[i],
                "home_score": hS[i],
                "away_score": aS[i],
                "prediction": predictions[i],
                "confidence": round(float(probabilities[i]*100), 2)
            })
        return jsonify({"games": games})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
