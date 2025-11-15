'use client';

import { useEffect, useState } from 'react';

export default function Home() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:5000/api/predictions')
    .then(res=> res.json())
    .then(data => {
      setGames(data.games);
      setLoading(false);
    })
    .catch(err => {
      console.error('Error fetching predictions:', err);
      setLoading(false);
    })
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <h1 className="text-4xl font-bold mb-2">NBA Game Predictor</h1>
        <p className="text-gray-400 mb-8">64.8% accuracy • Powered by ML</p>

        {/* Games list placeholder */}
        {loading ? (
          <div className="text-center text-gray-500">Loading predictions...</div>
        ) : games.length === 0 ? (
          <div className="text-center text-gray-500">No predictions available.</div>
        ) : (
        <div className="space-y-4">
          {games.map((game: any, i: number) => (
            <GameCard
              key = {i}
              homeTeam = {game.home_team}
              awayTeam = {game.away_team}
              prediction = {game.prediction}
              confidence = {game.confidence}
            />
          ))}
        </div>
        )}
      </div>
    </div>
  );
}

// Game card component
function GameCard({ 
  homeTeam, 
  awayTeam, 
  prediction, 
  confidence 
}: {
  homeTeam: string;
  awayTeam: string;
  prediction: string;
  confidence: number;
}) {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex items-center justify-between">
        {/* Teams */}
        <div className="flex items-center gap-8">
          <div className="text-xl text-blue-500 font-semibold">{homeTeam}</div>
          <div className="text-gray-500">vs</div>
          <div className="text-xl text-red-500 font-semibold">{awayTeam}</div>
        </div>

        {/* Prediction */}
        <div className="text-right">
          <div className="text-2xl font-bold text-blue-400">{prediction}</div>
          <div className="text-sm text-gray-400">{confidence}% confidence</div>
        </div>
      </div>
    </div>
  );
}