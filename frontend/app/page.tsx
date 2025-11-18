'use client';

import { useEffect, useState } from 'react';

export default function Home() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('https://nba-predictor-backend.onrender.com/api/predictions')
      .then(res => res.json())
      .then(data => {
        setGames(data.games);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching predictions:', err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="min-h-screen p-8 
      bg-gradient-to-br from-gray-900 via-gray-950 to-black 
      text-white">
      
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <h1 className="text-5xl font-extrabold mb-2 tracking-tight">
          NBA Predictor
        </h1>
        <p className="text-gray-400 mb-10 text-lg">
          64.8% accuracy • Machine Learning powered - Akash Kothari
        </p>

        {/* List */}
        {loading ? (
          <div className="text-center text-gray-500">Loading predictions...</div>
        ) : games.length === 0 ? (
          <div className="text-center text-gray-500">No predictions available.</div>
        ) : (
          <div className="space-y-5">
            {games.map((game: any, i: number) => (
              <GameCard
                key={i}
                homeTeam={game.home_team}
                awayTeam={game.away_team}
                time={game.time}
                homeScore={game.home_score}
                awayScore={game.away_score}
                prediction={game.prediction}
                confidence={game.confidence}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function GameCard({
  homeTeam,
  awayTeam,
  prediction,
  confidence,
  time,
  homeScore,
  awayScore
}: {
  homeTeam: string;
  awayTeam: string;
  prediction: string;
  confidence: number;
  time: string;
  homeScore: number | null;
  awayScore: number | null;
}) {

  const isLive = time.includes("Q") || time.includes("OT");
  const isFinal = time.toLowerCase().includes("final");

  // Cleaner color palette
  const colorHome = "text-blue-300";
  const colorAway = "text-red-300";

  const predColor = prediction === homeTeam ? colorHome : colorAway;

  return (
    <div className="bg-gray-800/40 backdrop-blur-sm 
      rounded-2xl p-6 border border-gray-700/50
      shadow-[0_0_15px_rgba(0,0,0,0.4)]
      hover:shadow-[0_0_25px_rgba(0,0,0,0.6)]
      hover:bg-gray-800/60 transition duration-200">

      <div className="flex items-center justify-between">
        
        {/* LEFT SIDE */}
        <div className="flex flex-col gap-3">

          {/* Team Row */}
          <div className="flex items-center gap-12">
            
            {/* HOME */}
            <div className="flex flex-col items-center">
              <div className={`text-xl font-bold ${colorHome}`}>
                {homeTeam}
              </div>
              {homeScore !== null && (
                <div className="mt-1 text-3xl font-semibold text-blue-200 drop-shadow-sm">
                  {homeScore}
                </div>
              )}
            </div>

            <div className="text-gray-500 text-lg">vs</div>

            {/* AWAY */}
            <div className="flex flex-col items-center">
              <div className={`text-xl font-bold ${colorAway}`}>
                {awayTeam}
              </div>
              {awayScore !== null && (
                <div className="mt-1 text-3xl font-semibold text-red-200 drop-shadow-sm">
                  {awayScore}
                </div>
              )}
            </div>
          </div>

          {/* STATUS */}
          <div className="mt-1 text-sm font-medium tracking-wide">
            {isFinal ? (
              <span className="text-red-400">FINAL</span>
            ) : isLive ? (
              <span className="text-green-400">LIVE • {time}</span>
            ) : (
              <span className="text-gray-400">Starts at {time}</span>
            )}
          </div>
        </div>

        {/* PREDICTION */}
        <div className="text-right flex flex-col items-end">
          <div className={`text-3xl font-extrabold ${predColor}`}>
            {prediction}
          </div>
          <div className="text-sm text-gray-400">{confidence}% confidence</div>
        </div>

      </div>
    </div>
  );
}