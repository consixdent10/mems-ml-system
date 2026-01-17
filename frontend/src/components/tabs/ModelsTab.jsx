import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
    ScatterChart, Scatter, ReferenceLine
} from 'recharts';

/**
 * Models Tab Component
 * Displays ML model training interface, leaderboard, and performance charts
 */
const ModelsTab = ({
    modelResults,
    bestModel,
    predictionsSample,
    isTraining,
    trainModels,
    getFeatureImportance
}) => {
    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h3 className="text-xl font-semibold">ML Model Training & Comparison</h3>
                <button
                    onClick={trainModels}
                    disabled={isTraining}
                    className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-6 py-2 rounded-lg font-medium transition"
                >
                    {isTraining ? 'Training...' : 'Train All Models'}
                </button>
            </div>

            {modelResults.length > 0 && (
                <>
                    {/* Leaderboard Table */}
                    <div className="bg-slate-700 rounded-lg p-4 mb-6">
                        <h4 className="text-lg font-semibold mb-3 text-blue-400">📊 Model Leaderboard (RUL Regression)</h4>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-slate-600">
                                        <th className="text-left py-2 px-3 text-gray-400">Model</th>
                                        <th className="text-right py-2 px-3 text-gray-400">MAE ↓</th>
                                        <th className="text-right py-2 px-3 text-gray-400">RMSE ↓</th>
                                        <th className="text-right py-2 px-3 text-gray-400">R² ↑</th>
                                        <th className="text-right py-2 px-3 text-gray-400">Time</th>
                                        <th className="text-center py-2 px-3 text-gray-400">Best</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {modelResults.map((model, idx) => (
                                        <tr key={idx} className={`border-b border-slate-600 ${model.modelType === bestModel ? 'bg-green-900/30' : ''}`}>
                                            <td className="py-2 px-3 font-medium">{model.modelType}</td>
                                            <td className="py-2 px-3 text-right">{model.mae?.toFixed(2) || 'N/A'}</td>
                                            <td className="py-2 px-3 text-right">{model.rmse?.toFixed(2) || 'N/A'}</td>
                                            <td className="py-2 px-3 text-right">{model.r2Score?.toFixed(4) || 'N/A'}</td>
                                            <td className="py-2 px-3 text-right">{model.trainingTime?.toFixed(2)}s</td>
                                            <td className="py-2 px-3 text-center">
                                                {model.modelType === bestModel && <span className="text-green-400">✅</span>}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {bestModel && (
                            <p className="mt-3 text-sm text-green-400">
                                🏆 Best Model: <strong>{bestModel}</strong> (Lowest RMSE)
                            </p>
                        )}
                    </div>

                    {/* Model Cards - Regression Metrics Only */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {modelResults.map((model, idx) => (
                            <div key={idx} className={`bg-slate-700 rounded-lg p-4 border-l-4 ${model.modelType === bestModel ? 'border-green-500' : 'border-blue-500'}`}>
                                <div className="flex justify-between items-start mb-3">
                                    <h4 className="text-lg font-semibold text-blue-400">{model.modelType}</h4>
                                    {model.modelType === bestModel && (
                                        <span className="bg-green-600 text-xs px-2 py-1 rounded-full">Best</span>
                                    )}
                                </div>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">MAE:</span>
                                        <span className="font-medium text-yellow-400">{model.mae?.toFixed(2) || 'N/A'}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">RMSE:</span>
                                        <span className="font-medium text-orange-400">{model.rmse?.toFixed(2) || 'N/A'}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">R² Score:</span>
                                        <span className={`font-medium ${model.r2Score > 0.7 ? 'text-green-400' : model.r2Score > 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
                                            {model.r2Score?.toFixed(4) || 'N/A'}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Training Time:</span>
                                        <span className="font-medium">{model.trainingTime?.toFixed(2)}s</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* MAE vs RMSE Comparison */}
                        <div>
                            <h3 className="text-xl font-semibold mb-4">MAE vs RMSE Comparison</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={modelResults}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis dataKey="modelType" stroke="#9CA3AF" angle={-15} textAnchor="end" height={80} />
                                    <YAxis stroke="#9CA3AF" />
                                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                    <Legend />
                                    <Bar dataKey="mae" fill="#F59E0B" name="MAE" />
                                    <Bar dataKey="rmse" fill="#EF4444" name="RMSE" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-4">Feature Importance</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={getFeatureImportance()} layout="vertical">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis type="number" stroke="#9CA3AF" />
                                    <YAxis dataKey="feature" type="category" stroke="#9CA3AF" />
                                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                    <Bar dataKey="importance" fill="#8B5CF6" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Actual vs Predicted Scatter Plot */}
                    {predictionsSample && predictionsSample.actual && (() => {
                        const scatterData = predictionsSample.actual.map((a, i) => ({ actual: a, predicted: predictionsSample.predicted[i] }));
                        const bestModelData = modelResults.find(m => m.modelType === bestModel);
                        const bestR2 = bestModelData?.r2Score || 0;
                        return (
                            <div className="mt-6">
                                <h3 className="text-xl font-semibold mb-2">Actual vs Predicted RUL (Best Model: {bestModel})</h3>
                                <p className="text-sm text-gray-400 mb-4">Model R²: <span className={bestR2 >= 0.7 ? 'text-green-400' : 'text-yellow-400'}>{bestR2.toFixed(4)}</span></p>
                                <ResponsiveContainer width="100%" height={350}>
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis type="number" dataKey="actual" name="Actual RUL%" stroke="#9CA3AF" domain={[0, 100]} label={{ value: 'Actual RUL%', position: 'insideBottom', offset: -10 }} />
                                        <YAxis type="number" dataKey="predicted" name="Predicted RUL%" stroke="#9CA3AF" domain={[0, 100]} label={{ value: 'Predicted RUL%', angle: -90, position: 'insideLeft' }} />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                        {/* Ideal diagonal line y=x */}
                                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke="#10B981" strokeWidth={2} strokeDasharray="5 5" />
                                        <Scatter
                                            name="Predictions"
                                            data={scatterData}
                                            fill="#3B82F6"
                                        />
                                    </ScatterChart>
                                </ResponsiveContainer>
                                <p className="text-sm text-gray-400 text-center mt-2">Green diagonal = ideal (y=x). Points near line = accurate predictions.</p>
                            </div>
                        );
                    })()}

                    {/* Performance Metrics Radar - Normalized Regression Scores */}
                    <div className="mt-6">
                        <h3 className="text-xl font-semibold mb-4">Normalized Performance Scores</h3>
                        <ResponsiveContainer width="100%" height={400}>
                            <RadarChart data={(() => {
                                // Normalize metrics to 0-100 scale (higher = better)
                                const maxMae = Math.max(...modelResults.map(m => m.mae || 0));
                                const maxRmse = Math.max(...modelResults.map(m => m.rmse || 0));
                                return modelResults.slice(0, 4).map(m => ({
                                    model: m.modelType,
                                    'MAE Score': maxMae > 0 ? (1 - (m.mae || 0) / maxMae) * 100 : 50,
                                    'RMSE Score': maxRmse > 0 ? (1 - (m.rmse || 0) / maxRmse) * 100 : 50,
                                    'R² Score': Math.max(0, Math.min(100, ((m.r2Score || 0) + 1) / 2 * 100))
                                }));
                            })()}>
                                <PolarGrid stroke="#374151" />
                                <PolarAngleAxis dataKey="model" stroke="#9CA3AF" />
                                <PolarRadiusAxis stroke="#9CA3AF" domain={[0, 100]} />
                                <Radar name="MAE Score" dataKey="MAE Score" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.5} />
                                <Radar name="RMSE Score" dataKey="RMSE Score" stroke="#EF4444" fill="#EF4444" fillOpacity={0.5} />
                                <Radar name="R² Score" dataKey="R² Score" stroke="#10B981" fill="#10B981" fillOpacity={0.5} />
                                <Legend />
                            </RadarChart>
                        </ResponsiveContainer>
                        <p className="text-sm text-gray-400 text-center mt-2">Higher scores = better performance (normalized to 0-100)</p>
                    </div>
                </>
            )}
        </div>
    );
};

export default ModelsTab;
