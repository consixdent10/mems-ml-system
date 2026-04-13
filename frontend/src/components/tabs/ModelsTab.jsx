import React from 'react';
import {
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ScatterChart, Scatter, ReferenceLine, BarChart, Bar, Cell
} from 'recharts';

/**
 * Models Tab Component
 * Displays ML model training interface, leaderboard with accuracy,
 * fault classification results, and actual vs predicted chart
 */
const ModelsTab = ({
    modelResults,
    bestModel,
    predictionsSample,
    isTraining,
    trainModels,
    trainError,
    onDownloadModel,
    classifierResults,
    isTrainingClassifier,
    trainClassifier,
    classifierError
}) => {
    return (
        <div className="space-y-6">
            <div className="flex flex-col">
                <div className="flex justify-between items-center">
                    <h3 className="text-xl font-semibold">ML Model Training & Comparison</h3>
                    <div className="flex gap-3">
                        <button
                            onClick={trainModels}
                            disabled={isTraining}
                            className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-6 py-2 rounded-lg font-medium transition"
                        >
                            {isTraining ? 'Training...' : 'Train All Models'}
                        </button>
                        {bestModel && onDownloadModel && (
                            <button
                                onClick={onDownloadModel}
                                className="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg font-medium transition flex items-center gap-2"
                            >
                                ⬇️ Download Best Model
                            </button>
                        )}
                    </div>
                </div>
                {trainError && (
                    <div className="mt-3 p-3 rounded bg-red-900/40 border border-red-500 text-red-200 text-sm">
                        {trainError}
                    </div>
                )}
            </div>

            {modelResults.length > 0 && (
                <>
                    {/* Leaderboard Table with Accuracy */}
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
                                        <th className="text-right py-2 px-3 text-gray-400 font-bold text-yellow-300">Accuracy ↑</th>
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
                                            <td className="py-2 px-3 text-right font-bold text-yellow-300 text-base">
                                                {model.accuracy != null ? `${model.accuracy}%` : 'N/A'}
                                            </td>
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
                                🏆 Best Model: <strong>{bestModel}</strong> (Highest Prediction Accuracy)
                            </p>
                        )}
                    </div>

                    {/* Actual vs Predicted Scatter Plot */}
                    {predictionsSample && predictionsSample.actual && (() => {
                        const scatterData = predictionsSample.actual.map((a, i) => ({ actual: a, predicted: predictionsSample.predicted[i] }));
                        const bestModelData = modelResults.find(m => m.modelType === bestModel);
                        const bestR2 = bestModelData?.r2Score || 0;
                        const bestAcc = bestModelData?.accuracy || 0;
                        return (
                            <div className="mt-6">
                                <h3 className="text-xl font-semibold mb-2">Actual vs Predicted RUL (Best Model: {bestModel})</h3>
                                <p className="text-sm text-gray-400 mb-4">
                                    R²: <span className={bestR2 >= 0.7 ? 'text-green-400' : 'text-yellow-400'}>{bestR2.toFixed(4)}</span>
                                    {' | '}
                                    Prediction Accuracy: <span className="text-yellow-300 font-bold">{bestAcc}%</span>
                                </p>
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
                </>
            )}

            {/* ============ FAULT CLASSIFICATION SECTION ============ */}
            <div className="border-t border-slate-600 pt-6 mt-6">
                <div className="flex justify-between items-center mb-4">
                    <div>
                        <h3 className="text-xl font-semibold">🔍 Fault Classification (Multi-Dataset)</h3>
                        <p className="text-sm text-gray-400 mt-1">
                            Trains on CWRU + ADI MEMS + NASA IMS datasets to classify bearing fault type
                        </p>
                    </div>
                    <button
                        onClick={trainClassifier}
                        disabled={isTrainingClassifier}
                        className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 px-6 py-2 rounded-lg font-medium transition"
                    >
                        {isTrainingClassifier ? 'Training Classifier...' : 'Train Fault Classifier'}
                    </button>
                </div>

                {classifierError && (
                    <div className="p-3 rounded bg-red-900/40 border border-red-500 text-red-200 text-sm mb-4">
                        {classifierError}
                    </div>
                )}

                {classifierResults && (
                    <>
                        {/* Classifier Leaderboard */}
                        <div className="bg-slate-700 rounded-lg p-4 mb-6">
                            <h4 className="text-lg font-semibold mb-3 text-orange-400">🎯 Classifier Leaderboard</h4>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-slate-600">
                                            <th className="text-left py-2 px-3 text-gray-400">Classifier</th>
                                            <th className="text-right py-2 px-3 text-yellow-300 font-bold">Accuracy ↑</th>
                                            <th className="text-right py-2 px-3 text-gray-400">Precision</th>
                                            <th className="text-right py-2 px-3 text-gray-400">Recall</th>
                                            <th className="text-right py-2 px-3 text-gray-400">F1 Score</th>
                                            <th className="text-right py-2 px-3 text-gray-400">Time</th>
                                            <th className="text-center py-2 px-3 text-gray-400">Best</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {classifierResults.classifiers?.map((clf, idx) => (
                                            <tr key={idx} className={`border-b border-slate-600 ${clf.modelType === classifierResults.bestClassifier ? 'bg-orange-900/20' : ''}`}>
                                                <td className="py-2 px-3 font-medium">{clf.modelType}</td>
                                                <td className="py-2 px-3 text-right font-bold text-yellow-300 text-base">
                                                    {clf.accuracy}%
                                                </td>
                                                <td className="py-2 px-3 text-right">{(clf.precision * 100).toFixed(1)}%</td>
                                                <td className="py-2 px-3 text-right">{(clf.recall * 100).toFixed(1)}%</td>
                                                <td className="py-2 px-3 text-right">{(clf.f1Score * 100).toFixed(1)}%</td>
                                                <td className="py-2 px-3 text-right">{clf.trainingTime?.toFixed(2)}s</td>
                                                <td className="py-2 px-3 text-center">
                                                    {clf.modelType === classifierResults.bestClassifier && <span className="text-orange-400">🏅</span>}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {classifierResults.bestClassifier && (
                                <p className="mt-3 text-sm text-orange-400">
                                    🏅 Best Classifier: <strong>{classifierResults.bestClassifier}</strong>
                                    {classifierResults.crossValidation && (
                                        <span className="text-gray-400">
                                            {' '}| Cross-Validation: {classifierResults.crossValidation.accuracy}% 
                                            (±{classifierResults.crossValidation.std}%, {classifierResults.crossValidation.folds}-fold)
                                        </span>
                                    )}
                                </p>
                            )}
                        </div>

                        {/* Feature Importance Bar Chart */}
                        {classifierResults.featureImportance?.length > 0 && (
                            <div className="bg-slate-700 rounded-lg p-4 mb-6">
                                <h4 className="text-lg font-semibold mb-3 text-blue-400">📊 Feature Importance (Random Forest)</h4>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={classifierResults.featureImportance} layout="vertical" margin={{ left: 100 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis type="number" stroke="#9CA3AF" />
                                        <YAxis type="category" dataKey="feature" stroke="#9CA3AF" width={90} tick={{ fontSize: 12 }} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none' }} />
                                        <Bar dataKey="importance" fill="#F59E0B" radius={[0, 4, 4, 0]}>
                                            {classifierResults.featureImportance.map((entry, idx) => (
                                                <Cell key={idx} fill={idx < 3 ? '#F59E0B' : '#6B7280'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        )}

                        {/* Dataset Summary */}
                        {classifierResults.classSummary && (
                            <div className="bg-slate-700 rounded-lg p-4">
                                <h4 className="text-lg font-semibold mb-3 text-green-400">📋 Training Data Summary</h4>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <div className="text-2xl font-bold text-blue-400">{classifierResults.classSummary.totalSamples}</div>
                                        <div className="text-gray-400">Total Samples</div>
                                    </div>
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <div className="text-2xl font-bold text-green-400">{classifierResults.classSummary.labels?.length}</div>
                                        <div className="text-gray-400">Fault Classes</div>
                                    </div>
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <div className="text-2xl font-bold text-purple-400">{classifierResults.classSummary.datasetsUsed?.length}</div>
                                        <div className="text-gray-400">Datasets Used</div>
                                    </div>
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <div className="text-2xl font-bold text-yellow-400">
                                            {classifierResults.classSummary.trainSize}/{classifierResults.classSummary.testSize}
                                        </div>
                                        <div className="text-gray-400">Train/Test Split</div>
                                    </div>
                                </div>
                                <div className="mt-3 text-xs text-gray-500">
                                    Datasets: {classifierResults.classSummary.datasetsUsed?.join(', ')} | 
                                    Classes: {classifierResults.classSummary.labels?.join(', ')}
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default ModelsTab;
