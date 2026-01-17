import React from 'react';

/**
 * StatCard Component
 * Reusable statistics card with icon, value, and optional trend
 */
const StatCard = ({
    title,
    value,
    unit = '',
    icon: Icon,
    iconColor = 'text-blue-400',
    bgGradient = 'from-slate-800 to-slate-700',
    trend = null,
    trendUp = true,
    subtitle = null
}) => {
    return (
        <div className={`bg-gradient-to-br ${bgGradient} rounded-lg p-5 border border-slate-600`}>
            <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm text-gray-400">{title}</h4>
                {Icon && <Icon className={iconColor} size={20} />}
            </div>
            <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold">{value}</span>
                {unit && <span className="text-sm text-gray-400">{unit}</span>}
            </div>
            {trend !== null && (
                <div className={`text-sm mt-2 ${trendUp ? 'text-green-400' : 'text-red-400'}`}>
                    {trendUp ? '↑' : '↓'} {trend}
                </div>
            )}
            {subtitle && (
                <p className="text-xs text-gray-500 mt-2">{subtitle}</p>
            )}
        </div>
    );
};

export default StatCard;
