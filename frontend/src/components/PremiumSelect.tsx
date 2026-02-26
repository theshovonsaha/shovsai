import React, { useState, useRef, useEffect } from 'react';

interface PremiumSelectProps {
    value: string;
    options: Record<string, string[]>;
    onChange: (value: string) => void;
    label?: string;
    placeholder?: string;
}

export const PremiumSelect: React.FC<PremiumSelectProps> = ({ value, options, onChange, label, placeholder }) => {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleSelect = (provider: string, model: string) => {
        onChange(`${provider}:${model}`);
        setIsOpen(false);
    };

    // Helper to extract clean name for display
    const getDisplayValue = () => {
        if (!value) return null;
        if (value.includes(':')) return value.split(':')[1];
        return value;
    };

    const hasOptions = Object.values(options).some(list => list.length > 0);

    return (
        <div className="premium-select-container" ref={containerRef}>
            {label && <label className="premium-select-label">{label}</label>}
            <div
                className={`premium-select-trigger ${isOpen ? 'active' : ''}`}
                onClick={() => setIsOpen(!isOpen)}
            >
                <div className="premium-select-value">
                    {getDisplayValue() || <span className="placeholder">{placeholder || 'Select...'}</span>}
                </div>
                <div className="premium-select-arrow">▾</div>
            </div>

            {isOpen && (
                <div className="premium-select-dropdown">
                    {!hasOptions ? (
                        <div className="premium-select-no-options">No models available</div>
                    ) : (
                        Object.entries(options).map(([provider, models]) => (
                            models.length > 0 && (
                                <div key={provider} className="provider-group">
                                    <div className="provider-header">{provider.toUpperCase()}</div>
                                    {models.map((opt) => {
                                        const fullValue = `${provider}:${opt}`;
                                        const isSelected = value === fullValue || (provider === 'ollama' && value === opt);
                                        return (
                                            <div
                                                key={opt}
                                                className={`premium-select-option ${isSelected ? 'selected' : ''}`}
                                                onClick={() => handleSelect(provider, opt)}
                                            >
                                                <span className="opt-text">{opt}</span>
                                                {isSelected && <span className="opt-check">✓</span>}
                                            </div>
                                        );
                                    })}
                                </div>
                            )
                        ))
                    )}
                </div>
            )}
        </div>
    );
};
