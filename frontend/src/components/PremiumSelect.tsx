import React, { useState, useRef, useEffect } from 'react';

interface PremiumSelectProps {
    value: string;
    options: string[];
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

    const handleSelect = (opt: string) => {
        onChange(opt);
        setIsOpen(false);
    };

    return (
        <div className="premium-select-container" ref={containerRef}>
            {label && <label className="premium-select-label">{label}</label>}
            <div
                className={`premium-select-trigger ${isOpen ? 'active' : ''}`}
                onClick={() => setIsOpen(!isOpen)}
            >
                <div className="premium-select-value">
                    {value || <span className="placeholder">{placeholder || 'Select...'}</span>}
                </div>
                <div className="premium-select-arrow">▾</div>
            </div>

            {isOpen && (
                <div className="premium-select-dropdown">
                    {options.length === 0 ? (
                        <div className="premium-select-no-options">No options available</div>
                    ) : (
                        options.map((opt) => (
                            <div
                                key={opt}
                                className={`premium-select-option ${opt === value ? 'selected' : ''}`}
                                onClick={() => handleSelect(opt)}
                            >
                                <span className="opt-text">{opt}</span>
                                {opt === value && <span className="opt-check">✓</span>}
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
};
