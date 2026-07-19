import React, { useState, useEffect } from 'react';
import { MoonIcon, SunIcon } from './icons';
import '../App.css';

const ThemeToggle = () => {
    const [isDark, setIsDark] = useState(() => {
        const saved = localStorage.getItem('theme');
        return saved ? saved === 'dark' : true;
    });

    useEffect(() => {
        if (isDark) {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
        }
    }, [isDark]);

    return (
        <button
            className="theme-toggle"
            onClick={() => setIsDark(!isDark)}
            aria-label="Toggle theme"
            title={isDark ? "Dark Mode Active (Click for Light)" : "Light Mode Active (Click for Dark)"}
        >
            {isDark ? <MoonIcon size={20} color="#f59e0b" /> : <SunIcon size={20} color="#eab308" />}
        </button>
    );
};

export default ThemeToggle;
