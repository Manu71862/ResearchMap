import React from 'react';

/**
 * ResearchMapLogo — Distinctive blue research network logo.
 * Concept: Interconnected scientific evidence nodes with a central verification hub.
 * Palette: Sophisticated deep blue, royal blue, sky blue, and white accents on transparent background.
 */
export default function ResearchMapLogo({ size = 32, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{ display: 'block', flexShrink: 0 }}
      aria-label="ResearchMap Logo"
    >
      <defs>
        <linearGradient id="rmEdgeGradComponent" x1="14" y1="12" x2="50" y2="54" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.95" />
          <stop offset="50%" stopColor="#2563eb" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#1d4ed8" stopOpacity="0.95" />
        </linearGradient>
        <filter id="rmGlowComponent" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      {/* Orbit ring */}
      <circle cx="32" cy="33" r="25" stroke="url(#rmEdgeGradComponent)" strokeWidth="1" strokeDasharray="2 3" opacity="0.4" />

      {/* Knowledge graph lattice */}
      <path
        d="M32 12 L50 22.5 L50 43.5 L32 54 L14 43.5 L14 22.5 Z"
        stroke="url(#rmEdgeGradComponent)"
        strokeWidth="2.2"
        strokeLinejoin="round"
      />

      {/* Central spokes */}
      <line x1="32" y1="12" x2="32" y2="33" stroke="url(#rmEdgeGradComponent)" strokeWidth="1.8" strokeLinecap="round" />
      <line x1="50" y1="22.5" x2="32" y2="33" stroke="url(#rmEdgeGradComponent)" strokeWidth="1.8" strokeLinecap="round" />
      <line x1="50" y1="43.5" x2="32" y2="33" stroke="url(#rmEdgeGradComponent)" strokeWidth="1.8" strokeLinecap="round" />
      <line x1="32" y1="54" x2="32" y2="33" stroke="url(#rmEdgeGradComponent)" strokeWidth="1.8" strokeLinecap="round" />
      <line x1="14" y1="43.5" x2="32" y2="33" stroke="url(#rmEdgeGradComponent)" strokeWidth="1.8" strokeLinecap="round" />
      <line x1="14" y1="22.5" x2="32" y2="33" stroke="url(#rmEdgeGradComponent)" strokeWidth="1.8" strokeLinecap="round" />

      {/* Outer Evidence Nodes */}
      <circle cx="32" cy="12" r="3.8" fill="#090d16" stroke="#60a5fa" strokeWidth="1.8" />
      <circle cx="32" cy="12" r="1.6" fill="#ffffff" />

      <circle cx="50" cy="22.5" r="3.4" fill="#090d16" stroke="#3b82f6" strokeWidth="1.8" />
      <circle cx="50" cy="22.5" r="1.4" fill="#93c5fd" />

      <circle cx="50" cy="43.5" r="3.8" fill="#090d16" stroke="#2563eb" strokeWidth="1.8" />
      <circle cx="50" cy="43.5" r="1.6" fill="#ffffff" />

      <circle cx="32" cy="54" r="3.4" fill="#090d16" stroke="#3b82f6" strokeWidth="1.8" />
      <circle cx="32" cy="54" r="1.4" fill="#93c5fd" />

      <circle cx="14" cy="43.5" r="3.8" fill="#090d16" stroke="#2563eb" strokeWidth="1.8" />
      <circle cx="14" cy="43.5" r="1.6" fill="#ffffff" />

      <circle cx="14" cy="22.5" r="3.4" fill="#090d16" stroke="#60a5fa" strokeWidth="1.8" />
      <circle cx="14" cy="22.5" r="1.4" fill="#93c5fd" />

      {/* Central Verification Synthesis Hub */}
      <circle cx="32" cy="33" r="5.5" fill="#0f172a" stroke="#38bdf8" strokeWidth="2.2" filter="url(#rmGlowComponent)" />
      <circle cx="32" cy="33" r="2.6" fill="#ffffff" />
    </svg>
  );
}
